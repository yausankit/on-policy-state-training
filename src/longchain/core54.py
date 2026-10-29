"""R54 shared code: data loading, frozen document encoding, GPU chain retrieval, gold states and negative pools.

Reuses the exact R29 encoder, state text (text19) and query prompt, so training and inference match the
MuSiQue main experiments. Data layout: ROOT/data/<dataset>/inputs/{train,dev,test}/ (R26 format; train and dev
share one corpus, see prepare_data54.py).
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Encoder, text19, norm  # noqa: E402

ROOT = Path(os.environ.get('LONGCHAIN_ROOT', Path(__file__).resolve().parent))
MAX_STATES_PER_QUESTION = 4      # cap on gold states per question (comparisons have 6-8 gold passages)
NEG_POOL = 100


def read(p):
    return json.loads(Path(p).read_text())


def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False) + '\n')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def split_dir(ds, split):
    return ROOT / 'data' / ds / 'inputs' / split


def load_split(ds, split):
    d = split_dir(ds, split)
    questions = read(d / 'questions.json')
    raw = {r['_id']: r for r in read(d / 'raw.json')}
    gold = {k: [int(x) for x in v] for k, v in read(d / 'ordered_gold.json').items()}
    corpus_file = d / 'corpus.json' if (d / 'corpus.json').exists() else split_dir(ds, 'train') / 'corpus.json'
    corp = [norm(c['title']) + '\n' + norm(c['text']) for c in read(corpus_file)]
    return questions, raw, gold, corp


def emb_path(ds, split):
    d = split_dir(ds, split)
    return d / 'doc_emb.npy' if (d / 'corpus.json').exists() else split_dir(ds, 'train') / 'doc_emb.npy'


def encode_corpus(corp, path):
    """Frozen R8 document encoding: raw passage text, no prompt, last-token pooling, L2-normalised, fp16."""
    enc = Encoder(frozen=True)
    order = np.argsort([len(c) for c in corp], kind='stable')
    out = np.empty((len(corp), 1024), np.float16)
    with torch.inference_mode():
        for s in range(0, len(corp), 16):
            ids = order[s:s + 16]
            toks = enc.tokenizer([corp[i] for i in ids], padding=True, truncation=True, max_length=32768,
                                 return_tensors='pt').to('cuda')
            h = enc.model(**toks).last_hidden_state[:, -1]
            out[ids] = torch.nn.functional.normalize(h.float(), p=2, dim=1).cpu().numpy().astype(np.float16)
            if s % 64000 == 0:
                print('encode', s, len(corp), flush=True)
    enc.release()
    np.save(path, out)
    return out


def load_docs(ds, split):
    return torch.as_tensor(np.load(emb_path(ds, split)), device='cuda', dtype=torch.float16)


def encode_queries(enc, texts):
    return torch.as_tensor(enc.encode(texts), device='cuda', dtype=torch.float16)


def chain_retrieve(enc, questions, corp, D, T, gold=None, single=False, frozen_view=False, k_out=50):
    """Greedy chain retrieval (R29 S-only), V** readout. Returns {qid: ranking}, and the visited states
    (question + passages written so far, with the remaining gold) when gold is given."""
    rankings, states = {}, []
    ctx = enc.model.disable_adapter() if frozen_view else torch.inference_mode()
    with ctx, torch.inference_mode():
        for s in range(0, len(questions), 32):
            qs = questions[s:s + 32]
            written = [[] for _ in qs]
            agg = torch.full((len(qs), D.shape[0]), -1e4, device='cuda', dtype=torch.float16)
            for t in range(1 if single else T):
                texts = [q['question'] if single else text19(q['question'], w, corp) for q, w in zip(qs, written)]
                if gold is not None:
                    for q, w, text in zip(qs, written, texts):
                        g = gold[q['qid']]
                        rem = [d for d in g if d not in w]
                        if rem:
                            states.append(dict(qid=q['qid'], evidence=list(w), gold=g, remaining=rem,
                                               positives=rem[:1], text=text, node=t))
                qv = encode_queries(enc, texts)
                sc = qv @ D.T
                for i, w in enumerate(written):
                    if w:
                        sc[i, torch.as_tensor(w, device='cuda')] = -1e4
                agg = torch.maximum(agg, sc)
                if single:
                    break
                choice = sc.argmax(dim=1).tolist()
                for w, c in zip(written, choice):
                    w.append(int(c))
            top = agg.topk(k_out + T, dim=1).indices.tolist()
            for q, w, row in zip(qs, written, top):
                seen = set(w)
                rankings[q['qid']] = list(w) + [d for d in row if d not in seen][:max(0, k_out - len(w))]
    return rankings, states


def gold_states(questions, gold, corp, kind, rng):
    """RN: E = random subset of the gold, target = next gold not in E (hop order).
    MDR: E = gold prefix, target = next gold. At most MAX_STATES_PER_QUESTION subset sizes per question."""
    rows = []
    for q in questions:
        g = gold[q['qid']]
        sizes = list(range(len(g)))
        if len(sizes) > MAX_STATES_PER_QUESTION:
            sizes = sorted(rng.sample(sizes, MAX_STATES_PER_QUESTION))
        for k in sizes:
            e = g[:k] if kind == 'MDR' else sorted(rng.sample(g, k), key=g.index)
            rem = [d for d in g if d not in e]
            rows.append(dict(qid=q['qid'], evidence=e, gold=g, remaining=rem, positives=rem[:1],
                             text=text19(q['question'], e, corp), node=len(e)))
    return rows


def negative_pools(enc, rows, D, frozen_view=True):
    """Top-NEG_POOL passages for each state text under the frozen encoder, excluding every gold passage of the
    question (R29/R37-1 rule)."""
    ctx = enc.model.disable_adapter() if frozen_view else torch.inference_mode()
    with ctx, torch.inference_mode():
        for s in range(0, len(rows), 64):
            part = rows[s:s + 64]
            sc = encode_queries(enc, [r['text'] for r in part]) @ D.T
            for i, r in enumerate(part):
                sc[i, torch.as_tensor(r['gold'], device='cuda')] = -1e4
            top = sc.topk(NEG_POOL, dim=1).indices.tolist()
            for r, t in zip(part, top):
                r['negative_top100'] = [int(x) for x in t]
    return rows


def metrics(rankings, gold, raw, ks=(5, 10, 20, 50)):
    per_type, overall = {}, []
    for qid, rk in rankings.items():
        g = gold[qid]
        m = {}
        for k in ks:
            hit = len(set(g) & set(rk[:k]))
            m[f'R@{k}'] = 100 * hit / len(g)
            m[f'C@{k}'] = 100.0 * (hit == len(g))
        overall.append(m)
        per_type.setdefault(raw[qid]['type'], []).append(m)
    avg = lambda rows: {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return dict(overall=avg(overall), n=len(overall), by_type={t: dict(n=len(v), **avg(v)) for t, v in per_type.items()})
