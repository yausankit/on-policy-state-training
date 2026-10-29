"""Appendix F.3: beam decoding for chain dense retrieval with the matched teacher-forced and on-policy encoders.
Each beam is a partial chain scored by the sum of the similarities of its passages. At each step every beam is
encoded, its top-B passages outside its history are expanded, and the B best partial chains are kept. The ranking
is the best final chain, then the remaining passages by their maximum score over all encoded states (the readout
of greedy decoding, which beam width 1 reproduces). Usage:
beam.py <dataset musique|hotpotqa_long|2wiki_long> <width> <T> <out.json> name=adapter ..."""
import sys, json, time, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from core import Encoder, text19, readout, norm
ds, B, T, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ROOT = Path(os.environ.get('OPST_ROOT', Path(__file__).resolve().parents[2] / 'src')); LONG = Path(os.environ.get('LONGCHAIN_DATA', 'data'))
if ds == 'musique':
    d = ROOT / 'inputs/design'; corp_rows = json.load(open(d / 'corpus.json')); raw = json.load(open(d / 'raw.json'))
    idx = {}
    for i, c in enumerate(corp_rows): idx.setdefault((norm(c['title']), norm(c['text'])), i)
    qs = [(r['id'], r['question']) for r in raw]
    gold = {r['id']: [idx[(norm(p['title']), norm(p['paragraph_text']))] for p in r['paragraphs'] if p['is_supporting']] for r in raw}
    emb = d / 'doc_emb.npy'
else:
    d = LONG / f'{ds}_ref/inputs/test'; corp_rows = json.load(open(d / 'corpus.json'))
    qs = [(q['qid'], q['question']) for q in json.load(open(d / 'questions.json'))]
    gold = {k: [int(x) for x in v] for k, v in json.load(open(d / 'ordered_gold.json')).items()}
    emb = d / 'doc_emb.npy'
corp = [norm(c['title']) + '\n' + norm(c['text']) for c in corp_rows]
D = torch.as_tensor(np.load(emb), device='cuda', dtype=torch.float16)
res = dict(dataset=ds, width=B, T=T, questions=len(qs))
for spec in sys.argv[5:]:
    name, path = spec.split('=', 1); enc = Encoder(path); enc.model.eval(); rankings = {}; encodings = 0; tick = time.monotonic()
    with torch.inference_mode():
        for s in range(0, len(qs), 16):
            part = qs[s:s + 16]
            beams = [[((), 0.0)] for _ in part]
            agg = torch.full((len(part), D.shape[0]), -1e4, device='cuda', dtype=torch.float32)
            for t in range(T):
                flat = [(i, h, sc) for i, bs in enumerate(beams) for h, sc in bs]
                q = torch.as_tensor(enc.encode([text19(part[i][1], list(h), corp) for i, h, _ in flat]), device='cuda', dtype=torch.float16)
                encodings += len(flat)
                S = (q @ D.T).float()
                for k, (i, h, _) in enumerate(flat):
                    if h: S[k, torch.as_tensor(list(h), device='cuda')] = -1e4
                for k, (i, h, _) in enumerate(flat): agg[i] = torch.maximum(agg[i], S[k])
                top = S.topk(B, dim=1)
                cand = [[] for _ in part]
                for k, (i, h, sc) in enumerate(flat):
                    for v, dd in zip(top.values[k].tolist(), top.indices[k].tolist()): cand[i].append((h + (dd,), sc + v))
                beams = []
                for c in cand:
                    seen, keep = set(), []
                    for h, sc in sorted(c, key=lambda x: -x[1]):
                        if frozenset(h) in seen: continue
                        seen.add(frozenset(h)); keep.append((h, sc))
                        if len(keep) == B: break
                    beams.append(keep)
            for i, (qid, _) in enumerate(part):
                best = beams[i][0][0]
                rankings[qid] = readout({t: [dd] for t, dd in enumerate(best)}, agg[i].cpu().numpy())
    secs = time.monotonic() - tick
    m = {}
    for k in (5, 10):
        r = [len(set(gold[q]) & set(rankings[q][:k])) / len(gold[q]) for q, _ in qs]; c = [set(gold[q]) <= set(rankings[q][:k]) for q, _ in qs]
        m[f'R@{k}'] = round(100 * float(np.mean(r)), 1); m[f'C@{k}'] = round(100 * float(np.mean(c)), 1)
    m['encodings_per_question'] = round(encodings / len(qs), 2); m['seconds_per_question'] = round(secs / len(qs), 3)
    res[name] = m; print(name, json.dumps(m), flush=True)
    json.dump(rankings, open(out.replace('.json', f'_{name}_rank.json'), 'w'))
    enc.release(); torch.cuda.empty_cache()
json.dump(res, open(out, 'w'), indent=1)
