"""Fixed-state recovery probe on MuSiQue (Section 5.4, Figure 3b, Table 5).

States are taken from the saved chain traces of one source retriever: (question, passages written before step t)
for every step at which a gold passage is still missing. Every probed encoder then scores the same states and writes
its next passage (argmax over the corpus minus the history, as in chain retrieval). Reported: rate at which that write
is a missing gold passage, by the number of non-gold passages in the history. Usage:
probe.py <traces_dir> <source_name> <out.json> <name=adapter_path> ...
"""
import json, sys, collections
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Encoder, text19, norm, ROOT

D29 = ROOT / 'inputs/design'
corp_rows = json.load(open(D29 / 'corpus.json')); raw = json.load(open(D29 / 'raw.json'))
corp = [norm(c['title']) + '\n' + norm(c['text']) for c in corp_rows]
idx = {}
for i, c in enumerate(corp_rows): idx.setdefault((norm(c['title']), norm(c['text'])), i)
gold = {r['id']: [idx[(norm(p['title']), norm(p['paragraph_text']))] for p in r['paragraphs'] if p['is_supporting']] for r in raw}
question = {r['id']: r['question'] for r in raw}
D = torch.as_tensor(np.load(D29 / 'doc_emb.npy'), device='cuda').float()
traces_dir, source, out = sys.argv[1], sys.argv[2], sys.argv[3]
traces = json.load(open(Path(traces_dir) / f'{source}.json'))
states = []
for t in traces:
    g = set(gold[t['qid']]); hist = []
    for s in range(4):
        w = t['writes'][str(s)][0]
        if g - set(hist):
            states.append((t['qid'], list(hist), sum(d not in g for d in hist), w))
        hist.append(w)
result = dict(source=source, states=len(states))
for spec in sys.argv[4:]:
    name, path = spec.split('=', 1)
    enc = Encoder(path)
    picks = []
    with torch.inference_mode():
        for b in range(0, len(states), 32):
            part = states[b:b + 32]
            q = torch.as_tensor(enc.encode([text19(question[qid], h, corp) for qid, h, _, _ in part]), device='cuda').float()
            sc = q @ D.T
            for i, (_, h, _, _) in enumerate(part):
                if h: sc[i, torch.as_tensor(h, device='cuda')] = -1e9
            picks += sc.argmax(dim=1).tolist()
    rec = collections.defaultdict(list); agree = []
    for (qid, h, wrong, w), p in zip(states, picks):
        rec[min(wrong, 2)].append(p in set(gold[qid]) - set(h)); agree.append(p == w)
    result[name] = {str(k): dict(rate=round(100 * sum(v) / len(v), 1), n=len(v)) for k, v in sorted(rec.items())}
    result[name]['agreement_with_source_writes'] = round(100 * sum(agree) / len(agree), 1)
    enc.release(); torch.cuda.empty_cache()
json.dump(result, open(out, 'w'), indent=1); print(json.dumps(result), flush=True)
