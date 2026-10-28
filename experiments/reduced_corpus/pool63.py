"""R63 candidate-pool corpus for HippoRAG 2 on the long-chain test sets.

Pool = every gold passage of the test questions + top-100 passages of each question under the frozen encoder
(single query). Writes a new R54 dataset <dataset>_pool (test split) with the pool corpus, remapped ordered gold
and the matching rows of the existing document embeddings, so every retriever can be re-evaluated on the pool.
Usage: pool63.py <dataset>   (e.g. hotpotqa_long_ref)
"""
import json, sys
import numpy as np
import torch
from core54 import ROOT, Encoder, read, save, load_split, load_docs, encode_queries, split_dir

ds = sys.argv[1]
qs, raw, gold, corp = load_split(ds, 'test')
D = load_docs(ds, 'test')
enc = Encoder(frozen=True)
keep = set(d for g in gold.values() for d in g)
with torch.inference_mode():
    for s in range(0, len(qs), 32):
        sc = encode_queries(enc, [q['question'] for q in qs[s:s + 32]]) @ D.T
        keep.update(int(x) for x in sc.topk(100, dim=1).indices.flatten().tolist())
ids = sorted(keep)
new = {old: i for i, old in enumerate(ids)}
src = split_dir(ds, 'test'); out = ROOT / 'data' / (ds.replace('_ref', '') + '_pool') / 'inputs' / 'test'
out.mkdir(parents=True, exist_ok=True)
full = read(src / 'corpus.json')
json.dump([full[i] for i in ids], open(out / 'corpus.json', 'w'), ensure_ascii=False)
json.dump(read(src / 'questions.json'), open(out / 'questions.json', 'w'), ensure_ascii=False)
json.dump(read(src / 'raw.json'), open(out / 'raw.json', 'w'), ensure_ascii=False)
json.dump({k: [new[d] for d in v] for k, v in gold.items()}, open(out / 'ordered_gold.json', 'w'))
np.save(out / 'doc_emb.npy', np.load(src / 'doc_emb.npy')[ids])
save(out / 'pool_stats.json', dict(source=ds, source_corpus=len(full), pool=len(ids),
                                   gold_passages=len(set(d for g in gold.values() for d in g)), top_k=100))
print(ds, 'pool', len(ids), flush=True)
