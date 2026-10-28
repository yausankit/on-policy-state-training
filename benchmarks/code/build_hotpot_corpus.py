"""Closed retrieval corpus for HotpotQA-Long (like 2Wiki-Long): original HotpotQA contexts of the source split
+ gold pages of every chain + the top-20 full-Wikipedia BM25 neighbours of each gold page as hard distractors.
Usage: build_hotpot_corpus.py <raw_split.json> <candidates.jsonl> <out_corpus.jsonl>   (runs on 239, r53env)"""
import json, sys
import bm25s, Stemmer
import build_hotpot as bh

raw, cand_path, out_path = sys.argv[1:4]
idx = bh.ROOT / 'claude/hotpotqa/bm25_index'
retriever = bm25s.BM25.load(str(idx))
wtitles = json.load(open(idx / 'titles.json'))
pages = bh.load_pages()
corpus = {}
for r in json.load(open(raw)):
    for t, s in r['context']:
        corpus.setdefault(t, t + ' ' + ''.join(s))
n_ctx = len(corpus)
gold = {}
for c in map(json.loads, open(cand_path)):
    for h in c['hops']:
        gold[h['gold_title']] = h['gold_title'] + ' ' + h['gold_text']
for t, x in gold.items():
    corpus[t] = x                                   # gold pages use the abstract text the chain was built on
stem = Stemmer.Stemmer('english')
gt = list(gold)
docs, _ = retriever.retrieve(bm25s.tokenize([gold[t][:1000] for t in gt], stopwords='en', stemmer=stem,
                                            show_progress=False), k=21, show_progress=False)
added = 0
for t, row in zip(gt, docs):
    for i in row:
        nt = wtitles[i]
        if nt != t and nt not in corpus and nt in pages:
            corpus[nt] = nt + ' ' + ' '.join(pages[nt][0]); added += 1
with open(out_path, 'w') as f:
    for t, x in corpus.items():
        f.write(json.dumps({'title': t, 'text': x}, ensure_ascii=False) + '\n')
print(f'contexts {n_ctx}  gold pages {len(gold)}  hard distractors added {added}  total {len(corpus)}', flush=True)
