"""Retrieval shortcut check for HotpotQA-Long over the full 2017 abstracts corpus (5.2M pages).
Builds (or loads) a BM25 index once; a chain is a shortcut if its final page is in the question-only top-10.
Usage: shortcut_bm25_hotpot.py <candidates.jsonl> <results.jsonl> <out.jsonl>
"""
import json, sys
from pathlib import Path
import bm25s, Stemmer
import build_hotpot as bh

INDEX = bh.ROOT / 'claude/hotpotqa/bm25_index'
stem = Stemmer.Stemmer('english')
if (INDEX / 'titles.json').exists():
    retriever = bm25s.BM25.load(str(INDEX))
    titles = json.load(open(INDEX / 'titles.json'))
else:
    pages = bh.load_pages()
    titles = list(pages)
    texts = [t + ' ' + ' '.join(pages[t][0]) for t in titles]
    del pages
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords='en', stemmer=stem, show_progress=False), show_progress=False)
    retriever.save(str(INDEX))
    json.dump(titles, open(INDEX / 'titles.json', 'w'))
print('corpus', len(titles), flush=True)
if len(sys.argv) < 4:
    sys.exit(0)
cand_path, res_path, out_path = sys.argv[1:4]
cands = {c['cid']: c for c in map(json.loads, open(cand_path))}
passed = [r for r in map(json.loads, open(res_path)) if r['passed']]
docs, _ = retriever.retrieve(bm25s.tokenize([r['question'] for r in passed], stopwords='en', stemmer=stem,
                                             show_progress=False), k=10, show_progress=False)
with open(out_path, 'w') as f:
    for r, top in zip(passed, docs):
        c = cands[r['cid']]
        top_titles = {titles[i] for i in top}
        gold = [h['gold_title'] for h in c['hops']]
        f.write(json.dumps(dict(cid=r['cid'], bm25_top10_hits=[g in top_titles for g in gold],
                                bm25_shortcut=gold[-1] in top_titles)) + '\n')
print('checked', len(passed), flush=True)
