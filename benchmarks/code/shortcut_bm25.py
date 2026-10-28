"""Retrieval shortcut check: is the evidence reachable from the question alone?

Runs on 239 (r53env). For each passed candidate, BM25 over the split's shared corpus
(all original context paragraphs, de-duplicated by title). A chain is a shortcut if its
final-hop gold paragraph is in the question-only top-10; a comparison is a shortcut if
all of its gold paragraphs are in the top-10.
Usage: shortcut_bm25.py <raw_split.json> <candidates.jsonl> <results.jsonl> <out.jsonl>
"""
import json, sys
import bm25s, Stemmer

raw, cand_path, res_path, out_path = sys.argv[1:5]
titles, texts, seen = [], [], set()
for r in json.load(open(raw)):
    for t, sents in r['context']:
        if t not in seen:
            seen.add(t)
            titles.append(t)
            texts.append(t + ' ' + ' '.join(sents))
cands = {c['cid']: c for c in map(json.loads, open(cand_path))}
for c in cands.values():                      # gold paragraphs are part of the corpus too
    for h in c['hops']:
        if h['gold_title'] not in seen:
            seen.add(h['gold_title'])
            titles.append(h['gold_title'])
            texts.append(h['gold_title'] + ' ' + h['gold_text'])
print('corpus', len(titles), flush=True)
stem = Stemmer.Stemmer('english')
retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(texts, stopwords='en', stemmer=stem, show_progress=False), show_progress=False)
results = [json.loads(l) for l in open(res_path)]
passed = [r for r in results if r['passed']]
docs, _ = retriever.retrieve(bm25s.tokenize([r['question'] for r in passed], stopwords='en', stemmer=stem,
                                             show_progress=False), k=10, show_progress=False)
with open(out_path, 'w') as f:
    for r, top in zip(passed, docs):
        c = cands[r['cid']]
        top_titles = {titles[i] for i in top}
        gold = [h['gold_title'] for h in c['hops']]
        if c['kind'] == 'chain':
            shortcut = gold[-1] in top_titles
        else:
            shortcut = set(gold) <= top_titles
        f.write(json.dumps(dict(cid=r['cid'], bm25_top10_hits=[g in top_titles for g in gold],
                                bm25_shortcut=shortcut)) + '\n')
print('checked', len(passed), flush=True)
