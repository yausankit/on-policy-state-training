"""Question-only BM25 over a corpus JSONL; reports R@k / C@k overall and by group. Usage: <corpus> <questions> <group_key>"""
import json, sys, collections
import bm25s, Stemmer
corpus_path, q_path, gk = sys.argv[1:4]
titles, texts = [], []
for l in open(corpus_path):
    r = json.loads(l); titles.append(r['title']); texts.append(r['text'])
qs = [json.loads(l) for l in open(q_path)]
stem = Stemmer.Stemmer('english')
r = bm25s.BM25(); r.index(bm25s.tokenize(texts, stopwords='en', stemmer=stem, show_progress=False), show_progress=False)
docs, _ = r.retrieve(bm25s.tokenize([q['question'] for q in qs], stopwords='en', stemmer=stem, show_progress=False), k=50, show_progress=False)
rows = []
for q, d in zip(qs, docs):
    ranked = [titles[i] for i in d]; m = {}
    for k in (5, 10, 20, 50):
        hit = sum(g in set(ranked[:k]) for g in q['gold']); m[f'R@{k}'] = hit / len(q['gold']); m[f'C@{k}'] = float(hit == len(q['gold']))
    rows.append((q[gk], m))
rep = lambda lab, rs: print(lab, ' '.join(f'{k}={100 * sum(x[k] for x in rs) / len(rs):.1f}' for k in rs[0]))
print('corpus', len(titles)); rep('BM25', [m for _, m in rows])
for g in sorted({g for g, _ in rows}, key=str): rep(f'  {g}', [m for gg, m in rows if gg == g])
