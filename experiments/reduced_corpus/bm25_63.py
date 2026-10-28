"""R63 BM25 baseline (question only) on MuSiQue 1000 and the refined long-chain test sets. Runs on 239 (r53env).
bm25s with English stopwords and the PyStemmer English stemmer; passage text = title + text."""
import json, sys
import bm25s, Stemmer
stem = Stemmer.Stemmer('english')
def norm(t): return ' '.join(str(t or '').split())
def run(name, corpus, questions, gold, types):
    bm = bm25s.BM25(); bm.index(bm25s.tokenize([norm(c['title']) + ' ' + norm(c['text']) for c in corpus], stopwords='en',
                                              stemmer=stem, show_progress=False), show_progress=False)
    top, _ = bm.retrieve(bm25s.tokenize([q for _, q in questions], stopwords='en', stemmer=stem, show_progress=False),
                         k=10, show_progress=False)
    rows = []
    for (qid, _), t in zip(questions, top):
        g = set(gold[qid]); t = [int(x) for x in t]
        rows.append({f'R@{k}': 100 * len(g & set(t[:k])) / len(g) for k in (5, 10)} | {f'C@{k}': 100.0 * (g <= set(t[:k])) for k in (5, 10)})
    avg = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in ('R@5', 'R@10', 'C@5', 'C@10')}
    print(name, len(rows), avg, flush=True)
mq_c = json.load(open(sys.argv[1])); mq = json.load(open(sys.argv[2]))
idx = {}
for i, c in enumerate(mq_c): idx.setdefault((norm(c['title']), norm(c['text'])), i)
gold = {r['id']: [idx[(norm(p['title']), norm(p['paragraph_text']))] for p in r['paragraphs'] if p['is_supporting']] for r in mq}
run('musique', mq_c, [(r['id'], r['question']) for r in mq], gold, None)
for ds in ('hotpotqa_long', '2wiki_long'):
    d = f'{sys.argv[3]}/{ds}/test'
    corpus = json.load(open(f'{d}/corpus.json')); qs = json.load(open(f'{d}/questions.json')); g = json.load(open(f'{d}/ordered_gold.json'))
    run(ds, corpus, [(q['qid'], q['question']) for q in qs], g, None)
