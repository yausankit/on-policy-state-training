"""Refined R53 benchmark split (step 3 of the refinement): 4x closed corpus, BM25 shortcut check, sampling.

Runs on 239 (r53env). Pool = verified questions (passed the LLM gate, naturalness >= 4, and the independent
gpt-5.4-mini re-check). Corpus = every gold page of the pool + full-pool BM25 nearest neighbours of each gold page
(round-robin over gold pages) + random fill, up to a fixed size. Neighbour source: HotpotQA 2017 abstracts for
HotpotQA-Long, all original 2WikiMultiHopQA context paragraphs (train + dev) for 2Wiki-Long. A question is dropped
if question-only BM25 over this corpus puts its final-hop gold (comparisons: all gold) in the top 10.
Sampling: balanced over question types; for 2Wiki-Long at most 40% genealogy questions.
Usage: finalize_refined.py <dataset> <split> <target_n> <corpus_size> <pool.json>
       (pool.json = [[source_split, cid], ...] with candidates/results/verify files under full/ and refine/)
"""
import collections, json, random, sys
from pathlib import Path
import bm25s, Stemmer

HERE = Path(__file__).parent
KIN = {'father', 'mother', 'spouse', 'child', 'sibling'}
stem = Stemmer.Stemmer('english')
dataset, split, target, size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pool = json.load(open(sys.argv[5]))
key = f'{dataset}_{split}'


def kin(c):
    if c['kind'] == 'comparison':
        return c['relation'] in KIN
    return c['kind'] == 'chain' and all(h['relation'] in KIN for h in c['hops'][:-1])


def bucket(c):
    return 'comparison' if c['kind'] == 'comparison' else f"chain{c['hops_n']}"


# ---- verified pool
cands, results, verified = {}, {}, {}
for src in sorted({s for s, _ in pool}):
    cands.update({c['cid']: c for c in map(json.loads, open(HERE / 'full' / f'{dataset}_{src}_candidates.jsonl'))})
    results.update({r['cid']: r for r in map(json.loads, open(HERE / 'full' / f'{dataset}_{src}_results.jsonl'))})
    vf = HERE / 'refine' / f'{key}_{src}_verify.jsonl'
    verified.update({v['cid']: v for v in map(json.loads, open(vf))})
source = {cid: s for s, cid in pool}
ok = [cid for _, cid in pool if verified.get(cid, {}).get('verified')]

# ---- corpus
corpus = {}
for cid in ok:
    for h in cands[cid]['hops']:
        corpus[h['gold_title']] = h['gold_text']
gold_titles = list(corpus)
if dataset == 'hotpotqa':
    import build_hotpot as bh
    idx = bh.ROOT / 'claude/hotpotqa/bm25_index'
    big = bm25s.BM25.load(str(idx)); big_titles = json.load(open(idx / 'titles.json'))
    pages = bh.load_pages(); big_text = lambda t: ' '.join(pages[t][0])
else:
    texts = {}
    for f in ('train.json', 'dev.json'):
        for r in json.load(open('/home/dmir/qiuchenjie2023/revision_20260928/round17/data_prep/2wiki/raw/' + f)):
            for t, s in r['context']:
                texts.setdefault(t, ' '.join(s))
    big_titles = list(texts); big_text = texts.get
    big = bm25s.BM25(); big.index(bm25s.tokenize([t + ' ' + texts[t] for t in big_titles], stopwords='en', stemmer=stem,
                                                show_progress=False), show_progress=False)
docs, _ = big.retrieve(bm25s.tokenize([(g + ' ' + corpus[g])[:1000] for g in gold_titles], stopwords='en', stemmer=stem,
                                      show_progress=False), k=200, show_progress=False)
n_gold = len(corpus)
for rank in range(200):
    if len(corpus) >= size:
        break
    for row in docs:
        if len(corpus) >= size:
            break
        t = big_titles[row[rank]]
        if t not in corpus and big_text(t):
            corpus[t] = big_text(t)
n_neigh = len(corpus) - n_gold
rest = [t for t in big_titles if t not in corpus]
random.Random(0).shuffle(rest)
for t in rest:
    if len(corpus) >= size:
        break
    corpus[t] = big_text(t)
titles = list(corpus)
pos = {t: i for i, t in enumerate(titles)}

# ---- BM25 shortcut check against this corpus
bm = bm25s.BM25(); bm.index(bm25s.tokenize([t + ' ' + corpus[t] for t in titles], stopwords='en', stemmer=stem,
                                         show_progress=False), show_progress=False)
qs = [results[cid]['question'] for cid in ok]
tops, _ = bm.retrieve(bm25s.tokenize(qs, stopwords='en', stemmer=stem, show_progress=False), k=10, show_progress=False)
kept = []
for cid, top in zip(ok, tops):
    c = cands[cid]; top_t = {titles[i] for i in top}; gold = [h['gold_title'] for h in c['hops']]
    if not ((gold[-1] in top_t) if c['kind'] != 'comparison' else set(gold) <= top_t):
        kept.append(cid)

# ---- balanced sampling (2Wiki: at most 40% genealogy)
rng = random.Random(0)
by = collections.defaultdict(list)
for cid in kept:
    by[(bucket(cands[cid]), dataset == '2wiki' and kin(cands[cid]))].append(cid)
for v in by.values():
    rng.shuffle(v)
cap_kin = int(0.4 * target) if dataset == '2wiki' else target
chosen, n_kin = [], 0
keys = sorted(by)
while len(chosen) < target and any(by[k] for k in keys):
    progressed = False
    for k in keys:
        if len(chosen) >= target or not by[k]:
            continue
        if k[1] and n_kin >= cap_kin:
            continue
        chosen.append(by[k].pop()); n_kin += k[1]; progressed = True
    if not progressed:
        break

# ---- write
out = HERE / 'refined' / f'{dataset}_long' / split
out.mkdir(parents=True, exist_ok=True)
questions, raw, ordered = [], [], {}
for cid in chosen:
    c, r = cands[cid], results[cid]
    questions.append({'qid': cid, 'question': r['question']})
    ordered[cid] = [pos[g] for g in dict.fromkeys(h['gold_title'] for h in c['hops'])]
    ev = [dict({'title': h['gold_title']}, **({'subject': h['subject'], 'relation': h['relation'], 'object': h['object']}
                                            if 'relation' in h else {'link_sentence': h.get('link_sentence')}))
          for h in c['hops']]
    raw.append({'_id': cid, 'question': r['question'], 'answer': r['answer'], 'type': bucket(c), 'hops': c['hops_n'],
                'genealogy': kin(c) if dataset == '2wiki' else None, 'source_split': source[cid],
                'origin_split': 'original-dev' if source[cid] == 'test' else 'original-train',
                'evidence': ev, 'naturalness': r.get('naturalness'),
                'closed_book_answerable': r.get('tags', {}).get('closed_book_answerable')})
json.dump(questions, open(out / 'questions.json', 'w'), ensure_ascii=False)
json.dump(raw, open(out / 'raw.json', 'w'), ensure_ascii=False)
json.dump(ordered, open(out / 'ordered_gold.json', 'w'))
json.dump([{'title': t, 'text': corpus[t]} for t in titles], open(out / 'corpus.json', 'w'), ensure_ascii=False)
stats = dict(dataset=dataset, split=split, pool=len(pool), verified=len(ok), not_shortcut=len(kept), selected=len(chosen),
             by_type=collections.Counter(bucket(cands[c]) for c in chosen),
             genealogy=sum(kin(cands[c]) for c in chosen) if dataset == '2wiki' else None,
             origin=collections.Counter(source[c] for c in chosen), corpus=len(titles), corpus_gold_pages=n_gold,
             corpus_bm25_neighbours=n_neigh, corpus_random=len(titles) - n_gold - n_neigh)
json.dump(stats, open(out / 'stats.json', 'w'), indent=1, default=dict)
print(json.dumps(stats, default=dict), flush=True)
