"""Assemble a final split of 2Wiki-Long / HotpotQA-Long in the R26 input format.

Runs on 239 (r53env). Steps: build the split's closed corpus (original contexts of the source split + gold
pages; for HotpotQA also the top-20 full-Wikipedia BM25 neighbours of every gold page), run the
question-only BM25 shortcut check against that corpus, keep candidates that passed the LLM gate and are
not shortcuts, sample to the target size with balanced question types, and write
inputs/{split}/{questions,raw,corpus,ordered_gold}.json plus a stats file.

Usage: finalize53.py <dataset> <split> <raw_source.json> <target> <candidates.jsonl>... -- <results.jsonl>...
"""
import collections, json, random, sys
from pathlib import Path
import bm25s, Stemmer

HERE = Path(__file__).parent
stem = Stemmer.Stemmer('english')
args = sys.argv[1:]
dataset, split, raw_path, target = args[0], args[1], args[2], int(args[3])
cut = args.index('--')
cand_paths, res_paths = args[4:cut], args[cut + 1:]
cands = {c['cid']: c for p in cand_paths for c in map(json.loads, open(p))}
results = {r['cid']: r for p in res_paths for r in map(json.loads, open(p))}

# ---- corpus
corpus = {}
for r in json.load(open(raw_path)):
    for t, s in r['context']:
        corpus.setdefault(t, ''.join(s) if dataset == 'hotpotqa' else ' '.join(s))
n_ctx = len(corpus)
for c in cands.values():
    for h in c['hops']:
        corpus[h['gold_title']] = h['gold_text']            # gold text is the version the chain was built on
n_added = 0
if dataset == 'hotpotqa':
    import build_hotpot as bh
    idx = bh.ROOT / 'claude/hotpotqa/bm25_index'
    wiki = bm25s.BM25.load(str(idx)); wtitles = json.load(open(idx / 'titles.json')); pages = bh.load_pages()
    gold = list(dict.fromkeys(h['gold_title'] for c in cands.values() for h in c['hops']))
    docs, _ = wiki.retrieve(bm25s.tokenize([(g + ' ' + corpus[g])[:1000] for g in gold], stopwords='en', stemmer=stem,
                                           show_progress=False), k=21, show_progress=False)
    for g, row in zip(gold, docs):
        for i in row:
            t = wtitles[i]
            if t != g and t not in corpus and t in pages:
                corpus[t] = ' '.join(pages[t][0]); n_added += 1
    del pages, wiki
titles = list(corpus)
texts = [t + ' ' + corpus[t] for t in titles]
pos = {t: i for i, t in enumerate(titles)}

# ---- question-only BM25 shortcut check against this corpus
passed = [r for r in results.values() if r.get('passed')]
bm = bm25s.BM25(); bm.index(bm25s.tokenize(texts, stopwords='en', stemmer=stem, show_progress=False), show_progress=False)
docs, _ = bm.retrieve(bm25s.tokenize([r['question'] for r in passed], stopwords='en', stemmer=stem,
                                     show_progress=False), k=10, show_progress=False)
kept = []
for r, top in zip(passed, docs):
    c = cands[r['cid']]
    top_t = {titles[i] for i in top}
    gold = [h['gold_title'] for h in c['hops']]
    shortcut = (gold[-1] in top_t) if c['kind'] != 'comparison' else set(gold) <= top_t
    if not shortcut:
        kept.append((r, c, [g in top_t for g in gold]))


def bucket(c):
    return 'comparison' if c['kind'] == 'comparison' else f"chain{c['hops_n']}"


# ---- balanced sampling to the target size
by = collections.defaultdict(list)
for x in kept:
    by[bucket(x[1])].append(x)
rng = random.Random(0)
for v in by.values():
    rng.shuffle(v)
    v.sort(key=lambda x: -(x[0].get('naturalness') or 0))     # most natural questions first
chosen, buckets = [], sorted(by)
quota = {b: 0 for b in buckets}
while len(chosen) < target and any(quota[b] < len(by[b]) for b in buckets):
    for b in sorted(buckets, key=lambda b: quota[b]):
        if quota[b] < len(by[b]) and len(chosen) < target:
            chosen.append(by[b][quota[b]]); quota[b] += 1

# ---- write R26-format inputs
out = HERE / 'final' / f'{dataset}_long' / 'inputs' / split
out.mkdir(parents=True, exist_ok=True)
questions, raw, ordered = [], [], {}
for r, c, bm_hits in chosen:
    qid = c['cid']
    questions.append({'qid': qid, 'question': r['question']})
    gold = list(dict.fromkeys(h['gold_title'] for h in c['hops']))
    ordered[qid] = [pos[g] for g in gold]
    ev = []
    for h in c['hops']:
        e = {'title': h['gold_title']}
        if 'relation' in h:
            e.update(subject=h['subject'], relation=h['relation'], object=h['object'])
        if h.get('link_sentence'):
            e['link_sentence'] = h['link_sentence']
        ev.append(e)
    raw.append({'_id': qid, 'question': r['question'], 'answer': r['answer'], 'type': bucket(c),
                'hops': c['hops_n'], 'evidence': ev, 'naturalness': r.get('naturalness'),
                'closed_book_answerable': r.get('tags', {}).get('closed_book_answerable'),
                'bm25_top10_hits': bm_hits})
json.dump(questions, open(out / 'questions.json', 'w'), ensure_ascii=False)
json.dump(raw, open(out / 'raw.json', 'w'), ensure_ascii=False)
json.dump(ordered, open(out / 'ordered_gold.json', 'w'))
json.dump([{'title': t, 'text': corpus[t]} for t in titles], open(out / 'corpus.json', 'w'), ensure_ascii=False)
stats = dict(dataset=dataset, split=split, candidates=len(cands), llm_gate_processed=len(results),
             llm_gate_passed=len(passed), not_shortcut=len(kept), selected=len(chosen),
             selected_by_type=collections.Counter(bucket(c) for _, c, _ in chosen),
             available_by_type={b: len(v) for b, v in by.items()},
             corpus=len(titles), corpus_original_contexts=n_ctx, corpus_hard_distractors=n_added,
             gold_passages=sum(len(v) for v in ordered.values()),
             distinct_gold_passages=len({i for v in ordered.values() for i in v}))
json.dump(stats, open(out / 'stats.json', 'w'), indent=1, default=dict)
print(json.dumps(stats, default=dict), flush=True)
