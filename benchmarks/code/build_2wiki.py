"""Extract k-hop chains and multi-entity comparisons from 2WikiMultiHopQA evidence triples.

Runs on 239. Output: one JSONL of candidates per split (train / dev / test), each with the
path, the gold paragraph per hop, and a draft prompt payload for question writing.
"""
import collections, json, random, re, sys
from pathlib import Path

RAW = Path('/home/dmir/qiuchenjie2023/revision_20260928/round17/data_prep/2wiki/raw')
OUT = Path('/home/dmir/qiuchenjie2023/revision_20260928/round53/claude/2wiki')

# Relations whose object is a person/work entity that can continue a chain.
BRIDGE = {'director', 'father', 'mother', 'spouse', 'performer', 'composer', 'child', 'sibling',
          'producer', 'creator', 'founded by', 'editor', 'publisher', 'presenter', 'student of',
          'doctoral advisor'}
# Relations allowed on the last hop (answer may be a value).
TERMINAL = BRIDGE | {'date of birth', 'date of death', 'place of birth', 'place of death',
                     'country of citizenship', 'cause of death', 'place of burial', 'educated at',
                     'award received', 'employer', 'occupation', 'publication date', 'inception',
                     'country of origin', 'country'}
# consecutive relation pairs that collapse to a shorter path (e.g. mother's husband = father)
DEGENERATE = {('mother', 'spouse'), ('father', 'spouse'), ('spouse', 'child'), ('child', 'father'),
              ('child', 'mother'), ('sibling', 'father'), ('sibling', 'mother'), ('spouse', 'spouse'),
              ('child', 'sibling'), ('sibling', 'sibling'), ('sibling', 'child')}
KIN = {'father', 'mother', 'spouse', 'child', 'sibling'}
DATE_ATTRS = {'date of birth', 'date of death', 'publication date', 'inception'}
TARGET = {'train': 5000, 'dev': 1250, 'test': 1250}
MONTHS = {m: i + 1 for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                          'august', 'september', 'october', 'november', 'december'])}


def parse_date(s):
    s = s.strip().lower()
    m = re.fullmatch(r'(\d{1,2}) ([a-z]+) (\d{3,4})', s)
    if m and m.group(2) in MONTHS:
        return int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))
    m = re.fullmatch(r'(\d{3,4})', s)
    return (int(m.group(1)), 0, 0) if m else None


def build_graph(records):
    para = {}            # entity id -> (title, text)
    name = {}            # entity id -> surface name
    edges = collections.defaultdict(set)   # (sid, rel) -> {oid}
    for r in records:
        ctx = {t: ' '.join(s) for t, s in r['context']}
        titles = list(dict.fromkeys(t for t, _ in r['supporting_facts']))
        subs = list(dict.fromkeys(e[0] for e in r['evidences_id']))
        if len(titles) != len(subs):
            continue
        for sid, title in zip(subs, titles):
            if title in ctx:
                para.setdefault(sid, (title, ctx[title]))
                name[sid] = re.sub(r'\s*\([^)]*\)$', '', title)   # display name from the article title
        for (sn, rel, on), (sid, _, oid) in zip(r['evidences'], r['evidences_id']):
            if not re.fullmatch(r'Q\d+', str(oid)):
                oid = 'LIT:' + on          # literal value (date, number): key by its text
            name.setdefault(sid, sn)
            name.setdefault(oid, on)
            edges[(sid, rel)].add(oid)
    # keep only functional edges (one object), so every hop has a unique answer
    func = {k: next(iter(v)) for k, v in edges.items() if len(v) == 1}
    out = collections.defaultdict(list)
    for (sid, rel), oid in func.items():
        if sid in para:
            out[sid].append((rel, oid))
    return para, name, out


def chains(para, name, out, k, rng, limit, per_start=1, banned=frozenset()):
    starts = [e for e in out if e in para] * per_start
    rng.shuffle(starts)
    res, used_start, used_answer = [], collections.Counter(), collections.Counter()
    kin_only = 0
    for e0 in starts:
        if len(res) >= limit:
            break
        if used_start[e0] >= per_start:
            continue
        path = [e0]
        rels = []
        ok = True
        for hop in range(k):
            cur = path[-1]
            last = hop == k - 1
            opts = [(r, o) for r, o in out.get(cur, []) if o not in path and (cur, r, o) not in banned
                    and not (rels and (rels[-1], r) in DEGENERATE)
                    and (r in TERMINAL if last else (r in BRIDGE and o in para and out.get(o)))]
            if not opts:
                ok = False
                break
            r, o = rng.choice(opts)
            path.append(o)
            rels.append(r)
        if not ok or used_answer[path[-1]] >= 8 or len(set(rels)) == 1 and k > 2:
            continue
        is_kin = all(r in KIN for r in rels[:-1])
        if is_kin and kin_only >= limit // 2:        # at most half of the chains are pure genealogy
            continue
        kin_only += is_kin
        used_start[e0] += 1
        used_answer[path[-1]] += 1
        hops = [dict(subject=name[path[i]], subject_id=path[i], relation=rels[i], object=name[path[i + 1]],
                     object_id=path[i + 1], gold_title=para[path[i]][0], gold_text=para[path[i]][1])
                for i in range(k)]
        res.append(dict(kind='chain', hops_n=k, start=name[e0], answer=name[path[-1]],
                        hidden=[name[x] for x in path[1:-1]], hops=hops))
    return res


def comparisons(para, name, out, rng, limit, banned=frozenset()):
    # two-hop items: x -r1-> p -attr-> date, with x and p both having paragraphs
    items = collections.defaultdict(list)
    for x, lst in out.items():
        for r1, p in lst:
            if r1 not in BRIDGE or p not in para or (x, r1, p) in banned:
                continue
            for attr, v in out.get(p, []):
                if attr in DATE_ATTRS and parse_date(name[v]) and (p, attr, v) not in banned:
                    items[(r1, attr)].append((x, p, v))
    res, used = [], collections.Counter()
    keys = [k for k, v in items.items() if len(v) >= 4]
    tries = 0
    while len(res) < limit and keys and tries < 200 * limit:
        tries += 1
        nonkin = [k for k in keys if k[0] not in KIN]
        key = rng.choice(nonkin if nonkin and rng.random() < 0.6 else keys)   # favour films, works, companies
        pool = items[key]
        m = rng.choice([3, 4])
        pick = rng.sample(pool, m)
        xs = {x for x, _, _ in pick}
        dates = [parse_date(name[v]) for _, _, v in pick]
        if len(xs) < m or len(set(dates)) < m or len({p for _, p, _ in pick}) < m or any(used[x] >= 3 for x in xs):
            continue
        used.update(xs)
        best = min(range(m), key=lambda i: dates[i])
        want = rng.choice(['earliest', 'latest'])
        if want == 'latest':
            best = max(range(m), key=lambda i: dates[i])
        hops = []
        for x, p, v in pick:
            hops.append(dict(subject=name[x], subject_id=x, relation=key[0], object=name[p], object_id=p,
                             gold_title=para[x][0], gold_text=para[x][1]))
            hops.append(dict(subject=name[p], subject_id=p, relation=key[1], object=name[v], object_id=v,
                             gold_title=para[p][0], gold_text=para[p][1]))
        res.append(dict(kind='comparison', hops_n=2 * m, entities=[name[x] for x, _, _ in pick],
                        relation=key[0], attribute=key[1], direction=want, answer=name[pick[best][0]],
                        hidden=[name[p] for _, p, _ in pick], hops=hops))
    return res


def components(c):
    return {(h['subject_id'], h['relation'], h['object_id']) for h in c['hops']}


def generate(recs, n, per_start, n_comp=None, banned=frozenset(), seed=0):
    para, name, out = build_graph(recs)
    cands, seen = [], set()
    for k in (3, 4, 5):
        for c in chains(para, name, out, k, random.Random(k + seed), n // 4, per_start, banned):
            key = tuple(sorted(components(c)))
            if key not in seen and not components(c) & banned:
                seen.add(key)
                cands.append(c)
        print('  chains', k, len(cands), flush=True)
    cands += [c for c in comparisons(para, name, out, random.Random(7 + seed), n_comp or n // 4, banned) if not components(c) & banned]
    print('  records', len(recs), 'entities', len(para), collections.Counter((c['kind'], c['hops_n']) for c in cands), flush=True)
    return cands


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('test pool (full original dev)', flush=True)
    test = generate(json.load(open(RAW / 'dev.json')), 6000, 30, n_comp=2500)
    test_comp = set().union(*(components(c) for c in test))
    train_raw = json.load(open(RAW / 'train.json'))
    print('dev (original train graph, avoiding test components)', flush=True)
    dev = generate(train_raw, 1250 * 4 // 3, 1, n_comp=420, banned=frozenset(test_comp))
    dev_comp = set().union(*(components(c) for c in dev))
    print('train (original train graph, avoiding dev and test components)', flush=True)
    train = generate(train_raw, 6000, 20, n_comp=2200, banned=frozenset(test_comp | dev_comp), seed=1000)
    train_comp = set().union(*(components(c) for c in train))
    for split, cands in (('train', train), ('dev', dev), ('test', test)):
        with open(OUT / f'{split}_candidates.jsonl', 'w') as f:
            for i, c in enumerate(cands):
                c['cid'] = f'2wl-{split}-{i:05d}'
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(split, 'final', len(cands), collections.Counter((c['kind'], c['hops_n']) for c in cands), flush=True)
    print('component overlap train-dev', len(train_comp & dev_comp), 'train-test', len(train_comp & test_comp),
          'dev-test', len(dev_comp & test_comp), flush=True)


if __name__ == '__main__':
    main()
