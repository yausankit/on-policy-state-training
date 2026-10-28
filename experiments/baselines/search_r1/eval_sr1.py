"""Metrics exactly as round54/core54.py::metrics. Usage: eval_sr1.py <name> <data_dir> <rankings.json> <out.json>"""
import json, sys
name, data, rk_path, out = sys.argv[1:5]
norm = lambda t: ' '.join(str(t or '').split())
rk = json.load(open(rk_path))
raw = json.load(open(f'{data}/raw.json'))
if name == 'musique':
    corp = json.load(open(f'{data}/corpus.json'))
    idx = {}
    for i, c in enumerate(corp): idx.setdefault((norm(c['title']), norm(c['text'])), i)
    gold = {r['id']: [idx[(norm(p['title']), norm(p['paragraph_text']))] for p in r['paragraphs'] if p['is_supporting']] for r in raw}
    rawd = {r['id']: dict(type=r['id'].split('__')[0][:4]) for r in raw}  # 2hop / 3hop / 4hop
else:
    gold = json.load(open(f'{data}/ordered_gold.json'))
    rawd = {r['_id']: r for r in raw}


def metrics(rankings, gold, raw, ks=(5, 10)):
    per_type, overall = {}, []
    for qid, r in rankings.items():
        g = gold[qid]
        m = {}
        for k in ks:
            hit = len(set(g) & set(r[:k]))
            m[f'R@{k}'] = 100 * hit / len(g)
            m[f'C@{k}'] = 100.0 * (hit == len(g))
        overall.append(m)
        per_type.setdefault(raw[qid]['type'], []).append(m)
    avg = lambda rows: {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return dict(overall=avg(overall), n=len(overall), by_type={t: dict(n=len(v), **avg(v)) for t, v in sorted(per_type.items())})


m = metrics(rk, gold, rawd)
m['mean_distinct'] = sum(len(v) for v in rk.values()) / len(rk)
json.dump(m, open(out, 'w'), indent=1)
print(json.dumps(m))
