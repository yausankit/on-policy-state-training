"""Metrics as round54/core54.py::metrics; usage: eval69.py name rankings.json"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common69 import load_questions
name, path = sys.argv[1], sys.argv[2]
rk = json.load(open(path))['rankings']; qs = load_questions(name)
gold = {q: g for q, _, g, _ in qs}; typ = {q: t for q, _, _, t in qs}
rows, by = [], {}
for qid, r in rk.items():
    g = gold[qid]; m = {}
    for k in (5, 10):
        hit = len(set(g) & set(r[:k])); m[f'R@{k}'] = 100 * hit / len(g); m[f'C@{k}'] = 100.0 * (hit == len(g))
    rows.append(m); by.setdefault(typ[qid], []).append(m)
avg = lambda rows: {k: round(sum(r[k] for r in rows) / len(rows), 1) for k in ('R@5', 'R@10', 'C@5', 'C@10')}
print(json.dumps(dict(overall=avg(rows), n=len(rows), by_type={t: dict(n=len(v), **avg(v)) for t, v in sorted(by.items())})))
