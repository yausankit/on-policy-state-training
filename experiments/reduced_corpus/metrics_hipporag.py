"""R63: HippoRAG 2 top-20 pool rankings -> R@5/R@10/C@5/C@10 overall and by type (same formula as round54/core54.py::metrics)."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent
import os
L = Path(os.environ.get('HIPPORAG2_DIR', 'hipporag2'))


def metrics(rankings, gold, raw, ks=(5, 10)):
    per_type, overall = {}, []
    for qid, rk in rankings.items():
        g = gold[qid]
        m = {}
        for k in ks:
            hit = len(set(g) & set(rk[:k]))
            m[f'R@{k}'] = 100 * hit / len(g)
            m[f'C@{k}'] = 100.0 * (hit == len(g))
        overall.append(m)
        per_type.setdefault(raw[qid]['type'], []).append(m)
    avg = lambda rows: {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return dict(overall=avg(overall), n=len(overall), by_type={t: dict(n=len(v), **avg(v)) for t, v in sorted(per_type.items())})


out = {}
for sp in sys.argv[1:]:
    gold = json.loads((R / 'gold' / f'{sp}_ordered_gold.json').read_text())
    raw = {v['_id']: v for v in json.loads((L / 'inputs' / sp / 'raw.json').read_text())}
    rk = json.loads((L / 'h3' / f'ranking_ids_{sp}.json').read_text())
    assert set(rk) == set(gold) == set(raw), sp
    out[sp] = metrics(rk, gold, raw)
    print(sp, {k: round(v, 1) for k, v in out[sp]['overall'].items()})
    for t, v in out[sp]['by_type'].items():
        print('   ', t, v['n'], {k: round(x, 1) for k, x in v.items() if k != 'n'})
(R / 'hipporag_metrics.json').write_text(json.dumps(out, indent=2) + '\n')
