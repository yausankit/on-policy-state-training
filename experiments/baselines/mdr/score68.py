"""Score R@k=|gold∩top-k|/|gold|, C@k=all gold in top-k, from saved ranking_ids.json vs ordered_gold.json (corpus indices)."""
import json,sys
from pathlib import Path
inp,out=Path(sys.argv[1]),Path(sys.argv[2])
gold=json.load(open(inp/'ordered_gold.json'));raw={str(r['_id']):r for r in json.load(open(inp/'raw.json'))};rank=json.load(open(out/'ranking_ids.json'))
rows={}
for q,g in gold.items():
 r=rank[q];g=set(g);row=dict(type=raw[q]['type'],n_gold=len(g),n_ranked=len(r))
 for k in (5,10):h=len(g&set(r[:k]));row[f'R@{k}']=h/len(g);row[f'C@{k}']=float(h==len(g))
 rows[q]=row
M=['R@5','R@10','C@5','C@10']
def agg(v):return dict(n=len(v),**{m:round(100*sum(x[m] for x in v)/len(v),1) for m in M})
s=dict(all=agg(list(rows.values())),by_type={t:agg([x for x in rows.values() if x['type']==t]) for t in sorted({x['type'] for x in rows.values()})},min_ranked=min(x['n_ranked'] for x in rows.values()),n_short=sum(x['n_ranked']<10 for x in rows.values()))
json.dump(rows,open(out/'per_question.json','w'),indent=1);json.dump(s,open(out/'summary.json','w'),indent=1);print(json.dumps(s))
