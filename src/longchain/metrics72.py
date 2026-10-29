"""R@5, R@10, C@5 and C@10 from saved rankings. MuSiQue: gold = supporting paragraph texts; long-chain: ordered_gold ids."""
import json,os,numpy as np
from pathlib import Path
W=Path(os.environ.get("RESULTS_DIR","."));R57=Path(os.environ.get("LONGCHAIN_ROOT",Path(__file__).resolve().parent))
def read(p):return json.loads(Path(p).read_text())
def norm(t):return " ".join(str(t).split())
def score(ranks,goldsets):
 rows=[]
 for q,g in goldsets.items():
  r=ranks[q];row={}
  for k in (5,10):n=len(g&set(r[:k]));row[f"R@{k}"]=100*n/len(g);row[f"C@{k}"]=100.0*(n==len(g))
  rows.append(row)
 return dict(n=len(rows),**{m:float(np.mean([x[m] for x in rows])) for m in ["R@5","R@10","C@5","C@10"]})
out={}
raw=read(W/"inputs/design/raw.json");corp=read(W/"inputs/design/corpus.json");ctext=[norm(c["title"]+"\n"+c["text"]) for c in corp]
gt={str(r["id"]):{norm(v["title"]+"\n"+v["paragraph_text"]) for v in r["paragraphs"] if v["is_supporting"]} for r in raw}
for c in ["Dense","RN","OP-refresh"]:
 p=W/"evaluations"/c/"13/design/4090/ranking_ids.json"
 if p.exists():
  ranks={q:[ctext[d] for d in v] for q,v in read(p).items()};out[("musique",c)]=score(ranks,{q:gt[q] for q in ranks})
for ds in ["hotpotqa_long_ref","2wiki_long_ref"]:
 gold={k:set(int(x) for x in v) for k,v in read(W/"long/data"/ds/"inputs/test/ordered_gold.json").items()}
 for c,src in [("Dense",W/"long/evaluations"/ds/"Frozen-SQ/13"),("RN",W/"long/evaluations"/ds/"RN/13"),("OP-refresh",W/"long/evaluations"/ds/"OP-refresh/13"),("Qwen3 RN",R57/"evaluations"/ds/"ABL-RN/13"),("Qwen3 OP-refresh",R57/"evaluations"/ds/"ABL-OP/13"),("Qwen3 Dense",R57/"evaluations"/ds/"Frozen-SQ/13")]:
  if (src/"rankings.json").exists():ranks=read(src/"rankings.json");out[(ds,c)]=score(ranks,{q:gold[q] for q in ranks})
for k,v in out.items():print(k,{m:round(x,1) if m!="n" else x for m,x in v.items()})
json.dump({f"{a}|{b}":v for (a,b),v in out.items()},open(W/"metrics72.json","w"),indent=1)
