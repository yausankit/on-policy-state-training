"""R@5, R@10, C@5 and C@10 on MuSiQue from saved ranking_ids.json. argv: <gpu dir name> <condition> ..."""
import json,sys,numpy as np
from pathlib import Path
import os
R=Path(os.environ.get("OPST_ROOT",Path(__file__).resolve().parent))
def norm(t):return " ".join(t.split())
out={}
for sp in ["design"]:
 raw=json.load(open(R/"inputs"/sp/"raw.json"));corp=json.load(open(R/"inputs"/sp/"corpus.json"));txt=[norm(c["title"]+"\n"+c["text"]) for c in corp]
 for c in sys.argv[2:]:
  p=R/"evaluations"/c/"13"/sp/sys.argv[1]/"ranking_ids.json"
  if not p.exists():continue
  ranks=json.load(open(p));rows=[]
  for rec in raw:
   q=str(rec.get("_id") or rec.get("id") or rec.get("qid"))
   if q not in ranks:continue
   gold={norm(v["title"]+"\n"+v["paragraph_text"]) for v in rec["paragraphs"] if v["is_supporting"]};o=[txt[i] for i in ranks[q]];row={}
   for k in [5,10]:n=len(gold&set(o[:k]));row[f"R@{k}"]=n/len(gold);row[f"C@{k}"]=float(n==len(gold))
   rows.append(row)
  out.setdefault(c,{})[sp]=dict(n=len(rows),**{m:round(100*float(np.mean([r[m] for r in rows])),1) for m in ["R@5","R@10","C@5","C@10"]})
print(json.dumps(out,indent=1));json.dump(out,open(R/"metrics_"+sys.argv[1]+".json","w"),indent=1)
