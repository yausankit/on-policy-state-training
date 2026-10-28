"""Average overlap |shared ∩ per-state frozen top100| / 100 for state sets (argv: name=path ...)."""
import json,sys,collections
sh=json.load(open("shared_pools/train_question_top100.json"));res={}
for spec in sys.argv[1:]:
 name,path=spec.split("=",1);rows=json.load(open(path));by=collections.defaultdict(list)
 for v in rows:
  p=v.get("negative_top100_state",v["negative_top100"]);o=len(set(p)&set(sh[v["qid"]]))/100;by["all"].append(o);by["node%d"%v["node"]].append(o);by["wrong" if any(d not in v["gold"] for d in v["evidence"]) else "clean"].append(o)
 res[name]={k:dict(mean=round(100*sum(x)/len(x),1),n=len(x)) for k,x in sorted(by.items())}
print(json.dumps(res,indent=1))
