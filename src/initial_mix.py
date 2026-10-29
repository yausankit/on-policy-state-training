"""Epoch-1 mix of on-policy state training: roll in the teacher-forced encoder (lora/RN/13) on the training
questions, give every visited state its frozen top-100 pool, and pair half of the gold states with an equal number
of visited states. Writes states/OP/{states.json,onpolicy_states.json}."""
from core import *
setup();out=ROOT/"states/OP";out.mkdir(parents=True,exist_ok=True)
if (out/"complete.json").exists():raise SystemExit(0)
recs=read(ROOT/"inputs/train/questions.json");corp=corpus("train");docs=np.load(ROOT/"inputs/train/doc_emb.npy");D=torch.as_tensor(docs,device="cpu");rnrows=read(state_source("RN"));gold={v["qid"]:v["gold"] for v in rnrows};enc=Encoder(adapter("RN",13));rows=[];traces=[]
with torch.inference_mode():
 for s in range(0,len(recs),32):
  jobs=[dict(rec=r,written=[],states=[]) for r in recs[s:s+32]]
  for t in range(4):
   texts=[text19(j["rec"]["question"],j["written"],corp) for j in jobs];qs=enc.encode(texts)
   for j,text,q in zip(jobs,texts,qs):
    key=j["rec"]["qid"];e=j["written"].copy();remaining=[d for d in gold[key] if d not in e]
    if remaining:j["states"].append(dict(qid=key,node=t,evidence=e,gold=gold[key],remaining=remaining,positives=remaining[:1],text=text))
    scores=(torch.as_tensor(np.asarray([np.zeros_like(q),q]))@D.T).numpy()[1].copy();scores[e]=-np.inf;ids=np.flatnonzero(np.isfinite(scores));j["written"].append(int(ids[np.lexsort((ids,-scores[ids]))[0]]))
  values=[dict(qid=j["rec"]["qid"],writes=j["written"],states=j["states"]) for j in jobs]
  traces+=values;rows.extend(v for j in values for v in j["states"]);print("rollin",s,flush=True)
enc.release();enc=Encoder(frozen=True);order=np.arange(len(docs))
for s in range(0,len(rows),32):
 qs=enc.encode([v["text"] for v in rows[s:s+32]])
 for v,q in zip(rows[s:s+32],qs):
  scores=(torch.as_tensor(q)@D.T).numpy();rank=np.lexsort((order,-scores));v["negative_top100"]=[int(d) for d in rank if int(d) not in set(v["gold"])][:100]
 print("pools",s,flush=True)
rng=random.Random(13);n=len(rnrows)//2;ri=rng.sample(range(len(rnrows)),n);oi=rng.sample(range(len(rows)),len(rnrows)-n);mixed=[dict(rnrows[i],origin="R17-RN",source_index=i) for i in ri]+[dict(rows[i],origin="RN-on-policy",source_index=i) for i in oi];rng.shuffle(mixed)
save(out/"rollin_traces.json",traces);save(out/"onpolicy_states.json",rows);save(out/"states.json",mixed)
from collections import Counter
stats=dict(rollin_adapter=str(adapter("RN",13)),training_questions=len(recs),onpolicy_states=len(rows),mixed_states=len(mixed),r17_states=n,op_states=len(mixed)-n,step_distribution=dict(Counter(v["node"]+1 for v in rows)),evidence_wrong_fraction=sum(d not in v["gold"] for v in rows for d in v["evidence"])/sum(len(v["evidence"]) for v in rows),seed=13,**device_info());save(out/"state_stats.json",stats);save(out/"complete.json",stats)
