"""Shared negative pools (Appendix B): frozen encoder, question alone, top 100 non-gold passages per training question."""
from core import *
setup(13);recs=read(ROOT/"inputs/train/questions.json");gold={}
for v in read(state_source("RN")):gold.setdefault(v["qid"],v["gold"])
D=torch.as_tensor(np.load(ROOT/"inputs/train/doc_emb.npy"),device="cpu");order=np.arange(len(D));enc=Encoder(frozen=True);out={}
for s in range(0,len(recs),32):
 batch=recs[s:s+32];qs=enc.encode([r["question"] for r in batch])
 for r,q in zip(batch,qs):
  scores=(torch.as_tensor(q)@D.T).numpy();rank=np.lexsort((order,-scores));g=set(gold[r["qid"]]);out[r["qid"]]=[int(d) for d in rank if int(d) not in g][:100]
 print("shared",s,flush=True)
save(SHARED_POOL,out);save(ROOT/"shared_pools/info.json",dict(questions=len(out),missing_gold=[r["qid"] for r in recs if r["qid"] not in gold],view="frozen base encoder, QUERY_PROMPT+question (single-query), top100 excluding gold",**device_info()))
