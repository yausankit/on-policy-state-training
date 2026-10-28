from inference29 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--seed',type=int,default=13);a=p.parse_args();setup(13);out=outpath('Dense',a.split,13)
if (out/'complete.json').exists():raise SystemExit(0)
tick=time.monotonic();recs=read(ROOT/'inputs'/a.split/'questions.json');corp=corpus(a.split);D=torch.as_tensor(np.load(ROOT/'inputs'/a.split/'doc_emb.npy'));enc=Encoder(frozen=True);retrieve_single(enc,recs,corp,D,out);save(out/'complete.json',dict(condition='Dense',T=1,questions=len(recs),seconds=time.monotonic()-tick,**device_info()))
