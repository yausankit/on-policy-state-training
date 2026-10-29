"""Reference distribution for synthetic contamination (Static-corrupt / Static-refresh): the joint counts of
(wrong passages, gold passages) in the visited states of an OP-refresh run of the same seed, over all epochs.
Writes static_reference/<seed>.json. Usage: static_reference.py --seed 13"""
from core import *
import argparse,collections
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
valid=valid_qids();pairs=[];sources=[]
for f in sorted((ROOT/'states/OP-refresh'/str(a.seed)).glob('epoch_*/states.json')):
 sources.append(str(f));pairs+=[(sum(d not in v['gold'] for d in v['evidence']),sum(d in v['gold'] for d in v['evidence'])) for v in read(f) if v.get('origin')!='R17-RN' and v['qid'] not in valid]
save(ROOT/'static_reference'/f'{a.seed}.json',dict(source=sources,joint_samples=pairs,joint_counts=dict(collections.Counter(map(str,pairs)))))
