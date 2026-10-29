"""Visited states collected once (Fixed-OP and Pool-resample): one roll-in of a teacher-forced encoder of the same
seed, with frozen negative pools. Writes states/Fixed-OP/<seed>/pool/states.json.
Usage: rollin_fixed.py --seed 13   (adapter: fixed_models/RN/<seed>/adapter)"""
from states import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();setup(a.seed)
out=ROOT/'states/Fixed-OP'/str(a.seed)/'pool'
if (out/'complete.json').exists():raise SystemExit(0)
tick=time.monotonic();enc=Encoder(ROOT/'fixed_models/RN'/str(a.seed)/'adapter');rows=rollin(enc,out);rows=pools(enc,rows,out);save(out/'states.json',rows);save(out/'complete.json',dict(seconds=time.monotonic()-tick,source=str(ROOT/'fixed_models/RN'/str(a.seed)/'adapter'),generations=1,**device_info()))
