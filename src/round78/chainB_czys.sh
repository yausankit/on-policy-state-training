#!/bin/bash
# card 3: same-hardware design/heldout evaluation of RN-sharedneg and of the original RN / OP-refresh adapters
cd /data1/xuboyan/rev/round78; source env_czys.sh; export CUDA_VISIBLE_DEVICES=3 R8_BUS=0000:5A:00.0
step(){ echo "$(date +%F_%T) START $1" >> czys.status; shift; "$@" > logs/czysB_$(date +%s).log 2>&1; rc=$?; echo "$(date +%F_%T) END rc=$rc $*" >> czys.status; }
for c in RN RN-orig OP-refresh-orig; do for sp in design heldout; do step eval_${c}_${sp} $PY -B evaluate78.py --condition $c --seed 13 --split $sp; done; done
echo "$(date +%F_%T) B_DONE" >> czys.status
