#!/bin/bash
# card 2: train and evaluate Pool-resample, then the fixed-state probe; card 3: same-hardware evaluation of the original Fixed-OP and OP-refresh adapters
cd /data1/xuboyan/rev/round87; source /data1/xuboyan/rev/round78/env_czys.sh; mkdir -p logs
step(){ echo "$(date +%F_%T) START $1" >> status.txt; shift; "$@" > logs/$(date +%s%N).log 2>&1; rc=$?; echo "$(date +%F_%T) END rc=$rc $*" >> status.txt; return $rc; }
if [ "$1" = card2 ]; then export CUDA_VISIBLE_DEVICES=2 R8_BUS=0000:38:00.0
  step train_pool $PY -B train29.py --condition Pool-resample --seed 13 --lr 3e-4 || exit 1
  for sp in design heldout; do step eval_pool_$sp $PY -B evaluate29.py --condition Pool-resample --seed 13 --split $sp; done
  P=/data1/xuboyan/rev/round78
  step probe $PY -u $P/probe/probe78.py $P/probe/traces TF probe87_TF.json TF=$P/ref_adapters/RN/adapter Fixed-OP=lora/Fixed-OP-orig/13/adapter Pool-resample=lora/Pool-resample/13/adapter OP-refresh=lora/OP-refresh-orig/13/adapter
  echo "$(date +%F_%T) A_DONE" >> status.txt
else export CUDA_VISIBLE_DEVICES=3 R8_BUS=0000:5A:00.0
  for c in Fixed-OP-orig OP-refresh-orig; do for sp in design heldout; do step eval_${c}_$sp $PY -B evaluate29.py --condition $c --seed 13 --split $sp; done; done
  echo "$(date +%F_%T) B_DONE" >> status.txt
fi
