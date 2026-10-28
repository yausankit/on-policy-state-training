#!/bin/bash
# card 2: train and evaluate OP-refresh-sharedneg, then the fixed-state probe
cd /data1/xuboyan/rev/round78; source env_czys.sh; export CUDA_VISIBLE_DEVICES=2 R8_BUS=0000:38:00.0
step(){ echo "$(date +%F_%T) START $1" >> czys.status; shift; "$@" > logs/czys_$(date +%s).log 2>&1; rc=$?; echo "$(date +%F_%T) END rc=$rc $*" >> czys.status; [ $rc -eq 0 ] || exit $rc; }
step trainOP $PY -B train78.py --condition OP-refresh --seed 13 --lr 3e-4
step evalOP_design $PY -B evaluate78.py --condition OP-refresh --seed 13 --split design
step evalOP_heldout $PY -B evaluate78.py --condition OP-refresh --seed 13 --split heldout
step probe $PY -u probe/probe78.py probe/traces TF probe/probe_TF.json TF=ref_adapters/RN/adapter OP=ref_adapters/OP/adapter RN-sharedneg=lora/RN/13/adapter OP-refresh-sharedneg=lora/OP-refresh/13/adapter
echo "$(date +%F_%T) A_DONE" >> czys.status
