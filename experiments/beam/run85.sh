#!/bin/bash
cd /data1/xuboyan/rev/round85; source /data1/xuboyan/rev/round78/env_czys.sh
A=/data1/xuboyan/rev/round78/ref_adapters
if [ "$1" = card3 ]; then export CUDA_VISIBLE_DEVICES=3
  for b in 1 2 4; do $PY -u beam85.py musique $b 4 musique_b$b.json TF=$A/RN/adapter OP=$A/OP/adapter > musique_b$b.log 2>&1; echo "musique b$b exit $?" >> status.txt; done
else export CUDA_VISIBLE_DEVICES=2
  $PY -u encode85.py hotpotqa_long > enc_hotpotqa.log 2>&1; echo "encode hotpotqa exit $?" >> status.txt
  for b in 1 2 4; do $PY -u beam85.py hotpotqa_long $b 8 hotpotqa_b$b.json TF=$A/RN/adapter OP=$A/OP/adapter > hotpotqa_b$b.log 2>&1; echo "hotpotqa b$b exit $?" >> status.txt; done
fi
