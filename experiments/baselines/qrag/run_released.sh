#!/bin/bash
# Full-corpus evaluation of the released checkpoint Q-RAG/qrag-ft-e5-on-musique (tensors-only checkpoint loading).
Q=${WORK:-.}/round64/qrag
R=$Q/released/qrag-ft-e5-on-musique
cd $Q
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HOME=$Q/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p $Q/eval_released
for name in ${@:-musique_test hotpotqa_long 2wiki_long}; do
  free=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 2 | awk -F, '{print $2-$1}')
  echo "$(date) start $name (free $free MiB)"
  ./env/bin/python qrag_corpus_eval.py $Q/Q-RAG $R $Q/data/$name $name $Q/eval_released/$name --safe_load > $Q/eval_released/$name.log 2>&1 || { echo "EVAL_FAILED $name"; exit 1; }
  echo "$(date) done $name"
done
./env/bin/python score.py $Q/data $Q/eval_released $Q/eval_released/scores.json > $Q/eval_released/scores.txt 2>&1
echo ALL_DONE
