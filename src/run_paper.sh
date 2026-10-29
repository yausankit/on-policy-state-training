#!/bin/bash
# Commands behind the reported runs, in dependency order. Every script reads its data from $OPST_ROOT (default: src/).
# Seed 13 is the shared seed of the ablations and analyses; Table 1 repeats steps 1-3 for seeds 42-46.
set -e
S=${SEED:-13}; LR=3e-4

# 1. Teacher forcing (w/o on-policy states).
python train.py --condition RN --seed $S --lr $LR
# 2. Epoch-1 mix: roll in the teacher-forced encoder and pair half of the gold states with visited states.
python initial_mix.py
# 3. On-policy state training (visited half refreshed every epoch).
python train.py --condition OP-refresh --seed $S --lr $LR

# Table 3: w/o refresh, w/o chain.
mkdir -p fixed_models/RN/$S && cp -r lora/RN/$S/adapter fixed_models/RN/$S/
python rollin_fixed.py --seed $S
python train.py --condition Fixed-OP --seed $S --lr $LR
python train_question_only.py --condition SQ-LoRA-v2 --seed $S --lr $LR

# Table 4: synthetic contamination built once / resampled, visited states resampled from one roll-in.
python static_reference.py --seed $S
python train.py --condition Static-corrupt --seed $S --lr $LR
python train.py --condition Static-refresh --seed $S --lr $LR
python train.py --condition Pool-resample --seed $S --lr $LR

# Appendix C: mixing ratio.
python train.py --condition Mix-25 --seed $S --lr $LR
python train.py --condition Mix-75 --seed $S --lr $LR

# Evaluation on the MuSiQue test questions and zero-shot transfer to the long-chain benchmarks (T=8).
python evaluate_dense.py --split design
for c in RN OP-refresh Fixed-OP Static-corrupt Static-refresh Pool-resample Mix-25 Mix-75 SQ-LoRA-v2; do
  python evaluate.py --condition $c --seed $S --split design
done
python metrics.py ${R19_GPU:-gpu} RN OP-refresh Fixed-OP Static-corrupt Static-refresh Pool-resample Mix-25 Mix-75 SQ-LoRA-v2 Dense
for d in hotpotqa_long 2wiki_long; do for c in RN OP-refresh Fixed-OP; do
  python longchain/eval54.py --dataset $d --condition $c --seed $S
done; done

# Section 5.4: fixed-state recovery probe on the teacher-forced chain's states (traces from evaluate.py).
python probe.py probe_traces TF probe_TF.json TF=lora/RN/$S/adapter w/o-refresh=lora/Fixed-OP/$S/adapter \
  Pool-resample=lora/Pool-resample/$S/adapter OP-refresh=lora/OP-refresh/$S/adapter

# Appendix B: shared negative pools, in a separate root with the same inputs.
# OPST_ROOT=<shared root> python shared_pool.py
# OPST_ROOT=<shared root> OPST_SHARED_NEG=1 python train.py --condition RN --seed $S --lr $LR
# OPST_ROOT=<shared root> python initial_mix.py
# OPST_ROOT=<shared root> OPST_SHARED_NEG=1 python train.py --condition OP-refresh --seed $S --lr $LR
# python overlap.py per-state=<root>/states/OP/states.json
