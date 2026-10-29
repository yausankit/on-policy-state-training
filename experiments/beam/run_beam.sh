#!/bin/bash
# Appendix F.3: greedy (width 1) and beam decoding (widths 2 and 4) with the matched seed-13 encoders.
# TF / OP: adapters of teacher forcing (RN) and on-policy state training (OP-refresh).
TF=${TF:-../../src/lora/RN/13/adapter}; OP=${OP:-../../src/lora/OP-refresh/13/adapter}
python encode_corpus.py hotpotqa_long
for b in 1 2 4; do
  python beam.py musique $b 4 musique_b$b.json TF=$TF OP=$OP
  python beam.py hotpotqa_long $b 8 hotpotqa_b$b.json TF=$TF OP=$OP
done
