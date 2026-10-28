# src

- `round29/`: the training and evaluation code of the reported runs (seed 13 for ablations and analyses, seeds 42–46 for Table 1). Origin: experiment rounds 19–29.
  - `core29.py`: Qwen3-Embedding-0.6B query encoder with LoRA (r=16, alpha=32), state serialization x(s), frozen document embeddings, scoring.
  - `states29.py`: teacher-forced gold states (random gold subsets in chain order), greedy roll-ins, earliest-missing-gold labels, frozen top-100 negative pools, the 50/50 gold/visited mix.
  - `train29.py`: contrastive training (batch 16, 15 sampled negatives per state plus in-batch negatives, missing gold masked, tau=0.05, lr 3e-4), visited half refreshed every epoch, checkpoint selection by validation C@5.
  - `inference29.py`, `evaluate29.py`: greedy chain retrieval (T=4 on MuSiQue) and R@k / C@k.
  - `evaluate_dense29.py`: frozen encoder with the question alone (Dense).
  - `train_sq_v2_29.py`: question-only fine-tuning (w/o chain in Table 3).
  - `prepare_fixed29.py`, `prepare29.py`: fixed probe states and input preparation.
  - `qa_metrics29.py`: EM / F1.
- `longchain/`: zero-shot evaluation of MuSiQue-trained encoders on the long-chain benchmarks (T=8). Origin: round 54.
- `round78/`: the same recipe re-run on the final server (`core78`/`states78`/`train78`/`evaluate78`/`inference78`), shared negative pools (`shared78.py`, Appendix B), roll-in for the shared-pool regime (`op78.py`), fixed-state recovery probe (`probe78.py`, Sec. 5.4 and Figure 3b), pool overlap statistics (`overlap78.py`). Launchers: `chainA_czys.sh`, `chainB_czys.sh`.
- `round87/`: Table 4 controls. `states65.py`/`train65.py` add synthetic contamination resampled every epoch; `states29.py`/`train29.py` add visited states resampled from one roll-in. Launcher: `chain87.sh`.
