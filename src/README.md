# src

One implementation of the chain retriever and of every training condition in the paper. `run_paper.sh` lists the commands in dependency order.

| File | Role | Paper |
|---|---|---|
| `core.py` | Query encoder (Qwen3-Embedding-0.6B + query-side LoRA), state text x(s), chain readout, data paths | Sec. 3, App. A |
| `states.py` | Roll-ins, frozen top-100 negative pools, and the gold/visited mix of every condition | Sec. 4, Tables 3–4, App. C |
| `train.py` | Contrastive training with refresh, early stopping on validation C@5 | Sec. 4, App. A |
| `train_question_only.py` | Question-only fine-tuning (w/o chain) | Table 3 |
| `initial_mix.py` | Epoch-1 mix from a roll-in of the teacher-forced encoder | Sec. 4 |
| `rollin_fixed.py` | Visited states collected once (Fixed-OP, Pool-resample) | Tables 3–4 |
| `static_reference.py` | (wrong, gold) counts of visited states used to match synthetic contamination | Table 4 |
| `shared_pool.py`, `overlap.py` | Shared negative pools and their overlap with per-state pools | Sec. 5.5.2, App. B |
| `inference.py`, `evaluate.py`, `evaluate_dense.py` | Greedy chain retrieval (T=4), single-query retrieval, Dense baseline | Table 1 |
| `metrics.py` | R@5, R@10, C@5, C@10 on MuSiQue | Tables 1–4 |
| `probe.py` | Fixed-state recovery probe | Sec. 5.4, Fig. 3b, Table 11 |
| `longchain/` | Zero-shot evaluation on HotpotQA-Long and 2Wiki-Long (T=8) and their metrics | Tables 1, 3 |

## Conditions (`train.py --condition`)

| Condition | Training states | Paper |
|---|---|---|
| `RN` | gold states only (teacher forcing) | w/o on-policy states |
| `OP-refresh` | half gold, half visited, visited half refreshed every epoch | ours |
| `Fixed-OP` | visited half sampled once from one teacher-forced roll-in | w/o refresh; Visited, once, ✗ |
| `Pool-resample` | visited half redrawn every epoch from that roll-in | Visited, once, ✓ |
| `Static-corrupt` | synthetic contamination of gold states, built once | Synthetic, ✗ |
| `Static-refresh` | synthetic contamination resampled every epoch | Synthetic, ✓ |
| `Mix-25`, `Mix-75` | 25% / 75% visited states, refreshed | Mixing ratio |
| `SQ-LoRA-v2` (`train_question_only.py`) | the question alone | w/o chain |

## Data layout under `OPST_ROOT`

```
inputs/{train,design}/questions.json   qid, question
inputs/{train,design}/corpus.json      title, text
inputs/{train,design}/doc_emb.npy      frozen passage embeddings (float16, L2-normalized)
inputs/design/raw.json                 MuSiQue records (gold paragraphs) for metrics.py and probe.py
states/RN/states.json                  teacher-forced gold states: qid, node, evidence, gold, remaining, positives, text, negative_top100
validation_qids.json                   the 200 checkpoint-selection questions
checks/validation_gold.json            their gold passage ids
```

Gold states draw, for a question with n gold passages, n histories of random size k in {0..n-1} as random gold subsets in chain order (MuSiQue question decomposition), each with its earliest missing gold passage as target and the frozen encoder's top 100 for x(s), gold removed, as negative pool.

Environment variables: `OPST_ROOT` (data root), `R19_MODEL` (local path of Qwen3-Embedding-0.6B), `R29_CHECKPOINTING=1` (gradient checkpointing), `OPST_SHARED_NEG=1` (shared negative pools), `R19_GPU` (name of the evaluation output folder).
