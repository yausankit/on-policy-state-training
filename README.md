# On-Policy State Training for Chain Dense Retrieval

Code and benchmark data for *Retrieving from Your Own Mistakes: On-Policy Training of Embedding Models for Multi-Hop Retrieval-Augmented Generation*.

A chain dense retriever re-encodes the question together with the passages it has retrieved so far and retrieves the next passage. Teacher forcing trains it only on gold histories, while at test time its own mistakes stay in the query. On-policy state training rolls in the current encoder at every epoch, labels each visited state with the earliest gold passage still missing (other missing gold passages are masked in the contrastive denominator), and trains the query-side LoRA of Qwen3-Embedding-0.6B on an equal mix of these visited states and teacher-forced states, with a fixed document index and frozen-encoder negatives.

**Not included:** model weights, LoRA adapters, passage embeddings, the full long-chain corpora (over 50 MB; rebuild them with `benchmarks/code`), logs and intermediate outputs.

## Layout

| Folder | Contents | Paper |
|---|---|---|
| `src/round29` | Core retriever: encoder and state serialization (`core29.py`), roll-ins, gold/visited state mixing and negative pools (`states29.py`), training (`train29.py`), chain inference and evaluation (`inference29.py`, `evaluate29.py`), frozen dense baseline (`evaluate_dense29.py`), question-only fine-tuning (`train_sq_v2_29.py`, w/o chain), fixed probe states (`prepare_fixed29.py`) | Sec. 3–5, Tables 1–4 |
| `src/longchain` | Zero-shot evaluation on HotpotQA-Long and 2Wiki-Long with T=8 (`eval54.py`, `core54.py`), metric recomputation (`metrics72.py`) | Table 1, Table 3 |
| `src/round78` | Re-run of the recipe on the final server: shared negative pools (`shared78.py`), fixed-state recovery probe (`probe78.py`), pool overlap (`overlap78.py`), metrics | Sec. 5.4, 5.5.2, Appendix B |
| `src/round87` | Mechanism controls: synthetic contamination built once or resampled every epoch (`states65.py`, `train65.py`), visited states resampled from one roll-in (`states29.py`, `train29.py`) | Sec. 5.5.1, Table 4 |
| `experiments/beam` | Greedy and beam decoding with the matched encoders | Appendix F.3 |
| `experiments/qa` | Downstream QA with a fixed reader | Table 2, Appendix F.1 |
| `experiments/reduced_corpus` | Reduced corpora for HippoRAG 2 / G-Retriever, BM25 baseline | Appendix D |
| `experiments/baselines` | Search-R1, Q-RAG, MDR, G-Retriever and LinearRAG runs | Table 1 |
| `benchmarks` | Construction code, prompts and released questions of HotpotQA-Long and 2Wiki-Long | Sec. 5.1, Appendix E |
| `figures` | Plotting scripts | Figures 2–7 |

Each folder has its own README. Scripts are kept as they were run; the round number in a file name is the experiment round it comes from.

## Environment

Python 3.11, torch 2.5.1 (CUDA 12.4), transformers 4.51.3, peft 0.17.1, numpy 1.26.4, faiss 1.9.0, matplotlib; see `requirements.txt`. Baselines use their official repositories (HippoRAG 2, G-Retriever, LinearRAG, Search-R1, Q-RAG, MDR).

Scripts contain absolute paths of the machines they ran on (for example `/data1/xuboyan/...`, `/home/...`); adapt them, or the environment variables `R29_DIR`, `R19_MODEL`, `R70_BASE` that most scripts read. LLM calls read `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` from the environment.
