# experiments

| Folder | Contents | Paper |
|---|---|---|
| `beam/` | Greedy vs. beam decoding (widths 2 and 4) with the matched seed-13 encoders; `encode_corpus.py` encodes a long-chain corpus; `run_beam.sh` | App. F.3, Table 10 |
| `qa/` | Top-5 passages of each retriever given to a fixed reader (`QA_MODEL=deepseek-flash` or `qwen-flash`), EM / F1 | Table 2, Table 8 |
| `reduced_corpus/` | Reduced corpora (all gold passages + frozen top 100 per question) for HippoRAG 2 and G-Retriever, HippoRAG 2 metrics, BM25 | App. D, Table 1 |
| `baselines/` | Search-R1 (released 7B checkpoint, our frozen encoder as search engine), Q-RAG (released MuSiQue checkpoint; `local_patches.diff` patches the official code), MDR (released HotpotQA checkpoint; `core30.py` loads the official model classes), G-Retriever (PCST over the HippoRAG 2 graph), LinearRAG | Table 1 |

Baseline scripts wrap the official repositories; `WORK` points to the directory holding their checkouts and data. LLM calls read `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL`.
