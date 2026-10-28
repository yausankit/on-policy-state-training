# experiments

- `beam/` (round 85): greedy and beam decoding (beam 2 and 4) with the matched teacher-forced and on-policy encoders on MuSiQue and HotpotQA-Long; `encode85.py` re-encodes a long-chain corpus. Appendix F.3, Table 10.
- `qa/qa67.py` (round 67): top-5 passages of each retriever given to a fixed reader (deepseek-flash, temperature 0), EM / F1 on MuSiQue. Table 2 and Appendix F.1.
- `reduced_corpus/` (round 63): reduced corpora (all gold passages plus the frozen encoder's top 100 per question) for HippoRAG 2 and G-Retriever on the long-chain benchmarks, HippoRAG 2 metrics, BM25 baseline. Appendix D, Table 1 (BM25 row and dagger rows).
- `baselines/`: Search-R1 (round 64, released 7B checkpoint with our frozen encoder as search engine), Q-RAG (round 64, released MuSiQue checkpoint; `local_patches.diff` are memory patches to the official code), MDR (round 68, released HotpotQA checkpoint on MuSiQue and the long-chain sets), G-Retriever (round 69, PCST over the HippoRAG 2 graph), LinearRAG (round 69). Table 1.
