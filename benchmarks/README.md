# benchmarks: HotpotQA-Long and 2Wiki-Long

Evaluation-only multi-hop benchmarks with chains of three to five hops (2Wiki-Long also has comparison questions over six to eight passages). See `code/DATA_CARD.md` and Appendix E of the paper.

## data
- `data/*/test/questions.json`: test questions (300 per benchmark).
- `data/*/test/ordered_gold.json`: gold passage ids in chain order, indexing the corpus built by `code/finalize_refined.py`.
- `data/*/test/raw.json`: full records (answer, type, hops, gold passages with title and text, origin split, naturalness).
- `data/*/dev/dev_pool.json`: the verified development pool (160 HotpotQA-Long and 280 2Wiki-Long candidates with question, answer, filter outcomes and gold chain); `finalize_refined.py` samples the 100-question development split from it.

The corpora (554,812 and 220,000 passages, 228 MB and 102 MB) are not included. `code/finalize_refined.py` rebuilds them from the original HotpotQA 2017 Wikipedia abstracts and 2WikiMultiHopQA paragraphs: every gold passage of the pool, then BM25 nearest neighbours of each gold passage (round-robin), then random passages.

## code
- `build_hotpot.py`, `build_hotpot_corpus.py`: hyperlink chains over HotpotQA abstracts.
- `build_2wiki.py`: functional-relation chains and comparison questions over 2WikiMultiHopQA evidence triples.
- `llm_pipeline.py`: question writing and filters (no leak of intermediate entities, answerable from shuffled gold passages, unique answer, naturalness); contains the prompts.
- `verify53.py`: re-verification by a second model family.
- `shortcut_bm25.py`, `shortcut_bm25_hotpot.py`, `eval_bm25_corpus.py`: question-only BM25 shortcut filter and corpus checks.
- `finalize53.py`, `finalize_refined.py`: sampling, type balance and corpus construction of the released split.
