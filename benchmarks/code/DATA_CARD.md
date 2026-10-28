# Data card: 2Wiki-Long and HotpotQA-Long

Harder multi-hop retrieval benchmarks built from 2WikiMultiHopQA and HotpotQA. Every question needs 3–5 chained pieces of evidence (or 6–8 for multi-entity comparisons), the final evidence cannot be retrieved from the question alone, and train / dev / test share no single-hop component.


## Refined release (evaluation only, used in the paper)

The released benchmark is evaluation only: models are trained elsewhere (MuSiQue in the paper) and tested zero-shot. It is a refined subset of the version described below, built by `finalize_refined.py`.

| | 2Wiki-Long test | 2Wiki-Long dev | HotpotQA-Long test | HotpotQA-Long dev |
|---|---:|---:|---:|---:|
| Questions | 300 | 100 | 300 | 100 |
| 3-hop / 4-hop / 5-hop / comparison | 170 / 45 / 12 / 73 | 36 / 23 / 7 / 34 | 100 / 100 / 100 / – | 41 / 40 / 19 / – |
| Genealogy questions | 111 (37%) | 40 (40%) | – | – |
| Origin: original dev / original train | 111 / 189 | 0 / 100 | 165 / 135 | 0 / 100 |
| Corpus (passages) | 220,000 | 220,000 | 554,812 | 554,812 |
| Corpus: gold / BM25 neighbours / random | 1,017 / 95,880 / 123,103 | 582 / 65,402 / 154,016 | 2,374 / 328,648 / 223,790 | 558 / 91,731 / 462,523 |

Refinement steps on top of the filters below:
1. **Quality.** Only questions rated naturalness 4 or 5. Each question is then re-verified by a second model family (gpt-5.4-mini): it must answer correctly from the shuffled gold passages and judge the question unambiguous with the reference as its only answer. Pass rates: 2Wiki-Long 62% (test pool) and 51% (dev pool); HotpotQA-Long 98% and 97%. A 100-question human audit sheet per test set is in `refine/*_test_audit.csv`.
2. **Balance.** 2Wiki-Long caps genealogy questions (chains of only father/mother/spouse/child/sibling, or kinship comparisons) at 40%. Types are sampled round-robin.
3. **Harder corpus.** About 4× the size of the earlier test corpora: all gold pages of the verified pool, then BM25 nearest neighbours of every gold page (round-robin over gold pages, up to rank 200), then random fill from the source collection. The BM25 shortcut filter is re-run against this corpus (HotpotQA-Long test drops 617 → 491 before sampling).

Test questions come from both original splits because the original-dev pool alone was too small after verification; `origin_split` in `raw.json` records which. Since the release is evaluation only, this does not create train/test overlap. Each `raw.json` record also carries `type`, `hops`, `genealogy`, `evidence` and `naturalness`.

The sections below describe the full 2,000 / 500 / 500 build that the refined release is drawn from.

## Sizes

| Split | 2Wiki-Long | HotpotQA-Long |
|---|---:|---:|
| train | 2,000 | 2,000 |
| dev | 500 | 500 |
| test | 500 | 500 |

### Question types

| Split | 2Wiki-Long: 3-hop / 4-hop / 5-hop / comparison | HotpotQA-Long: 3-hop / 4-hop / 5-hop |
|---|---|---|
| train | 703 / 71 / 2 / 1,224 | 826 / 701 / 473 |
| dev | 142 / 142 / 75 / 141 | 211 / 162 / 127 |
| test | 144 / 137 / 53 / 166 | 192 / 192 / 116 |

Comparisons involve 3 or 4 entities, each reached in 2 hops (6–8 gold passages). 2Wiki-Long train is dominated by 3-hop chains and comparisons: long chains in 2Wiki are mostly genealogies, and dev/test consumed most of them under the component-disjointness rule.

### Corpora and gold passages

| Split | Corpus (passages) | Gold passages (with repeats / distinct) |
|---|---:|---:|
| 2Wiki-Long train | 369,378 | 10,946 / 7,271 |
| 2Wiki-Long dev | 369,378 | 2,337 / 1,702 |
| 2Wiki-Long test | 54,957 | 2,383 / 1,167 |
| HotpotQA-Long train | 640,333 | 7,647 / 6,852 |
| HotpotQA-Long dev | 529,109 | 1,916 / 1,854 |
| HotpotQA-Long test | 138,703 | 1,924 / 1,762 |

## Sources

- 2WikiMultiHopQA (official train / dev, with `evidences`, `entity_ids`, `supporting_facts`, `context`).
- HotpotQA (official train v1.1 / dev distractor) and the HotpotQA 2017 Wikipedia abstracts with hyperlinks (`enwiki-20171001-pages-meta-current-withlinks-abstracts`).
- Test is built only from the original dev split. Dev (used for early stopping) and train are built from the original train split.

## Construction

**2Wiki-Long.** All evidence triples of a source split form a graph (Wikidata entity IDs; literal values such as dates keyed by their text). Only functional edges (one object per subject and relation) are kept, so every hop has a unique answer.
- Chains: k-hop paths e0 → e1 → … → ek (k = 3, 4, 5). Every hop is supported by the subject's paragraph, which is that hop's gold passage. No entity repeats in a path. Degenerate relation pairs (for example mother → spouse, which is just the father) are banned. Pure genealogy chains are capped at half of the chains.
- Multi-entity comparisons: 3–4 entities, each mapped through one relation (director, father, founder, and so on) to a person, and then to a date. The question asks which entity's related person has the earliest or latest date. 60% use non-kinship relations.

**HotpotQA-Long.** Each chain starts from an original HotpotQA bridge question's two pages (A → B), then follows hyperlinks in the 2017 abstracts (B → C → …) to reach k = 3, 4 or 5 pages. Each hop keeps the sentence that carries the link. Hops go only to named-entity pages (concept pages, years, lists and disambiguation pages are skipped), never through list-like sentences (five or more links), and never to pages with in-degree above 300.

**Question writing.** deepseek-flash writes one question per chain. It may name only the starting entity (or, for comparisons, the compared entities), must use every relation in order, and may not mention any intermediate entity or the answer. For HotpotQA-Long, the model also picks the answer: a specific fact stated in the final page.

## Filters

1. **No leak.** The question contains no name of an intermediate entity or of the answer (string match).
2. **Answerable from the gold passages.** Given all gold passages (shuffled), the model's answer matches the reference (normalized EM, F1 ≥ 0.8, or an LLM equivalence judgment).
3. **Unique answer.** The model judges the question unambiguous, with the reference as its only correct answer.
4. **Natural.** The model rates naturalness 1–5. Questions rated 3 get one rewrite attempt, and the rewrite must pass checks 1–4 again. Only questions rated 3 or higher are kept; sampling prefers 4–5.
5. **No retrieval shortcut.** Question-only BM25 over the split's own corpus must not place the final-hop gold passage in its top 10 (for comparisons: must not place all gold passages in its top 10).

Closed-book answerability and partial-evidence answerability were also measured, but only recorded as tags. A strong LLM remembers many of these facts, which does not affect retrieval evaluation.

### What the LLM filters removed (candidates that reached the LLM gate)

| | 2Wiki-Long | HotpotQA-Long |
|---|---:|---:|
| Processed | 5,188 | 7,890 |
| Passed | 4,017 (77.4%) | 4,312 (54.7%) |
| Failed: leak | 24 (0.5%) | 563 (7.1%) |
| Failed: not answerable from gold | 231 (4.5%) | 57 (0.7%) |
| Failed: answer not unique | 895 (17.3%) | 746 (9.5%) |
| Failed: not natural | 21 (0.4%) | 2,212 (28.0%) |

The BM25 shortcut filter then removed a further 2–5% of 2Wiki-Long and 21–31% of HotpotQA-Long candidates.

## Splits

- **No shared single-hop component** (overlap 0 for every pair of splits). For 2Wiki-Long, a component is a triple (subject ID, relation, object ID); for HotpotQA-Long, it is a page-to-page link edge.
- **No shared question text.**
- **Test gold passages that are also gold in train:** 9.1% for 2Wiki-Long, 7.5% for HotpotQA-Long. These are the same paragraph used in a different chain.

## Corpus

- **Test:** closed pool made of the original dev split's context paragraphs plus all gold pages.
- **Dev / train:** closed pool made of the original train split's context paragraphs plus all gold pages.
- **HotpotQA-Long only:** the top-20 full-Wikipedia BM25 neighbours of every gold page are added as hard distractors, so pages reached by hyperlink are not easy to spot as unusual.
- Gold pages use the text the chain was built on.

## Format

Same as the R26 inputs: `inputs/{train,dev,test}/` hold `questions.json` ([{qid, question}]), `raw.json` (full records: answer, type, hops, per-hop evidence, naturalness, closed-book tag, BM25 hits), `corpus.json` ([{title, text}]) and `ordered_gold.json` ({qid: [corpus indices in hop order]}).

Location: `239:/home/dmir/qiuchenjie2023/revision_20260928/round53/claude/final/{2wiki_long,hotpotqa_long}/`.

## Known limitations

- 2Wiki-Long chains lean toward royal and aristocratic genealogy. Dynasty names that repeat across hops make some middle hops lexically easy.
- 2Wiki-Long train has very few 4–5-hop chains.
- One model family (deepseek-flash) both writes and checks the questions. Naturalness was judged by the LLM; a manual check of 24 sampled questions (12 per dataset, before the naturalness filter was added) found every answer supported by its evidence, with about 7 of 12 HotpotQA-Long questions reading fully naturally. A larger human audit is still needed.
- A strong LLM can answer many questions closed-book (about 85% of the 2Wiki-Long sample, about 50% of the HotpotQA-Long sample). Question-answering evaluations should use the tagged closed-book-unanswerable subset.
