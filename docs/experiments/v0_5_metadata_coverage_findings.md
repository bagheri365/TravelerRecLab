# v0.5 — Hotel metadata coverage and generalization audit: findings

## Status and provenance

**Status:** completed descriptive audit; no content model trained.
**Input:** `data/processed/next_clickout.jsonl`, `data/raw/item_metadata.csv`.
**Output:** `data/processed/hotel_metadata_audit_v05.json` (the full run provided for review).
**Chronology:** train = November 1–4, validation = November 5, test = November 6, 2018 (UTC target time).
**Population:** eligible constructed next-clickout candidate-ranking examples, **not** all raw clickouts and not production exposures.

Reproduce the source report:

```bash
python -m travelerreclab.data.audit_hotel_metadata_v05 \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/hotel_metadata_audit_v05.json
```

Counts below are copied or derived from that audit report; percentages may differ in the last decimal due to rounding. The report uses exact string `item_id` joins, first matching metadata row, and retained hotel-bearing history. Its train vocabulary contains hotel IDs from **training candidate lists or hotel-bearing training history**, with `min_count=1`; it is not restricted to clicked training targets.

## 1. Scope and key populations

| Quantity | Count |
|---|---:|
| Training eligible examples | 823,338 |
| Validation eligible examples | 229,804 |
| Test eligible examples | 223,074 |
| Training-vocabulary distinct hotel IDs | 738,063 |
| Distinct training-positive target hotel IDs | 210,093 |
| Metadata rows | 927,142 |
| Relevant unique IDs with metadata and nonempty properties | 825,663 |
| Distinct property tags among relevant IDs | 157 |
| Relevant IDs with duplicate metadata rows | 0 |

The 825,663 number counts distinct *relevant* IDs that successfully join to metadata, not the count of all metadata rows.

## 2. Validation and test coverage

| Measurement | Validation count / denominator | Validation rate | Test count / denominator | Test rate |
|---|---:|---:|---:|---:|
| Target with metadata | 229,701 / 229,804 | 99.955% | 222,984 / 223,074 | 99.960% |
| Target with nonempty properties | 229,701 / 229,804 | 99.955% | 222,984 / 223,074 | 99.960% |
| Target in train vocabulary | 226,145 / 229,804 | 98.408% | 219,381 / 223,074 | 98.344% |
| Target seen as positive training target | 195,134 / 229,804 | 84.913% | 188,620 / 223,074 | 84.555% |
| Target present in retained hotel history | 107,283 / 229,804 | 46.685% | 103,945 / 223,074 | 46.597% |
| Candidate **slots** with metadata | 5,295,680 / 5,297,735 | 99.961% | 5,140,147 / 5,142,220 | 99.960% |
| Candidate **slots** in train vocabulary | 5,206,260 / 5,297,735 | 98.273% | 5,049,624 / 5,142,220 | 98.199% |

Candidate-slot fractions are weighted by appearances in clickout candidate lists. They are **not** coverage rates across distinct candidate hotel IDs. Almost universal presence of metadata is a feasibility finding, not evidence that metadata improves ranking.

## 3. New to this session is different from unknown to the model

`new_hotel_after_exploration` means the clicked target did not occur in the **retained preceding hotel-ID history**. It does not imply that the hotel ID was missing during training.

| Test measurement for `new_hotel_after_exploration` | Count / 86,783 | Rate |
|---|---:|---:|
| Target absent from retained hotel history | 86,783 / 86,783 | 100% by definition |
| Target with metadata and nonempty properties | 86,735 / 86,783 | 99.945% |
| Target in training vocabulary | 85,177 / 86,783 | 98.149% |
| Target previously clicked as a training positive | 71,467 / 86,783 | 82.351% |
| Target **not** in train vocabulary | 1,606 / 86,783 | 1.851% |

Among this behavioral segment, only 1,606 targets are ID-vocabulary-unknown. Consequently, the v0.4 attention-GRU's poor 0.1720 test MRR on this segment **cannot be explained solely by cold-start target IDs**. Whether the failure instead reflects history matching bias, candidate representation limitations, search context omission, objective design, or optimization remains untested.

For comparison, test `no_prior_hotel_reference` has 32,346 examples, 32,335 metadata-matched targets (99.966%), 31,771 train-vocabulary targets (98.222%), and 27,651 train-positive targets (85.485%). These are behavior labels based on retained hotel history, not global cold-start categories.

## 4. Property content: coverage is not feature quality

The relevant joined IDs have **157 distinct property tags**. The empirical histogram of nonempty tag counts across **825,663** relevant IDs yields **17 median tags**, approximately **20.64 mean tags**, and **160,434 IDs with exactly one tag** (~19.43%). Frequent tags include `Satisfactory Rating`, `Good Rating`, `Car Park`, `WiFi (Rooms)`, `Air Conditioning`, `Hotel`, `House / Apartment`, `Luxury Hotel`, and `Business Hotel`.

Tags mix property amenities, accommodation types, star/category labels, and rating-related information. The audit does not establish whether these features were valid at the historical prediction time. Rating-derived tags warrant particular scrutiny for timestamp leakage. No analysis here establishes tag predictive power, within-city discrimination, or robustness to missingness.

## 5. Model-facing implications (hypotheses, not empirical improvements)

1. **Content is feasible:** near-universal target and candidate-slot metadata coverage and a compact 157-tag vocabulary justify a controlled ID-plus-content candidate encoding experiment.
2. **Main discovery weakness is not global ID cold start:** ~98.15% of new-to-history test targets have known training-vocabulary IDs. Improvements must be evaluated on new-to-history *known IDs* as well as unknown IDs.
3. **Candidate-position remains a separate feature family:** original clickout candidate order is from target rows; do not blend it into a nominal history-only representation comparison without labeling the benchmark appropriately.
4. **Content may help but is unproven:** metadata presence and overlapping tags are not measured evidence of better candidate discrimination or personalization.

Proposed **future** experiment: retain the attention-GRU history encoder and candidate-softmax objective; add a lightweight property-tag embedding aggregate to candidate encodings, keeping an explicit ID-only reference. Prefer one predetermined scale or a validation-only sweep. Track overall MRR/Recall@5/NDCG@5; immediate continuation; earlier returns; new-to-history; no-prior-hotel-history; training-vocabulary-known versus unknown targets; train-positive-known versus never-positive targets; and metadata-missing cases. Distinguish feature-matched from parameter-matched claims.

A simple candidate feature construction to test is:

\[
e_h=e_{\mathrm{ID}}(h)+\gamma\;\frac{1}{|P(h)|}\sum_{p\in P(h)}v_p
\]

with a defined missing-metadata fallback and vocabulary/OOV rules. The formula is a proposed design, not an implemented or evaluated result. Unknown-ID properties could distinguish unknown hotels in principle, but **only** if their property feature vectors are genuinely available and correctly joined.

## 6. Methodological cautions

- Candidate lists are retrieved from the **target clickout event**, not independently observed pre-click candidate exposures. This is a reconstructed candidate-reranking benchmark.
- Availability of snapshot metadata at the original event timestamp has **not** been proven. Historical backtesting with such data carries point-in-time leakage risk.
- Train vocabulary includes observed candidates and hotel-bearing history, not just training positives. Train-vocabulary-known is not evidence a hotel was clicked or its embedding adequately trained.
- Statistics use `min_count=1`; the effect of frequency thresholds was not measured.
- All taxonomy counts depend on retained 1–50 earlier events and provisionally typed hotel-bearing actions. Missing older references can change whether a hotel is called new-to-history.
- Multiple clickouts per session are correlated; coverage rates here are descriptive, not inferential statistics.
- No v0.5 content-model quality, serving viability, causal effect, or cold-start improvement has been measured.

## 7. Decision and checkpoint gate

**Accept the metadata feasibility audit as complete.** Proceed only to a *controlled* content-aware candidate-encoder proposal, contingent on labeling the metadata snapshot's point-in-time uncertainty and retaining ID-only v0.4 comparisons. See ADR-028. Do not claim that v0.5 has already solved novel-hotel ranking.
