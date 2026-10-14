# TravelerRecLab

**How should we represent evolving traveler intent from hotel interactions—and what can offline ranking experiments actually tell us about real-time personalization?**

TravelerRecLab is a reproducible research project on session-based hotel recommendation. It studies next-click-out candidate ranking using behavioral heuristics, learned hotel embeddings, GRU sequence encoders, candidate-aware attention, and hotel property features. It was motivated by an ML system design problem: maintaining useful traveler representations as searches, views, and click-outs arrive.

The research rule is simple:

> Define the prediction contract, change one modeling idea at a time, examine who benefits and who regresses, and preserve the evidence—including evaluation corrections and negative results.

**Project status:** Research checkpoint complete at **v0.5.1**. The implemented work is **offline**; streaming updates, retrieval, production serving, and online business impact remain outside the demonstrated scope.

## At a Glance

- **Research question:** Can a learned session representation improve hotel click-out ranking, and which modeling components address which traveler behaviors?
- **Dataset:** Trivago RecSys Challenge 2019 clickstream, approximately **15.93 million events**, **730,803 users**, and **1,586,586 raw click-outs** over six days in November 2018.
- **Benchmark:** **1,276,216 eligible next-click-out examples**, with 1–50 retained preceding events and candidates reconstructed from the target click-out row.
- **Chronological splits:** **823,338 train / 229,804 validation / 223,074 test**.
- **Best evaluated neural model:** Content-aware attention-GRU, **0.5212 test tie-neutral MRR**.
- **Strongest evaluated heuristic:** Position + last referenced hotel, **0.5980 test MRR**; importantly, this uses target-row candidate ordering, an additional signal whose historical pre-click availability is unverified.
- **Most informative neural finding:** Candidate-aware attention improved earlier-hotel-return MRR **0.2903 → 0.3591**, while preserving immediate continuation.
- **Main unresolved ranking failure:** New-to-history hotel selections remain difficult; adding property tags raised their test MRR only **0.1720 → 0.1732**.
- **Scope boundary:** Click-outs are **not bookings**, candidate sets are **not an independently verified pre-click exposure log**, and offline scores are **not online conversion lifts**.

Detailed research records: [`docs/experiments/`](docs/experiments/) and [`docs/decisions/`](docs/decisions/).

## Why This Project Exists

A traveler can examine several properties, return to a hotel considered earlier, switch destinations or preferences, and eventually click out to a hotel they have never interacted with in the session. A useful representation needs to retain immediate interest **without forgetting older alternatives or ignoring unseen-to-session candidates**.

An appealing sequence model is not sufficient evidence of a good personalization system. We therefore distinguish:

1. **Behavioral memory:** Which hotels appear in the session and how recently?
2. **Learned sequence representation:** Does ordering of hotel interactions improve the ranking signal?
3. **Candidate-specific retrieval from history:** Can a candidate recover relevance from earlier interactions?
4. **Content generalization:** Can shared hotel attributes help rank hotels that are not in the session history?
5. **Serving feasibility:** Could the required inputs be available before the decision, and could the representation be updated in real time?

The first four questions were studied offline. The fifth was identified as a design requirement, **not implemented or validated**.

## Key Findings

### Learned Representation Results

All figures below are **tie-neutral ranking metrics** on the same **223,074-example chronological test split**.

| Model | Test MRR | Recall@5 | NDCG@5 | Main lesson |
|---|---:|---:|---:|---|
| Mean-pooled hotel embeddings, v0.3 | 0.4662 | 0.5712 | 0.4638 | Learning hotel vectors helps, but averaging discards order. |
| GRU session encoder, v0.3.1 | 0.5151 | 0.6037 | 0.5109 | An order-sensitive history encoder improves ranking. |
| Candidate-aware attention-GRU, v0.4 | 0.5201 | 0.6134 | 0.5176 | Earlier-hotel recovery improves. |
| **ID + property attention-GRU, v0.5.1** | **0.5212** | **0.6151** | **0.5189** | Hotel tags add a small further gain, not a discovery breakthrough. |

**Separate heuristic reference:** Position + last-item scores **0.5980 MRR**. It uses the recorded candidate list order, whereas the neural systems above do not. This is a valuable *candidate-reranking diagnostic*, **not** a feature-matched neural baseline or proof of a deployable serving feature.

### Order Matters to the Trained GRU—but Unevenly

Using the **same saved GRU checkpoint**, randomly permuting the recognized hotel history (while preserving hotel IDs and counts) changed test MRR:

| Input to trained GRU | Test MRR |
|---|---:|
| Recorded order | **0.5151** |
| One deterministic shuffled order | 0.4499 |

The **0.0652 MRR drop** demonstrates order sensitivity under an inference-time perturbation. It does **not** prove that modeling order alone caused the GRU's gain over mean pooling: the architectures differ in capacity, and shuffled sequences are out of their training distribution.

The original-order advantage is also concentrated in immediate continuations; shuffled order *improves* the return-to-earlier-hotel slice. The final recurrent state appears strongly oriented toward recent hotel identity.

See [`v0_3_1_gru_behavioral_findings.md`](docs/experiments/v0_3_1_gru_behavioral_findings.md).

### Candidate-Aware Attention Addresses an Observed Failure

| Test behavioral slice | GRU v0.3.1 MRR | Attention-GRU v0.4 MRR | Content-aware v0.5.1 MRR |
|---|---:|---:|---:|
| Immediate continuation | 0.9470 | 0.9494 | **0.9502** |
| Return to earlier hotel | 0.2903 | **0.3591** | 0.3584 |
| New hotel after exploration | 0.1718 | 0.1720 | **0.1732** |
| No prior hotel reference | 0.2701 | 0.2742 | **0.2765** |
| **All examples** | 0.5151 | 0.5201 | **0.5212** |

The attention-GRU increased **earlier-return Recall@5 from 0.5246 to 0.6786**. This supports the targeted hypothesis: candidate-specific access to historical GRU states can recover previously considered hotels without sacrificing the strong continuation signal.

It does not isolate **attention alone** from the additional parameters and final-state skip connection.

See [`v0_4_attention_gru_findings.md`](docs/experiments/v0_4_attention_gru_findings.md) and [ADR-026](docs/decisions/ADR-026-attention-gru-behavioral-evidence.md).

### Property Tags Have High Coverage, but Only Modest Impact

The metadata audit found:

| Test coverage measure | Result |
|---|---:|
| Target hotel with metadata and nonempty tags | **99.96%** |
| Candidate slots with metadata | **99.96%** |
| New-to-history target already in training ID vocabulary | **98.15%** |
| New-to-history target previously clicked in training | **82.35%** |
| Distinct relevant property tags | **157** |

Being **new to a traveler's session** is not the same as having an **unseen training ID**. The weakness on new-to-history targets cannot be explained solely by shared `UNK` embeddings.

The v0.5.1 model added learned property-tag representations to hotel-ID vectors and a content-only scoring path. Overall test MRR rose **0.5201 → 0.5212**; the new-to-history slice rose only **0.1720 → 0.1732**.

Its diagnostic also found the following association between **target exposure during training** and test ranking quality:

| Test target group | Examples | v0.5.1 MRR |
|---|---:|---:|
| Clicked as a target in training | 188,620 | 0.5462 |
| Known training ID, never clicked as a target | 30,761 | 0.3972 |
| Outside the training ID vocabulary | 3,693 | 0.2754 |

These groups are not randomized or matched; the difference is **not** a causal estimate of the effect of historical clicks. We have not shown an improvement in genuine ID cold start versus v0.4 on identical cold-start slices.

See [`v0_5_metadata_coverage_findings.md`](docs/experiments/v0_5_metadata_coverage_findings.md), [`v0_5_1_content_behavioral_and_exposure_findings.md`](docs/experiments/v0_5_1_content_behavioral_and_exposure_findings.md), and [ADR-030](docs/decisions/ADR-030-content-model-evidence-and-context-gate.md).

## Benchmark and Prediction Contract

**Data:** [Trivago RecSys Challenge 2019](https://www.recsyschallenge.com/2019/), using `train.csv` (event log) and `item_metadata.csv` (hotel properties). The repository does not distribute the raw data.

For an eligible target click-out at time $t$:

- **Input history** $H_t$: strictly preceding events in the same `(user_id, session_id)`, capped at the 50 most recent events and requiring at least one retained event.
- **Candidates** $C_t$: hotel IDs from the `impressions` field on the **target click-out row**.
- **Label** $y_t$: its clicked-out hotel, required to belong to $C_t$.
- **Task**: score and rerank candidates to place $y_t$ as high as possible.

```math
 s_\theta(c\mid H_t,C_t),\qquad
 \mathcal L=-\log\frac{\exp(s_\theta(y_t))}{\sum_{c\in C_t}\exp(s_\theta(c))}.
```

Chronological evaluation splits by **target timestamp in UTC**:

| Split | Dates (2018 UTC) | Eligible examples |
|---|---|---:|
| Train | Nov 1–4 | 823,338 |
| Validation | Nov 5 | 229,804 |
| Test | Nov 6 | 223,074 |
| **Total** | **Nov 1–6** | **1,276,216** |

These examples are **not the entire raw click-out population**. Approximately 309,673 raw click-outs lack preceding history; 697 additional history-bearing click-outs fail the target/candidate eligibility checks. The population and sequence-ordering choices are explicitly audited in [`docs/study/`](docs/study/).

### Behavioral Evaluation Slices

The four mutually exclusive categories are derived from **retained recognized hotel history**, not complete lifetime user behavior.

| Test slice | Examples | Share |
|---|---:|---:|
| Immediate continuation | 93,005 | 41.7% |
| Return to earlier hotel | 10,940 | 4.9% |
| New hotel after exploration | 86,783 | 38.9% |
| No prior hotel reference | 32,346 | 14.5% |

These slices expose regressions hidden by the overall average. A model can be excellent at continuation and poor at selecting a new hotel.

### Ranking Metrics and Tie Handling

We report **MRR**, **Recall@5**, and **NDCG@5**, using *tie-neutral* scoring for equally scored candidates. For a target tied across ranks $a$ through $b$, its reciprocal rank is averaged across all tied positions:

```math
 RR_{\mathrm{neutral}}=\frac{1}{b-a+1}\sum_{r=a}^{b}\frac{1}{r}.
```

This distinction matters because using the logged candidate order to break score ties can silently inflate performance.

## Mathematical Formulation of the Methods

The experiments all solve the same **candidate-ranking** problem. For a retained event history $H=(a_1,\ldots,a_T)$, candidates $C=\{c_1,\ldots,c_K\}$, and clicked target $y\in C$, each method produces scores $s(c\mid H,C)$. Ranking metrics depend on the order of these scores, not on calibrated click probabilities.

### Fixed-Rule Controls (v0.2–v0.2.3)

Let $h_i$ denote the hotel reference, when recognized, in event $a_i$, and let $w(a_i)$ be its fixed action-type weight. The initial controls used the following scoring rules:

| Method | Candidate score $s(c)$ | Interpretation |
|---|---|---|
| Random | $r_{\mathrm{seed}}(c,\text{example})$ | Deterministic pseudo-random sanity control, independent of preferences. |
| Global popularity | $\sum_{n\in\mathcal D_{\mathrm{train}}}\mathbf 1[y_n=c]$ | Counts training click-out targets; ignores the current history. |
| Last referenced hotel | $\mathbf 1[c=h_T^*]$ | $h_T^*$ is the most recent **recognized hotel** in the retained history. |
| Action-weighted | $\sum_{i=1}^T w(a_i)\mathbf 1[h_i=c]$ | Aggregates matching hotel interactions with fixed action weights. |
| Recency-weighted | $\sum_{i=1}^T w(a_i)e^{-\Delta t_i/\tau}\mathbf 1[h_i=c]$ | Discounts older interactions; validation selected $\tau=300$ seconds. |

Here $\Delta t_i$ is the nonnegative time gap between the target event and past event $i$. Events without a recognized hotel reference do not contribute to the hotel-matching sums. A missing last hotel produces zero last-item scores for all candidates.

The position control uses the recorded **zero-based** candidate index $j$ in a list of length $K$:

```math
p(c_j)=1-\frac{j}{K-1}\quad(K>1),
\qquad p(c_1)=1\quad(K=1).
```

The additive hybrid is

```math
s_{\mathrm{hybrid}}(c_j)=p(c_j)+\lambda\,\mathbf 1[c_j=h_T^*],
```

with $\lambda\in\{0,0.25,0.5,1,2,4\}$ evaluated on validation. The selected $\lambda=2$ and $\lambda=4$ produced identical rankings and metrics in the reported sweep. This hybrid intentionally uses the **logged candidate position**, unlike the neural encoders.

### Learned Hotel Embeddings and Mean Pooling (v0.3)

Each train-vocabulary hotel has an embedding $e_h\in\mathbb R^d$ and candidate bias $b_h$, with $d=32$ in the full runs. For the $L$ recognized hotel references in the retained history:

```math
z_{\mathrm{mean}}=\frac{1}{L}\sum_{i=1}^{L}e_{h_i},\qquad
s_{\mathrm{mean}}(c)=\frac{z_{\mathrm{mean}}^\top e_c}{\sqrt d}+b_c.
```

An empty recognized-hotel history is handled separately by the implementation. The substantive limitation is **permutation invariance**: the hotel sequences $[A,B,C]$ and $[C,B,A]$ have the same average representation.

### GRU Session Encoder (v0.3.1)

The GRU replaces averaging with ordered recurrent updates:

```math
h_i=\operatorname{GRU}(e_{h_i},h_{i-1}),\qquad
z_{\mathrm{GRU}}=h_L,\qquad
s_{\mathrm{GRU}}(c)=\frac{z_{\mathrm{GRU}}^\top e_c}{\sqrt d}+b_c.
```

The GRU has trainable gates that can retain or update the session state as hotel interactions arrive. Unlike averaging, $h_L$ can depend on the order of the sequence. Our **inference-time shuffle** test kept hotel IDs and counts fixed while changing their order; it measured the sensitivity of this trained encoder, not an independently trained shuffled-history architecture.

### Candidate-Aware Attention over GRU States (v0.4)

Instead of scoring every candidate using only the final state, preserve $h_1,\ldots,h_L$. Each candidate $c$ generates a query $q_c=W_qe_c$ and attends to valid historical states:

```math
\alpha_{c,i}=\frac{\exp(q_c^\top h_i/\sqrt d)}{\sum_{j=1}^{L}\exp(q_c^\top h_j/\sqrt d)},
\qquad v_c=\sum_{i=1}^{L}\alpha_{c,i}h_i.
```

The implementation uses a learned **shared scalar blend**, $\beta=\sigma(g)$, to retain the final GRU state while incorporating candidate-specific context:

```math
s_{\mathrm{attn}}(c)=
\frac{e_c^\top(h_L+\beta v_c)}{\sqrt d}+b_c.
```

For empty hotel histories, it bypasses attention and uses a learned empty-history state. The v0.4 gain cannot be assigned exclusively to attention, because the new query projection, blend and additional parameters also change the architecture.

### ID + Property Content-Aware Attention (v0.5.1)

Let $P(c)$ be the available metadata tags for hotel $c$, with learned tag embeddings $v_p$. An empty property list contributes a zero vector. The hotel representation is:

```math
m(c)=\begin{cases}
\frac{1}{|P(c)|}\sum_{p\in P(c)}v_p,& |P(c)|>0,\\
0,&|P(c)|=0,
\end{cases}
\qquad
\widetilde e_c=e_{\mathrm{ID}(c)}+\gamma m(c),\quad\gamma=0.5.
```

The model uses $\widetilde e$ for **both historical hotel inputs and candidates**, then applies the same GRU and candidate-aware attention construction with the resulting states. Its candidate score additionally includes a learned linear *content-only prior*:

```math
s_{\mathrm{content}}(c)=
\frac{\widetilde e_c^\top(h_L+\beta v_c)}{\sqrt d}
+b_{\mathrm{ID}(c)}+w_{\mathrm{prior}}^\top m(c).
```

An ID absent from the training vocabulary maps to a shared unknown-ID embedding, but its properties can still produce a distinct $m(c)$. The added prior allows property-based ranking when no hotel history is recognized. It also means the experiment changes more than just the embedding input.

### Shared Training Objective and Evaluation

All neural systems optimize a candidate-restricted softmax loss on the clicked hotel:

```math
P_\theta(y\mid H,C)=\frac{\exp(s_\theta(y))}{\sum_{c\in C}\exp(s_\theta(c))},
\qquad
\mathcal L(\theta)=-\frac{1}{N}\sum_{n=1}^{N}\log P_\theta(y_n\mid H_n,C_n).
```

A candidate padding position receives score $-\infty$ and is excluded from the denominator. Model selection used validation tie-neutral MRR. For target rank $r_n$, the untied metric is $\mathrm{MRR}=N^{-1}\sum_n1/r_n$; tied target ranks are handled by the tie-neutral expected reciprocal rank defined above.

These equations describe the **implemented offline scoring families**, not a full probabilistic model of bookings, travel intent, or a streaming representation update service.

## Research Evolution

> audit the task → establish controls → learn embeddings → model order → retrieve earlier interests → test hotel content

| Milestone | Question | Result | Disposition |
|---|---|---|---|
| v0.0–v0.1 | What exactly is a valid prediction example? | Raw schema and semantics audits; 1.276M eligible sequences | Retain benchmark contract |
| v0.2–v0.2.3 | Which simple ranking signals are informative? | Last-item 0.5049; position + last-item 0.5980 tie-neutral MRR | Retain with feature-source caveat |
| v0.3 | Can a learned hotel embedding summarize history? | Mean pooling: 0.4662 MRR | Retain as nonsequential control |
| v0.3.1 | Does a GRU improve ranking and react to event order? | GRU 0.5151; shuffled-input diagnostic 0.4499 | Retain, document order sensitivity |
| v0.4 | Can candidate-aware attention recover older interests? | Earlier-return MRR 0.2903 → 0.3591 | Hypothesis supported at architecture level |
| v0.5 | Is hotel content available for candidate encoding? | 99.96% test target metadata coverage; 157 tags | Retain audit |
| **v0.5.1** | Does content improve new-hotel ranking? | Overall 0.5212; new-to-history 0.1732 | **Small positive result; close project here** |

## Negative Findings and Evaluation Corrections

TravelerRecLab keeps failures and protocol corrections visible:

- **Tie-breaking correction:** Last-item MRR appeared to be **0.5980** with logged-order ties but was **0.5049** using tie-neutral evaluation. The original ranking incorporated upstream list order through its tie-breaks.
- **Recent-item dominance:** The original GRU achieved **0.9470** immediate-continuation MRR but only **0.2903** earlier-return MRR. An attractive overall metric masked a specific weakness.
- **Shuffling exception:** A randomized history damaged overall GRU performance but **improved** the earlier-return segment. Order sensitivity is not synonymous with uniform benefit.
- **Diminishing returns:** Property embeddings made only a **+0.0011** absolute MRR improvement overall versus v0.4, despite almost complete metadata coverage.
- **Cold-start claim not established:** Performance on unseen-ID targets was measured for v0.5.1, but a matched v0.4 comparison on precisely those targets was not completed.

The test split was inspected repeatedly for **post-hoc diagnosis** over successive milestones. These results should be treated as **exploratory offline research**, not as a pristine one-shot confirmatory benchmark. There are no session-clustered confidence intervals or demonstrated online uplifts.

## Models and Experimental Protocol

**Non-neural controls:** random, global popularity, last referenced hotel, action-weighted history, recency-weighted history, position-only, and position-plus-last-item.

**Neural sequence:** train-only hotel-ID vocabulary → hotel vectors → mean pooling or GRU states → candidate-specific scoring → candidate-softmax training. v0.4 adds attention over historical GRU states; v0.5.1 adds property-tag embeddings and a content scoring path.

Experiments consistently use:

- training data for gradients and vocabulary membership;
- chronological validation data for checkpoint and predeclared parameter selection;
- the same eligible candidate-ranking population for full-run comparisons;
- tie-neutral metrics and behavioral breakdowns;
- saved metrics/checkpoints and experiment-specific decision records;
- explicit disclosure of unverified historical feature availability.

The comparisons are **not parameter-count matched**. The initial three-epoch neural runs were still improving on validation, so convergence is **not established**. Training examples were processed in file order rather than shuffled mini-batches.

## Reproducibility

Requires **Python 3.12**. Download the competition data separately and place the files as shown; they are intentionally excluded from Git.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,training]'
pytest
```

Build the candidate-ranking benchmark:

```bash
python -m travelerreclab.data.build_sequences \
  --input data/raw/train.csv \
  --output data/processed/next_clickout.jsonl \
  --max-history 50 --min-history 1
```

Run the metadata audit:

```bash
python -m travelerreclab.data.audit_hotel_metadata_v05 \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/hotel_metadata_audit_v05.json
```

Train the final researched model:

```bash
python -m travelerreclab.models.gru_attention_content \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051.json \
  --epochs 3 --batch-size 256 --dim 32
```

Evaluate saved-checkpoint behavioral and training-exposure slices:

```bash
python -m travelerreclab.models.gru_attention_content_diagnostics \
  --checkpoint data/processed/gru_attention_content_v051.pt \
  --input data/processed/next_clickout.jsonl \
  --metadata data/raw/item_metadata.csv \
  --output data/processed/gru_attention_content_v051_test_diagnostics.json \
  --split test
```

Note: running on the original Trivago data is resource-intensive; raw data, processed examples, and `.pt` weights are **not** included in this repository. Commands reproduce the experimental *procedure*, not a guarantee of byte-identical floating-point results across environments.

## Repository Structure

```text
TravelerRecLab/
├── configs/                         # configuration scaffolding
├── data/                            # local raw and processed data (not committed)
├── docs/
│   ├── decisions/                   # architecture decision records
│   ├── experiments/                 # protocols, findings, limitations
│   └── study/                       # raw-data semantics and behavior audits
├── src/travelerreclab/
│   ├── data/                        # sequence construction and data audits
│   ├── evaluation/                  # heuristics and ranking diagnostics
│   └── models/                      # embedding, GRU, attention, content models
├── tests/                            # regression and synthetic fixture tests
├── pyproject.toml
└── README.md
```

Selected research reports:

- [`v0_2_3_position_behavior_hybrid_findings.md`](docs/experiments/v0_2_3_position_behavior_hybrid_findings.md)
- [`v0_3_mean_pool_findings.md`](docs/experiments/v0_3_mean_pool_findings.md)
- [`v0_3_1_gru_behavioral_findings.md`](docs/experiments/v0_3_1_gru_behavioral_findings.md)
- [`v0_4_attention_gru_findings.md`](docs/experiments/v0_4_attention_gru_findings.md)
- [`v0_5_metadata_coverage_findings.md`](docs/experiments/v0_5_metadata_coverage_findings.md)
- [`v0_5_1_content_behavioral_and_exposure_findings.md`](docs/experiments/v0_5_1_content_behavioral_and_exposure_findings.md)

## From Offline Research to a Real-Time System

The motivating ML system design question concerns **live traveler state**, not merely an offline benchmark. A production design would connect event producers to a stream processor that deduplicates and orders events, maintains session state, updates a compact traveler representation, and makes it available to a candidate retrieval and ranking service. It would also require availability filtering, caching and fallbacks, freshness monitoring, point-in-time training joins, and online experimentation.

**None of these production capabilities were implemented or benchmarked here.** In particular, the project has not demonstrated incremental GRU-state updates, a streaming feature store, serving latency, full-catalog retrieval, bookings, or A/B-tested business impact.

The offline modeling work helps formulate those design choices. It does **not** establish that the current trained model is directly deployable.

## Limitations

- **Target-row candidate reconstruction:** Impressions and prices come from the click-out event; they are not independently verified to have been available before that event.
- **Label scope:** The task predicts click-outs among reconstructed candidates, not saves, bookings, satisfaction, or incremental business value.
- **Historical feature availability:** Property metadata is a snapshot, not a point-in-time-verified serving feature. Rating-derived property tags warrant particular care.
- **Incomplete intent/context:** The final model uses hotel IDs and property tags, not destination changes, filters, time gaps, prices, or full search context.
- **Population selection:** No-history raw click-outs are excluded; sessions are truncated to 50 earlier events; some event-order semantics remain provisional.
- **Evaluation exposure:** Test results were reused for descriptive diagnostics and later milestone interpretation; no untouched confirmatory holdout is claimed.
- **Uncertainty:** No clustered confidence intervals; multiple examples from the same session are correlated.
- **Training and model controls:** Three-epoch comparison does not prove convergence, models differ in parameter count, and explicit attribute-only and matched cold-start ablations remain incomplete.
- **Operational scope:** No production throughput, freshness, latency, state-store, or online experiment results.

## Current Conclusion

TravelerRecLab asks:

> How can we represent evolving traveler intent, and which signals can we trust when ranking hotels?

Our evidence points to three complementary lessons:

1. **Sequential behavior matters:** GRU history encoding outperformed mean pooling, and the trained GRU was sensitive to event order.
2. **Remembering the past matters:** Candidate-aware attention substantially improved recovery of hotels considered earlier in a session.
3. **History and static hotel tags are not enough for discovery:** Property features were abundant but delivered only a small lift on hotels new to the current session.

The final result is an **offline, documented modeling study**—not a deployed Expedia system. The natural next research question would be how to incorporate *causally available search intent* and maintain traveler state online; that work is deliberately left outside this project's completed scope.
