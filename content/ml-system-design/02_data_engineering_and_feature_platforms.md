# Module 02 — Data Engineering & Feature Platforms

## Why this module matters

"The model is 10% of the system" is the oldest cliché in ML engineering and it is still under-believed. In interviews, data questions ("where do labels come from?", "how do you keep features consistent?") expose juniors faster than any modeling question. In 2026 there's a second data discipline layered on top: **curating and synthesizing data for LLMs**, which has become a primary competitive moat (FineWeb, DCLM, and Nemotron showed that data curation moves benchmarks as much as architecture does).

## 1. Storage and pipelines

- **Lakehouse pattern:** raw data lands in object storage as Parquet, governed by a table format — **Apache Iceberg** has effectively won (Delta Lake remains in Databricks shops) — giving ACID transactions, schema evolution, and time travel over cheap storage. Time travel matters for ML specifically: it's how you reproduce "the dataset as of the March 3rd training run."
- **Batch processing:** Spark (or increasingly Polars/DuckDB/Ray Data for single-node-to-medium scale — don't reach for Spark in an interview if the data fits on one box; it signals cargo-culting).
- **Streaming:** Kafka (transport) + Flink (stateful compute) for features that must be fresh in seconds — fraud velocity counters, in-session behavior. **CDC** (change data capture, e.g., Debezium) streams operational-DB changes into the lake so analytics/ML never query production databases directly.
- The interview-relevant skill is choosing the **freshness tier** per feature: static (daily batch), near-real-time (minutes, micro-batch), real-time (seconds, streaming). Each tier costs roughly an order of magnitude more to operate than the previous one.

## 2. Feature stores

A feature store solves three problems at once: **reuse** (one definition of `user_7d_purchase_count` shared across teams), **online/offline consistency**, and **point-in-time correctness**.

- **Offline store** (lakehouse tables): serves training, supports **point-in-time joins** — for a training example at time *t*, join the feature values as they were at *t*, never after. Getting this wrong is *label leakage via features* and produces models that are great offline and dead online.
- **Online store** (Redis/DynamoDB/Cassandra): serves the same features at <10 ms for inference.
- **Materialization** jobs keep the two in sync. Open source: **Feast** (the de facto OSS standard); managed: Tecton, Databricks/SageMaker/Vertex feature stores. Know the architecture, not the vendor matrix.

For LLM systems the analogous concept is the **context platform**: the retrieval indexes, user memory, and document stores that get assembled into the prompt. Same consistency and freshness problems, new name.

## 3. Labels

- **Natural labels:** clicks, purchases, dwell time, "did the agent's draft get sent unedited?" — design the product so labels fall out of usage (this is the data flywheel introduced earlier). Beware **delayed labels** (chargebacks arrive 60 days later — train with importance-corrected or two-stage labels) and **degenerate feedback loops** (the model only gets labels on items it chose to show; fix with exploration traffic or logged propensities).
- **Programmatic/weak supervision:** labeling functions + denoising (Snorkel lineage) — still useful, but largely superseded in practice by:
- **LLM-as-labeler:** frontier model labels at scale, humans audit a sample, a small model is trained on the result (distillation). The 2026 default for bootstrapping classifiers. Critical discipline: measure LLM-labeler agreement against a human-gold subset *per slice*, not just overall.
- **Active learning:** spend the human budget on uncertain/disagreement examples. Pairs naturally with the LLM-labeler audit loop.

## 4. Synthetic data (a defining 2026 topic)

- **For training small/domain models:** generate inputs (rendered documents, simulated dialogs, code with tests) plus labels at near-zero marginal cost. Document AI is a flagship case: template-driven rendering (HTML→browser→image) produces pixel-perfect ground truth — coordinates, fields, reading order — that would cost dollars per page to hand-label.
- **Known failure mode:** distribution gap. Synthetic data tends to nail *visual/semantic surface realism* while missing the **physical degradation distribution** of real data — perspective distortion, crumple, blur, sensor noise. Models trained purely on clean synthetic data fall off a cliff on real captures; the fix is a measured augmentation pipeline (geometric warps first — they dominate the gap — then photometric noise) plus a small real-data fine-tuning set. In interviews, volunteering this failure mode is a strong senior signal.
- **For LLM pretraining/post-training:** distillation from stronger models (instruction data à la Alpaca→Nemotron lineage), rephrasing corpora (WRAP/Cosmopedia-style textbook synthesis), and RL-generated reasoning traces. Risks: **model collapse** under recursive training on own outputs (manage by anchoring on real data and deduplicating), license/ToS constraints on distilling from commercial APIs, and benchmark **contamination** (decontaminate by n-gram/embedding match against eval sets).

## 5. Data work specific to LLMs

- **Deduplication** at scale: MinHash-LSH near-dedup — dedup is one of the highest-leverage quality interventions in pretraining (and in fine-tuning sets, where near-duplicates silently overweight a pattern).
- **Quality filtering:** classifier-based (FineWeb-Edu style "educational value" scorers) beat hand-rules; data mixing/curriculum (what fraction code vs web vs math, upweighting high-quality sources late in training) is an active lever.
- **Tokenization-aware packing:** SFT examples are packed into fixed-length sequences with proper attention masking; chat-template errors are the #1 silent killer of fine-tunes (covered in the training chapter).

## Going deeper

- Point-in-time correctness is the concept to master first — it is the single most common source of the "great offline, dead online" failure.
- Large-scale data-curation results have made data quality a measurable science: careful deduplication and quality filtering can beat training on several times more unfiltered tokens. Study the published corpus-curation pipelines (near-dedup, classifier-based quality scoring, decontamination) as reference designs.
- Industrial synthetic-data generation for post-training is well documented; the recurring lesson is to measure transfer to real data and to guard against model collapse under recursive self-training.
- Open feature-store architectures (online/offline split, materialization, point-in-time joins) and real-time feature-pipeline write-ups are the best way to ground the concepts in this chapter.

## Project 02 — A point-in-time-correct mini feature platform

Build, on your laptop: (1) a DuckDB/Parquet "lakehouse" with simulated e-commerce events (views, purchases) carrying event timestamps; (2) three features at different freshness tiers — `user_30d_purchase_count` (daily batch), `user_session_view_count` (streaming, simulate with a Python consumer reading a Kafka topic or just a sorted event replay), `item_ctr_7d`; (3) a point-in-time training-set builder and a Redis online store, then **write a test proving offline and online values match** for sampled (user, timestamp) pairs; (4) deliberately introduce a time-travel bug (join features computed *after* the label event) and measure the offline-AUC inflation it causes on a toy purchase-prediction model. That last step gives you a war story for interviews.

**LLM-track alternative:** build a fine-tuning data pipeline — take 20k raw documents, run MinHash dedup, an LLM-quality-filter pass with a small local model, decontamination against an eval set, and produce packed SFT-ready sequences. Report what fraction each stage removed.

## Interview Q&A

**Q1. What is point-in-time correctness and what bug does it prevent?**
**A.** When constructing a training example for an event at time *t*, every feature must reflect only information available strictly before *t*. Violating it (e.g., joining today's `user_30d_purchase_count` onto last month's training labels) leaks future information into training: offline metrics look spectacular, online performance collapses because the future isn't available at serving time. A feature store enforces it by storing feature values with effective timestamps and performing as-of joins. Adjacent leakage bugs: computing normalization statistics over the full dataset including the test period, and including the label's own causal descendants as features (e.g., "refund issued" as a feature for fraud).

**Q2. Design the feature pipeline for real-time fraud detection on payments.**
**A.** Three freshness tiers. Static/daily batch: account age, historical chargeback rate, device reputation — Spark job → offline store → synced to online store. Near-real-time (minutes): merchant-level rolling fraud rate. Real-time (seconds): velocity features — transactions and distinct cards per device in the last 5 min/1 hr — computed by Flink over a Kafka stream of auth events with sliding windows, written to Redis with TTLs. At request time the scorer does one batched Redis multi-get (<10 ms), concatenates with request-level features (amount, geo-IP distance), and scores. Training reads point-in-time values of *the same definitions* from the offline store; additionally log the served feature vector with each decision (log-and-wait) to make skew detectable. Mention delayed labels: chargebacks arrive weeks later, so labels are joined back asynchronously and recent data is trained with a maturity cutoff or label-correction scheme.

**Q3. You have 5M unlabeled support tickets and budget for 5k human labels. How do you build a ticket classifier?**
**A.** (1) Write a precise labeling guide; have a frontier LLM label a stratified ~100k sample with structured output and rationale. (2) Spend ~2k human labels auditing the LLM labels across slices (language, product area, length) to measure agreement; iterate the prompt/taxonomy where agreement is low; reserve ~1k human labels as a pristine test set that no model or prompt ever touches. (3) Fine-tune a small model (e.g., a modern small encoder or a 0.5–4B decoder) on the LLM-labeled set — distillation — and use the remaining human budget for active learning on the distilled model's most uncertain examples. (4) Ship the small model (cheap, fast), keep the LLM as a fallback for low-confidence cases, and let production disagreements flow back as new audit candidates. Report test metrics only on the human-gold set.

**Q4. When does synthetic training data fail, and how do you de-risk it?**
**A.** It fails when the synthetic distribution diverges from deployment on the dimensions the model is actually sensitive to. Classic case: synthetic document images that are semantically and visually realistic but physically pristine — real captures add perspective distortion, crumpling, blur, and lighting, and geometric/spatial corruptions cause disproportionate accuracy drops compared to photometric ones. De-risking: (1) always measure **transfer**: train on synthetic, evaluate on a real held-out set — never report synthetic-on-synthetic numbers; (2) characterize the gap dimension-by-dimension with a corruption-robustness benchmark, then close it with targeted augmentation rather than more synthetic volume; (3) mix in even a small real fine-tuning set — it disproportionately recovers performance; (4) for text/LLM data, dedup and decontaminate, and avoid recursive self-training without fresh real-data anchors (model collapse).

**Q5. Why is deduplication so important for LLM training data?**
**A.** Duplicates distort training in three ways: they implicitly upweight repeated content (the model overfits to it at the expense of everything else), they inflate memorization and verbatim regurgitation risk (privacy/copyright), and they corrupt evaluation when near-copies of benchmark items hide in training data (contamination → fake capability gains). Exact dedup (hashing) is trivial; the real tool is near-dedup with MinHash-LSH over shingled documents, which scales to billions of docs. In fine-tuning sets the same applies at small scale: fifty paraphrases of one instruction silently dominate the gradient. Empirically, careful dedup+filtering (FineWeb/DCLM lineage) yields models that beat ones trained on several times more unfiltered tokens — data quality buys compute.
