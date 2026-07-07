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

### Tool Survival Guide: Feast

**Architecture.** Feast has four components: the **registry** (a lightweight metadata store — YAML files committed to git, or a SQL backend — holding feature definitions and nothing else); the **offline store** (Parquet/DuckDB locally, BigQuery/Redshift/Snowflake in production); the **online store** (Redis, DynamoDB, or SQLite locally); and **materialization jobs** that read from the offline store and write to the online store. There is no streaming path in the core loop — materialization is a periodic batch process, which is Feast's fundamental architectural constraint.

```python
# feature_repo/features.py  (representative as of 2026 — check current Feast docs)
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

user = Entity(name="user_id", join_keys=["user_id"])

user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",   # when this feature snapshot was computed
    created_timestamp_column="created",
)

user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="purchase_count_30d", dtype=Int64),
        Field(name="avg_order_value_30d", dtype=Float64),
    ],
    source=user_stats_source,
)
```

**`get_historical_features` and point-in-time joins.** Pass an entity DataFrame with your join key(s) and an `event_timestamp` column (timezone-aware UTC). Feast performs an as-of join: for each training row at time *t*, it retrieves the feature row with the largest `event_timestamp` ≤ *t*. Both the entity DataFrame timestamps and the feature source timestamps must be UTC-aware — a mismatched timezone is the classic silent-skew bug (see War Story below).

```python
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

entity_df = pd.DataFrame({
    "user_id": [42, 101, 77],
    "event_timestamp": pd.to_datetime(
        ["2024-03-01", "2024-03-02", "2024-03-03"], utc=True   # UTC is non-negotiable
    ),
    "label": [1, 0, 1],
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_stats:purchase_count_30d", "user_stats:avg_order_value_30d"],
).to_df()
```

**Real limitations.**

| Limitation | Failure mode | Mitigation |
| --- | --- | --- |
| Complex / nested types | No native serialization for lists, structs, or embeddings | Serialize to bytes (proto/JSON) before write; deserialize in model server |
| Sub-second freshness | Materialization is batch; streaming-push API has historically been unstable | For <1 s freshness, bypass Feast — write Flink → Redis directly; document the split |
| Materialization lag | Online store is stale by up to one full materialization interval | Monitor lag; alert when lag > SLA; keep TTLs shorter than acceptable staleness |
| Registry contention | Concurrent `feast apply` from multiple teams causes partial-apply or merge conflicts | One registry per domain; cross-team read access via shared feature-view references |
| Bulk online lookup loop | Calling `get_online_features` row-by-row causes N Redis round-trips | Always pass a list of entity rows; batch the lookup |

Cost-per-freshness-tier tables (batch vs. micro-batch vs. streaming operating costs) are in the economics chapter (Module 11) — refer there rather than duplicating pricing here.

**When NOT to use Feast.** A handful of slowly-changing lookup tables (product category, account tier) — a DuckDB join at training time and a Redis hash at serving time is less infrastructure with the same result. Streaming-first architectures where every feature is a sub-minute aggregate — Feast's materialization model fights you; build on Flink + Redis and add metadata governance separately once you have real cross-team reuse pressure. Feast earns its complexity when ≥5 teams share ≥50 feature definitions and point-in-time correctness is a compliance requirement, not just a best practice. In an interview, naming both the value *and* the cost of a feature store is a senior signal.

---

### War Story: the point-in-time timezone skew leak

**Setting.** A large e-commerce team trains a purchase-propensity model. Offline cross-validation AUC is strong — roughly 0.82 — and the team deploys to shadow mode. After a month, live lift over the baseline ranker is near zero. Business impact order-of-magnitude: a model expected to recover single-digit percentage revenue uplift that produces nothing is a costly quarter of engineering.

**Debugging path.**

1. **Feature distribution check.** Training vs. serving histograms look comparable. No obvious covariate drift — the team rules out distribution shift.
2. **Model version check.** The artifact in serving matches the one evaluated offline. No stale-deploy issue.
3. **Shadow-mode score comparison.** A week of logged dual-scores reveals the new model systematically overscores users who purchased *very recently* and underscores inactive users. That asymmetry is the skew fingerprint: a feature that conflates recent history with future events.
4. **Feature-by-feature trace.** A script samples ~500 training rows and recomputes each feature from raw events at the label's `event_timestamp`. One feature — `user_7d_purchase_count` — shows consistent positive bias in the training set relative to the recomputed ground truth.
5. **Root cause.** The offline data warehouse stores timestamps in local time (UTC−8 in winter). The label `event_timestamp` column in the training extract is UTC. A 7-day window computed relative to a UTC−8 timestamp, joined against a UTC label event, shifts the window forward by 8 hours — pulling in up to 8 hours of *future* purchase events relative to the true label time. The model learned to fire on a signal that does not exist at inference time.

**Fix.** Normalize all timestamps to UTC at ingestion in the feature-computation pipeline — one-line change. Re-materialize. Retrain. Offline AUC dips slightly (expected — the leaked signal is gone); live lift aligns with the corrected offline estimate.

**Prevention.** (1) Enforce UTC-aware timestamps as a schema rule in the feature store — raise at write time if `event_timestamp` is naive or non-UTC. (2) Write a **time-travel unit test** in CI: for a synthetic event at time *t*, assert that `user_7d_purchase_count` equals the count of events in the half-open interval [*t* − 7 d, *t*), not any shifted version. (3) Add the skew monitor in Project 02 to your deployment checklist — catch calibration drift in shadow mode before it wastes a month of serving. The broader lesson: **a skew bug that manifests as calibration error is harder to catch than one that shifts feature means.** Always log served feature vectors and compare to what the training set would have served for the same (user, timestamp) pair.

---

## Project 02 — A point-in-time-correct mini feature platform

Build, on your laptop: (1) a DuckDB/Parquet "lakehouse" with simulated e-commerce events (views, purchases) carrying event timestamps; (2) three features at different freshness tiers — `user_30d_purchase_count` (daily batch), `user_session_view_count` (streaming, simulate with a Python consumer reading a Kafka topic or just a sorted event replay), `item_ctr_7d`; (3) a point-in-time training-set builder and a Redis online store, then **write a test proving offline and online values match** for sampled (user, timestamp) pairs; (4) deliberately introduce a time-travel bug (join features computed *after* the label event) and measure the offline-AUC inflation it causes on a toy purchase-prediction model. That last step gives you a war story for interviews.

**LLM-track alternative:** build a fine-tuning data pipeline — take 20k raw documents, run MinHash dedup, an LLM-quality-filter pass with a small local model, decontamination against an eval set, and produce packed SFT-ready sequences. Report what fraction each stage removed.

### Project 02 — Concrete implementation guide

The spec above describes what to build. These are the working pieces.

#### Step 1 — Generate synthetic events (DuckDB)

```python
import logging
import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

con = duckdb.connect("feature_store.duckdb")
rng = np.random.default_rng(42)

n_users, n_events = 500, 20_000
events = pd.DataFrame({
    "user_id": rng.integers(0, n_users, n_events),
    "event_type": rng.choice(["view", "purchase"], n_events, p=[0.85, 0.15]),
    "event_timestamp": pd.to_datetime(
        rng.integers(
            int(pd.Timestamp("2024-01-01", tz="UTC").timestamp()),
            int(pd.Timestamp("2024-04-01", tz="UTC").timestamp()),
            n_events,
        ),
        unit="s",
        utc=True,
    ),
})
con.execute("CREATE OR REPLACE TABLE events AS SELECT * FROM events")
logger.info("Created events table: %d rows", n_events)
```

Expected output: `INFO Created events table: 20000 rows`

---

#### Step 2a — Point-in-time join (DuckDB ASOF JOIN)

DuckDB's ASOF JOIN is the right primitive: for each training row it finds the most recent feature snapshot whose `event_timestamp` ≤ the label's `event_timestamp`. NULL means no prior snapshot exists — handle with `COALESCE`.

```sql
-- training_events: (user_id, event_timestamp UTC, label)
-- user_feature_snapshots: (user_id, event_timestamp UTC, purchase_count_30d)
-- ASOF JOIN: per training row, pick the latest snapshot row where
-- snapshot.event_timestamp <= label.event_timestamp.

CREATE OR REPLACE TABLE training_set AS
SELECT
    e.user_id,
    e.event_timestamp,
    e.label,
    COALESCE(f.purchase_count_30d, 0) AS purchase_count_30d
FROM training_events AS e
ASOF JOIN user_feature_snapshots AS f
    ON  e.user_id         = f.user_id
    AND e.event_timestamp >= f.event_timestamp
ORDER BY e.event_timestamp;
```

Verify: `SELECT COUNT(*), AVG(purchase_count_30d) FROM training_set;` — row count must equal `training_events`; earliest rows will have `purchase_count_30d = 0` (no prior snapshot), later rows non-zero.

**Step 2b — Introduce the bug intentionally.** Flip `>=` to `<=` in the ASOF join condition. Train a toy logistic regression on both versions and record AUC. The buggy version will be noticeably inflated — this is your war-story number for the interview.

---

#### Step 3 — Offline/online skew monitor (Python)

```python
import logging
from typing import Sequence
import duckdb
import redis

logger = logging.getLogger(__name__)


def check_feature_skew(
    con: duckdb.DuckDBPyConnection,
    redis_client: redis.Redis,
    feature_name: str,
    sample_pairs: Sequence[tuple],   # [(user_id, event_timestamp_utc), ...]
    tolerance: float = 0.05,
) -> None:
    """
    Assert mean absolute relative difference between offline (DuckDB)
    and online (Redis) feature values is within tolerance.

    Missing values are warned and excluded. Raises AssertionError on breach.
    """
    diffs: list[float] = []
    missing = 0

    for user_id, event_ts in sample_pairs:
        row = con.execute(
            """
            SELECT feature_value
            FROM user_feature_snapshots
            WHERE user_id = ?
              AND feature_name = ?
              AND event_timestamp <= ?
            ORDER BY event_timestamp DESC
            LIMIT 1
            """,
            [user_id, feature_name, event_ts],
        ).fetchone()

        online_bytes = redis_client.hget(f"user:{user_id}", feature_name)

        if row is None or online_bytes is None:
            missing += 1
            logger.warning(
                "Missing value: user_id=%s feature=%s — excluded from skew calc",
                user_id, feature_name,
            )
            continue

        offline_val = float(row[0])
        online_val = float(online_bytes)
        denominator = abs(offline_val) if offline_val != 0.0 else 1.0
        diffs.append(abs(online_val - offline_val) / denominator)

    if not diffs:
        logger.error("No valid samples for feature=%s — skew check aborted", feature_name)
        return

    mean_diff = sum(diffs) / len(diffs)
    logger.info(
        "Skew check: feature=%s n_valid=%d n_missing=%d mean_rel_diff=%.4f tolerance=%.4f passed=%s",
        feature_name, len(diffs), missing, mean_diff, tolerance, mean_diff <= tolerance,
    )
    assert mean_diff <= tolerance, (
        f"Feature skew for '{feature_name}' exceeded tolerance: "
        f"mean_rel_diff={mean_diff:.4f} > {tolerance:.4f}"
    )
```

Expected output — healthy pipeline:

```text
INFO Skew check: feature=purchase_count_30d n_valid=198 n_missing=2 mean_rel_diff=0.0000 tolerance=0.0500 passed=True
```

Expected output — timezone bug active:

```text
INFO  Skew check: feature=purchase_count_30d n_valid=198 n_missing=2 mean_rel_diff=0.1840 tolerance=0.0500 passed=False
AssertionError: Feature skew for 'purchase_count_30d' exceeded tolerance: mean_rel_diff=0.1840 > 0.0500
```

---

#### Project 02 — Troubleshooting table

| Symptom | Likely cause | Check |
| --- | --- | --- |
| ASOF JOIN returns NULL for most rows | Snapshot and label timestamps are in different timezones | Inspect `typeof(event_timestamp)` in both tables; normalize both to UTC |
| Offline AUC inflated, skew monitor shows positive bias | Join direction wrong (`<=` instead of `>=`) — future-leaking join | Recheck join condition; `e.event_timestamp >= f.event_timestamp` is correct |
| Skew monitor shows non-zero error on fresh data | Materialization lag: online store hasn't caught up | Check last materialization timestamp; shorten interval or add freshness alert |
| Skew monitor shows 100% missing | Redis key schema mismatch (`user:{id}` vs `user_features:{id}`) | Print a sample key from the write path and the read path; make them match |
| AUC identical with and without the time-travel bug | Synthetic data too sparse — bug's effect diluted | Increase event density; confirm both training sets differ on `purchase_count_30d` mean |

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

## You can now

- Design a multi-tier feature pipeline and select the right freshness tier — batch, micro-batch, or streaming — per feature, citing the order-of-magnitude operating-cost difference between tiers.
- Implement point-in-time-correct training-set construction using DuckDB ASOF JOIN, and deliberately introduce and quantify the AUC inflation caused by a time-travel leakage bug.
- Build an offline/online skew monitor that measures mean absolute relative difference between store values, trace a timezone mismatch to a calibration failure, and write a time-travel unit test that catches it in CI.
- Decide when a feature store earns its complexity — naming the threshold (≥5 teams, ≥50 shared definitions) and the use cases where streaming-first or simple-lookup architectures make it the wrong tool.
- Curate LLM training data with MinHash near-dedup, classifier-based quality filtering, and decontamination, and articulate the model-collapse risk from recursive self-training on synthetic outputs.
