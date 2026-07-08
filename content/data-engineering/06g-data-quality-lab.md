# 06 — Advanced Topics: Data Quality and Observability — A Hands-On Lab

This is a supplementary advanced topic. The core track surveys data quality; here you actually build the gates. Everything you've learned about correctness (idempotency, exactly-once), contracts (ODCS), and CI/CD now gets pointed at one job: making bad data *unable* to reach the consumers downstream of you — dashboards, financial reports, and increasingly ML models whose training set is silently poisoned by a schema drift you never saw. This is the difference between a pipeline that runs and a pipeline you can trust at 3 AM.

> **Date-stamped claims.** API and product details below reflect the state as of mid-2026. Great Expectations, Soda, and dbt move fast; verify the current signature before you ship. In particular, GX shipped the **GX Core 1.x** API, which is a hard break from the pre-1.0 `great_expectations.yml` + CLI world you'll still find in old tutorials.

---

## The Problem — Silent Corruption

A pipeline failure that throws an exception is the good case. You get paged, you fix it. The expensive failures are *silent*: the job succeeds, the row count looks plausible, and the data is quietly wrong.

- An upstream team renames `user_id` to `userId`. Your join now matches nothing, produces zero rows for that segment, and the dashboard shows "revenue down 8%." Nobody knows for three weeks.
- A currency field starts arriving in cents instead of dollars. Every downstream aggregation is off by 100x, but the numbers are still numbers, so no test fails.
- A sensor's null rate creeps from 2% to 40% over a month. Your model's feature distribution shifts, offline metrics still look fine on the frozen eval set, and production accuracy silently rots.

The lesson: **quality must be enforced, not hoped for.** "The upstream team said they wouldn't change the schema" is not enforcement. A test that runs on every batch and halts the pipeline is enforcement. This lab builds four layers of it — declarative expectations, lightweight in-warehouse checks, statistical/anomaly monitors, and orchestration gates — and ties them back to the [ODCS data contracts](02-medium-guide.md) the producing team already publishes.

---

## Layer 1 — Declarative Expectations with Great Expectations (GX Core 1.x)

Great Expectations is the category-standard framework for *declarative* quality: you describe what good data looks like as a set of **Expectations**, group them into a **Suite**, bind a suite to a batch of data with a **Validation Definition**, and run one or more validation definitions from a **Checkpoint**. That vocabulary — Suite → Validation Definition → Checkpoint — is the whole GX Core 1.x mental model.

Start with a context and a data source. GX Core prefers a fluent, Python-first API; the old YAML project scaffold is gone.

```python
import great_expectations as gx
import pandas as pd

df = pd.read_parquet("orders_2026_07_08.parquet")

context = gx.get_context()  # ephemeral by default; file-backed if a project dir exists

# Source -> asset -> batch definition
data_source = context.data_sources.add_pandas("orders_source")
asset = data_source.add_dataframe_asset(name="orders")
batch_definition = asset.add_batch_definition_whole_dataframe("daily_batch")
```

Now build the **Expectation Suite** — the reusable, version-controlled description of "good."

```python
suite = context.suites.add(gx.ExpectationSuite(name="orders_quality"))

suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="amount_usd", min_value=0, max_value=100_000,
        # severity is advisory metadata: warnings don't fail the run
        severity="error",
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="status", value_set=["pending", "paid", "shipped", "refunded"]
    )
)
suite.add_expectation(
    gx.expectations.ExpectTableRowCountToBeBetween(min_value=1_000, max_value=5_000_000)
)
```

Bind the suite to the batch with a **Validation Definition**, wrap it in a **Checkpoint**, and run:

```python
validation_def = context.validation_definitions.add(
    gx.ValidationDefinition(
        name="orders_daily_validation",
        data=batch_definition,
        suite=suite,
    )
)

checkpoint = context.checkpoints.add(
    gx.Checkpoint(
        name="orders_checkpoint",
        validation_definitions=[validation_def],
        result_format="COMPLETE",  # return the failing rows, not just a boolean
        # actions fire on the result: Slack, email, store to Data Docs, etc.
        actions=[],
    )
)

result = checkpoint.run(batch_parameters={"dataframe": df})

if not result.success:
    raise ValueError("orders_checkpoint failed — see Data Docs for the failing expectations")
```

`checkpoint.run()` returns a result whose `.success` is your gate. GX also renders **Data Docs** — static HTML showing exactly which expectations failed and on which rows — which is what you link in the incident channel instead of describing the failure in prose.

### Fail the pipeline vs quarantine

The `raise` above is the **fail-fast** policy: nothing downstream runs on bad data. Correct for a payments table, where wrong is worse than late.

The alternative is **quarantine**: split the batch, let the clean rows flow, and route the failing rows to a `orders_quarantine` table for a human to triage. Use `result_format="COMPLETE"` to recover the unexpected values and filter:

```python
# Pseudocode for the quarantine pattern
res = result.run_results  # per-validation-definition results
bad_ids = extract_unexpected_index_list(res, expectation="ExpectColumnValuesToBeInSet")
clean = df[~df["order_id"].isin(bad_ids)]
quarantine = df[df["order_id"].isin(bad_ids)]
write_delta(clean, "orders");  write_delta(quarantine, "orders_quarantine")
```

Rule of thumb: **fail-fast for correctness-critical tables, quarantine for high-volume ingestion** where dropping 0.1% bad rows beats halting the whole load. Never silently drop — quarantine keeps the bad rows for forensics.

---

## Layer 2 — The Lighter Alternatives: Soda and dbt Tests

GX is powerful and correspondingly heavy: a Python framework with its own object model. For many teams that is more than the job needs. Two lighter options cover most cases.

### Soda (SodaCL)

Soda Core is a CLI + Python library that runs YAML checks (**SodaCL**) as aggregated SQL *where the data already lives* — no data movement. It reads almost like English, so analysts write checks too:

```yaml
# checks.yml — SodaCL
checks for orders:
  - row_count > 0
  - missing_count(order_id) = 0
  - duplicate_count(order_id) = 0
  - invalid_percent(status) < 1%:
      valid values: [pending, paid, shipped, refunded]
  - freshness(created_at) < 1d
  # built-in anomaly detection — no threshold, it learns the baseline
  - anomaly detection for row_count
```

```bash
soda scan -d warehouse -c configuration.yml checks.yml
```

Soda's strength is **continuous observability on a schedule** with built-in anomaly detection, and its SodaCL doubles as a data-contract-verification language.

### dbt tests (and dbt-expectations)

If your transformations already run in dbt, quality belongs *in the model layer* — this is "shift-left." dbt ships four generic tests (`unique`, `not_null`, `accepted_values`, `relationships`); the **dbt-expectations** package ports the GX expectation vocabulary into native dbt tests.

```yaml
# models/schema.yml
models:
  - name: orders
    columns:
      - name: order_id
        data_tests:
          - unique
          - not_null
      - name: status
        data_tests:
          - accepted_values:
              values: [pending, paid, shipped, refunded]
      - name: amount_usd
        data_tests:
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0
              max_value: 100000
    data_tests:
      - dbt_expectations.expect_table_row_count_to_be_between:
          min_value: 1000
          max_value: 5000000
```

```bash
dbt test --select orders   # fails the build if any test fails
```

**When each fits:**

| Tool | Best when |
|------|-----------|
| **dbt tests** | Transformations already in dbt; you want quality co-located with models and enforced in the same CI run. Lowest barrier if you know SQL. |
| **Soda** | Warehouse-first stack, checks on raw/landing tables *before* dbt, scheduled monitoring, non-engineers authoring checks, built-in anomaly detection. |
| **Great Expectations** | Python pipelines (Spark/pandas), pre-warehouse validation, need for rich failed-row reports (Data Docs), reusable suites across engines. |

They are complementary, not rivals: Soda or GX on ingestion (before dbt can see the data), dbt tests on the transformed models.

---

## Layer 3 — Freshness, Volume, Schema Drift, and Distribution

Column-level rules ("`amount` is positive") are necessary but miss the failures in the intro. The four checks that catch *silent* corruption operate on metadata and statistics:

1. **Freshness** — is the newest row recent enough? `MAX(updated_at)` should be within the SLA in the ODCS contract. A stalled upstream job produces no error, just stale data. `freshness(created_at) < 1d` in Soda, or a dbt `source freshness` block.
2. **Volume** — did roughly the expected number of rows arrive? A daily load that usually lands 1M rows and today landed 4,000 is broken even if every row is individually valid. `ExpectTableRowCountToBeBetween`, or Soda's `row_count` anomaly check.
3. **Schema drift** — did columns get added, dropped, or retyped? `ExpectTableColumnsToMatchSet` / `ExpectColumnToExist` catch the `user_id` → `userId` rename that silently zeroed a segment.
4. **Distribution** — did the *shape* of a column change? Null rate, mean, cardinality, category mix. `ExpectColumnMeanToBeBetween`, `ExpectColumnProportionOfUniqueValuesToBeBetween`, or a KS-test between today's batch and a reference window.

**Anomaly detection on metrics** is the generalization: instead of hand-setting `min/max` thresholds you'll get wrong, the tool learns the historical baseline of a metric (row count, null %, freshness lag) and flags statistically significant deviations. Soda ships this natively; Elementary computes it from dbt run metadata; the commercial platforms make it their headline feature. The tradeoff: fewer brittle thresholds to maintain, but you trade them for false positives during legitimate regime changes (a marketing campaign really did 5x the row count) — so route anomaly alerts to a human, not to a hard pipeline halt.

---

## Layer 4 — Quality Gates in Orchestration + the Data-Contract Tie-In

A check that only runs when you remember to run it is not a gate. Gates belong in the orchestrator so they run on every materialization and *block downstream work* on failure.

**Dagster** models this natively with **asset checks**: a check attached to an asset that runs when it materializes. A *blocking* check stops downstream assets from materializing when it fails — a true quality gate in one decorator.

```python
import dagster as dg

@dg.asset
def orders(): ...

@dg.asset_check(asset=orders, blocking=True)
def orders_not_null(context):
    n_bad = query("SELECT count(*) FROM orders WHERE order_id IS NULL")
    return dg.AssetCheckResult(
        passed=(n_bad == 0),
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"null_ids": n_bad},
    )
```

Dagster also integrates dbt tests, Soda, and GX directly, so you can wrap your Layer-1/2 checks as asset checks rather than reimplementing them.

**Airflow** has no native asset-check primitive; you express the gate as a task. The classic pattern is a "circuit breaker": a SQL-check operator (or a `PythonOperator` running your GX checkpoint) that raises on failure, placed *between* the load task and the publish task so a failure short-circuits the DAG before bad data is exposed.

```python
def run_gx_gate():
    result = checkpoint.run(batch_parameters={"dataframe": load_df()})
    if not result.success:
        raise AirflowException("GX gate failed — halting before publish")

gate = PythonOperator(task_id="gx_gate", python_callable=run_gx_gate)
load >> gate >> publish   # publish never runs if the gate raises
```

### Connecting gates to ODCS contracts

You already learned that the [Open Data Contract Standard (ODCS)](02-medium-guide.md) is the org-level promise a producing team publishes — "here is the schema and these are the guarantees." The gate is how that promise becomes *enforceable* rather than aspirational. The `quality` block of an ODCS contract lists rules the producer commits to; your job is to compile those rules into runnable checks.

- The contract says `order_id` is unique and non-null → generate the GX suite / SodaCL / dbt test from the contract.
- The contract states a freshness SLA of 1 hour → a freshness gate on `MAX(created_at)`.
- Soda can consume a data contract file directly and verify a dataset against it, closing the loop.

The deeper idea from the contracts lesson holds here: contracts move quality *upstream*. The producer's own CI runs the contract's checks, so a breaking change is **unshippable**, not merely detectable after it breaks you. That is the same enforcement muscle as [CI/CD for Data](04-next-steps.md) — run the same checks on the PR that produces the data as you run in production, so `dbt test` / `soda scan` / the GX checkpoint are steps in the GitHub Actions workflow, not a nightly afterthought.

---

## Observability Platforms (At a Mention)

Everything above is *testing* — you assert what you expect. **Data observability** is the complementary discipline: monitor the pipeline's health metrics broadly and let anomaly detection surface issues you *didn't* think to write a check for. The commercial category leader is **Monte Carlo** (freshness, volume, schema, distribution, and lineage monitoring with ML-driven anomaly detection, largely no-code). **Bigeye** and **Anomalo** are peers.

Open-source, in mid-2026:

- **Elementary** — observability built for dbt; anomaly detection and lineage from dbt run metadata. Tightest fit if your stack is dbt-centric, and effectively free.
- **Great Expectations** — the quality-testing standard, though not a full observability platform.
- Warehouse-native monitors — **Snowflake data quality monitors** and **Databricks DQ / Lakehouse Monitoring** — are increasingly the pragmatic "good enough" option when you're already all-in on one platform.

The honest tradeoff: open-source observability tools trail the commercial platforms on out-of-the-box coverage, alert routing, and lineage-driven incident triage. For a portfolio project or a dbt shop, **Elementary + dbt tests** is the cheapest credible observability story. For an F100 with hundreds of critical tables, a managed platform usually wins on total cost of incidents.

---

## Exercises

1. Take a Parquet extract with a deliberately corrupted column (inject nulls, an out-of-range value, and a renamed column). Write a GX Core 1.x suite that catches all three, run it via a Checkpoint, and implement both the fail-fast and the quarantine policy.
2. Rewrite the same three checks as SodaCL and as dbt-expectations tests. Note how many lines each takes and which you'd hand to an analyst.
3. Add a blocking Dagster asset check (or an Airflow circuit-breaker task) so the corrupted batch never materializes the downstream asset. Prove the downstream asset stays stale.
4. Take an ODCS contract's `quality` block and hand-compile it into a Soda checks file. Then wire `soda scan` into a GitHub Actions job so a PR that violates the contract fails CI.

---

## You can now

- Distinguish silent data corruption from loud failures, and name the four metadata/statistical checks (freshness, volume, schema drift, distribution) that catch the silent ones.
- Build a declarative quality gate in **GX Core 1.x** — Suite → Validation Definition → Checkpoint — and choose fail-fast vs quarantine with a defensible rule.
- Pick the right-weight tool: dbt tests for in-model shift-left, Soda for warehouse-native scheduled monitoring with anomaly detection, GX for Python pipelines needing rich failed-row reports.
- Wire checks into Dagster asset checks or an Airflow circuit-breaker so gates run on every materialization and block downstream work.
- Compile an ODCS contract's guarantees into runnable checks and run them in CI — turning the contract from a promise into enforcement — and place observability platforms (Monte Carlo, Elementary) correctly relative to testing.
