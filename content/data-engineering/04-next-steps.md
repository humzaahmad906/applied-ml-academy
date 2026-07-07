# 04 — Next Steps: Specialization for Fortune 100 Roles

The core track leaves you knowing Kestra, BigQuery, dbt, Spark, and Kafka. That's a respectable foundation. But Fortune 100 data platform teams use a wider and slightly different toolkit. This section is how you close that gap.

**Time:** 6–8 weeks at 10 hrs/week. Treat it as a second course, structured around the topics F100 interviews actually test.

## The Honest Map: Core Stack vs Fortune 100 Reality

| Layer | Core Track | Common at F100 |
|---|---|---|
| Cloud | GCP | AWS dominates, Azure second, GCP third |
| Orchestrator | Kestra | Airflow (overwhelmingly), Dagster (growing), some Prefect |
| Warehouse | BigQuery | Snowflake (most common), Databricks, BigQuery, Redshift |
| Transformation | dbt | dbt (same!) |
| Batch processing | Spark | Spark on Databricks/EMR/Dataproc |
| Streaming | Kafka | Kafka + Flink, Kinesis, Pub/Sub |
| Lakehouse | Introduced as bonus | Iceberg or Delta, becoming table stakes |
| Observability | Introduced lightly | Monte Carlo, Datafold, custom |
| Governance | Introduced lightly | Atlan, Collibra, Unity Catalog, custom |
| Quality | dbt tests | dbt tests + Great Expectations + Soda |
| CI/CD | Introduced lightly | GitHub Actions / GitLab CI heavily |

You're already strong on the layers where the core track and F100 practice agree (dbt, Spark, Kafka). The gaps to close: AWS, Airflow, Snowflake, lakehouse formats, observability, governance, and CI/CD. That's what this section covers.

---

## Phase 1 — AWS for Data Engineers (1 week)

GCP gave you the cloud mental model. AWS is the same patterns with different names and a steeper learning curve.

### What to Learn

1. **IAM** — roles, policies, the principle of least privilege. Spend a day here; IAM mistakes are the single biggest production data leak risk.
2. **S3** — the bucket model is similar to GCS but with more knobs (storage classes, lifecycle policies, versioning, replication).
3. **Athena** — Trino/Presto as a service. Query S3 directly with SQL. Roughly analogous to BigQuery on external tables.
4. **Glue** — AWS's managed catalog + Spark. The Glue Data Catalog is the metadata layer for Athena/EMR/Redshift Spectrum.
5. **EMR or Glue jobs** — running Spark on AWS. EMR is the older heavy approach; Glue Jobs is the lighter serverless option.
6. **Kinesis** — AWS's streaming service. Kinesis Data Streams ≈ Kafka, Kinesis Firehose ≈ "stream to S3 with no code."
7. **Redshift** — AWS's warehouse. Older than Snowflake but still very common at F100. Different cost model (cluster-based, not query-based).
8. **VPCs and Security Groups** — networking basics. You don't need to be a network engineer, but a DE who doesn't know what a VPC is gets stuck constantly.

### Cost Discipline

AWS bills are even more booby-trapped than GCP. Set up:

- Billing alerts at multiple thresholds ($5, $20, $50)
- AWS Budgets
- The Cost Explorer dashboard checked weekly

Common money pits: forgotten EMR clusters, NAT Gateways, cross-AZ data transfer. Tag every resource with a project name so you can attribute cost.

### What to Build

Take your Tier-2 medium project and port it to AWS:

- GCS → S3
- BigQuery → Athena (read), Redshift or Snowflake (warehouse)
- Kestra → Airflow (you'll do Airflow next anyway)
- Same dbt code (dbt is cloud-agnostic — this is the point)

You don't need to rebuild from scratch; just stand up the equivalent stack and load a representative dataset.

---

## Phase 2 — Airflow (1.5 weeks)

Airflow is the orchestrator the industry has settled on for the last decade. Kestra/Dagster/Prefect are gaining ground but Airflow still runs more F100 pipelines than the rest combined. You need to be fluent.

**Airflow 3.x is the current version as of 2025–2026.** If a tutorial is showing you `execution_date` in task context, it's Airflow 2. The field is now `logical_date`. If a tutorial has SubDAGs, they were removed in Airflow 3. Read what follows carefully — F100 shops running Airflow 3 will ask about these changes.

### What to Learn

1. **The DAG mental model**
   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime

   with DAG(
       'taxi_pipeline',
       start_date=datetime(2024, 1, 1),
       schedule='@daily',
       catchup=False,
   ) as dag:
       extract = PythonOperator(task_id='extract', python_callable=extract_fn)
       load = PythonOperator(task_id='load', python_callable=load_fn)
       transform = PythonOperator(task_id='transform', python_callable=transform_fn)

       extract >> load >> transform
   ```

2. **Operators** — built-in ones for everything: `BashOperator`, `PythonOperator`, `S3KeySensor`, `BigQueryOperator`, `KubernetesPodOperator`, the dbt operators. Plus the modern way: TaskFlow API with `@task` decorators.

3. **TaskFlow API (the modern style)**
   ```python
   from airflow.decorators import dag, task

   @dag(start_date=datetime(2024, 1, 1), schedule='@daily')
   def pipeline():
       @task
       def extract(): ...

       @task
       def load(raw_data): ...

       @task
       def transform(loaded_path): ...

       transform(load(extract()))

   pipeline()
   ```
   Much cleaner. This is how new Airflow code is written.

4. **XComs** — passing data between tasks. Use sparingly; XComs are for metadata (paths, IDs), not for actual data.

5. **Sensors** — tasks that wait for a condition (file exists, table updated). Use `reschedule` mode to free worker slots while waiting.

6. **Connections and Variables** — Airflow's secrets management. Don't put credentials in DAG code.

7. **Backfills** — `airflow dags backfill -s 2024-01-01 -e 2024-01-31 my_dag`. The killer feature. Re-process historical date ranges with idempotency.

8. **Executors** — LocalExecutor (laptop), CeleryExecutor (worker pool), KubernetesExecutor (each task in a pod). KubernetesExecutor is dominant for new deployments.

9. **The pitfall: dynamic DAGs**
   Airflow parses your DAG files on a schedule. If your DAG file makes API calls or reads from a database, every parse hits those — burns money and slows the scheduler. Keep DAG files lightweight.

### Airflow 3.x — What Changed and Why It Matters

**Airflow 3.0 went GA in April 2025.** It's the biggest architectural shift in Airflow's history. Here's what to know for interviews:

**DAG versioning.** Airflow 3 finally solves the "what code ran this historical run?" problem. Every DAG file change creates a new version; the UI shows historical runs against the exact code that executed them. Before 3.0, if you changed a DAG and someone asked "what logic ran last Tuesday?", the answer was "whatever was in git at that commit, maybe." Now it's native in the scheduler.

**Asset-aware scheduling.** Data assets (tables, files, anything with a URI) are first-class citizens. Instead of scheduling a DAG on a cron, you can schedule it to run *when an asset is updated* by an upstream DAG. The pattern:

```python
from airflow.sdk import Asset, dag, task

# Producer DAG — declares what it produces
taxi_raw = Asset("s3://lake/taxi/raw/")

@dag(schedule="@daily")
def ingest_taxi():
    @task(outlets=[taxi_raw])  # signals: I updated this asset
    def extract_and_load(): ...
    extract_and_load()

ingest_taxi()

# Consumer DAG — wakes up when the asset arrives, not on a cron
@dag(schedule=[taxi_raw])      # runs when taxi_raw is updated
def transform_taxi():
    @task
    def run_dbt(): ...
    run_dbt()

transform_taxi()
```

This replaces the fragile pattern of manually chaining DAGs with `TriggerDagRunOperator` or setting exact cron offsets. Asset dependencies are explicit, observable, and retry-safe. For new Airflow 3 code, prefer asset-based scheduling over cron wherever it makes sense.

**Task Execution API + edge executor (3.0).** Tasks can now run anywhere — not just on the Airflow workers. The edge executor sends work to remote execution environments over HTTP. This is how Airflow becomes viable for heterogeneous compute (run some tasks on Kubernetes, some on bare metal, some in cloud-managed runtimes) without committing to a single executor type.

**React UI (3.0).** The old Airflow UI was Flask/Jinja and notoriously painful. Airflow 3 ships a full React rewrite: faster, cleaner, proper lineage visualization, asset-centric views.

**Human-in-the-Loop tasks (3.1, September 2025).** Native support for tasks that require a human approval or input before proceeding. No more hacking `ExternalTaskSensor` to pause for a sign-off. Airflow 3.1 makes this first-class:

```python
@task.human_in_the_loop(
    subject="Approve deployment to prod",
    description="Review the dbt run output before pushing to production.",
)
def approve_production_deploy(context): ...
```

An approver gets a notification (email, Slack), clicks approve or reject in the UI, and the DAG resumes or fails gracefully. Critical for regulated industries and ML model deployment workflows.

**Asset partitioning (3.2, April 2026).** Assets can now be partitioned — a DAG that processes `taxi_raw` for `2026-04-01` produces a partition of that asset, not just a generic update. Downstream DAGs can subscribe to specific partitions. This closes the gap between Airflow's asset model and Dagster's partition-first model.

**Migration from Airflow 2 to 3 — the practical checklist:**

- `execution_date` in task context → rename to `logical_date` everywhere. It's a breaking change.
- SubDAGs are removed. Replace with TaskGroups (available since 2.x, the recommended migration path).
- The `ruff` linter ships an `AIR30` ruleset that auto-flags migration issues: `ruff check --select AIR30 your_dags/`. Run it before migrating. It catches the `execution_date` renames, deprecated operator imports, and removed config keys automatically.
- Test your DAG serialization — Airflow 3 DAG versioning requires serialized DAGs to be stable across parses.

### Airflow vs Dagster in 2026

> **The honest sidebar:** Which one should you learn?

**Airflow** is the brownfield F100 standard. In the most recent industry survey, 89% of data teams report they still plan to use Airflow. It runs more production pipelines at Fortune 500 companies than everything else combined. Every major cloud has a managed Airflow (AWS MWAA, Google Cloud Composer, Astronomer). If you're interviewing at any established enterprise, you need to be fluent.

**Dagster** is the greenfield ML default. **Dagster Components went GA in October 2025** — the new component system makes Dagster's asset-centric model composable: you define assets as typed components, wire them together, and the platform handles orchestration, lineage, and partitioning without glue code. It was asset-centric from day one (Airflow added assets in 2.4 and matured them in 3.0; Dagster built the whole model around them from the start). For a new data platform at a tech company or ML-heavy startup, Dagster is increasingly the default recommendation.

**The career play:** Learn Airflow first. It's what F100 interviews test. Spend a week with Dagster afterward — the asset model will click fast because Airflow 3 borrowed heavily from it. Mention both in interviews. The candidate who can discuss both and explain when each fits is noticeably more senior than the candidate who only knows one.

### What to Build

Take your Tier-2 medium project and re-orchestrate it with Airflow. Use asset-based scheduling where natural (your ingest DAG produces an asset; your dbt DAG consumes it). You'll feel the differences from Kestra: more boilerplate, more flexibility, way more community resources.

### Resources

- The official Airflow tutorial (make sure it's for Airflow 3)
- Marc Lamberti's Astronomer courses (free, excellent — he's been fast to update for v3)
- The Astronomer Cosmos package for running dbt natively as Airflow tasks (used at most F100s now)
- Airflow 3.0 migration guide: `ruff --select AIR30` before you start

---

## Phase 3 — Snowflake or Databricks (1.5 weeks — pick one to start)

These two are the dominant choices at F100 outside Google. Decide which to learn first based on signal from job postings you're targeting:

- **Snowflake**: Lighter, SQL-first, "data cloud." Strong in regulated industries (finance, healthcare).
- **Databricks**: Notebook-first, ML-friendly, the home of Delta Lake. Strong in tech and any company with serious ML investment.

Both are similar in concept (separated storage/compute, columnar). Differences in syntax and operational model.

### Snowflake — What to Learn

1. **Virtual warehouses** — the compute clusters. Multiple warehouses for different workload types. You pay per second of warehouse runtime.
2. **Roles and grants** — Snowflake's permission model is hierarchical and powerful. Learn it well; it's interview-bait.
3. **Time Travel** — query a table as of any point in the last 1–90 days. `SELECT * FROM mytable AT(OFFSET => -60*60)` to see the table 1 hour ago.
4. **Zero-copy cloning** — `CREATE TABLE foo CLONE bar`. Instant clone, only stores diffs. Game-changer for dev environments.
5. **Streams and Tasks** — change tracking + scheduling, the native CDC + orchestration features.
6. **Snowpark** — Python/Java/Scala execution inside Snowflake. The bridge into ML workloads.
7. **External tables and Iceberg integration** — query S3/GCS data directly; native Iceberg table support is rolling out.

### Databricks — What to Learn

1. **The notebook environment** — Python, SQL, Scala, R all in one notebook. Magic commands.
2. **Delta Lake** — the table format. ACID on object storage. `MERGE`, `OPTIMIZE`, `VACUUM`.
3. **Unity Catalog** — the governance layer. Tables, views, models, ML features all governed centrally.
4. **Delta Live Tables (DLT)** — declarative pipelines on Delta. Simpler than vanilla Spark.
5. **Workflows** — Databricks' built-in orchestrator. Less powerful than Airflow but tightly integrated.
6. **MLflow** — model lifecycle. Particularly relevant for DE/ML overlap roles.
7. **SQL Warehouses** — Photon engine for SQL queries. Competitive with Snowflake.

### Free Tier Reality

- Snowflake offers a 30-day free trial. Use it intensively.
- Databricks Community Edition is free but limited (no jobs, no MLflow). Better to use a paid trial or the free hours on AWS/GCP.

### What to Build

Port your medium-tier project to Snowflake (or Databricks). Same dbt code. Different connection profile. Compare cost and developer experience.

---

## Phase 4 — Lakehouse and Open Table Formats (1 week)

Covered conceptually in Tier 3. Now you actually build with them.

### Iceberg in Practice

Three ways to write Iceberg tables:

1. **PyIceberg** — Python library for table operations. Lightweight; good for ingestion.
2. **Spark + iceberg-spark-runtime** — for full table operations and big writes.
3. **Native engine support** — Snowflake, BigQuery, Trino, Databricks (writes coming) all read/write Iceberg natively now.

A useful exercise:

```python
# Using PyIceberg
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, LongType, TimestampType

catalog = load_catalog("default", **{
    "uri": "https://glue.us-east-1.amazonaws.com",
    "warehouse": "s3://my-bucket/warehouse/",
})

schema = Schema(
    NestedField(1, "order_id", StringType(), required=True),
    NestedField(2, "customer_id", StringType(), required=True),
    NestedField(3, "amount", LongType()),
    NestedField(4, "created_at", TimestampType(), required=True),
)

catalog.create_table("orders.orders", schema=schema)
```

Then query with Spark, Trino, or Snowflake — all reading the same data, all transactional. That moment is when "lakehouse" stops being marketing and starts being real.

### Time Travel and Snapshots

```sql
-- Query a table as it was 1 hour ago
SELECT * FROM orders.orders TIMESTAMP AS OF current_timestamp() - INTERVAL 1 HOUR;

-- Query a specific snapshot
SELECT * FROM orders.orders VERSION AS OF 12345;
```

This is why analysts love lakehouses. Bugs in your transformation? Roll back the table to before the bad write. Need to reproduce a number from last quarter? Query the snapshot from that date.

### Schema Evolution

Add a column to an Iceberg table without rewriting any files. Drop a column without breaking old queries. Change a column type with explicit promotion rules. The whole point of these formats is making schema change a non-event.

### What to Read

- The Iceberg [Table Spec](https://iceberg.apache.org/spec/) (skim, then refer back)
- Tabular's blog (the Iceberg creators)
- Databricks' Delta Lake docs and the "Delta vs Iceberg" community blog posts (read at least 3 to triangulate)

---

## Phase 5 — Data Quality and Observability (1 week)

### Why This Matters

dbt's built-in tests are a floor, not a ceiling. Real F100 data quality is multi-layered:

- **Unit-style tests** — dbt tests on column values
- **Anomaly detection** — row counts, value distributions, freshness
- **Schema change detection** — alert when source schemas drift
- **Lineage** — trace a downstream alert back to the upstream cause
- **Incident response** — when data is broken, what's the playbook?

### Tools to Know

1. **dbt tests + dbt_expectations** — your baseline
2. **Great Expectations** — heavier-weight, very flexible. Used at many F100s.
3. **Soda Core / Soda Cloud** — YAML-defined checks. Newer, lighter.
4. **Monte Carlo / Datafold / Anomalo** — commercial observability platforms. You won't have access, but you should know what they do.
5. **OpenLineage** — open standard for lineage events. Emitted by Airflow, dbt, Spark. Build a lineage graph from these events.

### What to Build

Add a real data quality layer to one of your existing projects:

- Define expectations for every source and mart table (row count ranges, null rates, value distributions)
- Add anomaly detection on at least one metric (alert when daily revenue is >3 stddev from rolling mean)
- Build a small lineage viewer using OpenLineage events from your Airflow + dbt runs

This is the kind of thing that turns a "I built a pipeline" interview answer into a "I built a *reliable* pipeline" answer.

### Incident Response Is a Discipline, Not a Dashboard

"How do you handle a 3 AM data incident?" is now a standard question in F100 senior DE interviews. Monte Carlo and Datafold give you observability. What you do *with* the alert is a separate skill — and most candidates have no answer because they've never been on call for data.

**On-call rotations for data teams.** Treat data infrastructure like software infrastructure: weekly rotations, primary + secondary, escalation paths documented before the incident happens. The days of "the pipeline broke, email Bob" are over at any serious F100 data team.

**Severity classification.** Not all incidents are equal. A reasonable classification:

| Severity | Example | Response time | Who gets woken up |
| --- | --- | --- | --- |
| S1 | Exec dashboard showing wrong revenue; compliance report corrupted | Immediate | On-call + data lead |
| S2 | Core fact table stale by >2 hours; ML feature store out of sync | <30 min | On-call engineer |
| S3 | Non-critical mart late; low-traffic report stale | Business hours | Ticket in queue |
| S4 | Cosmetic issue, low-priority pipeline slow | Next sprint | Backlog |

Define these before an incident — not during one. Disagreeing about severity at 3 AM is expensive.

**Runbook anatomy.** Every recurring pipeline should have a runbook. The format that works:

1. **Symptom** — what the alert looks like (exact alert name, what the metric was vs expected)
2. **Likely causes** — top 3 root causes for this alert, ranked by historical frequency
3. **Diagnostic checks** — the exact commands/queries to run first (with copy-pasteable SQL or CLI commands)
4. **Escalation path** — if checks A, B, C don't resolve it, who to page and what to tell them
5. **Resolution steps** — for each likely cause, the fix
6. **Post-incident** — link to postmortem template, which metrics to validate before declaring resolved

A runbook that requires you to figure out the structure at 3 AM is not a runbook.

**Blameless postmortems.** The standard from SRE culture applies to data teams. After any S1 or S2: within 48 hours, write up what happened, the timeline, contributing factors (never "human error" — always *why* the human was in a position to make that error), and action items. Postmortems that assign blame teach people to hide incidents. Postmortems that analyze systems teach people to improve them.

**Lineage-driven root-cause analysis.** This is the skill that separates senior DEs who've operated production systems from everyone else. When `marts.fct_revenue` is broken:

1. Pull the lineage for `fct_revenue` — every upstream model and source
2. Walk backward: which upstream ran most recently? What was the row count delta?
3. Check source freshness first (is the source late?), then transformation logic (did a model logic change?), then infrastructure (did a schema change break an expectation?)
4. OpenLineage + Elementary + dbt's lineage graph together give you this without manual investigation

Monte Carlo's incident management features and incident.io's runbook tooling are the commercial-grade implementations of this workflow. You don't need to use them, but you should know what problem they solve and be able to describe the same workflow with your OSS stack.

**What to build:** Add a `runbooks/` directory to your capstone project with at least two runbooks — one for "ingest job fails," one for "downstream mart row count anomaly." This is the kind of operational maturity that separates a "great project" from a "production-grade project" in an interview walkthrough.

---

## Phase 6 — Governance, Security, and Cost (1 week)

The single area separating "I can build pipelines" from "I can lead a data platform team."

### Governance

- **Data catalogs** — Atlan, Collibra, Alation, DataHub (open source), Unity Catalog (Databricks)
- **Lineage** — column-level lineage from your transformation tool
- **PII handling** — tagging, masking, role-based access
- **Glossary** — central definitions of business terms ("what is an active customer?")

You don't need to deeply use these tools. You need to know:

1. What problems they solve
2. Why F100 companies *must* have them (regulatory pressure)
3. How dbt integrates (exposures, contracts, semantic models)

### Security

- **Encryption at rest and in transit** — defaults at most cloud providers, but you need to know how to verify
- **Network isolation** — VPC peering, PrivateLink, no public endpoints
- **Secrets management** — Vault, AWS Secrets Manager, GCP Secret Manager. Never plaintext credentials.
- **Audit logging** — who queried what, when. Required in regulated industries.
- **PII detection and masking** — scan data for SSNs, credit cards, PII; mask or tokenize appropriately.

### Cost (FinOps)

Every senior DE eventually has a quarter where their main work product is a cost reduction project. Cost fluency is the single most visible differentiator between a senior and a junior in F100 data platform roles — juniors add features, seniors make features cheaper to run. Learn this now.

**The FinOps Foundation framework** (finops.org) is the industry standard for cloud cost management. The loop has three phases:

1. **Inform** — get visibility: who spent what, on what, when. You can't optimize what you can't measure. Tag every resource, query, and pipeline to a team and project.
2. **Optimize** — act on the data: right-size warehouses, kill idle resources, restructure expensive queries, tier cold storage.
3. **Operate** — embed cost into the engineering lifecycle: cost review in PR checks, budget alerts per team, a monthly cost retrospective.

**The FOCUS standard** (FinOps Open Cost and Usage Specification) is worth knowing by name. It's the vendor-neutral format for cloud billing data — same schema whether you're on AWS, Azure, GCP, or Snowflake. When an F100 runs multi-cloud, FOCUS is how they aggregate cost data without writing 4 different parsers. The FinOps Foundation publishes it; major cloud vendors have committed to emitting FOCUS-compliant data.

**FinOps for AI** is now a formal certification. As data platforms increasingly absorb GPU compute, embedding pipelines, and LLM API calls, the cost profile shifts dramatically. 78% of FinOps practices now report directly into the CTO or CIO — not Finance — because infrastructure cost is a product decision, not just an accounting one. If you're working anywhere near ML infrastructure, having FinOps literacy puts you in a small minority.

**Concrete practices:**

- **Per-query cost attribution** — tag queries to projects/teams; Snowflake's `QUERY_HISTORY` view and BigQuery's information schema both expose this
- **Slot/credit budgets** — guardrails so a runaway query doesn't burn $10K; set these before you need them
- **Storage tiering** — cold data to cheap storage classes; Iceberg's lifecycle policies + S3 Intelligent-Tiering are the right combo
- **Right-sizing compute** — Snowflake warehouses, Spark clusters, EMR — most are over-provisioned by 2x because nobody checked after initial setup
- **Materialization decisions** — every incremental model you add is a recurring cost; the question "is this materialization worth its refresh cost?" is a senior question

### What to Build

In your portfolio capstone, add a "Cost Dashboard" — a section in your dashboard layer that tracks per-day, per-pipeline cost. Even rough estimates demonstrate the right mindset. Name-drop the FinOps inform/optimize/operate loop in your README. The candidate who frames cost as a disciplined practice rather than an afterthought stands out.

---

## Phase 7 — CI/CD for Data (3 days)

Modern F100 data teams treat data code like software: PR reviews, CI tests, staged deploys, blue/green dbt environments.

### What to Learn

1. **dbt slim CI** — only run/test models affected by a PR. Massive cost savings on big projects.
2. **GitHub Actions workflows** for:
   - Running `dbt compile` and `dbt test` on every PR
   - Running unit tests on Python ingestion code
   - Deploying to dev → staging → prod on merge
3. **Environment separation** — separate datasets per environment. Snowflake zero-copy clones for PR environments are best-in-class.
4. **Data contracts** — dbt's `contract` feature, schema enforcement at boundaries between teams.

### What to Build

Add a `.github/workflows/` directory to your capstone:

- `pr.yml` — compile, test, lint on every PR
- `deploy.yml` — auto-deploy to prod on merge to `main`
- Branch protection on `main`

---

## Phase 8 — System Design for DE Interviews (1 week)

F100 senior DE interviews almost always include a system design round. The format: "Design a system that does X" — you're judged on clarity of thought, awareness of trade-offs, and the right vocabulary.

### Topics That Come Up

1. **"Design a data pipeline for [domain]"** — e.g., real-time fraud detection, marketing attribution, customer 360, ML feature store. Approach: clarify requirements → sketch the pipeline → discuss trade-offs at each layer.

2. **"How would you handle backfills?"** — Idempotency, partitioned writes, replay buffers (Kafka or S3), orchestrator support, the cost of double-runs.

3. **"How do you handle late-arriving data?"** — Event time vs processing time, watermarks, restating prior aggregates, incremental + merge patterns.

4. **"How do you ensure data quality?"** — Layered approach (source, staging, mart), automated tests, anomaly detection, lineage for root-cause analysis, incident playbooks.

5. **"How would you migrate from Redshift to Snowflake (or any X to Y)?"** — Parallel-run pattern, shadow writes, validation comparison, cutover criteria, rollback plan.

6. **"How would you reduce our $500K/year warehouse bill by 30%?"** — Audit query patterns, materialization decisions, partitioning fixes, compute right-sizing, cold-storage tiering.

### How to Prepare

- **DDIA (Designing Data-Intensive Applications)** by Martin Kleppmann. The single most important book in this space. Read it cover-to-cover. Twice. Most senior DE interviewers will assume you've read it.
- **The DE-specific system design interview prep:** *Data Pipeline Pocket Reference* by James Densmore, *Fundamentals of Data Engineering* by Joe Reis and Matt Housley. The latter is the closest thing to an industry consensus textbook.
- **Practice out loud.** Find a friend or use a mock interview service. Reading about system design and *doing* system design are different skills.

---

## A Word on Specialization

After this phase, you have an honest choice to make:

1. **Batch / warehouse specialist** — Spark, Snowflake/Databricks, dbt mastery, modeling depth. Most common DE archetype.
2. **Streaming specialist** — Kafka, Flink, real-time analytics, event-driven architecture. Smaller market, higher comp.
3. **Platform / infrastructure specialist** — Kubernetes, Terraform, Airflow operations, observability. Adjacent to SRE.
4. **Analytics engineer** — dbt, modeling, BI tools, semantic layer. Closest to analysts.
5. **ML platform / Feature Store engineer** — DE who lives next to ML teams. Your background fits.

Don't pick yet. Build one of the Fortune 100 portfolio projects in each of two or three of these specializations and see what energizes you. Then go deep on that.

---

## When You're Done with This Phase

You should have:

- A working stack on AWS (mirror of your GCP work)
- An Airflow orchestrator running your pipelines
- A Snowflake or Databricks port of your medium project
- An Iceberg or Delta table in at least one project
- A data quality and observability layer on at least one project
- CI/CD workflows on at least one project
- A bookshelf with DDIA, Fundamentals of DE, and one more book

Now you're ready to build something that lands you a Fortune 100 role — the portfolio projects come next.

---

## You can now

- Port a GCP-based stack to AWS — S3, Athena/Glue, EMR or Glue jobs, Redshift, Kinesis — and reason about IAM least-privilege and the cost traps unique to AWS.
- Orchestrate pipelines in Airflow 3.x with the TaskFlow API and asset-aware scheduling, and name what changed from Airflow 2 (`logical_date`, no SubDAGs, DAG versioning).
- Operate Snowflake or Databricks as your warehouse/lakehouse, and port a dbt project across warehouses by swapping only the connection profile.
- Work with open table formats in practice: create Iceberg tables, use time travel and snapshots, and evolve schemas without rewriting files.
- Layer real data quality, observability, incident runbooks, governance, FinOps cost discipline, and CI/CD onto a project — the operational maturity that separates senior from junior.
