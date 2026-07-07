# 03 — Advanced Guide: Distributed Processing, Streaming, and the Capstone — Part 3 of 3: Lakehouse Architecture and the Capstone

This is part 3 of the Advanced Guide lesson (Distributed Processing, Streaming, and the Capstone). [Part 1](03-advanced-guide.md) covered Spark and [Part 2](03b-advanced-guide.md) covered Kafka, streaming, and CDC; here we cover lakehouse table formats and catalogs, then the capstone project that ties the whole tier together.

## Week 3–4 — Lakehouse Architecture (Bonus)

So far we've mostly treated lakes (GCS) and warehouses (BigQuery) as separate things. The frontier of the industry is the **lakehouse** — combining the cheap storage of a lake with the ACID guarantees and query performance of a warehouse, using **open table formats**.

### Why You Need to Know This

In 2026, every Fortune 100 data platform team is having the "Iceberg vs Delta" conversation. If you can speak to it credibly in an interview, you're already in a different category from candidates who only know the fundamentals.

### The Three Open Table Formats

1. **Apache Iceberg** — Netflix-originated. Strong on schema evolution, snapshots, time travel. The fastest-growing of the three. Supported natively by Snowflake, BigQuery, Databricks, Trino, Spark.
2. **Delta Lake** — Databricks-originated. Strongest in the Databricks ecosystem. Now open source.
3. **Apache Hudi** — Uber-originated. Strongest on streaming upsert workloads. Niche but real.

All three solve the same problem: ACID transactions, schema evolution, time travel, and metadata management on top of object storage (S3/GCS/ADLS).

### How a Lakehouse Table Is Built

A lakehouse table is **Parquet files + a metadata layer**. The metadata layer tracks:

- Which files are part of the table right now
- The schema at this point in time
- Snapshots (so you can time travel)
- Statistics for query planning

When you `INSERT`, the engine writes new Parquet files and updates the metadata. When you read, it consults the metadata to know which files to actually scan.

### What to Read

- The Iceberg [spec overview](https://iceberg.apache.org/spec/) (skim — get the gist)
- Tabular's blog (the Iceberg founders) — pick any 2–3 posts
- Databricks' Delta Lake docs — same concepts, different vocabulary

You don't need to implement a lakehouse yet — you'll do that in the projects file. You need to be able to *discuss* it.

### The Catalog Wars — 2026's Real Battleground

The Iceberg-vs-Delta format war is over. Interoperability layers (Delta UniForm, Apache XTable) mean engines can read each other's tables. The folder layout stopped being the interesting question.

The new battleground is **catalogs** — the service that tracks "which tables exist, which schemas they have, who can read them, which snapshot is current." If you walk into a 2026 F100 system design round saying "we'll use Iceberg" without naming a catalog, you've just shown you stopped reading in 2024.

The four catalogs to know:

1. **Iceberg REST Catalog** — the open *protocol* spec. Any catalog can implement it; engines speak it as a lingua franca. **This is the standard you must know by name.**

2. **Apache Polaris** — Snowflake-donated reference implementation of the REST catalog; **graduated to a top-level Apache project in Feb 2026** (1.4 added production hardening: STS session tags, S3 KMS, CockroachDB backend). Vendor-neutral, OSS, deployable on your own cluster; Snowflake's Horizon Catalog runs on it, Dremio's Open Catalog is managed Polaris. The "safe default" recommendation in 2026.

3. **Unity Catalog (OSS)** — Databricks open-sourced their catalog in 2024. Supports both Iceberg and Delta tables. Tighter UX in the Databricks ecosystem; viable as a standalone too.

4. **Nessie** — Dremio-originated. The differentiator: **git-style branching** of your data — create a branch, mutate tables, merge or discard. Powerful for testing transformations against a snapshot of production without copying it.

Others worth a sentence:

- **AWS Glue Data Catalog** — the historical default in AWS shops. Still very common. Supports Iceberg/Delta via plugins.
- **Hive Metastore** — what catalogs are slowly replacing. Treat as legacy.
- **Tabular** — the Iceberg founders' catalog product; **acquired by Databricks in 2024** for ~$1B. Folded into Unity Catalog.

The senior takeaway: **format is mostly settled (Iceberg has the momentum), catalog is the design decision**. Pick a catalog that speaks REST so you can swap engines later.

### DuckLake — The Newest Entrant (Watch List)

**DuckLake** shipped v1.0 in April 2026. The idea: skip the JSON/Avro metadata files of Iceberg entirely and put the table metadata directly in a SQL database (Postgres, DuckDB itself, anything). Parquet still in object storage, but the catalog *is* the metadata store — no separate manifest layer. Simpler, transactional by construction (Postgres handles concurrency), and noticeably faster on small/medium tables.

Status check (mid-2026): **early production, no longer just a watch-list item.** v1.0 is stable, it's a top-10 DuckDB extension by downloads, and client support landed for Spark, Trino, DataFusion, and Pandas. Viable for *new* small/medium projects; still young for migrating a large existing lakehouse. Knowing its design tradeoff (SQL-database metadata vs manifest files) is a strong 2026 interview signal.

**MotherDuck — DuckDB as a cloud warehouse.** DuckLake and MotherDuck are complementary. MotherDuck is the serverless cloud runtime for DuckDB: same SQL dialect, same extensions, same file formats — but the database is hosted, shareable, and always-on. 10,000+ paying teams by Q1 2026. The practical implication: the old guidance "DuckDB is local-only, graduate to Snowflake for production" no longer holds cleanly. You can now run DuckDB locally for development, attach to a MotherDuck warehouse for production queries, and share results with a team — without changing a line of SQL. For small-to-medium data teams, this removes a whole tier of infrastructure complexity. **MotherDuck Flights** (June 2026) extends the platform with an agentic ingestion layer — more on this in the Agentic Data Engineering phase at the end of this guide.

### AI / DE Convergence — pgvector, LanceDB, and the Semantic Layer as AI Interface

The fastest-growing surface area in DE 2026 is the boundary with AI systems. You don't need to become an MLE, but you need to know the data-infrastructure side of this conversation.

**Vector stores you should know by name:**

- **pgvector** — Postgres extension. The default for <10M vectors. Zero new infrastructure if you're already on Postgres. Every F100 evaluation starts here and usually ends here.
- **LanceDB** — embedded, file-based vector store (Parquet + Lance format). Great for edge, notebooks, and small-app deployments.
- **Pinecone / Weaviate / Qdrant / Milvus** — standalone vector DBs. Reach for these only when pgvector hits a real ceiling (which is later than most people think).

**The DE-as-AI-infrastructure narrative.** When an LLM agent answers a business question over your warehouse, you want it pulling structured metric definitions from a **semantic layer**, not generating raw SQL against your bronze tables. Text-to-SQL accuracy has improved a lot but still loses to "text-to-metric" against a well-modeled semantic layer (dbt's, Cube's, Looker's). Cube supports MCP (Model Context Protocol) for native agent integration; dbt's semantic layer has its own JSON-RPC interface.

The 30-second elevator pitch you should be ready to give: *"DEs are the ones building the structured, governed, observable substrate that AI systems read from. The semantic layer is becoming the AI interface to the warehouse; vector stores extend that to unstructured corpora."*

---

## The Capstone Project

This is the big one. Industry-grade. A piece you'll be talking about in interviews two years from now.

### Spec (Minimum)

1. **A real data source** — pick something with genuine volume. APIs that work:
   - GitHub Events API (live firehose of activity)
   - Reddit API (high-volume comments and posts)
   - A public Kafka stream (Wikipedia edits, financial data)
   - Synthesized but realistic data (generate it with `faker` at scale)

2. **Both batch and streaming components**
   - Streaming: Kafka producer + consumer pipeline that handles live ingestion
   - Batch: Spark job that processes historical data weekly

3. **Lakehouse storage** — Iceberg or Delta tables, not raw Parquet. Use it for at least one fact table.

4. **A modern stack:**
   - Ingestion: dlt + Kafka (and at least *evaluate* one Airbyte connector, even if you don't use it — make a deliberate choice)
   - Storage: GCS (lake) + BigQuery or Snowflake (warehouse) + Iceberg tables (lakehouse) registered in a **REST-spec catalog** (Polaris or Unity Catalog OSS)
   - Processing: Spark for the heavy batch step; **DuckDB or Polars for at least one batch step under 100GB** — you want both on the resume; **Flink** for streaming (Kafka Streams or ksqlDB acceptable, but Flink is the credibility tool)
   - Transformation: dbt with proper layering (bonus: port one mart to SQLMesh as a comparison exercise and write up the trade-offs)
   - Orchestration: Airflow (yes, switch to Airflow for this — it's what F100 uses)
   - Observability: structured logs + at least one metric per pipeline + alerts on failure + **OpenLineage** events emitted from dbt/Airflow/Spark; at minimum, **Elementary** wired into the dbt project
   - Governance: at least one **data contract** YAML on the most critical upstream source
   - IaC: Terraform for all cloud resources

5. **A visualization layer** — at least one dashboard (Metabase, Looker Studio, Superset). The dashboard isn't the point, but you need it to demonstrate the pipeline produces something useful.

6. **A `README` that's a tech blog post.** Architecture diagram. Decisions and trade-offs. What you'd do differently. Cost analysis. Lessons learned.

### Acceptance Criteria

- The whole stack comes up with `make up` or equivalent (one command)
- A reviewer can ingest live data within 5 minutes of starting
- Failure of any single component is detected and recovered from gracefully
- The streaming side handles at least 1000 events/second (load test it and document the result)
- The batch side processes at least 100GB without falling over (or you simulate it and explain)
- The `README` would survive being read by a senior engineer who's deciding whether to interview you

### Why This Capstone Matters

It's substantial enough that talking about it for 30 minutes in an interview is natural. It demonstrates batch + streaming + modeling + orchestration + IaC + observability. There's no realistic data engineering job at a Fortune 100 that touches *none* of these. By the time you finish it, you have an answer to every variant of "tell me about a project you're proud of."

---

## You can now

- Reason about Spark execution — driver, executors, partitions, shuffles — locate why a job is slow, and fix it with broadcast joins, early filtering, or AQE skew handling.
- Decide when a single-node engine (DuckDB, Polars) beats a Spark cluster, and articulate that threshold in an interview.
- Stand up Kafka, design partitioning for throughput and ordering, manage consumer groups, and evolve schemas safely with Avro + Schema Registry.
- Build a log-based CDC pipeline with Debezium — including the outbox pattern and the WAL-retention gotcha — and land changes into an Iceberg table via MERGE.
- Compare table formats (Iceberg/Delta/Hudi) and, more importantly, catalogs (REST spec, Polaris, Unity, Nessie), and explain why the catalog is the sharper 2026 decision.

---

## Confidence Checks Before Moving On

1. You can sketch the Spark execution model (driver, executors, partitions, tasks, shuffles) on a whiteboard.
2. You can explain when you'd broadcast a join vs use the default sort-merge.
3. You can describe what a Kafka consumer group does and what happens when one consumer dies.
4. You can explain why exactly-once semantics across heterogeneous systems is hard.
5. You understand the difference between event time, processing time, and watermarks.
6. You can describe what an Iceberg snapshot is and why it enables time travel.
7. Your capstone runs, end-to-end, with one command.
8. You can explain when DuckDB or Polars beat Spark, and conversely when Spark is still the right tool.
9. You can name three streaming engines beyond Kafka Streams, and the one-line trade-off for each (Flink, RisingWave, Materialize / Bytewax / Spark Structured Streaming).
10. You can explain the difference between a *table format* (Iceberg, Delta) and a *catalog* (Polaris, Unity, Nessie, REST spec), and why the catalog is the more interesting 2026 decision.
11. You can explain what OpenLineage emits and why a team would standardize on it across dbt + Airflow + Spark.
12. You can describe the role of pgvector / LanceDB in a 2026 data platform and why a DE would care.

When these are all solid, you're past the core track. Move on to the specialization phase in the next section.
