# 01 — Beginner Guide: Foundations and Your First Pipeline

**Topics covered:** Docker, Postgres, Terraform, GCP; workflow orchestration with Kestra; data ingestion with dlt

**Time:** 3–4 weeks at 8–10 hrs/week
**Goal:** Build genuine confidence by completing a real end-to-end pipeline that runs on a schedule.

## What You Will Be Able to Do After This Tier

- Spin up databases and tools in Docker containers without thinking twice
- Write a Python script that ingests data from somewhere and lands it in Postgres
- Provision cloud infrastructure (GCS buckets, BigQuery datasets) with Terraform
- Build a Kestra workflow that orchestrates a pipeline on a schedule
- Use `dlt` to ingest from APIs with incremental loading

By the end, you'll have a portfolio-worthy project even though you're only a quarter through the curriculum.

---

## Week 1 — Docker and Postgres

### Why Docker Matters in Data Engineering

Every tool you use as a DE — Postgres, Spark, Kafka, Airflow, dbt — runs in containers in production. Locally, Docker lets you spin up the entire production stack on your laptop without polluting your system. If you can't use Docker comfortably, you can't be a data engineer in 2026.

### What to Learn

1. **The Docker mental model**
   - Image vs container (image is a template, container is a running instance)
   - Layers and caching
   - The Dockerfile — a recipe for building an image
   - Networks and volumes

2. **Essential commands**
   ```bash
   docker run -it python:3.12 bash           # interactive container
   docker ps                                  # list running containers
   docker images                              # list images
   docker build -t myimage .                  # build from a Dockerfile
   docker exec -it <container> bash           # shell into a running container
   docker logs <container>                    # see what it's printing
   docker volume create mydata                # persistent storage
   ```

3. **Your first useful container — Postgres**
   ```bash
   docker run -it \
     -e POSTGRES_USER=root \
     -e POSTGRES_PASSWORD=root \
     -e POSTGRES_DB=ny_taxi \
     -v $(pwd)/ny_taxi_postgres_data:/var/lib/postgresql/data \
     -p 5432:5432 \
     postgres:15
   ```
   Read every flag. `-e` sets environment variables. `-v` mounts a host directory as a volume (so your data survives container restarts). `-p` maps a host port to a container port. This is the canonical command and you should be able to write it from memory by the end of the week.

4. **Connecting to Postgres**
   Install `pgcli` (better than `psql` for daily use):
   ```bash
   pip install pgcli
   pgcli -h localhost -p 5432 -u root -d ny_taxi
   ```

5. **Loading data with Python**
   Write a script that downloads the NYC taxi data, reads it with pandas (or pyarrow, which is much faster), and writes it to Postgres using SQLAlchemy. Walk through it step by step — type it yourself.

   **2026 upgrade:** Try the same exercise with **Polars** (`pl.read_parquet(...).write_database(...)`). On a 1GB file the difference is dramatic — usually 5–10x faster than pandas, and Polars hands you a clean Arrow buffer that Postgres' `COPY` protocol consumes efficiently. Pandas is still fine for tiny files and legacy code; reach for Polars when the file makes your laptop fan spin up.

### Exercises (Do All of Them)

1. Spin up Postgres in a container, load a CSV of your choice, run three SELECT queries against it.
2. Stop the container without `-v` mounted. Start a fresh one. Confirm your data is gone. Now do the same with the volume mounted, restart, confirm data persists. This teaches you why volumes matter.
3. Write a `Dockerfile` that bakes your Python ingestion script into an image. Run the ingestion *as a container*, not as a host script.

### Common Pitfalls

- **Port conflict on 5432.** If you already have Postgres installed on your host, the container can't bind to 5432. Use `-p 5433:5432` instead, then connect on 5433.
- **Permission errors on the volume.** Especially on Linux. The Postgres process inside the container runs as UID 999 and needs to be able to write to your volume directory. `chmod 777` works as a quick fix; figure out proper permissions later.
- **Cached layers when you don't want them.** `docker build --no-cache` if you suspect a layer is stale.

---

## Week 1 — DuckDB and Polars (Critical 2026 Additions)

In 2026, you can't credibly interview for an F100 DE role without these. They're the single biggest tooling shift in the last three years.

### DuckDB — The OLAP Engine in Your Process

DuckDB is to analytics what SQLite is to transactions: an embedded, single-file, zero-install database. It speaks columnar storage natively, runs PostgreSQL-flavored SQL, and reads Parquet/CSV/JSON directly from disk or S3/GCS — no loading step required.

**Why it matters:** In 2026 benchmarks (Decathlon case study; the widely-shared "650GB on S3" post), DuckDB and Polars regularly beat Spark on sub-100GB data running on a single laptop. Every senior DE now reaches for DuckDB *first* for exploration and only escalates to Spark when data genuinely doesn't fit.

**The five things to learn:**

```python
import duckdb

# 1. Query a Parquet file directly — no loading
duckdb.sql("SELECT COUNT(*) FROM 'yellow_tripdata_2024-01.parquet'").show()

# 2. Query a remote Parquet file from object storage
duckdb.sql("SELECT * FROM 's3://bucket/trips/*.parquet' LIMIT 10").show()

# 3. Join Parquet + CSV + Postgres in one query
duckdb.sql("""
    INSTALL postgres;
    LOAD postgres;
    ATTACH 'postgresql://root:root@localhost:5432/ny_taxi' AS pg;

    SELECT t.*, z.borough
    FROM 'trips.parquet' t
    JOIN pg.public.zones z ON t.pickup_zone_id = z.zone_id
""")

# 4. Round-trip with pandas / polars
df = duckdb.sql("SELECT * FROM 'trips.parquet'").df()       # to pandas
pl = duckdb.sql("SELECT * FROM 'trips.parquet'").pl()       # to polars

# 5. Persist to a single .duckdb file
con = duckdb.connect("warehouse.duckdb")
con.execute("CREATE TABLE trips AS SELECT * FROM 'trips.parquet'")
```

**When to reach for DuckDB:**

- Local development and unit testing of dbt models (instead of hitting BigQuery)
- Exploring a Parquet directory before you build a pipeline
- One-off analytics that don't justify spinning up a warehouse
- The transformation step in small/medium pipelines

**When not to:** truly distributed workloads (>~500GB at a single node), multi-user concurrent writes, anything that needs an external query API for a BI tool. Those still go to BigQuery/Snowflake — or **MotherDuck**.

**MotherDuck — DuckDB in the cloud.** The "DuckDB is local-only" limitation has an answer now. MotherDuck is a serverless DuckDB cloud warehouse: the same SQL dialect, the same file format, the same extensions — but your DuckDB database is hosted and queryable by a team, not locked to your laptop. More than 10,000 paying teams were using it by Q1 2026. Connecting is one line:

```python
import duckdb
con = duckdb.connect("md:my_warehouse")  # md: prefix routes to MotherDuck
con.sql("SELECT * FROM trips LIMIT 10").show()
```

Your local DuckDB files and your MotherDuck warehouse share the same dialect — zero relearning. **MotherDuck Flights** (launched June 2026) adds an agentic natural-language ingestion layer: describe what you want to ingest, Flights figures out the schema, fetches the data, and lands it in a DuckDB table. It's early, but it signals where the product is heading.

The updated guidance: DuckDB scales from a laptop exploration tool to a cloud data warehouse *without changing SQL dialects*. Start local, graduate to MotherDuck when you need sharing or always-on queries. This is a genuinely different position from "DuckDB is a toy; use Snowflake for production."

### Polars — The Modern DataFrame

Polars is what pandas would be if you redesigned it in 2020. It's columnar (Arrow-backed), multi-threaded by default, has a lazy execution engine, and the API is consistent and predictable (no `axis=0` vs `axis=1` lottery).

```python
import polars as pl

# Eager — reads immediately
df = pl.read_parquet("yellow_tripdata_2024-01.parquet")

# Lazy — builds a plan, executes only on collect()
result = (
    pl.scan_parquet("trips/*.parquet")
      .filter(pl.col("fare_amount") > 0)
      .group_by("vendor_id")
      .agg([
          pl.len().alias("trip_count"),
          pl.col("fare_amount").sum().alias("revenue"),
      ])
      .sort("revenue", descending=True)
      .collect()
)
```

The lazy API (`scan_parquet` + `collect`) is the killer feature — Polars builds a query plan, pushes predicates and projections down into the Parquet reader, and only materializes what you actually need. Same idea as Spark, but on a single machine with zero JVM overhead.

**The two-DataFrame rule for 2026:**

- **pandas** when you're gluing to legacy code or libraries that demand it
- **Polars** for anything new, especially anything performance-sensitive

### Exercises

1. Install DuckDB. Download a year of NYC taxi Parquet files. Run `SELECT COUNT(*), AVG(fare_amount) FROM 'yellow_tripdata_*.parquet' WHERE pickup_datetime > '2024-06-01'`. Note that it didn't read the early months.
2. Same query in pandas. Note the RAM usage and time.
3. Same query in Polars lazy mode. Compare.
4. Use DuckDB's `httpfs` extension to query Parquet directly from a public GCS bucket without downloading first.
5. Build a tiny Polars pipeline that reads a CSV, applies 3 transformations, and writes Parquet. Then re-do the same pipeline in lazy mode and `explain()` the plan.

---

## Week 1 — Docker Compose

A single container is fine for one tool. Real pipelines involve multiple tools talking to each other. Docker Compose is how you declare a multi-container stack.

### What to Learn

1. **`docker-compose.yml` structure** — services, networks, volumes, dependencies
2. **Service-to-service networking** — containers in the same compose file can reach each other by service name, not by `localhost`
3. **Bringing up the stack** — `docker compose up -d`, `docker compose down`, `docker compose logs -f`

### Your First Compose Stack

A `docker-compose.yml` for Postgres + pgAdmin:

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: root
      POSTGRES_PASSWORD: root
      POSTGRES_DB: ny_taxi
    volumes:
      - ./ny_taxi_postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: root
    ports:
      - "8080:80"
```

Run with `docker compose up -d`. Open `http://localhost:8080`. Connect pgAdmin to Postgres using the **service name** `postgres` as the host — *not* `localhost`. This is the moment Compose clicks.

---

## Week 2 — GCP and Terraform

### GCP Setup (Don't Skip Any Step)

1. Create a Google Cloud account at console.cloud.google.com (free $300 trial credit)
2. Create a new project — give it a memorable name like `de-course-2026`
3. Enable APIs you'll need: BigQuery API, Cloud Storage, IAM
4. Create a service account with these roles:
   - BigQuery Admin
   - Storage Admin
   - Viewer (project-level)
5. Generate a JSON key and download it. **Treat this file like a password.** Add the path to `.gitignore` immediately.
6. Install `gcloud` CLI, authenticate:
   ```bash
   gcloud auth activate-service-account --key-file=~/.gcp/your-key.json
   gcloud auth application-default login
   ```
7. **Set a billing alert at $5 immediately.** GCP charges add up fast if you leave a Dataproc cluster running.

### Terraform Basics

Terraform is "Infrastructure as Code" — instead of clicking buttons in a cloud console, you declare your infrastructure in `.tf` files and Terraform makes the real world match your declaration.

**Why this matters for DE:** Your data pipelines depend on buckets, datasets, IAM roles, network configs. The moment you have more than one environment (dev, staging, prod), you need IaC or you'll spend half your life debugging environment drift.

### Minimal Terraform Project

`main.tf`:
```hcl
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  credentials = file(var.credentials)
  project     = var.project
  region      = var.region
}

resource "google_storage_bucket" "data_lake" {
  name          = "${var.project}-data-lake"
  location      = var.region
  force_destroy = true

  lifecycle_rule {
    condition { age = 30 }
    action { type = "Delete" }
  }
}

resource "google_bigquery_dataset" "raw" {
  dataset_id = "raw"
  location   = var.region
}
```

`variables.tf`:
```hcl
variable "credentials" { default = "~/.gcp/your-key.json" }
variable "project"     { default = "de-course-2026" }
variable "region"      { default = "us-central1" }
```

Workflow:
```bash
terraform init      # download providers
terraform plan      # show what will change
terraform apply     # actually make the changes
terraform destroy   # tear it all down
```

### Exercises

1. Provision a bucket and a dataset with Terraform.
2. Modify the bucket's lifecycle rule, run `plan`, see Terraform detect the drift.
3. Add a second dataset (`staging`) to the same `.tf` file. Apply.
4. Destroy everything. Re-apply from scratch. Same state in 30 seconds. This is the magic.

---

## Week 2–3 — Workflow Orchestration with Kestra

### The Core Idea

Up to now, you've been running scripts manually. In production:

- Pipelines must run on schedules (daily, hourly)
- When they fail, they should retry (or alert someone)
- When task B depends on task A, B must wait for A to succeed
- You need a UI to see what ran when and what broke
- You need to be able to backfill historical date ranges

This is what a workflow orchestrator does. We use Kestra here; the industry uses Airflow more, but the concepts transfer 1:1.

### What to Learn

1. **Install Kestra in Docker**
   Use a `docker-compose.yml` to bring it up. UI is at `http://localhost:8080`.

2. **The anatomy of a Kestra flow**
   ```yaml
   id: hello_world
   namespace: dev

   tasks:
     - id: print_hello
       type: io.kestra.plugin.core.log.Log
       message: "Hello from Kestra!"
   ```
   Flows are YAML. Tasks have an `id` and a `type`. Tasks run in order unless you specify otherwise.

3. **Variables and inputs**
   ```yaml
   inputs:
     - id: date
       type: STRING
       defaults: "2024-01-01"

   tasks:
     - id: log_date
       type: io.kestra.plugin.core.log.Log
       message: "Processing data for {{ inputs.date }}"
   ```

4. **Triggers**
   ```yaml
   triggers:
     - id: daily_schedule
       type: io.kestra.plugin.core.trigger.Schedule
       cron: "0 6 * * *"   # 6 AM every day
   ```

5. **Real pipeline patterns**
   - Download a file → upload to GCS → load to BigQuery
   - Handle failures with retries
   - Parametrize by date for backfills

### Exercises

1. Build a flow that downloads NYC taxi data for a given month and uploads it to your GCS bucket.
2. Add a second task that loads the file from GCS to BigQuery.
3. Add a schedule that runs the flow daily.
4. Deliberately break the flow (wrong filename). Watch it fail. Add a retry policy. Watch it retry.
5. Trigger a backfill for three historical months. Confirm all three runs succeed.

### Data Lake vs Data Warehouse — Conceptual Pause

You're now writing to both GCS (lake) and BigQuery (warehouse). Make sure you understand the difference:

- **Data lake (GCS, S3, ADLS):** Raw files in object storage. Cheap. Schemaless. Anything goes.
- **Data warehouse (BigQuery, Snowflake, Redshift):** Structured tables. Optimized for analytics. More expensive per GB but vastly faster to query.

The modern pattern: land *raw* data in the lake, then load *modeled* data into the warehouse. The lake is your source of truth and replay buffer; the warehouse is your query layer.

---

## Week 3–4 — Data Ingestion with dlt

### The Ingestion Landscape — Where dlt Sits

Before diving in: there are two flavors of ingestion tooling in 2026, and you should know both exist.

1. **Code-first (dlt, Meltano).** You write Python. The tool handles schema inference, incremental state, retries, type coercion. Maximum flexibility, lowest abstraction tax.
2. **Connector-first (Airbyte, Fivetran, Estuary Flow).** You pick from a catalog of 350+ pre-built connectors and configure them via UI or YAML. Maximum coverage of weird SaaS APIs, less control.

**Airbyte** is the OSS leader of the connector-first camp and the most common ingestion tool you'll see at Fortune 100. **Fivetran** is the commercial leader (they acquired dbt Labs and Census in 2025, so they're now a consolidated transformation + activation + ingestion vendor). **dlt** is the code-first leader and what we use here — it's the right tool for learning because you see every concept exposed in Python rather than hidden behind a UI.

Senior interview signal: knowing *when* to write a dlt pipeline vs reach for an Airbyte connector vs pay Fivetran. The rule of thumb — if a robust Airbyte connector exists for your source, use it; if not, dlt; if the source is weird and high-volume streaming, look at Estuary.

### Why dlt Exists

Hand-rolling ingestion code seems easy until you hit:

- Nested JSON that needs flattening
- Schemas that evolve over time
- Incremental loading (only pull rows newer than last run)
- Type coercion (string `"2024-01-01"` → date)
- Retries, paging, rate limits

`dlt` (data load tool) handles all of this. You write a Python generator that yields records; dlt handles everything else.

### Minimal dlt Pipeline

```python
import dlt
import requests

@dlt.resource(write_disposition="append")
def nyc_taxi():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    # ... yield records from the parquet
    yield from records

pipeline = dlt.pipeline(
    pipeline_name="nyc_taxi",
    destination="duckdb",
    dataset_name="raw"
)

load_info = pipeline.run(nyc_taxi())
print(load_info)
```

That's a complete pipeline. Run it, query the resulting DuckDB file, see your data.

### Key Concepts to Internalize

1. **Resources and sources** — a `resource` is one table; a `source` groups multiple related resources
2. **Write dispositions** — `append`, `replace`, `merge` (for upserts)
3. **Incremental loading** — `dlt.sources.incremental("updated_at", initial_value="2024-01-01")` — dlt tracks state automatically
4. **Destinations** — DuckDB locally, BigQuery in production, Postgres, Snowflake, etc.

### Exercises

1. Build a dlt pipeline against a public REST API of your choice (GitHub events, Wikipedia, CoinGecko, NWS weather).
2. Implement incremental loading on a timestamp column.
3. Run it against DuckDB locally.
4. Change the destination to BigQuery, run again — same code, different warehouse.

---

## The Beginner Tier Project

This is what you build before moving to Tier 2.

### Spec

A scheduled pipeline that:

1. Runs daily via a Kestra flow
2. Uses `dlt` to ingest from a public API (your choice — but pick something with real volume, not a 50-row toy)
3. Lands raw data in GCS as Parquet
4. Loads to BigQuery (raw schema)
5. Has at least one transformation in SQL that produces a simple aggregated table (e.g., daily counts by category)
6. Provisions the GCS bucket and BigQuery datasets via Terraform
7. All in a Git repo with a clear README

### Acceptance Criteria

- A reviewer can clone your repo, follow your README, and have the pipeline running in under 30 minutes
- The pipeline survives a deliberate failure injection (kill the API mid-run, restart, no duplicates)
- Cost is tracked — your README has a "this costs $X/month to run" estimate

### What This Project Proves

- You can stand up infrastructure as code
- You can orchestrate scheduled work
- You can ingest from real APIs
- You can document well enough that someone else can run your code

These four skills are the floor for a junior DE role. You're now on the floor.

---

## You can now

- Spin up Postgres and multi-container stacks in Docker with the correct volume, port, and env-var flags — and explain why the volume is what makes data survive a restart.
- Query Parquet/CSV directly with DuckDB, including joins that span files, Postgres, and object storage, with no loading step.
- Choose deliberately between pandas and Polars, and use Polars' lazy `scan_parquet` + `collect` to push filters and projections down into the reader.
- Provision GCS buckets and BigQuery datasets as code with Terraform, reasoning about `plan` vs `apply` and drift detection.
- Build a scheduled Kestra flow — with inputs, triggers, retries, and backfills — that orchestrates a `dlt` API ingest into your lake and warehouse.

---

## Confidence Checks Before Tier 2

Don't move on until you can answer these without googling:

1. What's the difference between a Docker image and a container?
2. Why does Postgres in Docker need a volume mount to persist data?
3. What does `terraform plan` show that `apply` doesn't?
4. In a Docker Compose file with services `postgres` and `pgadmin`, what host name does pgadmin use to reach postgres?
5. What's the difference between a data lake and a data warehouse, in one sentence?
6. In Kestra, how do you make a task run only after another task succeeds?
7. In dlt, what's the difference between `append`, `replace`, and `merge` write dispositions?
8. Why do we use Parquet instead of CSV for the lake layer?
9. When would you use DuckDB instead of standing up a full Postgres + Python stack?
10. What does Polars' lazy API (`scan_parquet` + `collect`) do that eager pandas can't?
11. When would you reach for Airbyte instead of dlt, and vice versa?

If any of these is shaky, go back and re-read the relevant section. You're not behind; you're being thorough.

When they all feel solid, move on to the medium tier.
