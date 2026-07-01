# 00 — Overview and Prerequisites

A complete data engineering curriculum: a full core track covering the foundational tools, followed by post-core specialization and portfolio projects designed to land Fortune 100 data engineering roles.

## The Chapters in This Curriculum

| Chapter | What It Covers | Estimated Time |
|---|---|---|
| Overview and Prerequisites | Orientation, prereqs, study habits | 1 day |
| Beginner Guide | Docker, Postgres, Terraform, Kestra, dlt | 3–4 weeks |
| Medium Guide | BigQuery, dbt, dimensional modeling, data platforms | 3–4 weeks |
| Advanced Guide | Spark, Kafka, streaming, lakehouse, capstone | 4–6 weeks |
| Next Steps | Specialization: Airflow, Snowflake, Iceberg, observability, governance | 6–8 weeks |
| Fortune 100 Projects | 7 portfolio projects engineered to demonstrate F100-level competence | 6+ months total |

**Total realistic timeline:** 6 months of focused part-time study (10–15 hrs/week) to finish the core track plus the specialization. Then 6–12 months building 2–3 portfolio projects deeply. So roughly 12–18 months from a serious start to credibly interviewing for F100 DE roles.

## Prerequisites — What You Must Have Before You Start

This is non-negotiable. If you start without these, you'll spend most of your time fighting basics instead of learning DE.

### 1. Python — Solid Intermediate Level

You need to be comfortable with:

- Functions, classes, modules, imports
- List/dict/set comprehensions
- Virtual environments (`venv` or `uv` — I recommend `uv` in 2026; it's 10–100x faster than `pip` and replaces `pip`/`virtualenv`/`poetry` with a single Rust binary)
- `pip install` and `requirements.txt` (or `pyproject.toml` + `uv pip install`)
- Reading JSON, CSV, and Parquet files (pandas is fine, but try `polars` too — it's the modern columnar DataFrame library and much faster on medium data)
- Context managers (`with open(...)`)
- Basic error handling (`try/except`)
- f-strings and pathlib
- **Ruff** for linting/formatting (same Astral team as `uv`, Rust-based, replaces `flake8`/`black`/`isort`). Using `pip + flake8` in 2026 is a dated signal.
- **Pydantic v2** basics for data validation — typed models, `BaseModel`, field validators. Modern ingestion pipelines use it at boundaries (API responses, config files) instead of raw `dict` juggling.

You do **not** need to be a Python expert. You need to be able to write a 100-line script that reads data, transforms it, and writes it somewhere without googling syntax every five minutes.

**If you're not there yet:** Spend a week on the official Python tutorial, then build three small CLI scripts of your choosing (download a file, parse it, save somewhere).

### 2. SQL — Solid Foundation

This one matters more than people admit. Bad data engineers write bad SQL. Good ones write SQL that survives at scale.

You need to be comfortable with:

- `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`
- All four `JOIN` types (INNER, LEFT, RIGHT, FULL)
- `UNION` vs `UNION ALL`
- Subqueries and CTEs (`WITH`)
- Aggregation functions (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`)
- `CASE WHEN` expressions
- `DISTINCT` and `COUNT(DISTINCT ...)`

If you can write a SQL query that joins 4 tables, filters on a date range, groups by two columns, and produces a sensible result — you're ready. Window functions come later, in the medium tier.

**If you're not there yet:** [SQLZoo](https://sqlzoo.net/) or [SQLBolt](https://sqlbolt.com/). Pick one, finish it, move on. (The old Mode Analytics SQL tutorial is gone — Mode was absorbed into ThoughtSpot's Analyst Studio; skip it.)

**Better SQL practice option for 2026:** Install **DuckDB** (`pip install duckdb` or `brew install duckdb`) and practice against real Parquet/CSV files on disk. DuckDB is an embedded OLAP engine (think "SQLite for analytics") that speaks PostgreSQL-flavored SQL very close to BigQuery's. You'll use it again throughout the curriculum, so getting fluent now pays off twice.

### 3. Command Line — Comfortable, Not Expert

You should be able to:

- Navigate directories (`cd`, `ls`, `pwd`)
- Create/move/delete files (`touch`, `mv`, `rm`, `mkdir`)
- Read files (`cat`, `less`, `head`, `tail`)
- Pipe and redirect (`|`, `>`, `>>`)
- Set environment variables (`export`)
- SSH into a remote machine
- Use a text editor (vim survival skills, or just use VS Code)

### 4. Git — Basic Workflow

- `clone`, `add`, `commit`, `push`, `pull`
- Branch and merge
- Resolve a simple merge conflict
- Read a diff

If git intimidates you, [Learn Git Branching](https://learngitbranching.js.org/) is the best interactive tutorial available.

### 5. A Computer That Won't Hold You Back

- 16GB RAM minimum, 32GB strongly recommended once you hit Spark
- ~100GB free disk space
- A Linux/macOS development environment (WSL2 if you're on Windows — don't try to do this on raw Windows)

You also need a **Google Cloud account** with billing enabled. The course uses the free tier extensively; you should never spend more than $5–10 total if you're careful. Set up budget alerts on day one.

### 6. Data Formats Primer — Know These Before You Start

You'll see these constantly. Skim them now so the material doesn't pause to explain:

- **CSV** — row-oriented text. Universal, slow, no schema, no types. Fine for tiny files; awful for analytics.
- **JSON / JSON Lines (NDJSON)** — semi-structured, self-describing, slow to parse. JSONL (one JSON object per line) is the streaming-friendly variant.
- **Parquet** — columnar binary format. Compressed, typed, splittable. The default "data lake" file format. Reading one column of a Parquet file doesn't scan the others — that's why warehouses love it.
- **Avro** — row-oriented binary with a schema. Used in streaming (Kafka) because the schema can evolve safely.
- **ORC** — Hadoop-era columnar format. Still in some shops; Parquet won the mindshare battle.
- **Arrow** — in-memory columnar format. Not a storage format; a *transport* format. Polars, DuckDB, and pandas 2.x all speak Arrow internally, so they can pass data to each other with zero copies.

Rule of thumb: **Parquet for storage, Arrow for in-memory, Avro for streaming.** CSV only when something external forces your hand.

## Study Habits That Actually Work

### Type the code

Every example. Don't copy-paste. Don't just "follow along." Your hands need to learn the syntax even if your brain already knows it. This is the difference between people who finish the course and people who don't.

### Build a `notes/` directory

One folder, one markdown file per module. Write down:

- What the module taught you (in your own words)
- What broke and how you fixed it
- Three questions you couldn't answer yet
- A code snippet you want to remember

After six months, this folder is more valuable than any course material.

### Use AI as a tutor, not a crutch

When something doesn't work, your first move should be: *read the error*. Your second: *check the docs*. Your third: *ask AI to explain the error, not to fix it for you*. The moment you let AI write the code, you stop learning.

A good prompt pattern: *"Here's an error I'm getting. Here's what I think is happening. Here's what I tried. Can you tell me what concept I'm missing?"*

### Build in public (optional but high-leverage)

A weekly LinkedIn post or blog entry about what you learned. Three benefits:

1. Writing forces you to clarify your understanding
2. You build an audience of DE people before you need one
3. When you apply for jobs, "here are 20 posts about my journey" beats a bullet-point resume

### Don't compare to others

Online DE communities are full of people finishing tiers in days. Most of them are senior engineers brushing up. Your pace is your pace. The only comparison that matters is you-today vs you-six-months-ago.

## What This Curriculum Is Built to Do

Plenty of material teaches the tools. This curriculum is designed to go further and:

1. Tell you *what to skip* if you already know it
2. Show you which topics are foundational versus critical-but-often-missed
3. Give you projects substantial enough to anchor a Fortune 100 interview
4. Bridge from "I learned the tools" to "I'm employable at senior level"

The core track teaches the fundamentals; everything after it is the scaffolding that turns fundamentals into a hireable profile.

## When to Move On

When you've checked off everything in the **Prerequisites** section above. Move on to the beginner tier and start there.

If even one prereq is shaky, fix it first. Two weeks of solid prep saves three months of confused thrashing later.
