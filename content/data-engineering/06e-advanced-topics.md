# 06 — Advanced Topics: Everything Else Worth Knowing — Part 5 of 5: The Bookshelf, Agentic Data Engineering, and What's Next

This is part 5 of 5 of the Advanced Topics reference. [Part 4](06d-advanced-topics.md) covered disaster recovery, compliance, and architectural patterns (Phases 14–16); here we close out with the reading list (Phase 17), the 2026 shift toward agentic data engineering (Phase 18), and where to go from here.

---

## Phase 17 — The Bookshelf

A reading list to develop senior-level taste, in priority order.

### Tier 1 — Read These

1. **Designing Data-Intensive Applications** — Martin Kleppmann. The bible. Twice.
2. **Fundamentals of Data Engineering** — Joe Reis & Matt Housley. The closest thing to an industry-consensus DE textbook.
3. **The Data Warehouse Toolkit** — Kimball. The dimensional modeling foundation. Skim parts; the core chapters are essential.

### Tier 2 — Strongly Recommended

4. **Data Pipelines Pocket Reference** — James Densmore. Concise, practical.
5. **Building Event-Driven Microservices** — Adam Bellemare. Best book on Kafka in practice.
6. **Streaming Systems** — Tyler Akidau, Slava Chernyak, Reuven Lax. The book on streaming. From the people who built it.
7. **Database Internals** — Alex Petrov. Deep on storage engines.

### Tier 3 — When You're Ready

8. **The Data Engineer's Guide to Apache Spark and Delta Lake** — Bill Chambers, Matei Zaharia. By Spark's creator.
9. **Practical Data Privacy** — Katharine Jarmul. Compliance + technical implementation.
10. **97 Things Every Data Engineer Should Know** — collection of essays. Skim.

### Papers Worth Reading

- The Dynamo paper (Amazon)
- The Spanner paper (Google)
- The Kafka paper (LinkedIn)
- The Iceberg paper / overview essays (Netflix)
- Tyler Akidau's "Streaming 101" and "Streaming 102" essays (free, online)
- The Datadog "Architecture" blog series (free, online, immensely educational)

### Newsletters / Blogs Worth Following

- Benn Stancil (Mode/Substack) — strategy and meta
- The Pragmatic Engineer (Gergely Orosz) — engineering culture
- Seattle Data Guy (newsletter) — practical DE
- The dlt blog — concrete patterns
- The DataDev blog (Joe Reis)
- The Tabular blog (Iceberg)
- The Databricks engineering blog

### Conferences (Watch Talks Online)

- **Data + AI Summit** (Databricks) — free recordings, dozens of senior-engineer talks per year
- **Snowflake Summit** — same for Snowflake's world
- **Current** (Confluent) — Kafka and streaming
- **DataEngBytes** — community talks
- **Coalesce** (dbt Labs) — analytics engineering

---

## Phase 18 — Agentic Data Engineering (The 2026 Shift)

> **Date-stamped claims in this section.** This area is moving faster than any other in data engineering. Everything below reflects the state as of June 2026. Treat specific product claims as directionally correct; verify current capability before acting on them.

The most significant structural shift in data engineering since the cloud transition is underway: agents are building, debugging, and maintaining pipelines. The DE role is not disappearing — it is shifting from writing pipelines to specifying and reviewing them.

### What Is Actually in Production

**Databricks Genie Code** (launched March 2026) and **Snowflake Cortex Code** (launched April 2026) are the two clearest production signals. Both are agent systems that operate on your data platform: they can generate pipeline code from a natural-language spec, diagnose a failing job by reading logs and lineage, propose fixes, and (with approval) apply them. These are not demos. F100 companies are running them in pilot programs against real pipelines.

**dlt** reports that approximately 91% of new dlt pipelines are now agent-built as of mid-2026. The pattern: an agent scaffolds the pipeline (source connection, schema inference, incremental logic, destination config), a human reviews and approves, the pipeline ships. The agent writes the first draft; the engineer owns the output.

**MCP (Model Context Protocol)** is shipping in data tools as the integration layer. Cube and Qlik have MCP servers in production. The MCP ecosystem crossed 1,000 registered servers by mid-2026. MCP lets an LLM agent call into your semantic layer, run queries, inspect metadata, and read lineage — without you writing custom tool integrations for each agent. **MotherDuck Flights** (June 2026) uses this model: describe what you want to ingest in natural language, the agent resolves schema and data source, lands data in DuckDB.

### What to Actually Learn vs Watch

**Learn now:**

1. **Prompt-driven pipeline scaffolding review.** When an agent generates a dlt pipeline or a Spark job, you need to be able to review it the same way you review a junior engineer's PR. That means: does the incremental logic handle late arrivals? Is the deduplication key correct? Are secrets handled correctly? Is the schema evolution strategy safe? The agent writes fast; you catch the bugs.

2. **Agent-output code review patterns.** Agent-generated SQL has characteristic failure modes: it often produces technically correct but semantically wrong joins (especially fan-out joins that multiply row counts), misidentifies grain, and gets incremental filter predicates subtly wrong. Build a checklist for reviewing agent SQL the same way you have a checklist for reviewing human SQL.

3. **Guardrails on agent-run SQL.** When an agent can execute SQL against your warehouse autonomously, the blast radius of a mistake is large. Standard guardrails:
   - Read-only grants for agent service accounts (never write access by default)
   - Row caps on agent queries (`LIMIT 10000` enforced at the connection level)
   - Query cost budget per agent session
   - Dry-run / EXPLAIN before execution
   - Human approval gate before any DDL (CREATE, DROP, ALTER)

4. **Spec-writing as a core skill.** The DE's primary output is shifting from code to specifications: clear, unambiguous descriptions of what a pipeline should do, what the schema should be, what the quality constraints are, and what the failure behavior should be. A vague spec produces a plausible-looking but wrong pipeline. A precise spec produces a reviewable first draft. This is the new "write clean SQL" — the fundamental craft.

**Watch but don't learn yet:**

- Autonomous pipeline maintenance (agents that detect failures, diagnose root cause, and open PRs with fixes — shipping in Genie Code and Cortex Code but still early for unattended production use)
- Agent-to-agent orchestration (multiple specialized agents coordinating on a complex pipeline build — research-grade in mid-2026)
- Fully autonomous data contract generation from schema inference

### The Role Shift — What This Means for Your Career

The candidate who in 2024 differentiated by knowing Airflow and dbt now differentiates by understanding *when to trust agent output and when not to*. The F100 interviews that ask "write me a Spark job" are becoming "review this agent-generated Spark job and tell me what's wrong with it."

The skills that become more valuable: systems thinking (understanding what a correct pipeline looks like end-to-end), data contract and spec writing, cost and security review of generated code, and knowing the failure modes of LLM code generation deeply enough to catch them quickly.

The skills that become less valuable: remembering API syntax, boilerplate pipeline scaffolding, writing the same incremental dlt resource for the 50th time.

The honest summary: data engineering is not being automated away. The hard parts — designing the right data model, specifying correct incremental logic, owning quality and cost, responding to incidents — all require judgment that agents don't have. The easy parts — scaffolding, boilerplate, format conversion — are largely gone. Senior DEs who adapt become more productive. Junior DEs who only knew the easy parts have a harder path.

---

## You can now

- Reason from distributed-systems first principles — CAP/PACELC, consistency models, replication and partitioning strategies — and slot any new datastore into that mental model in minutes.
- Write the SQL that separates senior from mid: sessionization, recursive CTEs, MERGE/upsert, and reading a query plan to find the expensive step.
- Explain storage internals (Parquet row groups, statistics, encodings; Arrow zero-copy) well enough to make file-layout and compression choices that change query cost.
- Compare lakehouse formats and catalogs, operate Flink for stateful exactly-once streaming, and pick a real-time OLAP store (ClickHouse/Druid/Pinot) when latency demands it.
- Put guardrails around agent-generated SQL and route LLM queries through a semantic layer — the DE-owned surface of the 2026 AI/DE convergence.

---

## A Closing Note

You'll never finish this curriculum. New tools appear monthly; old ones become obsolete. The point isn't to know everything; the point is to internalize the **underlying patterns** so deeply that any new tool slots into your existing mental model in a day.

The patterns that recur:

- Storage and compute separation, repeatedly reinvented
- Lazy evaluation and predicate pushdown, in every modern engine
- Eventually-consistent replication with strong-consistency islands
- Idempotency as the primary defense against distributed failures
- Schema evolution as a first-class operational concern
- Cost as a function of bytes scanned, shuffles, and compute time

Master these patterns and the rest is vocabulary.

---

## What to Do Next

You've now seen the full landscape. The honest path forward:

1. **Finish the core track.** The beginner, medium, and advanced tiers.
2. **Pick a specialization** from the next-steps phase and go deep on it. Don't sprinkle attention.
3. **Build two portfolio projects** from the Fortune 100 projects. Slowly. Deeply.
4. **Use these advanced topics as a reference** when problems push you into new territory.
5. **Read DDIA at least twice.** I'm serious.

The compound interest on solid fundamentals over 18 months is genuinely transformative. Most candidates skip the fundamentals and end up mid-level forever. Don't be most candidates.
