# 06 — Advanced Topics: Everything Else Worth Knowing — Part 1 of 5: Distributed Systems, SQL, Storage, and the Lakehouse

The earlier sections cover the core curriculum, the F100 specialization tools, and the portfolio projects. This section covers the remaining body of knowledge that distinguishes a strong senior data engineer from a competent mid-level one — distributed systems theory, deep SQL, storage internals, alternative architectures, and the operational concerns that don't show up until you've shipped real systems.

Treat this as a **post-graduate curriculum**. You don't sit down and march through it. You consult it as you build the projects, as you encounter new problems at work, and as you prepare for interviews.

**How to use this section:** It's organized in 18 phases, split across 5 parts for readability. Phase 1–4 (this part) are foundational theory — work through them sequentially. Phase 5 onward (later parts) are specialized topics — read in any order based on the role you're targeting and the problems you're solving.

---

## Phase 1 — Distributed Systems Foundations

Most DE problems are distributed systems problems wearing different clothes. Without this foundation, you'll keep re-learning the same lessons by hitting the same walls.

### What to Learn

#### CAP and PACELC

**CAP theorem:** In the presence of a network **P**artition, a system must choose between **C**onsistency and **A**vailability. You can't have all three.

CAP is famous and slightly misleading. The better mental model is **PACELC**: in the presence of a Partition (P), choose Availability (A) or Consistency (C); Else (E), choose Latency (L) or Consistency (C). Real systems make this second trade-off constantly — even when there's no partition, faster reads usually mean weaker consistency.

**Examples to internalize:**
- Postgres (single-node): CA (no partitions to worry about)
- Cassandra: AP/EL — prioritizes availability and low latency, eventual consistency
- Spanner: CP/EC — strong consistency everywhere, accepts higher latency
- DynamoDB: tunable — you pick per request

#### Consistency Models

From strongest to weakest:

- **Linearizability** — every operation appears to happen instantaneously at some point between its call and return. The gold standard.
- **Sequential consistency** — operations from each client appear in order, but across clients the global order can be reordered.
- **Causal consistency** — operations that are causally related appear in the right order; unrelated ones can be in any order.
- **Read-your-writes** — you can always read what you just wrote.
- **Eventual consistency** — eventually, replicas agree. No guarantees about *when*.

For DE: most warehouses are eventually consistent for replicated reads but linearizable within a single write transaction. Most lakehouse formats provide *serializable* isolation (slightly weaker than linearizable but still strong).

#### Replication Strategies

- **Single-leader (primary-replica):** Postgres replication, most relational systems. Writes go to leader; reads can go anywhere with eventual lag.
- **Multi-leader:** rare and complex; needed for geo-distributed writes. Conflict resolution is the hard part.
- **Leaderless:** Cassandra, DynamoDB. Writes go to multiple nodes; quorum-based reads. R + W > N for strong reads.

#### Partitioning (Sharding) Strategies

- **Range partitioning** — partition by a key range. Good for ordered scans, bad for hotspots (e.g., "all today's events" land on one shard).
- **Hash partitioning** — hash the key, mod by shard count. Even distribution, terrible for range scans.
- **Composite** — hash on a high-cardinality field, range within (e.g., DynamoDB partition key + sort key).
- **Geo-partitioning** — by user region. Required for data residency compliance.

Every distributed database is a remix of these choices. Kafka uses hash partitioning on the message key. BigQuery uses range partitioning on a date/integer column. Snowflake uses micro-partitions automatically (range-like, but managed). The vocabulary is the same.

#### Consensus (Conceptually)

You don't need to implement Raft. You need to know:

- **Why consensus matters:** Multiple nodes agreeing on a single value (e.g., "who's the leader?", "what's the next log entry?") in the presence of failures.
- **Raft is the easier-to-understand modern consensus algorithm.** Watch the [Raft visualization](https://raft.github.io/).
- **Where you see it in DE:** Kafka's controller election (KRaft mode, post-ZooKeeper), etcd, every modern distributed database's metadata layer.

### The Reading

Three resources, in order of priority:

1. **Designing Data-Intensive Applications** by Martin Kleppmann. Chapters 5–9 are the distributed systems heart. If you only read one thing in this entire file, it's these chapters.
2. **The Dynamo paper** (Amazon, 2007) — foundational for eventually-consistent stores.
3. **The Spanner paper** (Google, 2012) — TrueTime and globally consistent transactions.

### Exercises

1. Sketch the architecture of three DE systems you've used (Postgres, Kafka, BigQuery, Snowflake — pick any). For each, identify: leader topology, replication strategy, consistency model, partitioning strategy.
2. Write a 500-word explanation of why exactly-once semantics across heterogeneous systems is hard, citing CAP and the two-phase commit cost.
3. Pick one paper from the [Papers We Love DE list](https://github.com/papers-we-love/papers-we-love) and write a one-page summary.

---

## Phase 2 — SQL Mastery Beyond the Basics

The medium-tier guide covered window functions and CTEs. This phase pushes deeper into the SQL that shows up in F100 interviews and real platform work.

### Window Functions — The Hard Patterns

```sql
-- Running totals partitioned and ordered, with frame specification
SELECT
  user_id,
  event_time,
  amount,
  SUM(amount) OVER (
    PARTITION BY user_id
    ORDER BY event_time
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_total,

  -- 7-day rolling window — RANGE is time-aware
  SUM(amount) OVER (
    PARTITION BY user_id
    ORDER BY event_time
    RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
  ) AS amount_last_7d,

  -- Find session boundaries: a new session starts after 30 min of inactivity
  CASE
    WHEN event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) > INTERVAL '30 minutes'
    THEN 1 ELSE 0
  END AS session_start,

  -- Then cumulatively sum the boundaries to get a session ID
  SUM(CASE
    WHEN event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) > INTERVAL '30 minutes'
    THEN 1 ELSE 0
  END) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id

FROM events;
```

Sessionization is the canonical "you can do this in SQL?" interview question. Internalize the pattern.

### Recursive CTEs

For hierarchies and graph traversal:

```sql
-- Find the full reporting chain for every employee
WITH RECURSIVE org_chain AS (
  -- Base case: top-level employees (no manager)
  SELECT employee_id, manager_id, name, 0 AS depth, ARRAY[name] AS chain
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  -- Recursive case: employees whose manager we've already found
  SELECT e.employee_id, e.manager_id, e.name, oc.depth + 1, oc.chain || e.name
  FROM employees e
  JOIN org_chain oc ON e.manager_id = oc.employee_id
)
SELECT * FROM org_chain;
```

Used for: org hierarchies, category trees, dependency resolution, graph shortest paths.

### MERGE and UPSERT Patterns

The single most useful statement for incremental data loads:

```sql
MERGE INTO target t
USING source s
  ON t.id = s.id
WHEN MATCHED AND t.updated_at < s.updated_at THEN
  UPDATE SET name = s.name, updated_at = s.updated_at
WHEN NOT MATCHED THEN
  INSERT (id, name, updated_at) VALUES (s.id, s.name, s.updated_at)
WHEN MATCHED AND s.is_deleted THEN
  DELETE;
```

This is what dbt's `incremental` materialization compiles to. Every CDC sink uses it. Postgres calls it `INSERT ... ON CONFLICT DO UPDATE`. BigQuery, Snowflake, Spark SQL, and Iceberg all support `MERGE`.

### Query Plans

Every database has an `EXPLAIN` command. Reading query plans is one of the highest-leverage skills a DE has. Three plan elements to focus on:

1. **Scan types:** sequential scan (reads everything), index scan (uses an index), partition pruning (only some partitions), broadcast vs shuffle.
2. **Join types:** hash join, nested loop join, merge join, broadcast hash join. The optimizer picks one; you can sometimes hint.
3. **Rows estimated vs actual:** When these differ by 10x+, the optimizer made bad decisions. Update statistics or rewrite the query.

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;  -- Postgres: shows actual execution
EXPLAIN FORMAT=JSON SELECT ...;          -- BigQuery
EXPLAIN ANALYZE SELECT ...;              -- Snowflake (returns query profile)
```

Spend a day reading plans for queries you've written. It changes how you write SQL forever.

### JSON in Modern SQL

Most warehouses now have first-class JSON support:

```sql
-- Postgres / BigQuery / Snowflake all support similar syntax
SELECT
  raw->>'event_type' AS event_type,           -- extract text
  (raw->'properties'->>'amount')::numeric,    -- extract and cast
  JSON_ARRAY_LENGTH(raw->'items') AS item_count,
  raw @> '{"status": "completed"}' AS is_complete  -- contains check
FROM raw_events;
```

You'll often land raw JSON in a `VARIANT`/`JSON` column and use SQL to extract structured columns. Especially common for webhook ingestion.

### Set Operations Beyond UNION

```sql
-- Rows in A but not B
SELECT * FROM events_today
EXCEPT
SELECT * FROM events_yesterday;

-- Rows in both
SELECT * FROM events_today
INTERSECT
SELECT * FROM events_yesterday;
```

`EXCEPT` is the easiest way to do data comparison ("what changed between these two tables?"). Use it for migration validation.

### Exercises

1. Write a sessionization query on any event stream.
2. Use a recursive CTE to compute the depth of every node in a hierarchy.
3. Write a `MERGE` that handles inserts, updates, and soft deletes.
4. Pick a slow query you've written and read its query plan. Identify one inefficiency.
5. Use `EXCEPT` to find rows that differ between two versions of a table.

---

## Phase 3 — File Formats and Storage Internals

DEs talk about Parquet constantly without knowing what's inside one. This matters because most of your job's performance comes down to file layout decisions.

### Parquet Internals

A Parquet file is structured as:

```
File
└── Row Groups (typically 128MB–512MB each)
    └── Column Chunks (one per column, per row group)
        └── Pages (typically 1MB each)
            └── Data Pages + Dictionary Pages + Index Pages
```

Key properties:

- **Column-oriented within a row group.** Reading 3 of 100 columns reads ~3% of bytes.
- **Statistics per row group:** min/max/null count for each column. Used for predicate pushdown — readers skip row groups where the filter can't match.
- **Dictionary encoding:** repeated values stored once; the column stores indexes. Massive wins for low-cardinality columns.
- **Run-length and bit-packing encoding** on top of dictionary.
- **Compression** (Snappy by default; ZSTD increasingly common) applied per page.

#### What This Means in Practice

1. **Row group size matters.** Too small → tons of metadata overhead. Too large → can't parallelize reads across them. 128MB is a reasonable default.
2. **Sort within row groups.** If readers filter on column X, sorting by X within each row group makes min/max statistics tight — readers skip most row groups.
3. **Column count matters.** 1000-column Parquet files are slower than 100-column files due to metadata.
4. **Dictionary encoding only works for low-cardinality columns.** Random UUIDs as column values produce no compression.

### ORC vs Parquet

ORC was developed by Hortonworks for Hive. Similar concept, different details:

- ORC has more aggressive lightweight indexing within row groups
- ORC supports ACID better via the Hive Streaming API
- Parquet has broader ecosystem support (every modern engine reads it)

In 2026, Parquet has won outside the Hadoop ecosystem. Iceberg, Delta, Hudi all use Parquet.

### Apache Arrow — The In-Memory Standard

Arrow is the columnar in-memory format. Parquet is the on-disk format. They're complementary.

The killer property of Arrow: **zero-copy data exchange between processes/languages**. A Python process and a Java process can share an Arrow buffer with no serialization cost. This is why the modern stack (Polars, DataFusion, DuckDB, Snowflake's Snowpark) is converging on Arrow.

Implications:

- A Spark → Pandas conversion that used to serialize each row now copies a single Arrow buffer
- A DuckDB query result can be passed to a NumPy ML model with zero serialization
- Cross-language data pipelines no longer pay serialization tax

### Compression Trade-offs

| Codec | Compression Ratio | CPU Cost | When to Use |
|---|---|---|---|
| Snappy | Low | Very low | Default for most warehouses. Fast read/write. |
| ZSTD | High | Medium | Cold storage, archives, anywhere bytes matter |
| GZIP | High | High | Legacy systems only |
| LZ4 | Low | Very low | Streaming/real-time |

For lakehouse tables: Snappy for hot data, ZSTD for cold partitions. Iceberg/Delta let you specify per-table.

### File Size Optimization

The "small files problem" is the single most common lakehouse pathology. Symptoms: 100K tiny files per partition, queries that spend 90% of their time listing files.

Fix: **compaction**. A scheduled job that rewrites a partition into fewer, larger files. Target file size: 128MB–1GB for analytics workloads.

```sql
-- Iceberg
CALL system.rewrite_data_files(table => 'db.orders', target_file_size_bytes => 536870912);

-- Delta
OPTIMIZE my_table;
```

### Exercises

1. Take a CSV file, write it as Parquet with different row group sizes (10MB, 128MB, 1GB), and compare query times.
2. Write the same data sorted vs. unsorted by a filter column. Time a filtered query against both.
3. Run a workload that produces small files. Then run a compaction job. Compare query times.
4. Use `pyarrow` to inspect a Parquet file's metadata directly. Look at the row group statistics.

---

## Phase 4 — The Lakehouse Deep Dive

Earlier files introduced Iceberg/Delta/Hudi. This phase goes deeper — the comparison that comes up in every senior interview.

### The Three Formats

| Feature | Iceberg | Delta Lake | Hudi |
|---|---|---|---|
| Origin | Netflix | Databricks | Uber |
| Strongest at | Schema evolution, snapshot isolation, multi-engine | Databricks ecosystem, performance | Streaming upserts |
| Native engines | Spark, Trino, Snowflake, BigQuery, DuckDB | Spark, Databricks (native), recent open additions | Spark, Flink |
| Catalog | Pluggable: Glue, Nessie, Polaris, REST | Mostly Hive metastore or Unity Catalog | Hive metastore, plus its own |
| Time travel | Yes (snapshots) | Yes (versions) | Yes (commits) |
| Schema evolution | Best in class | Good | Good |
| Partition evolution | Yes (unique to Iceberg) | No | Limited |
| Hidden partitioning | Yes (unique) | No | No |

### Iceberg-Specific Features Worth Knowing

**Hidden Partitioning.** In Hive-style tables, your `WHERE` clause must match the partition column literally. In Iceberg, you partition by a transform of a column (e.g., `days(event_time)`), and the optimizer applies the transform automatically. You write `WHERE event_time > '2024-01-01'` and Iceberg figures out which partitions to scan.

**Partition Evolution.** You can change a partition scheme without rewriting data. Old data stays in old partitions; new data uses the new scheme; queries handle both transparently. Game-changer for tables that grow into new partitioning needs.

**Snapshot Isolation.** Every write produces a new snapshot. Readers see a consistent snapshot regardless of concurrent writes. Time travel is just querying an older snapshot.

**Equality vs Position Deletes.**
- *Position deletes*: "delete row N in file X." Cheap to apply, requires reading the deletion file plus the data file.
- *Equality deletes*: "delete all rows where id = 42." More flexible, slightly more expensive to apply.

Modern Iceberg uses both depending on the operation. Streaming upserts often use equality deletes; batch deletes often use position deletes.

### Catalog Options

The catalog tracks "what tables exist and where their metadata is."

- **Glue Data Catalog** (AWS): default on AWS. Mature, but vendor-locked.
- **Hive Metastore**: the legacy choice. Still everywhere in Hadoop-era deployments.
- **Nessie**: git-like semantics for table metadata. Branches, tags, time travel via branches. Very interesting.
- **REST Catalog**: the modern open standard. Iceberg's REST catalog spec is now the canonical way.
- **Polaris** (Snowflake's open-source): REST-based, Snowflake-friendly.
- **Unity Catalog** (Databricks): Databricks' centerpiece. Strong governance features.

The trend: REST catalog implementations are converging. Multiple vendors (Snowflake, Databricks, AWS, Tabular/Databricks acquired) provide them. The lock-in is decreasing.

### Compaction and Maintenance

Lakehouse tables need ongoing maintenance:

1. **Compaction:** Rewrite small files into larger ones. Weekly is typical.
2. **Snapshot expiration:** Drop old snapshots beyond your retention window. Keep storage costs in check.
3. **Orphan file cleanup:** Failed writes can leave files no snapshot references. Periodic cleanup.
4. **Statistics refresh:** Update min/max stats after compaction.

```sql
-- Iceberg
CALL system.rewrite_data_files(table => 'orders.fct_orders');
CALL system.expire_snapshots(table => 'orders.fct_orders', older_than => TIMESTAMP '2024-01-01');
CALL system.remove_orphan_files(table => 'orders.fct_orders');
```

### Exercises

1. Create the same dataset as an Iceberg table and a Delta table. Run the same queries. Compare.
2. Implement partition evolution: change the partitioning of an Iceberg table, write new data, query across the old and new schemes.
3. Set up a Nessie catalog and use its branch feature: create a branch, write to it, merge it back.
4. Write a scheduled compaction job and run it on a table that has lots of small files. Measure before/after query times.

---

## You can now

- Reason from distributed-systems first principles — CAP/PACELC, consistency models, replication and partitioning strategies — and slot any new datastore into that mental model in minutes.
- Write the SQL that separates senior from mid: sessionization, recursive CTEs, MERGE/upsert, and reading a query plan to find the expensive step.
- Explain storage internals (Parquet row groups, statistics, encodings; Arrow zero-copy) well enough to make file-layout and compression choices that change query cost.
- Compare Iceberg, Delta, and Hudi, and name the catalogs (Glue, Hive Metastore, Nessie, REST, Polaris, Unity) that decide how those tables get discovered and governed.

This is part 1 of the Advanced Topics reference. Next: streaming and modern engines (Phases 5–10) in [Part 2](06b-advanced-topics.md).
