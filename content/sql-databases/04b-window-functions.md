# 04b — Window Functions

`GROUP BY` answers "what's the total per group?" — but it does so by *collapsing* each group into a single row. That's exactly what you want for a summary, and exactly what you *don't* want when the question is "how does each row compare to its neighbors?". Ranking every salesperson within their region, computing a running total, or finding the most recent event per user all need each original row to survive while still seeing across a set of related rows. That's what **window functions** do. This lesson is one of the highest-leverage things in the whole SQL track: window functions are non-negotiable for point-in-time feature engineering and they show up in nearly every SQL interview.

We'll use a small `sales` table:

| sale_id | region | rep     | amount | sale_date  |
|---------|--------|---------|--------|------------|
| 1       | East   | Ada     | 100    | 2026-01-01 |
| 2       | East   | Ada     | 300    | 2026-01-03 |
| 3       | East   | Alan    | 250    | 2026-01-02 |
| 4       | West   | Grace   | 400    | 2026-01-01 |
| 5       | West   | Katherine | 150  | 2026-01-04 |

## The core idea

A window function computes a value across a **set of rows related to the current row** — the "window" — and returns one result *per row*, without collapsing anything. The row count of your result stays the same as the input.

Contrast the two directly. `GROUP BY` collapses:

```sql
SELECT region, SUM(amount) AS region_total
FROM sales
GROUP BY region;
```

| region | region_total |
|--------|--------------|
| East   | 650          |
| West   | 550          |

Two input regions, two output rows. The individual sales are gone.

The window version keeps every row *and* attaches the regional total to each:

```sql
SELECT
  sale_id, region, amount,
  SUM(amount) OVER (PARTITION BY region) AS region_total
FROM sales;
```

| sale_id | region | amount | region_total |
|---------|--------|--------|--------------|
| 1       | East   | 100    | 650          |
| 2       | East   | 300    | 650          |
| 3       | East   | 250    | 650          |
| 4       | West   | 400    | 550          |
| 5       | West   | 150    | 550          |

All five rows survive. Each one now carries its region's total alongside its own amount — so you can immediately compute "what share of my region did this sale represent?" without a self-join.

## Anatomy of OVER()

The `OVER()` clause is what turns an ordinary function into a window function. It has three parts, all optional:

```sql
function(...) OVER (
  PARTITION BY <columns>   -- split rows into independent windows
  ORDER BY <columns>       -- order rows within each window
  <frame clause>           -- which rows around the current one to include
)
```

- **`PARTITION BY`** divides the rows into groups that are processed independently — like `GROUP BY`, but without collapsing. Omit it and the whole table is one window.
- **`ORDER BY`** sets the order *inside* each partition. Ranking and offset functions need it; running totals use it to decide "up to which row."
- The **frame clause** (covered later) narrows the window to a sliding range of rows.

An empty `OVER ()` means "the entire result set as one window" — handy for adding a grand total to every row.

## Ranking: ROW_NUMBER, RANK, DENSE_RANK

These three assign a position to each row within its partition, ordered by whatever you specify. They differ only in how they handle ties.

```sql
SELECT
  region, rep, amount,
  ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS row_num,
  RANK()       OVER (PARTITION BY region ORDER BY amount DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS dense_rnk
FROM sales;
```

Suppose the East region had two sales tied at 300. The three functions diverge:

| region | amount | row_num | rnk | dense_rnk |
|--------|--------|---------|-----|-----------|
| East   | 300    | 1       | 1   | 1         |
| East   | 300    | 2       | 1   | 1         |
| East   | 250    | 3       | 3   | 2         |
| East   | 100    | 4       | 4   | 3         |

- **`ROW_NUMBER()`** always gives a distinct number, breaking ties arbitrarily (1, 2, 3, 4).
- **`RANK()`** gives ties the same rank, then *skips* the next values (1, 1, 3, 4).
- **`DENSE_RANK()`** gives ties the same rank but does *not* skip (1, 1, 2, 3).

### The "latest record per group" / dedup pattern

This is the single most useful window pattern in day-to-day data work. To get one row per group — say, each rep's single largest sale, or the most recent event per user — number the rows within each partition and keep number 1:

```sql
SELECT *
FROM (
  SELECT
    region, rep, amount, sale_date,
    ROW_NUMBER() OVER (PARTITION BY rep ORDER BY sale_date DESC) AS rn
  FROM sales
) ranked
WHERE rn = 1;
```

You can't put a window function directly in `WHERE` (windows are computed *after* `WHERE` runs), so you wrap the query in a subquery — or a CTE — and filter on the alias outside. Ordering by `sale_date DESC` and keeping `rn = 1` gives the newest row per rep. Flip to `ASC` for the earliest, or partition by a key column and order by anything to deduplicate down to one row per key.

## Offset functions: LAG and LEAD

`LAG` and `LEAD` reach into *other* rows relative to the current one — `LAG` looks backward, `LEAD` looks forward — within the ordered partition. The classic use is period-over-period change. Here's each rep's day-over-day change:

```sql
SELECT
  rep, sale_date, amount,
  LAG(amount) OVER (PARTITION BY rep ORDER BY sale_date) AS prev_amount,
  amount - LAG(amount) OVER (PARTITION BY rep ORDER BY sale_date) AS day_over_day
FROM sales;
```

For Ada's two sales (100 on the 1st, 300 on the 3rd):

| rep | sale_date  | amount | prev_amount | day_over_day |
|-----|------------|--------|-------------|--------------|
| Ada | 2026-01-01 | 100    | NULL        | NULL         |
| Ada | 2026-01-03 | 300    | 100         | 200          |

The first row has no prior row, so `LAG` returns `NULL`. You can supply a default — `LAG(amount, 1, 0)` returns 0 instead of NULL — and the second argument is the offset (`LAG(amount, 2)` looks two rows back). `LEAD` works identically in the opposite direction.

## Running aggregates and the frame clause

Add an `ORDER BY` to an aggregate window and it becomes *cumulative* — it aggregates from the start of the partition up to the current row. That's a running total:

```sql
SELECT
  rep, sale_date, amount,
  SUM(amount) OVER (PARTITION BY rep ORDER BY sale_date) AS running_total
FROM sales;
```

| rep | sale_date  | amount | running_total |
|-----|------------|--------|---------------|
| Ada | 2026-01-01 | 100    | 100           |
| Ada | 2026-01-03 | 300    | 400           |

The **frame clause** makes the window explicit. It defines *which rows around the current one* the function sees:

```sql
SUM(amount) OVER (
  PARTITION BY rep
  ORDER BY sale_date
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
```

`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` — everything from the partition start through the current row — is the default frame when you supply `ORDER BY`, which is why the running total above worked without writing it out. Change the bounds to get a **moving average**, e.g. a trailing 3-row window:

```sql
SELECT
  rep, sale_date, amount,
  AVG(amount) OVER (
    ORDER BY sale_date
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS moving_avg_3
FROM sales;
```

Frame bounds you'll use most: `UNBOUNDED PRECEDING` (partition start), `N PRECEDING`, `CURRENT ROW`, `N FOLLOWING`, and `UNBOUNDED FOLLOWING` (partition end). `ROWS` counts physical rows; `RANGE` works on value distances of the `ORDER BY` column. This syntax is standard and behaves the same across Postgres, BigQuery, and DuckDB.

## Why this matters for ML

Window functions are how you build **point-in-time-correct features** — features that only ever look at data available *at the moment of prediction*. This is the front line of the fight against data leakage.

Say you're predicting whether a customer churns, and you want a "total spend so far" feature. If you use `SUM(amount) OVER (PARTITION BY customer)` — no `ORDER BY` — every training row sees the customer's *entire* lifetime spend, including purchases that happened *after* the prediction point. That's leakage: the model learns from the future and looks brilliant in training, then collapses in production. The fix is a frame that only sees the past:

```sql
SUM(amount) OVER (
  PARTITION BY customer_id
  ORDER BY event_date
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
```

Now each row's feature reflects only rows at or before its own timestamp. `LAG` (last purchase amount), running counts, and trailing moving averages are all leakage-safe for the same reason — they're anchored to the current row's position in time.

The dedup pattern powers **most-recent-value-per-entity feature tables**: `ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY updated_at DESC)` then `WHERE rn = 1` gives you exactly one current feature row per user, product, or account — the standard shape for a serving-time feature lookup.

## Key takeaways

- Window functions compute across a set of related rows *without collapsing them* — unlike `GROUP BY`, the row count is preserved.
- `OVER()` has three optional parts: `PARTITION BY` (independent windows), `ORDER BY` (order within a window), and a frame clause (sliding range of rows).
- `ROW_NUMBER` always gives distinct numbers; `RANK` skips after ties; `DENSE_RANK` doesn't skip. `ROW_NUMBER() ... = 1` in a subquery is the go-to "latest/one row per group" and dedup pattern.
- `LAG`/`LEAD` read neighboring rows for period-over-period change; the first/last row yields `NULL` unless you pass a default.
- Adding `ORDER BY` to an aggregate makes it cumulative; the frame clause (`ROWS BETWEEN ... AND ...`) controls running totals and moving averages.
- For ML, an `ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` frame keeps features point-in-time-correct and leakage-free; the dedup pattern builds most-recent-value feature tables.

## Try it

Using the `sales` table:

1. For each region, rank reps by total `amount` using `DENSE_RANK`, highest first. (Hint: you may need a `GROUP BY` inside a subquery before ranking.)
2. Write a query returning, for each rep ordered by `sale_date`, the change in `amount` from their previous sale using `LAG`.
3. Add a `running_total` column that accumulates `amount` per region over `sale_date`, then explain in one sentence why that same expression *without* `ORDER BY` would leak future information if used as an ML feature.
