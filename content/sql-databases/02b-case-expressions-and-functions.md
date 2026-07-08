# 02b — CASE Expressions and Built-in Functions

Filtering picks *which rows* you see. This lesson is about *transforming the values* inside those rows: labelling them, reshaping them, cleaning them up. These are the everyday tools of feature engineering — the step where raw columns become the inputs a model or a report actually uses. We'll cover conditional logic with `CASE`, date and string functions, and how to handle missing values gracefully.

We'll use a small `sales` table:

| sale_id | customer | amount | sale_date  | channel |
|---------|----------|--------|------------|---------|
| 1       | Ada      | 42.00  | 2026-01-05 | web     |
| 2       | Grace    | 18.50  | 2026-01-20 | store   |
| 3       | Ada      | 99.99  | 2026-02-11 | web     |
| 4       | Linus    | 5.00   | 2026-02-28 | store   |
| 5       | Grace    | NULL   | 2026-03-02 | web     |

## CASE: conditional logic in SQL

`CASE` is SQL's version of if/else. It looks at a value or a condition and returns a result. There are two forms.

The **simple** form compares one expression against fixed values:

```sql
SELECT customer, channel,
  CASE channel
    WHEN 'web'   THEN 'online'
    WHEN 'store' THEN 'in person'
    ELSE 'unknown'
  END AS channel_label
FROM sales;
```

| customer | channel | channel_label |
|----------|---------|---------------|
| Ada      | web     | online        |
| Grace    | store   | in person     |
| Ada      | web     | online        |
| Linus    | store   | in person     |
| Grace    | web     | online        |

The **searched** form is more flexible: each `WHEN` is a full condition, so you can use ranges and comparisons. This is the form you'll reach for most often. Here we bucket each sale into a size tier:

```sql
SELECT sale_id, amount,
  CASE
    WHEN amount >= 50 THEN 'large'
    WHEN amount >= 20 THEN 'medium'
    ELSE 'small'
  END AS size_bucket
FROM sales;
```

| sale_id | amount | size_bucket |
|---------|--------|-------------|
| 1       | 42.00  | medium      |
| 2       | 18.50  | small       |
| 3       | 99.99  | large       |
| 4       | 5.00   | small       |
| 5       | NULL   | small       |

`CASE` checks conditions top to bottom and stops at the first match, so order matters — put the most specific conditions first. If nothing matches and there's no `ELSE`, the result is `NULL`. (Notice row 5: `NULL >= 50` is unknown, not true, so it falls through to `ELSE`.)

### CASE inside aggregates: conditional counts and pivots

`CASE` becomes powerful when you put it *inside* an aggregate function. Because `SUM` and `COUNT` ignore `NULL`, a `CASE` that returns `1` or `NULL` lets you count only the rows you care about:

```sql
SELECT
  COUNT(*) AS total_sales,
  SUM(CASE WHEN channel = 'web' THEN 1 ELSE 0 END) AS web_sales,
  SUM(CASE WHEN amount >= 50 THEN 1 ELSE 0 END) AS large_sales
FROM sales;
```

| total_sales | web_sales | large_sales |
|-------------|-----------|-------------|
| 5           | 3         | 1           |

This "SUM of a CASE" idiom is one of the most useful patterns in analytical SQL. Extended one step further, it lets you **pivot** rows into columns — turning a category column into one column per category:

```sql
SELECT customer,
  SUM(CASE WHEN channel = 'web'   THEN amount ELSE 0 END) AS web_total,
  SUM(CASE WHEN channel = 'store' THEN amount ELSE 0 END) AS store_total
FROM sales
GROUP BY customer;
```

| customer | web_total | store_total |
|----------|-----------|-------------|
| Ada      | 141.99    | 0.00        |
| Grace    | 0.00      | 18.50       |
| Linus    | 0.00      | 5.00        |

Each customer becomes one row, with channel spend spread across columns. This is exactly how you'd build wide feature tables for a model from long transactional data.

## Date and time functions

Dates are more than text — the database understands them, so you can pull them apart and do arithmetic.

`EXTRACT` pulls a single component (year, month, day, hour) out of a date. It's part of the SQL standard and works the same across Postgres and BigQuery:

```sql
SELECT sale_id,
  EXTRACT(YEAR  FROM sale_date) AS yr,
  EXTRACT(MONTH FROM sale_date) AS mo
FROM sales;
```

| sale_id | yr   | mo |
|---------|------|----|
| 1       | 2026 | 1  |
| 3       | 2026 | 2  |

`DATE_TRUNC` rounds a date down to the start of a unit — handy for grouping daily rows into months. **Here dialects diverge**, so read the target engine's docs:

```sql
-- Postgres: unit is a quoted string, and comes first
SELECT DATE_TRUNC('month', sale_date) AS month_start FROM sales;

-- BigQuery: unit is a keyword, and comes second
SELECT DATE_TRUNC(sale_date, MONTH) AS month_start FROM sales;
```

Both return `2026-01-01` for the January sales, `2026-02-01` for February, and so on. Grouping by the truncated date is the standard way to build a monthly time series.

For the current date, Postgres uses `CURRENT_DATE` (no parentheses) while BigQuery writes it as `CURRENT_DATE()`. Date arithmetic also differs: in Postgres you can subtract dates directly (`CURRENT_DATE - sale_date` gives an integer number of days), whereas BigQuery uses `DATE_DIFF(CURRENT_DATE(), sale_date, DAY)`. Computing a customer's "days since last purchase" is a common feature — just confirm the syntax for your engine. (Note: MySQL has no `DATE_TRUNC` at all; you'd use `DATE_FORMAT` instead.)

## String functions

Text columns almost always need cleaning before they're useful. The common functions:

| Function                     | What it does                          |
|------------------------------|---------------------------------------|
| `UPPER(x)` / `LOWER(x)`      | change case                           |
| `TRIM(x)`                    | strip leading/trailing whitespace     |
| `SUBSTRING(x FROM 1 FOR 3)`  | take part of a string                 |
| `REPLACE(x, 'a', 'b')`       | swap one substring for another        |
| `CONCAT(a, b)` or `a || b`   | join strings together                 |

```sql
SELECT
  UPPER(customer)               AS customer_upper,
  CONCAT(customer, ' (', channel, ')') AS labelled
FROM sales
WHERE sale_id = 1;
```

| customer_upper | labelled    |
|----------------|-------------|
| ADA            | Ada (web)   |

Normalising case with `LOWER` before comparing or grouping is a routine cleaning step — `'Web'`, `'web'`, and `'WEB'` should not be treated as three different channels. And recall `LIKE` from the previous lesson: `WHERE customer LIKE 'A%'` finds every name starting with A. The `||` concatenation operator is standard SQL and works in Postgres; BigQuery supports `CONCAT` (and `||`), while some engines only offer one — another dialect check.

## Handling NULLs: COALESCE and NULLIF

`NULL` propagates: any arithmetic with a `NULL` returns `NULL`, which quietly corrupts sums and averages. Two functions tame it.

`COALESCE` returns the first non-`NULL` argument — a clean way to supply a default:

```sql
SELECT sale_id, COALESCE(amount, 0) AS amount_filled
FROM sales;
```

| sale_id | amount_filled |
|---------|---------------|
| 4       | 5.00          |
| 5       | 0.00          |

Row 5's missing amount becomes `0`. Filling `NULL`s with a sentinel (0, a mean, "unknown") is one of the most common feature-engineering steps of all.

`NULLIF(a, b)` does the reverse — it returns `NULL` when `a` equals `b`. Its classic use is guarding against divide-by-zero: `total / NULLIF(count, 0)` yields `NULL` instead of an error when `count` is `0`.

## Deduplication: DISTINCT

Real data has duplicates. `DISTINCT` collapses repeated rows down to unique ones:

```sql
SELECT DISTINCT customer FROM sales;
```

| customer |
|----------|
| Ada      |
| Grace    |
| Linus    |

`DISTINCT` applies to *all* selected columns together, so `SELECT DISTINCT customer, channel` returns unique *combinations*, not unique customers. That makes `DISTINCT` good for "what distinct values exist" but a poor tool for "keep the latest row per customer" — it can't say which duplicate to keep. For that "keep-latest" deduplication you need `ROW_NUMBER()`, a window function we cover in lesson 04b.

## Key takeaways

- `CASE` is SQL's if/else: the **simple** form matches fixed values, the **searched** form takes full conditions. It checks top to bottom and stops at the first match.
- `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` gives conditional counts; extended with amounts it **pivots** rows into columns — a core feature-engineering trick.
- `EXTRACT` and `DATE_TRUNC` pull apart and group dates; syntax for `DATE_TRUNC`, `CURRENT_DATE`, and date subtraction differs between Postgres and BigQuery, so check your engine.
- String functions (`UPPER`/`LOWER`, `TRIM`, `SUBSTRING`, `REPLACE`, `CONCAT`/`||`) clean and reshape text; normalise case before grouping.
- `COALESCE` supplies defaults for `NULL`; `NULLIF` creates `NULL` (e.g. to avoid divide-by-zero).
- `DISTINCT` finds unique rows/combinations but can't keep-latest — use `ROW_NUMBER()` (lesson 04b) for that.

## Try it

Using the `sales` table:

1. Write a query that labels each sale as `'weekday'` or `'weekend'`. (Hint: `EXTRACT(DOW FROM sale_date)` in Postgres returns 0 for Sunday and 6 for Saturday.)
2. Using `SUM(CASE ...)`, produce one row per month showing the count of web sales and the count of store sales side by side.
3. Write a query that returns each customer's total spend, treating a `NULL` amount as `0`.
4. Write a query that returns the distinct list of channels, lowercased and trimmed of stray whitespace.
