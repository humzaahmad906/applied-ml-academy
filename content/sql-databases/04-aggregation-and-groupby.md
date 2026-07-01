# 04 — Aggregation and GROUP BY

So far every query has returned rows more or less as they're stored. But often the interesting question isn't "show me the rows" — it's "how many?", "what's the total?", or "what's the average?". Answering those means **aggregation**: collapsing many rows into a single summary value. This lesson covers aggregate functions, `GROUP BY`, and `HAVING`.

We keep using `customers` and `orders`.

## Aggregate functions

An **aggregate function** takes many values and returns one. The core five are:

| Function   | What it does                    |
|------------|---------------------------------|
| `COUNT()`  | counts rows                     |
| `SUM()`    | adds up numeric values          |
| `AVG()`    | averages numeric values         |
| `MIN()`    | smallest value                  |
| `MAX()`    | largest value                   |

To count all orders:

```sql
SELECT COUNT(*)
FROM orders;
```

With our three orders, this returns `3`. `COUNT(*)` counts rows regardless of their contents.

To find the total revenue across all orders:

```sql
SELECT SUM(total) AS total_revenue
FROM orders;
```

This adds 42.00 + 18.50 + 99.99 and returns `160.49`. The `AS total_revenue` gives the result column a readable name — otherwise it might be labeled something like `sum`.

You can compute several aggregates at once:

```sql
SELECT
  COUNT(*)   AS num_orders,
  SUM(total) AS total_revenue,
  AVG(total) AS avg_order,
  MIN(total) AS smallest,
  MAX(total) AS largest
FROM orders;
```

This returns a single row: 3 orders, 160.49 total, ~53.50 average, 18.50 smallest, 99.99 largest.

## COUNT and NULLs

There's a subtle but important difference:

- `COUNT(*)` counts every row.
- `COUNT(column)` counts only rows where that column is *not* NULL.

So if some customers had a missing email, `COUNT(email)` would be smaller than `COUNT(*)`. This is handy for counting "how many rows actually have a value here."

`COUNT(DISTINCT column)` counts *unique* non-null values:

```sql
SELECT COUNT(DISTINCT customer_id) AS customers_who_ordered
FROM orders;
```

Our orders belong to customers 1, 1, and 3 — so this returns `2`.

## Grouping rows with GROUP BY

A single total across the whole table is useful, but usually you want totals *per category*: revenue per customer, orders per city, sales per day. That's what `GROUP BY` does. It splits rows into groups that share a value, then computes the aggregate *for each group*.

Total spent by each customer:

```sql
SELECT customer_id, SUM(total) AS total_spent
FROM orders
GROUP BY customer_id;
```

The database gathers all rows with the same `customer_id`, then sums each group:

| customer_id | total_spent |
|-------------|-------------|
| 1           | 60.50       |
| 3           | 99.99       |

Customer 1's two orders (42.00 + 18.50) collapse into one row totaling 60.50. Customer 3 has a single order.

The golden rule of `GROUP BY`: every column in your `SELECT` must either be **grouped** (listed in `GROUP BY`) or **aggregated** (wrapped in a function). You can't select a raw column that isn't grouped, because there'd be many possible values per group and no single answer.

## Grouping combined with joins

Aggregation and joins together answer rich questions. "How much has each customer *by name* spent, including customers who spent nothing?"

```sql
SELECT c.name, COUNT(o.order_id) AS num_orders, SUM(o.total) AS total_spent
FROM customers AS c
LEFT JOIN orders AS o
  ON c.customer_id = o.customer_id
GROUP BY c.name;
```

The left join keeps every customer. The result:

| name         | num_orders | total_spent |
|--------------|------------|-------------|
| Ada Lovelace | 2          | 60.50       |
| Alan Turing  | 0          | NULL        |
| Grace Hopper | 1          | 99.99       |

Notice Alan Turing shows `num_orders = 0`. This is exactly why `COUNT(o.order_id)` is better than `COUNT(*)` here: `COUNT(*)` would count his single left-join row and wrongly report `1`, while counting the non-null `order_id` correctly gives `0`.

## Filtering groups with HAVING

You already know `WHERE` filters rows. But what if you want to filter *groups* — say, only customers who've spent more than 80? You can't use `WHERE` for that, because `WHERE` runs *before* grouping and doesn't know about the aggregate. For filtering after aggregation, use `HAVING`:

```sql
SELECT customer_id, SUM(total) AS total_spent
FROM orders
GROUP BY customer_id
HAVING SUM(total) > 80;
```

This returns only customer 3 (99.99), because customer 1's 60.50 doesn't clear the threshold.

The distinction is worth memorizing:

- **`WHERE`** filters individual rows *before* grouping.
- **`HAVING`** filters groups *after* aggregating.

You can use both in one query. This one considers only orders from 2026 (a row filter), then keeps only customers whose 2026 total exceeds 50 (a group filter):

```sql
SELECT customer_id, SUM(total) AS total_spent
FROM orders
WHERE order_date >= '2026-01-01'
GROUP BY customer_id
HAVING SUM(total) > 50;
```

## Putting the clauses in order

A query with all these pieces runs in this logical order:

1. `FROM` / `JOIN` — assemble the rows.
2. `WHERE` — filter individual rows.
3. `GROUP BY` — form groups.
4. `HAVING` — filter groups.
5. `SELECT` — compute the output columns.
6. `ORDER BY` — sort.
7. `LIMIT` — trim.

Remembering this order explains a lot of "why doesn't my query work" moments — for instance, why you can't reference a `SELECT` alias inside `WHERE`.

## Key takeaways

- Aggregate functions (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) collapse many rows into one value.
- `COUNT(*)` counts rows; `COUNT(column)` ignores NULLs; `COUNT(DISTINCT ...)` counts uniques.
- `GROUP BY` computes aggregates *per group*; non-aggregated selected columns must be grouped.
- Joins plus `GROUP BY` answer per-entity questions like revenue per customer.
- `WHERE` filters rows before grouping; `HAVING` filters groups after aggregating.

## Try it

Using `customers` and `orders`:

1. Write a query returning the number of orders and average order value for each customer_id.
2. Write a query that lists each city and how many customers live in it.
3. Write a query that finds which customers have placed more than one order, using `GROUP BY` and `HAVING`.
