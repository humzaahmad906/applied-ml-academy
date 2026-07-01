# 05 — Subqueries and CTEs

Sometimes one query isn't enough — the answer depends on the *result* of another query. "Which customers spent more than the average?" needs you to compute the average first, then compare against it. SQL lets you nest a query inside another (a **subquery**) or name intermediate results up front (a **CTE**). Both let you build complex answers out of simple pieces.

We continue with `customers` and `orders`.

## What a subquery is

A **subquery** is a `SELECT` statement wrapped in parentheses and used inside another query. The inner query runs, produces a result, and the outer query uses it. Subqueries show up in three main places.

## Subqueries that return a single value

The simplest case: the subquery returns one number, and you compare against it. "Which orders are above the average order total?"

```sql
SELECT order_id, total
FROM orders
WHERE total > (SELECT AVG(total) FROM orders);
```

The inner `(SELECT AVG(total) FROM orders)` computes the average (about 53.50). The outer query then keeps orders above it — order 102 (99.99). The inner query runs once and its single value slots into the comparison.

This is powerful because you can't do it in one flat query: `WHERE total > AVG(total)` is illegal, since an aggregate can't sit directly in a `WHERE`. The subquery gives the average its own moment to compute.

## Subqueries that return a list

A subquery can also return a *column* of values, used with `IN`. "Show the names of customers who have placed at least one order":

```sql
SELECT name
FROM customers
WHERE customer_id IN (SELECT customer_id FROM orders);
```

The inner query returns the set of customer ids that appear in orders (1 and 3). The outer query keeps customers whose id is in that set — Ada and Grace. Flip it to `NOT IN` to find customers with no orders (Alan).

## Correlated subqueries

A **correlated subquery** references the outer query and runs once *per outer row*. "Show each customer alongside their order count":

```sql
SELECT
  c.name,
  (SELECT COUNT(*)
   FROM orders o
   WHERE o.customer_id = c.customer_id) AS num_orders
FROM customers c;
```

For each customer row, the inner query counts that customer's orders. Note the inner query mentions `c.customer_id` from the outer query — that's the "correlation." The result gives Ada 2, Alan 0, Grace 1. This is often expressible as a join with `GROUP BY` too; both are valid, and joins are usually faster on large tables.

## Subqueries in FROM

You can treat a subquery's result as a temporary table to query further. "What's the average amount spent *per customer*?" First total by customer, then average those totals:

```sql
SELECT AVG(customer_total) AS avg_per_customer
FROM (
  SELECT customer_id, SUM(total) AS customer_total
  FROM orders
  GROUP BY customer_id
) AS per_customer;
```

The inner query produces one row per customer (60.50 and 99.99). The outer query averages those two numbers (about 80.25). A subquery in `FROM` must be given an alias — here, `per_customer`.

## The readability problem

Subqueries work, but deeply nested ones become hard to read. Reading them means starting from the innermost parentheses and working outward — the opposite of top-to-bottom. When a query has two or three levels of nesting, it turns into a puzzle. That's exactly the problem CTEs solve.

## Common Table Expressions (CTEs)

A **Common Table Expression** is a named, temporary result you define *before* your main query using the `WITH` keyword. It's the same idea as a subquery in `FROM`, but pulled out front and given a name, so the query reads top to bottom.

Here's the "average per customer" query rewritten as a CTE:

```sql
WITH per_customer AS (
  SELECT customer_id, SUM(total) AS customer_total
  FROM orders
  GROUP BY customer_id
)
SELECT AVG(customer_total) AS avg_per_customer
FROM per_customer;
```

Read it in order: first define `per_customer` (total per customer), then use it. Same result, far clearer. The name `per_customer` documents *what* the intermediate step represents.

## Chaining multiple CTEs

The real payoff is chaining several CTEs, each building on the last. Separate them with commas:

```sql
WITH per_customer AS (
  SELECT customer_id, SUM(total) AS customer_total
  FROM orders
  GROUP BY customer_id
),
big_spenders AS (
  SELECT customer_id
  FROM per_customer
  WHERE customer_total > 80
)
SELECT c.name, p.customer_total
FROM big_spenders b
JOIN per_customer p ON b.customer_id = p.customer_id
JOIN customers c    ON c.customer_id = b.customer_id;
```

Step one computes each customer's total. Step two picks the big spenders (over 80). The final query joins back to names. Each step is small and readable, and you could inspect any CTE on its own while debugging. This returns Grace Hopper with 99.99.

## Subquery or CTE — which to use?

They're often interchangeable. Guidelines:

- **Reach for a CTE** when the logic has multiple steps, when a step is reused, or when readability matters (which is almost always).
- **A simple subquery is fine** for a one-off single-value comparison like `WHERE total > (SELECT AVG(total) ...)` — pulling that into a CTE would be overkill.
- Modern databases optimize both similarly, so choose based on clarity, not a vague performance hunch.

## Key takeaways

- A **subquery** is a query nested inside another; it can return a single value, a list (with `IN`), or a table (in `FROM`).
- A **correlated subquery** runs once per outer row and references the outer query.
- **CTEs** (`WITH name AS (...)`) name intermediate results so queries read top to bottom.
- Chain multiple CTEs with commas to build multi-step logic in readable pieces.
- Prefer CTEs for multi-step or reused logic; a small subquery is fine for a quick comparison.

## Try it

Using `customers` and `orders`:

1. Write a subquery that returns the names of customers who placed an order worth more than 90.
2. Rewrite the "big spenders" example as a single query using only subqueries (no `WITH`), then compare which version you find easier to read.
3. Write a CTE that computes each customer's order count, then in the main query return only customers with zero orders.
