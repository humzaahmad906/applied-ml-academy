# 03 — Joins

In the relational model, related data lives in separate tables connected by keys. But when you ask a real question — "which customers placed which orders?" — you need data from *both* tables at once. **Joins** are how you combine rows from multiple tables into a single result. This is where relational databases really start to shine.

We continue with `customers` and `orders`. As a reminder, `orders.customer_id` is a foreign key pointing to `customers.customer_id`.

## The idea behind a join

A join stitches two tables together by matching rows. You tell the database *which columns should match* — usually a foreign key on one side and a primary key on the other. For each match, the database produces a combined row containing columns from both tables.

The matching condition goes in an `ON` clause:

```sql
SELECT customers.name, orders.order_id, orders.total
FROM customers
JOIN orders
  ON customers.customer_id = orders.customer_id;
```

For each order, the database finds the customer whose `customer_id` matches, and glues the rows together. The result:

| name         | order_id | total |
|--------------|----------|-------|
| Ada Lovelace | 100      | 42.00 |
| Ada Lovelace | 101      | 18.50 |
| Grace Hopper | 102      | 99.99 |

Notice Ada appears twice — once per order — and Alan Turing doesn't appear at all, because he has no orders. That behavior is the key to understanding join *types*.

## Table aliases

Writing `customers.name` and `orders.order_id` everywhere gets tedious. Give tables short aliases with `AS` (or just a space):

```sql
SELECT c.name, o.order_id, o.total
FROM customers AS c
JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

This is exactly the same query, just easier to read. Most SQL you'll encounter uses aliases.

## INNER JOIN: only matching rows

The plain `JOIN` above is an **inner join** (you can write `INNER JOIN` to be explicit). An inner join returns only rows that have a match on *both* sides. Because Alan Turing has no matching order, he's excluded. Because every order happens to have a valid customer, every order appears.

Use an inner join when you only care about records that are connected — "customers who have actually ordered."

## LEFT JOIN: keep everything on the left

Often you *do* want the unmatched rows. "List every customer and their orders, including customers who haven't ordered yet." That's a **left join**:

```sql
SELECT c.name, o.order_id, o.total
FROM customers AS c
LEFT JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

A left join keeps *every* row from the left table (`customers`), and attaches matching rows from the right table (`orders`) where they exist. Where there's no match, the right-side columns come back as `NULL`:

| name         | order_id | total |
|--------------|----------|-------|
| Ada Lovelace | 100      | 42.00 |
| Ada Lovelace | 101      | 18.50 |
| Alan Turing  | NULL     | NULL  |
| Grace Hopper | 102      | 99.99 |

Now Alan Turing appears, with `NULL` in the order columns to signal "no orders." This is extremely useful. To find customers with *no* orders at all, filter on the NULLs:

```sql
SELECT c.name
FROM customers AS c
LEFT JOIN orders AS o
  ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

This returns just Alan Turing.

## RIGHT JOIN: keep everything on the right

A **right join** is the mirror image: it keeps every row from the *right* table and attaches matches from the left. This query keeps all orders even if somehow an order had no matching customer:

```sql
SELECT c.name, o.order_id, o.total
FROM customers AS c
RIGHT JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

In practice, right joins are rarer, because you can always rewrite a right join as a left join by swapping the table order. Many people pick one direction (usually left) and stick with it for consistency.

## FULL JOIN: keep everything on both sides

A **full join** (or `FULL OUTER JOIN`) keeps *all* rows from both tables. Where a row matches, it's combined; where it doesn't, the missing side is filled with `NULL`.

```sql
SELECT c.name, o.order_id, o.total
FROM customers AS c
FULL JOIN orders AS o
  ON c.customer_id = o.customer_id;
```

With our data this shows all customers (including Alan with NULL order) *and* all orders. Full joins are useful for reconciliation — finding mismatches on either side, such as customers with no orders and orders with no customer.

## A picture in words

It helps to picture the two tables as overlapping circles:

- **INNER JOIN** = only the overlap (rows that match on both sides).
- **LEFT JOIN** = the whole left circle, plus the overlap.
- **RIGHT JOIN** = the whole right circle, plus the overlap.
- **FULL JOIN** = both circles entirely.

The overlap is always the matching rows. The difference is how much of the *non*-matching rows you keep.

## Joining more than two tables

Real queries often join three or more tables. You just chain `JOIN` clauses. If we had an `order_items` table, we could go customers → orders → items in one query, each `JOIN` adding its own `ON` condition. The same matching logic applies at every step.

## Key takeaways

- A **join** combines rows from multiple tables by matching columns in an `ON` clause.
- **INNER JOIN** returns only rows that match on both sides.
- **LEFT JOIN** keeps all left-table rows; unmatched right columns become `NULL`.
- **RIGHT JOIN** is the mirror of left; **FULL JOIN** keeps everything from both.
- Filtering `WHERE ... IS NULL` after a left join finds rows with *no* match.
- Use short **table aliases** to keep join queries readable.

## Try it

Using `customers` and `orders`:

1. Write an inner join that lists each customer's name alongside every order total they placed.
2. Write a left join that lists all customers and the number of order columns — then identify which customer has no orders.
3. Explain in one sentence what would change in query 1 if you switched `INNER JOIN` to `LEFT JOIN`.
