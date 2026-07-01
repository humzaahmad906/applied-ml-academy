# 02 — SELECT and Filtering

Now that you understand tables and keys, it's time to actually ask the database questions. In SQL, reading data is done with the `SELECT` statement. It is the single most-used command in all of SQL, and this lesson covers it end to end: choosing columns, filtering rows, sorting, and limiting results.

We'll keep using the `customers` and `orders` tables from the previous lesson.

## The simplest query

To read every column and every row of a table:

```sql
SELECT *
FROM customers;
```

The `*` means "all columns." `FROM customers` says which table to read. This returns the full customers table — all three rows, all four columns.

In real work you usually want only *some* columns. List them by name instead of using `*`:

```sql
SELECT name, city
FROM customers;
```

This returns just two columns: each customer's name and city. Selecting only the columns you need makes queries clearer and faster.

## Filtering rows with WHERE

Most of the time you don't want every row — you want the rows that match some condition. That's what `WHERE` is for.

```sql
SELECT name, city
FROM customers
WHERE city = 'London';
```

This returns only Ada Lovelace, because she's the one customer in London. The `WHERE` clause keeps rows where the condition is true and discards the rest.

Note that text values go in single quotes (`'London'`), while numbers do not.

## Comparison operators

`WHERE` supports the operators you'd expect:

| Operator | Meaning                  |
|----------|--------------------------|
| `=`      | equal to                 |
| `<>`     | not equal to             |
| `<`      | less than                |
| `>`      | greater than             |
| `<=`     | less than or equal to    |
| `>=`     | greater than or equal to |

For example, orders worth more than 40:

```sql
SELECT order_id, total
FROM orders
WHERE total > 40;
```

This returns orders 100 (42.00) and 102 (99.99). Order 101 (18.50) is filtered out.

## Combining conditions: AND, OR, NOT

You can chain conditions together:

```sql
SELECT order_id, total, order_date
FROM orders
WHERE total > 20
  AND order_date >= '2026-01-01';
```

`AND` requires *both* conditions to be true. `OR` requires *at least one*. `NOT` flips a condition. Use parentheses to make the grouping explicit when you mix them:

```sql
SELECT *
FROM orders
WHERE (total > 90 OR total < 20)
  AND order_date >= '2026-01-01';
```

## Handy operators: BETWEEN, IN, LIKE

A few special operators make common filters cleaner.

**BETWEEN** checks a range (inclusive on both ends):

```sql
SELECT order_id, total
FROM orders
WHERE total BETWEEN 18 AND 45;
```

This matches orders 100 and 101.

**IN** checks against a list of values:

```sql
SELECT name, city
FROM customers
WHERE city IN ('London', 'New York');
```

This is shorthand for `city = 'London' OR city = 'New York'`.

**LIKE** matches text patterns. The `%` symbol stands for "any sequence of characters":

```sql
SELECT name, email
FROM customers
WHERE email LIKE '%@example.com';
```

This finds every customer whose email ends in `@example.com`.

## Handling missing values: NULL

Sometimes a column has no value at all. This is represented by `NULL`, which means "unknown" — not zero, not an empty string. Because `NULL` is unknown, you cannot test it with `=`. Use `IS NULL` or `IS NOT NULL`:

```sql
SELECT name
FROM customers
WHERE email IS NULL;
```

This finds customers with no recorded email. Getting this wrong is a classic beginner mistake: `WHERE email = NULL` never matches anything.

## Sorting results with ORDER BY

Query results come back in no guaranteed order unless you ask for one. `ORDER BY` sorts them:

```sql
SELECT name, city
FROM customers
ORDER BY name;
```

By default sorting is ascending (A to Z, small to large). Add `DESC` for descending:

```sql
SELECT order_id, total
FROM orders
ORDER BY total DESC;
```

This lists the biggest orders first: 102 (99.99), then 100 (42.00), then 101 (18.50).

You can sort by multiple columns — the second breaks ties in the first:

```sql
SELECT customer_id, order_date, total
FROM orders
ORDER BY customer_id, order_date DESC;
```

## Limiting the number of rows

When a table is large, you often want just the top few rows. `LIMIT` caps the number returned:

```sql
SELECT order_id, total
FROM orders
ORDER BY total DESC
LIMIT 1;
```

This returns only the single largest order. Combining `ORDER BY` with `LIMIT` is the standard way to answer "top N" questions — top 10 customers, most recent 5 orders, and so on. Always pair `LIMIT` with `ORDER BY`, otherwise which rows you get is arbitrary.

## The logical order of a query

SQL statements are written in a fixed order — `SELECT`, then `FROM`, then `WHERE`, then `ORDER BY`, then `LIMIT`. It helps to remember that the database *reads* the table (`FROM`), *filters* it (`WHERE`), *chooses columns* (`SELECT`), *sorts* (`ORDER BY`), and finally *trims* (`LIMIT`).

## Key takeaways

- `SELECT` chooses columns; `FROM` names the table; `*` means all columns.
- `WHERE` filters rows using comparison operators and `AND` / `OR` / `NOT`.
- `BETWEEN`, `IN`, and `LIKE` express ranges, lists, and text patterns.
- `NULL` means unknown — test it with `IS NULL`, never `=`.
- `ORDER BY` sorts (add `DESC` to reverse); `LIMIT` caps the row count.
- Pair `ORDER BY` with `LIMIT` to answer "top N" questions.

## Try it

Using the `orders` table:

1. Write a query that returns every order placed in January 2026, sorted from newest to oldest.
2. Write a query that returns the two smallest orders by total.
3. Write a query using `IN` to find orders belonging to customer 1 or customer 3, showing `order_id`, `customer_id`, and `total`.
