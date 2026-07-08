# 05b — Writing Data and Transactions

So far every query has *read* data. But an ML engineer's job is often to *write* it: populating a feature table, refreshing yesterday's aggregates, correcting a mislabeled row. Reading is only half of SQL. This lesson covers the four statements that change data — `INSERT`, `UPDATE`, `DELETE`, and the upsert — and the safety mechanism that keeps multi-step changes from corrupting your tables: **transactions**.

We keep using the `customers` and `orders` tables, and introduce a small `features` table later on.

## INSERT — adding rows

`INSERT` adds new rows. The basic form names the columns, then supplies values:

```sql
INSERT INTO customers (customer_id, name, city, email)
VALUES (4, 'Katherine Johnson', 'Hampton', 'katherine@example.com');
```

Listing the columns explicitly is a good habit — it keeps your statement correct even if someone later adds a column to the table. After this runs, `customers` has a fourth row.

You can insert several rows in one statement by separating the value lists with commas:

```sql
INSERT INTO customers (customer_id, name, city, email)
VALUES
  (5, 'Dorothy Vaughan', 'Hampton',  'dorothy@example.com'),
  (6, 'Mary Jackson',    'Hampton',  'mary@example.com');
```

One statement, two rows — faster than two separate `INSERT`s and treated as a single unit.

You can also insert the *result of a query*. This is how feature tables get populated: compute something with a `SELECT`, and pipe it straight into a table. Here we build a per-customer order summary:

```sql
INSERT INTO customer_features (customer_id, order_count, total_spent)
SELECT customer_id, COUNT(*), SUM(total)
FROM orders
GROUP BY customer_id;
```

There are no `VALUES` here — the `SELECT` supplies the rows directly. Its columns must line up, in order, with the column list.

## UPDATE — changing existing rows

`UPDATE` modifies rows that already exist. The `SET` clause says which columns change, and — critically — the `WHERE` clause says which rows:

```sql
UPDATE customers
SET city = 'Washington'
WHERE customer_id = 4;
```

This changes the city for customer 4 only. Before, `city` was `'Hampton'`; after, it is `'Washington'`. Every other row is untouched.

### The disaster of a missing WHERE

Here is the single most important warning in this lesson. **The `WHERE` clause in an `UPDATE` is optional to the database but essential to you.** If you omit it, the `UPDATE` applies to *every row in the table*:

```sql
-- DANGER: no WHERE clause
UPDATE customers
SET city = 'Washington';
```

This does not error. It silently sets *every* customer's city to `'Washington'`, overwriting all the real values. There is no undo button. Once the statement commits, the original cities are gone — the only way back is restoring from a backup, if one exists. The same trap applies to `DELETE`: a `DELETE FROM customers` with no `WHERE` empties the entire table.

Get in the habit of writing the `WHERE` clause *first*, then the `SET` or the `DELETE`. Better still, run a `SELECT` with the same `WHERE` first to see exactly which rows you're about to touch, then swap `SELECT` for `UPDATE`/`DELETE` once the row set looks right.

## DELETE vs TRUNCATE

`DELETE` removes rows that match a condition:

```sql
DELETE FROM orders
WHERE order_date < '2026-01-01';
```

This removes only the pre-2026 orders. Because it is row-by-row and condition-based, it is the tool you want almost every time you remove data.

`TRUNCATE` is different. It removes *all* rows from a table in one fast operation:

```sql
-- DANGER: removes every row in the table
TRUNCATE TABLE staging_features;
```

`TRUNCATE` takes no `WHERE` clause — it is all or nothing. It is faster than `DELETE` on a large table because it doesn't scan and remove rows individually, which makes it tempting for clearing out a staging table between pipeline runs. But treat it with respect: it wipes the entire table, and in many setups it cannot be rolled back the way a `DELETE` can. Reserve it for tables you genuinely intend to empty completely, and never point it at a table holding data you can't regenerate.

## UPSERT — insert or update, atomically

Pipelines re-run. A feature job that runs nightly will, sooner or later, run twice on the same day — after a retry, a backfill, or a crash. If it blindly `INSERT`s, the second run either errors on a duplicate key or creates duplicate rows. What you want is: insert the row if it's new, update it if it already exists. That's an **upsert**, and in Postgres it's `INSERT ... ON CONFLICT DO UPDATE`:

```sql
INSERT INTO customer_features (customer_id, order_count, total_spent)
VALUES (1, 5, 320.00)
ON CONFLICT (customer_id)
DO UPDATE SET
  order_count = EXCLUDED.order_count,
  total_spent = EXCLUDED.total_spent;
```

`ON CONFLICT (customer_id)` names the column with the unique constraint. If a row with that `customer_id` already exists, the insert would violate the constraint — so instead of failing, Postgres runs the `DO UPDATE`. The special `EXCLUDED` table holds the values you *tried* to insert, so `EXCLUDED.order_count` is the new `5`.

Say customer 1 already had `order_count = 2`. After this statement:

| customer_id | order_count | total_spent |
|-------------|-------------|-------------|
| 1           | 5           | 320.00      |

If customer 1 hadn't existed, the same statement would have simply inserted the row. Either way you end up with exactly one correct row. This makes the write **idempotent**: running it once or five times leaves the table in the same state. That property is what lets a pipeline retry safely.

Postgres also offers `ON CONFLICT DO NOTHING`, which skips the row on a conflict instead of updating it — useful when the first write wins and later duplicates should be ignored.

## Transactions — all or nothing

Some changes span multiple statements, and a half-finished change is worse than no change at all. The classic example is a money transfer: subtract from one account, add to another.

```sql
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

COMMIT;
```

`BEGIN` starts a **transaction**. The two `UPDATE`s happen, but nothing is made permanent until `COMMIT`. If the database crashes between the two statements, the whole transaction is discarded — you never end up in the nightmare state where 100 left account 1 but never arrived at account 2. The transfer happens completely or not at all.

If you detect a problem partway through, `ROLLBACK` throws away everything since `BEGIN`:

```sql
BEGIN;
DELETE FROM orders WHERE customer_id = 3;
-- realize that was the wrong customer
ROLLBACK;
```

After `ROLLBACK`, it's as if the `DELETE` never ran. Wrapping any multi-statement change in a transaction gives you this escape hatch, and it's why you should reach for `BEGIN` whenever a single logical change requires more than one statement.

## ACID in brief

Transactions give four guarantees, abbreviated **ACID**:

- **Atomicity** — a transaction is all-or-nothing. Every statement commits together, or none does. The money transfer above relies on this.
- **Consistency** — a transaction moves the database from one valid state to another, never violating constraints like unique keys or foreign keys along the way.
- **Isolation** — concurrent transactions don't step on each other. Each runs as if it were alone, so two pipelines writing at once don't interleave into garbage.
- **Durability** — once a transaction commits, it survives crashes and power loss. The data is written to disk, not just held in memory.

Isolation has degrees, called **isolation levels** (`READ COMMITTED`, `REPEATABLE READ`, `SERIALIZABLE`), which trade strictness against concurrency; Postgres defaults to `READ COMMITTED`, and you rarely need to change it early on.

## ML tie-in — writing feature tables safely

Everything here converges on one common ML task: refreshing a feature table. A robust refresh does the write inside a transaction, so a failure halfway through never leaves consumers reading a partially-updated table:

```sql
BEGIN;

INSERT INTO customer_features (customer_id, order_count, total_spent)
SELECT customer_id, COUNT(*), SUM(total)
FROM orders
GROUP BY customer_id
ON CONFLICT (customer_id)
DO UPDATE SET
  order_count = EXCLUDED.order_count,
  total_spent = EXCLUDED.total_spent;

COMMIT;
```

The transaction gives you atomicity — training code never sees a half-written table. The upsert gives you idempotency — if the job retries, it corrects the existing rows instead of erroring or duplicating them. Together they turn a fragile "hope it doesn't run twice" script into a pipeline step you can re-run without fear.

## Key takeaways

- `INSERT` adds rows — single, multi-row, or from a `SELECT` (the usual way to populate feature tables).
- `UPDATE ... SET ... WHERE` changes rows; **an `UPDATE` or `DELETE` with no `WHERE` hits every row and cannot be undone.**
- `DELETE` removes matching rows; `TRUNCATE` empties a whole table fast and is not reliably reversible — reserve it for tables you mean to wipe.
- `INSERT ... ON CONFLICT DO UPDATE` (upsert) makes writes idempotent, so pipelines can retry safely.
- Wrap multi-statement changes in `BEGIN ... COMMIT`; use `ROLLBACK` to abandon a transaction.
- **ACID** = Atomicity, Consistency, Isolation, Durability — the guarantees that make transactions trustworthy.

## Try it

Using `customers`, `orders`, and a `customer_features (customer_id, order_count, total_spent)` table:

1. Write an `INSERT ... SELECT` that fills `customer_features` with each customer's order count and total spent.
2. Write an upsert that sets customer 2's `order_count` to 3 and `total_spent` to 75.00, updating the row if it already exists.
3. Write a transaction that deletes all orders for customer 1 and then inserts a single replacement order, so the two changes commit together — and say what `ROLLBACK` would do if you ran it instead of `COMMIT`.

Sources: [PostgreSQL: Documentation — INSERT](https://www.postgresql.org/docs/current/sql-insert.html), [PostgreSQL Upsert: INSERT ON CONFLICT Guide (dbvis)](https://www.dbvis.com/thetable/postgresql-upsert-insert-on-conflict-guide/)
