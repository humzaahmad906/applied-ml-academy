# 06 — Indexing and Performance

Your queries return correct answers — but on a real database with millions of rows, *correct* isn't enough. A query that takes ten minutes is nearly useless. This lesson explains why queries get slow, what an **index** is, how it speeds things up, and how to read what the database is doing with `EXPLAIN`. You don't need to be a performance expert; you need the core intuition.

We keep using `customers` and `orders`, but imagine them scaled up to millions of rows.

## Why a query can be slow

Consider this familiar query:

```sql
SELECT *
FROM customers
WHERE email = 'ada@example.com';
```

With three rows, it's instant. But how does the database *find* the matching row? Without any help, it does a **full table scan**: it reads *every single row*, checks the email, and keeps the matches. Three rows — fine. Ten million rows — it reads all ten million, even though exactly one matches. That's the fundamental source of slowness: reading far more data than the answer requires.

## What an index is

An **index** is a separate data structure the database maintains to find rows quickly, without scanning everything. The classic analogy is the index at the back of a book. To find every mention of "London," you *could* read all 500 pages. Or you flip to the index, find "London — pages 12, 88, 240," and jump straight there. The book index is sorted alphabetically, so *finding* the term is fast, and it *points* to the exact locations.

A database index works the same way. If you create an index on `customers.email`, the database keeps a sorted structure of email values, each pointing to its row. Now the email lookup becomes: find the value in the sorted index (fast, because sorted data is searchable by halving the search space repeatedly), then jump directly to the row. Ten million rows, but only a handful of steps to find the one you want.

You create an index like this:

```sql
CREATE INDEX idx_customers_email
ON customers (email);
```

## What to index

Indexes help most on columns you frequently **search**, **join**, or **sort** by:

- Columns in `WHERE` conditions (like `email` above).
- Foreign keys used in joins (like `orders.customer_id`). Joining is essentially repeated lookups, so an index on the join column speeds up joins dramatically.
- Columns in `ORDER BY`, since an index is already sorted.

Primary keys are indexed automatically — that's part of what makes them fast, reliable handles.

## Why not index everything?

If indexes make lookups faster, why not index every column? Because indexes aren't free:

- **They take space.** Each index is an extra structure stored on disk.
- **They slow down writes.** Every `INSERT`, `UPDATE`, or `DELETE` must also update every affected index to keep it sorted. Ten indexes means ten extra bookkeeping steps per write.
- **Unused indexes are pure cost.** An index on a column you never query just wastes space and slows writes for no benefit.

So indexing is a tradeoff: faster reads in exchange for slower writes and more storage. Index the columns your real queries actually use, and no more.

## When an index won't help

Indexes aren't magic. A few cases where they don't help:

- **Low-selectivity columns.** An index on a `status` column that's only ever `'active'` or `'inactive'` barely helps — half the table matches either way, so a scan is about as good.
- **Functions on the column.** `WHERE UPPER(email) = 'ADA@EXAMPLE.COM'` can't use a plain index on `email`, because the index stores the raw values, not the uppercased ones.
- **Leading wildcards.** `WHERE email LIKE '%example.com'` can't use the index, because the index is sorted by the *start* of the value, and you haven't given it a start to search from. `LIKE 'ada%'` *can* use it.

## Seeing what the database does: EXPLAIN

How do you know whether your query uses an index or does a full scan? You ask the database to describe its plan, with `EXPLAIN`:

```sql
EXPLAIN
SELECT *
FROM customers
WHERE email = 'ada@example.com';
```

`EXPLAIN` doesn't run the query — it shows the **query plan**, the step-by-step strategy the database chose. The exact output varies by database, but you're looking for a few key signals in plain language:

- A **sequential scan** / **full table scan** means it's reading every row. On a big table with a filter, that's a warning sign.
- An **index scan** / **index seek** means it's using an index to jump to the rows. That's usually what you want for a selective filter.
- Plans also estimate **cost** and **row counts**, so you can spot a step that expects to process far more rows than it should.

The typical workflow: a query feels slow, you run `EXPLAIN` on it, you notice a full scan on a large table, you add an index on the filtered column, then you run `EXPLAIN` again and confirm it now uses an index scan. That loop — measure, change, measure again — is the whole game.

## A few extra habits that help

Beyond indexing, some simple habits keep queries fast:

- **Select only the columns you need.** `SELECT *` pulls every column, including big ones you may not use.
- **Filter early and specifically.** The more rows `WHERE` eliminates up front, the less work everything downstream does.
- **Be wary of functions in `WHERE`** if they block index use, as shown above.

## Key takeaways

- Without help, a filter does a **full table scan**, reading every row — slow on big tables.
- An **index** is a sorted lookup structure (like a book's index) that finds rows fast.
- Index columns used in `WHERE`, joins, and `ORDER BY`; primary keys are indexed automatically.
- Indexes cost storage and slow down writes, so index only what your queries use.
- Indexes can't help with functions on the column, leading wildcards, or low-selectivity columns.
- Use `EXPLAIN` to see the query plan and confirm whether an index is being used.

## Try it

No database required — reason it through:

1. For the query `SELECT * FROM orders WHERE customer_id = 1`, which column would you index, and why would it help on a million-row table?
2. Explain in your own words why adding ten indexes to a heavily-written table might make the system *slower* overall.
3. Given `WHERE LOWER(name) = 'ada lovelace'`, explain why a plain index on `name` won't be used, and suggest one way to make the lookup indexable.
