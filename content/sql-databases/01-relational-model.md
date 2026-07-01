# 01 — The Relational Model

Before you write a single query, it helps to understand what a database is actually storing. Almost every business system you've used — an online store, a bank, a ticket booking site — keeps its data in a **relational database**. This lesson explains the mental model behind those databases so the rest of the course makes sense.

## Tables, rows, and columns

A relational database organizes data into **tables**. A table is a lot like a spreadsheet: it has named **columns** and any number of **rows**.

Imagine a small online shop. We might have a `customers` table:

| customer_id | name         | email                | city      |
|-------------|--------------|----------------------|-----------|
| 1           | Ada Lovelace | ada@example.com      | London    |
| 2           | Alan Turing  | alan@example.com     | Cambridge |
| 3           | Grace Hopper | grace@example.com    | New York  |

Each **column** has a name (`name`, `email`) and a **type** (text, number, date). Every value in a column must match that type — you can't put the word "hello" in a numeric column. Each **row** is one record: one complete customer.

We'll also need an `orders` table:

| order_id | customer_id | order_date | total  |
|----------|-------------|------------|--------|
| 100      | 1           | 2026-01-05 | 42.00  |
| 101      | 1           | 2026-01-09 | 18.50  |
| 102      | 3           | 2026-02-01 | 99.99  |

These two tables — `customers` and `orders` — are the sample schema we'll use throughout the whole course, so it's worth getting comfortable with them now.

## Primary keys

Notice the `customer_id` and `order_id` columns. Each gives every row a unique label. This is called a **primary key**: a column (or set of columns) whose value is unique for every row and never empty.

Why does this matter? Two customers could easily both be named "Alan Turing." If you needed to update one of them, "the row where name is Alan Turing" is ambiguous. But "the row where customer_id is 2" always points to exactly one person. A primary key is the reliable handle for a row.

Good primary keys are stable (they don't change) and meaningless (a plain number, not an email that a person might want to update later).

## Foreign keys and relationships

Look again at the `orders` table. It has a `customer_id` column too — but here it's not the order's own identity. It's a reference to *which customer placed the order*. Order 100 belongs to customer 1 (Ada). Orders 100 and 101 both belong to Ada.

A column that points to the primary key of another table is a **foreign key**. It is how tables connect. The foreign key `orders.customer_id` links each order back to a row in `customers`.

This link is the "relational" part of a relational database. Instead of stuffing a customer's name, email, and city into every single order, we store the customer once and reference them by id. If Ada changes her email, we update one row in `customers`, and every order still points to the correct, current information.

## Kinds of relationships

Relationships between tables come in a few shapes:

- **One-to-many:** One customer can have many orders, but each order belongs to exactly one customer. This is the most common relationship, and it's exactly what our two tables show.
- **One-to-one:** One row in a table matches at most one row in another. For example, a `customers` table and a separate `customer_settings` table, one settings row per customer.
- **Many-to-many:** Rows on both sides can match many rows on the other. An order can contain many products, and a product can appear in many orders. These are modeled with a third table in between (often called a *junction* or *join* table), which we'll cover in the schema design lesson.

## Why not just one giant table?

A beginner's instinct is to put everything in one big table: customer name, email, and every order all in one place. This causes real problems:

- **Repetition.** Ada's name and email get copied into every order row.
- **Update anomalies.** Change her email in one order row but forget the others, and now your data disagrees with itself.
- **Wasted space and errors.** More copies means more chances for typos and inconsistency.

Splitting data into focused tables and linking them with keys avoids all of this. Each fact is stored in exactly one place. This principle is the heart of good database design, and we'll formalize it later as *normalization*.

## The rules that keep data honest

The relational model also lets the database *enforce* rules so bad data can't sneak in:

- A **primary key constraint** stops two rows from sharing the same id.
- A **foreign key constraint** stops an order from referencing a customer that doesn't exist.
- A **NOT NULL constraint** requires a column to always have a value.

The database checks these automatically on every insert and update. That's a huge advantage over spreadsheets, where nothing stops you from typing nonsense.

## Key takeaways

- Data lives in **tables** made of **columns** (typed fields) and **rows** (records).
- A **primary key** uniquely identifies each row; it should be stable and unique.
- A **foreign key** references another table's primary key and creates a **relationship**.
- Relationships are **one-to-many**, **one-to-one**, or **many-to-many**.
- Splitting data into linked tables avoids repetition and keeps facts in one place.
- The database enforces **constraints** so invalid data is rejected automatically.

## Try it

On paper (no computer needed), design the tables for a small library:

1. Sketch a `books` table and a `members` table. Give each a sensible primary key and a few columns.
2. Now model borrowing: when a member borrows a book, where does that fact live? Create a `loans` table with foreign keys pointing to both `books` and `members`, plus a `borrowed_date`.
3. Ask yourself: is the relationship between members and books one-to-many or many-to-many? Explain why your `loans` table handles it correctly.
