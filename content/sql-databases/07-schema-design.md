# 07 — Schema Design Basics

You've learned to query data. Now comes the question that determines whether querying is pleasant or painful: how should the tables be *shaped* in the first place? **Schema design** is the practice of deciding what tables exist, what columns they hold, and how they relate. A good schema makes correct queries easy and bad data impossible. A poor one makes every query a fight. This lesson covers the essentials, using our familiar shop as the running example.

## Start from the things and the facts

Good schema design begins by identifying the **entities** — the distinct "things" your system cares about — and the **facts** about them. In our shop the entities are clearly *customers*, *orders*, and *products*. Each entity becomes a table. Each fact about it becomes a column: a customer has a name, an email, a city; an order has a date and a total.

The guiding instinct: **one kind of thing per table, one fact per column.** When you find yourself cramming two ideas into one column ("London, UK" as a single field) or mixing two kinds of things in one table (customer details repeated inside order rows), that's a signal to split.

## Keys, revisited

Every table needs a **primary key** — a column that uniquely and permanently identifies each row. A plain auto-incrementing integer (`customer_id`, `order_id`) is the common, safe choice: it's small, never changes, and carries no meaning a user might want to edit later. Avoid using something like an email address as a primary key; people change emails, and a changing key breaks every reference to it.

**Foreign keys** wire the tables together. `orders.customer_id` references `customers.customer_id`, expressing "this order belongs to this customer." Declaring it as a real foreign key constraint also lets the database *enforce* the link — it will refuse an order that points to a non-existent customer, so your data can't drift into nonsense.

## The problem normalization solves

Suppose you designed a single flat table with everything in it:

| order_id | customer_name | customer_email  | customer_city | product     | total |
|----------|---------------|-----------------|---------------|-------------|-------|
| 100      | Ada Lovelace  | ada@example.com | London        | Notebook    | 42.00 |
| 101      | Ada Lovelace  | ada@example.com | London        | Pen set     | 18.50 |

Ada's name, email, and city are **repeated** on every order she places. Three problems follow:

- **Update anomaly.** Ada changes her email. Now you must find and update *every* order row. Miss one, and your data contradicts itself.
- **Insertion anomaly.** You can't record a new customer until they place an order, because customer data only lives inside order rows.
- **Deletion anomaly.** Delete Ada's only order and you lose her email entirely.

**Normalization** is the process of splitting data to eliminate this repetition, so each fact is stored in exactly one place.

## Normalization in plain terms

The formal theory has "normal forms," but the practical heart of it is three ideas:

1. **No repeating groups; atomic columns.** Each column holds a single value, not a list. Don't store `"Notebook, Pen set"` in one field — that belongs in separate rows.
2. **Every non-key column depends on the *whole* key.** A fact should belong to the thing the table is about. A customer's city describes the *customer*, not the *order*, so it belongs in `customers`, not `orders`.
3. **No column depends on another non-key column.** If you stored both `city` and `country` in orders and country is really determined by the customer, you've got a fact hiding in the wrong place.

Apply these to the flat table and it naturally splits back into `customers` and `orders` — exactly the schema we've used all course. Customer facts live once in `customers`; each order references the customer by id. Now Ada's email changes in one row, full stop.

## Modeling a many-to-many relationship

Our schema so far handles one-to-many (one customer, many orders). But an order can contain *many products*, and a product can appear in *many orders* — a **many-to-many** relationship. You can't express that with a single foreign key on either side. The solution is a **junction table** (also called a join or link table) that sits between them:

```sql
CREATE TABLE order_items (
  order_id   INTEGER NOT NULL REFERENCES orders(order_id),
  product_id INTEGER NOT NULL REFERENCES products(product_id),
  quantity   INTEGER NOT NULL,
  PRIMARY KEY (order_id, product_id)
);
```

Each row of `order_items` links one order to one product, with a quantity. An order with three products has three rows here; a product sold in fifty orders appears in fifty rows. The primary key is the *combination* of `order_id` and `product_id` (a **composite key**), which prevents listing the same product twice on one order. This little table is the standard way every many-to-many relationship is modeled.

## Choosing column types and constraints

While shaping tables, pin down each column's **type** and **constraints**:

- Pick the narrowest type that fits: integers for counts and ids, a decimal type for money (not floating point, to avoid rounding surprises), dates for dates.
- Mark columns `NOT NULL` when a value is always required — an order must have a date.
- Add `UNIQUE` where duplicates make no sense — two customers shouldn't share an email even though it's not the primary key.

These constraints are the database doing quality control for you, rejecting bad data at the door rather than letting it rot inside.

## A note on when to bend the rules

Fully normalized schemas are the right default: they keep data correct and are easy to reason about. But sometimes, on very large read-heavy systems, teams deliberately *denormalize* — storing a bit of redundant data — to avoid expensive joins. That's a considered tradeoff made after measuring, not a starting point. As a beginner: **normalize first.** Introduce redundancy only when you have a real, measured performance reason, and understand the update anomalies you're taking back on.

## Key takeaways

- Design starts by finding **entities** (tables) and **facts** (columns): one thing per table, one fact per column.
- Use stable, meaningless **primary keys**; wire tables with **foreign keys** and let the database enforce them.
- Repeating data causes **update, insertion, and deletion anomalies**; **normalization** removes the repetition.
- Practical normalization: atomic columns, and every fact stored with the thing it actually describes.
- Model **many-to-many** relationships with a **junction table** and a composite key.
- Add types and `NOT NULL` / `UNIQUE` constraints so the database rejects bad data.
- Normalize first; denormalize only as a measured, deliberate tradeoff.

## Try it

On paper, design a schema for a simple blog:

1. Identify the entities. At minimum you'll have authors and posts. Give each table a primary key and sensible columns, and connect posts to authors with a foreign key.
2. Now add **tags**: a post can have many tags, and a tag can apply to many posts. Design the junction table that makes this work, including its composite primary key.
3. Point to one column in your design where a `NOT NULL` or `UNIQUE` constraint would prevent a realistic data-entry mistake, and explain which mistake.
