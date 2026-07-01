# 06 — Merging and Reshaping

Data rarely arrives in one perfect table. You'll have sales in one file and product details in another, and you'll need to stitch them together. Other times a single table is shaped wrong for the question you're asking — too wide, or too long — and you need to reshape it. This lesson covers combining tables (`merge`, `concat`) and reshaping them (`pivot`, `melt`).

Our two starting tables — sales, and a small lookup of item categories:

```python
import pandas as pd

sales = pd.DataFrame({
    "item": ["latte", "espresso", "muffin", "tea"],
    "qty": [2, 1, 3, 4],
    "revenue": [9.00, 3.00, 9.75, 11.00],
})

items = pd.DataFrame({
    "item": ["latte", "espresso", "muffin", "cookie"],
    "category": ["drink", "drink", "food", "food"],
})
```

## Merging: joining tables on a key

`merge` combines two DataFrames by matching values in a shared column — the **key**. Here both tables have an `item` column, so we join on it:

```python
combined = sales.merge(items, on="item")
print(combined)
```

```
       item  qty  revenue category
0     latte    2     9.00    drink
1  espresso    1     3.00    drink
2    muffin    3     9.75     food
```

Each sales row picked up its `category` from the `items` table. Notice `tea` disappeared (it's not in `items`) and `cookie` never appeared (no sales). That's because the default is an **inner join**: keep only keys present in *both* tables.

## The four join types

The `how` argument controls what happens to non-matching rows:

- `how="inner"` (default) — keep only keys in **both** tables.
- `how="left"` — keep **all** rows from the left table; fill missing right-side values with `NaN`.
- `how="right"` — keep all rows from the right table.
- `how="outer"` — keep **every** key from either table.

A left join is the most common in practice, because you usually have a main table and want to enrich it without losing any of its rows:

```python
sales.merge(items, on="item", how="left")
```

Now `tea` stays, but its `category` is `NaN` because `items` had no match for it. That `NaN` is a useful signal — it tells you your lookup table is incomplete.

If the key columns have different names in each table, use `left_on` and `right_on`:

```python
sales.merge(items, left_on="item", right_on="product_name", how="left")
```

## Concat: stacking tables

`merge` joins tables *side by side* on a key. `concat` **stacks** them — useful when you have the same columns split across files (say, January sales and February sales) and want one combined table:

```python
jan = pd.DataFrame({"item": ["latte", "tea"], "qty": [2, 4]})
feb = pd.DataFrame({"item": ["muffin", "latte"], "qty": [3, 1]})

all_sales = pd.concat([jan, feb], ignore_index=True)
print(all_sales)
```

```
     item  qty
0   latte    2
1     tea    4
2  muffin    3
3   latte    1
```

`ignore_index=True` renumbers the rows 0–3; without it you'd keep the original indices (0, 1, 0, 1), which is usually confusing. Use `concat` to add rows; use `merge` to add columns.

## Wide vs long, and why it matters

The same data can be stored in two shapes. **Long** format has one measurement per row:

```
item     month  qty
latte    Jan    2
latte    Feb    1
tea      Jan    4
```

**Wide** format spreads a category across columns:

```
item    Jan  Feb
latte    2    1
tea      4    0
```

Neither is "correct" — but different tasks want different shapes. Humans read wide tables more easily; most plotting and grouping code prefers long. `pivot` and `melt` convert between them.

## Pivot: long to wide

`pivot` reshapes long data into a wide grid. You tell it what becomes the row index, what becomes the columns, and what fills the cells:

```python
long = pd.DataFrame({
    "item": ["latte", "latte", "tea", "tea"],
    "month": ["Jan", "Feb", "Jan", "Feb"],
    "qty": [2, 1, 4, 3],
})

wide = long.pivot(index="item", columns="month", values="qty")
print(wide)
```

```
month  Feb  Jan
item
latte    1    2
tea      3    4
```

If several rows map to the same cell and need combining (say you have multiple sales per item per month), use `pivot_table`, which aggregates — it takes an `aggfunc` like `"sum"` or `"mean"`:

```python
long.pivot_table(index="item", columns="month", values="qty", aggfunc="sum")
```

## Melt: wide to long

`melt` is the inverse — it collapses columns back into rows. Point it at the columns to keep (`id_vars`) and it turns the rest into two columns, a variable name and its value:

```python
back_to_long = wide.reset_index().melt(
    id_vars="item",
    var_name="month",
    value_name="qty",
)
print(back_to_long)
```

```
    item month  qty
0  latte   Feb    1
1    tea   Feb    3
2  latte   Jan    2
3    tea   Jan    4
```

You've returned to the tidy long shape — the format most pandas operations and plotting libraries expect. The `reset_index()` first moves `item` out of the index so `melt` can treat it as a normal column to keep.

## Key takeaways

- `merge` joins two tables **side by side** on a shared key column; `concat` **stacks** tables with the same columns.
- Join type matters: `inner` keeps only matches, `left` keeps all left rows, `outer` keeps everything. `NaN`s after a left join reveal missing lookups.
- **Long** format (one measurement per row) suits grouping and plotting; **wide** format is easier to read.
- `pivot` (or `pivot_table` when you need to aggregate) turns long into wide; `melt` turns wide back into long.

## Try it

Using `sales` and `items` from the top of the lesson:

1. Left-join `items` onto `sales`. Which item ends up with a `NaN` category, and why?
2. Create `jan` and `feb` sales tables and `concat` them into one, with clean 0-based row numbers.
3. Take the `long` table, `pivot` it to wide (items as rows, months as columns), then `melt` it back to long. Do you recover the original?
