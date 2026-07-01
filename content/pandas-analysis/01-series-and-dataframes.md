# 01 — Series and DataFrames

Welcome to pandas. If you already know a little Python — lists, dictionaries, loops — you're ready. Pandas is the library almost everyone reaches for when they need to work with tabular data: spreadsheets, CSV exports, database dumps, survey results. Before we load real files, let's meet the two objects you'll use in every single lesson: the **Series** and the **DataFrame**.

## The mental model

Think of a spreadsheet. A single column is a **Series**. The whole grid of columns side by side is a **DataFrame**. That's really it. Everything else in pandas is built on top of these two.

By convention, we import pandas as `pd`:

```python
import pandas as pd
```

You'll see `pd` everywhere. Just go with it.

## A Series is a labeled column

A Series is a one-dimensional sequence of values, each attached to a label called the **index**.

```python
temps = pd.Series([18, 21, 19, 23])
print(temps)
```

This prints something like:

```
0    18
1    21
2    19
3    23
dtype: int64
```

The left column (0, 1, 2, 3) is the index — pandas made it automatically. The right column holds your values. At the bottom, `dtype: int64` tells you these are 64-bit integers. Pandas gives every Series a single data type.

You can supply your own index labels, which is where Series start to feel powerful:

```python
temps = pd.Series([18, 21, 19, 23], index=["Mon", "Tue", "Wed", "Thu"])
print(temps["Tue"])
```

That prints `21`. The index behaves a lot like a dictionary key, but a Series also supports fast math across all its values at once:

```python
print(temps + 5)
```

Every value goes up by 5 — no loop required. This "do it to the whole column at once" behavior is called **vectorization**, and it's the heart of why pandas is both fast and pleasant to write.

## A DataFrame is a table of Series

A DataFrame is a collection of Series that share the same index — columns lined up in a grid. The most common way to build one by hand is from a dictionary, where each key becomes a column name.

We'll use one small dataset for the whole course: a handful of sales records from a coffee shop.

```python
sales = pd.DataFrame({
    "item": ["latte", "espresso", "muffin", "latte", "tea"],
    "size": ["M", "S", "L", "L", "M"],
    "price": [4.50, 3.00, 3.25, 5.00, 2.75],
    "qty": [2, 1, 3, 1, 4],
})
print(sales)
```

The output looks like a clean table:

```
       item size  price  qty
0     latte    M   4.50    2
1  espresso    S   3.00    1
2    muffin    L   3.25    3
3     latte    L   5.00    1
4       tea    M   2.75    4
```

Each row got an automatic integer index (0–4). Each column is a Series you can pull out by name:

```python
print(sales["price"])
```

That gives you the price column, on its own, as a Series. Notice the bracket syntax matches how you'd index a dictionary — column name in, column out.

## Peeking at structure

Even with a tiny table, you'll want habits that scale to big ones. Two attributes tell you the shape of things:

```python
print(sales.shape)     # (5, 4)  -> 5 rows, 4 columns
print(sales.columns)   # the column names
```

`shape` returns a tuple of `(rows, columns)`. `columns` lists the names. There's also `sales.index`, which shows the row labels.

## Creating a new column

Because columns are just Series, and Series do vectorized math, adding a computed column is a one-liner. Let's compute revenue per row:

```python
sales["revenue"] = sales["price"] * sales["qty"]
print(sales)
```

Pandas multiplies the two columns element by element and stores the result as a new column named `revenue`. The `muffin` row, for example, becomes `3.25 * 3 = 9.75`. No loop, no manual bookkeeping — you describe the operation on whole columns and pandas fills in every row.

## Why not just use lists and dicts?

You could store this data in plain Python. But the moment you want to filter rows, group by category, handle missing values, or compute summaries, you'd be writing loops by hand. Pandas gives you all of that as short, readable expressions — and it runs far faster because the heavy lifting happens in optimized C code under the hood.

## Key takeaways

- A **Series** is a one-dimensional labeled array — think of one column.
- A **DataFrame** is a set of Series sharing an index — think of a whole table.
- The **index** labels rows; **columns** have names. Both let you look things up by label, not just position.
- Operations are **vectorized**: you act on entire columns at once instead of looping.
- Build a DataFrame quickly from a dictionary of lists, and add computed columns with simple arithmetic.

## Try it

Recreate the `sales` DataFrame above. Then:

1. Print just the `item` column.
2. Add a column called `is_drink` that is `True` for every item except `"muffin"`. (Hint: `sales["item"] != "muffin"` returns a Series of True/False values.)
3. Print `sales.shape` before and after adding the column. Which number changed, and why?
