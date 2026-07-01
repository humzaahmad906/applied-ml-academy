# 03 — Selection and Filtering

Loading data is step one. Almost immediately you'll want a *subset* of it: certain columns, certain rows, or the rows that meet some condition. This lesson covers the three tools that handle nearly all of it — plain bracket selection, the `loc` and `iloc` accessors, and boolean masks.

We'll keep using the coffee-shop data, now with a revenue column already added:

```python
import pandas as pd

sales = pd.DataFrame({
    "item": ["latte", "espresso", "muffin", "latte", "tea"],
    "size": ["M", "S", "L", "L", "M"],
    "price": [4.50, 3.00, 3.25, 5.00, 2.75],
    "qty": [2, 1, 3, 1, 4],
})
sales["revenue"] = sales["price"] * sales["qty"]
```

## Selecting columns

One column with brackets gives you a Series:

```python
sales["item"]
```

A list of columns gives you a smaller DataFrame — note the double brackets:

```python
sales[["item", "revenue"]]
```

The inner `[...]` is a Python list of column names; the outer `[...]` is the selection. Beginners mix these up constantly: single brackets with a name = Series, double brackets with a list = DataFrame.

## loc and iloc: the two ways to grab rows

Pandas gives you two labeled accessors, and the distinction matters:

- **`loc`** selects by **label** — the index values and column names.
- **`iloc`** selects by **integer position** — 0, 1, 2, like list indexing.

With our default integer index they look similar, but they mean different things. The general form is `df.loc[rows, columns]`.

```python
sales.loc[2]                       # the row labeled 2, as a Series
sales.loc[2, "item"]               # single cell -> "muffin"
sales.loc[0:2, ["item", "price"]]  # rows 0,1,2 and two columns
```

An important quirk: with `loc`, `0:2` includes **both** endpoints (rows 0, 1, and 2), because it's slicing by label. With `iloc`, slicing follows normal Python rules and the end is **excluded**:

```python
sales.iloc[0:2]        # rows at positions 0 and 1 only
sales.iloc[0, 0]       # top-left cell -> "latte"
sales.iloc[-1]         # last row, just like list[-1]
```

Rule of thumb: use `loc` when you're thinking in terms of names and labels, `iloc` when you're thinking in terms of position (like "the first three rows").

## Boolean masks: the real workhorse

Filtering by condition is where pandas shines. Write a comparison on a column and you get back a Series of `True`/`False`, one per row — a **boolean mask**:

```python
mask = sales["price"] > 3.50
print(mask)
```

```
0     True
1    False
2    False
3     True
4    False
```

Feed that mask back into the DataFrame with brackets, and pandas keeps only the `True` rows:

```python
expensive = sales[sales["price"] > 3.50]
print(expensive)
```

You get the two rows where price exceeds 3.50 (the two lattes). This pattern — build a condition, use it to filter — is probably the single most common thing you'll do in pandas.

## Combining conditions

To combine masks, use `&` (and), `|` (or), and `~` (not). You **must** wrap each condition in parentheses, because these operators bind more tightly than comparisons:

```python
# Medium-sized drinks that sold more than one unit
sales[(sales["size"] == "M") & (sales["qty"] > 1)]
```

That returns the latte and the tea. Forgetting the parentheses is the number-one filtering bug — you'll get a confusing error, and the fix is almost always "add the parens."

A cleaner way to test membership in a set of values is `.isin`:

```python
sales[sales["item"].isin(["latte", "tea"])]
```

This keeps every row whose item is a latte or a tea, without chaining several `|` conditions.

## Filtering with loc, and modifying safely

You can combine a mask with `loc` to select rows *and* columns in one expression:

```python
sales.loc[sales["price"] > 3.50, ["item", "revenue"]]
```

`loc` is also the correct way to *assign* to a filtered subset. Say tea is on sale and we want to bump its quantity:

```python
sales.loc[sales["item"] == "tea", "qty"] = 5
```

Use `loc` for this rather than chained brackets like `sales[mask]["qty"] = 5`. The chained version may modify a temporary copy instead of the original and silently do nothing (pandas often warns you with a "SettingWithCopyWarning"). The single-`loc` form always writes to the real DataFrame.

## Putting it together

A typical exploration flows like this:

```python
# What did we sell for more than $8 in revenue, and what were they?
big = sales.loc[sales["revenue"] > 8, ["item", "size", "revenue"]]
print(big)
```

You built a condition, applied it, and narrowed to the columns you cared about — all in one readable line.

## Key takeaways

- `df["col"]` returns a Series; `df[["a", "b"]]` returns a DataFrame.
- `loc` selects by **label** (endpoints included in slices); `iloc` selects by **position** (end excluded).
- A comparison on a column produces a **boolean mask**; `df[mask]` keeps the `True` rows.
- Combine masks with `&`, `|`, `~` and **always parenthesize** each condition; use `.isin` for membership.
- Use `df.loc[mask, "col"] = value` to assign to a subset safely and avoid the copy-vs-view trap.

## Try it

Starting from the `sales` DataFrame:

1. Select just the rows where `size` is `"L"`. How many are there?
2. Use `iloc` to grab the first two rows and only the first three columns.
3. Write one expression that keeps rows where the item is a latte **and** the quantity is greater than 1. Then rewrite the item condition using `.isin`.
