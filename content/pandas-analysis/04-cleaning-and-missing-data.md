# 04 — Cleaning and Missing Data

Real data is messy. Numbers arrive as text, values go missing, rows get duplicated, and categories are spelled three different ways. Cleaning is where analysts spend a surprising share of their time — and doing it carefully is what separates trustworthy results from garbage. This lesson covers the core cleaning moves: fixing dtypes, handling missing values, and removing duplicates.

Here's a deliberately messy version of the coffee-shop data:

```python
import pandas as pd
import numpy as np

sales = pd.DataFrame({
    "item": ["latte", "Espresso", "muffin", "latte", "tea", "tea"],
    "size": ["M", "S", "L", "L", None, "M"],
    "price": ["4.50", "3.00", "3.25", "5.00", "2.75", "2.75"],
    "qty": [2, 1, 3, 1, np.nan, 4],
})
```

Notice the traps: `price` is stored as strings, `size` has a `None`, `qty` has a `NaN`, `Espresso` is capitalized differently from the other lowercase items, and the last two rows look like the same tea sale entered twice.

## Checking and fixing dtypes

Run `info()` first — it's your diagnostic:

```python
sales.info()
```

You'll see `price` listed as `object`, which means text. You can't do math on text prices, so convert it:

```python
sales["price"] = sales["price"].astype(float)
```

`astype` converts a column's type. It works when every value is convertible. When some values are junk (say `"n/a"` mixed in with numbers), `astype` will error; use `pd.to_numeric` with `errors="coerce"` instead, which turns anything unconvertible into `NaN`:

```python
sales["price"] = pd.to_numeric(sales["price"], errors="coerce")
```

The same idea applies to dates via `pd.to_datetime`. The lesson: don't trust the dtype pandas guessed — verify it and convert deliberately.

## What is NaN?

Missing values show up as `NaN` ("not a number") for numeric columns and `None`/`NaN` for object columns. `NaN` is contagious in a specific way: any arithmetic involving it produces `NaN`, and it is *never* equal to anything, even itself. So you can't test for it with `==`. Instead, pandas gives you dedicated tools.

To find missing values:

```python
print(sales.isna())          # True/False for every cell
print(sales.isna().sum())    # count of missing values per column
```

That second line is the one you'll actually use. It gives you a per-column tally:

```
item     0
size     1
price    0
qty      1
```

So `size` and `qty` each have one missing value.

## Handling missing values

There's no universally correct fix — it depends on what the data means. You have three main options.

**Drop rows** with any missing value:

```python
sales.dropna()
```

This removes rows 4 (missing size and qty). Dropping is honest but wasteful if you lose a lot of data. You can restrict it: `sales.dropna(subset=["qty"])` drops only rows missing `qty`.

**Fill with a value** using `fillna`:

```python
sales["qty"] = sales["qty"].fillna(0)          # assume a missing qty means 0
sales["size"] = sales["size"].fillna("Unknown")
```

**Fill with a statistic**, common for numeric columns, is to substitute the median or mean:

```python
median_qty = sales["qty"].median()
sales["qty"] = sales["qty"].fillna(median_qty)
```

Choose based on meaning. If a missing quantity truly means "none sold," `0` is right. If it means "we didn't record it," the median is often a safer placeholder than an invented `0` that would drag your averages down. State your assumption either way.

## Standardizing text

The `Espresso` vs `espresso` mismatch will break grouping and counting later — pandas treats them as two different items. String methods live under `.str`:

```python
sales["item"] = sales["item"].str.lower().str.strip()
```

`str.lower()` lowercases every value; `str.strip()` removes leading/trailing whitespace, another silent troublemaker. Now all four `espresso`/`latte`/`tea`/`muffin` values are consistent. You can chain `.str` methods just like this.

## Removing duplicates

Duplicate rows inflate counts and totals. Find and remove them:

```python
print(sales.duplicated())     # True for rows that repeat an earlier row
sales = sales.drop_duplicates()
```

`duplicated()` marks the *second and later* occurrences as `True`, keeping the first. If two rows are identical, one gets dropped. You can also dedupe based on specific columns — for example, treat rows as duplicates if the item and size match, regardless of other columns:

```python
sales = sales.drop_duplicates(subset=["item", "size"])
```

Be deliberate here: sometimes two identical-looking rows are genuinely two separate sales, and dropping them would be wrong. Look before you drop.

## Renaming for clarity

While cleaning, you'll often tidy column names too:

```python
sales = sales.rename(columns={"qty": "quantity"})
```

## A cleaning checklist

For any new dataset, walk through this in order:

1. `info()` — check dtypes; convert the ones that are wrong.
2. `isna().sum()` — find missing values; decide drop vs fill per column.
3. Standardize text with `.str.lower()` / `.str.strip()` where categories should match.
4. `duplicated()` / `drop_duplicates()` — remove genuine repeats.
5. Re-run `info()` to confirm everything is now the type and shape you expect.

## Key takeaways

- Verify dtypes with `info()`; convert with `astype` or `pd.to_numeric(..., errors="coerce")`.
- `NaN` marks missing data and can't be compared with `==`; use `isna()` / `isna().sum()` to find it.
- Handle missing values by dropping (`dropna`) or filling (`fillna`) — choose based on what the missingness *means*, and state your assumption.
- Standardize text with `.str.lower()` and `.str.strip()` so categories match.
- Remove genuine repeats with `drop_duplicates`, but confirm they're really duplicates first.

## Try it

Recreate the messy `sales` DataFrame above, then:

1. Convert `price` to a float and confirm with `info()`.
2. Report how many missing values each column has. Fill the missing `qty` with the column's median and the missing `size` with `"Unknown"`.
3. Lowercase the `item` column, then run `drop_duplicates`. How many rows remain, and which row got removed?
