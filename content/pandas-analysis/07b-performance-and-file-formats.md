# 07b — Performance and File Formats

You can now load, clean, transform, and plot data. This lesson is about doing all of that *fast* and *storing it well*. Two things trip up beginners once their data grows past a few thousand rows: writing pandas code that secretly runs a slow Python loop, and saving everything as CSV out of habit. Both have easy fixes, and both matter more than you'd expect.

We'll use a slightly bigger DataFrame so timings are visible:

```python
import pandas as pd
import numpy as np

n = 1_000_000
df = pd.DataFrame({
    "city": np.random.choice(["Lahore", "Karachi", "Islamabad"], size=n),
    "price": np.random.uniform(1, 100, size=n),
    "qty": np.random.randint(1, 10, size=n),
})
```

## The performance ladder: vectorization vs apply vs iterrows

Say we want a `revenue` column, `price * qty`. There are three ways to write it, and they differ in speed by *orders of magnitude*.

**Bottom rung — `iterrows` (a real Python loop):**

```python
%%timeit
rev = []
for _, row in df.iterrows():
    rev.append(row["price"] * row["qty"])
# ~30 seconds for 1M rows
```

**Middle rung — `apply` (a Python loop in disguise):**

```python
%%timeit
df.apply(lambda row: row["price"] * row["qty"], axis=1)
# ~10 seconds
```

**Top rung — vectorization (the whole column at once):**

```python
%%timeit
df["price"] * df["qty"]
# ~5 milliseconds
```

That's not a typo: vectorization is thousands of times faster here. Why? Pandas columns are backed by NumPy arrays, and NumPy runs the multiplication in compiled C over the entire array in one shot — no per-row Python overhead. `iterrows` and `apply(axis=1)` both hand you one row at a time as a Python object, so the interpreter does a million round-trips. `apply` *looks* clean and functional, but with `axis=1` it is a loop wearing a nice coat.

The rule: **express operations on whole columns, not row by row.** Arithmetic, comparisons, and most string and datetime operations all vectorize:

```python
df["revenue"] = df["price"] * df["qty"]        # arithmetic
df["expensive"] = df["price"] > 50             # comparison
df["city_upper"] = df["city"].str.upper()      # vectorized strings via .str
```

Reach for `apply` only when there's genuinely no vectorized equivalent — and even then, prefer `apply` on a single column (which is faster) over `apply(axis=1)`. Treat `iterrows` as a last resort you'll almost never need.

A quick way to build the habit: before writing a loop, ask "is there a version of this that touches the whole column?" For conditional logic across columns, `np.where` and `np.select` vectorize what you might reach for `apply` to do:

```python
# instead of apply-ing a lambda that returns "high"/"low" per row:
df["tier"] = np.where(df["price"] > 50, "high", "low")
```

The payoff isn't just speed — vectorized code is usually shorter and reads more like the operation you're actually describing.

## Categorical dtype: cheap wins on repeated strings

Look at `city`: a million rows, but only three distinct values repeated over and over. Stored as normal strings ("object" dtype), pandas keeps a full copy of the text in every row. The `category` dtype instead stores the three unique labels once and replaces each row with a small integer code pointing at them.

```python
print(df["city"].memory_usage(deep=True))   # ~60 MB as strings

df["city"] = df["city"].astype("category")
print(df["city"].memory_usage(deep=True))   # ~1 MB as category
```

That's a large memory drop, and groupbys and comparisons on the column get faster too because the engine works with integers. Use `category` whenever a column has many rows but few distinct values — cities, product types, status flags, country codes.

**Downcasting numerics** is the same idea for numbers. Pandas defaults to 64-bit types, but `qty` only holds 1–9, which fits comfortably in an 8-bit integer:

```python
print(df["qty"].dtype)                        # int64 → 8 bytes/row
df["qty"] = pd.to_numeric(df["qty"], downcast="integer")
print(df["qty"].dtype)                        # int8 → 1 byte/row
```

`downcast="integer"` (or `"float"`) picks the smallest type that holds the data. On wide datasets, combining categoricals and downcasting can cut memory by more than half — which often decides whether a dataset fits in RAM at all. A good moment to do this is right after loading: run `df.info(memory_usage="deep")` to see where the bytes are going, then convert the obvious offenders. One caution — downcasting is lossy if the data later grows beyond the small type's range (an `int8` maxes out at 127), so only downcast columns whose range you understand.

## File formats: CSV vs Parquet

CSV is everywhere, human-readable, and opens in Excel — but as a storage format it's genuinely bad for data work. It's plain text, so every number is re-parsed from a string on load; it stores no types (was that column an int or a string? pandas has to guess every time); and it doesn't compress. **Parquet** fixes all three, and in 2026 it's the default format for moving data between tools.

Parquet is *columnar* (values of one column stored together, which compresses well and lets tools read just the columns you ask for), *typed* (dtypes are saved in the file — no re-guessing), and *compressed* by default. That columnar layout is why analytical tools love it: if a query only needs two of fifty columns, Parquet reads just those two off disk and skips the rest, whereas CSV has to scan every character of every row. Compare:

```python
df.to_csv("data.csv", index=False)
df.to_parquet("data.parquet")               # needs pyarrow installed

import os
print(os.path.getsize("data.csv"))          # ~40 MB
print(os.path.getsize("data.parquet"))      # ~8 MB
```

```python
%%timeit
pd.read_csv("data.csv")                      # ~600 ms
```

```python
%%timeit
pd.read_parquet("data.parquet")              # ~60 ms
```

Smaller on disk, roughly ten times faster to read, and — crucially — your `category` and `int8` dtypes survive the round trip, which they would silently lose through CSV. Use CSV only when a human or a spreadsheet needs to read the file. For anything you'll load back into code, use Parquet:

```python
df.to_parquet("data.parquet")
df2 = pd.read_parquet("data.parquet", columns=["city", "revenue"])  # read only what you need
```

## Honest aside: pandas isn't the only game

Pandas is the right default for learning and for datasets that fit in memory, but by 2026 it shares the stage with two tools worth knowing about:

- **Polars** is a newer DataFrame library with a pandas-like API, written in Rust. It's multi-core by default and supports *lazy evaluation* — you describe a chain of transforms and Polars optimizes the whole pipeline before running it. It's often several times faster than pandas on medium-to-large data and handles datasets larger than pandas comfortably will.
- **DuckDB** lets you run SQL directly against Parquet files or even a pandas DataFrame sitting in memory, without loading everything first. It's excellent for join-heavy, warehouse-style queries and can chew through files bigger than RAM.

```python
import duckdb
# SQL straight over a Parquet file — no full load
duckdb.sql("SELECT city, AVG(price) FROM 'data.parquet' GROUP BY city").df()
```

You don't have to choose one forever. A common 2026 workflow mixes all three: DuckDB to query and join big files down to a manageable size, Polars for heavy multi-step pipelines, and pandas for the final exploration and plotting where its huge ecosystem shines. Pandas itself is closing part of the gap: pandas 2.x can use a **PyArrow backend** (`pd.read_parquet(..., dtype_backend="pyarrow")`) for faster, more memory-efficient columns, especially strings.

The practical takeaway: learn pandas well first — its concepts transfer directly to Polars and to DataFrame thinking in general. When a dataset starts feeling slow or too big for memory, that's your signal to reach for Polars or DuckDB, not to give up.

## Key takeaways

- **Vectorize.** Operate on whole columns (`df["a"] * df["b"]`), not row by row. `apply(axis=1)` and `iterrows` are Python loops and can be thousands of times slower.
- Use the **`category`** dtype for columns with many rows but few distinct string values — big memory savings and faster grouping.
- **Downcast** numeric columns with `pd.to_numeric(..., downcast=...)` to shrink 64-bit defaults to the smallest type that fits.
- Prefer **Parquet** over CSV for data you'll reload: smaller, much faster, and it preserves dtypes. Keep CSV only for humans and spreadsheets.
- Pandas isn't alone: **Polars** (fast, lazy, multi-core) and **DuckDB** (SQL over files/DataFrames) handle larger data; pandas 2.x adds a **PyArrow backend**. Learn pandas first, reach for the others when size or speed demands it.

## Try it

Using the million-row `df` from the top of the lesson:

1. Time `df["price"] * df["qty"]` against the same calculation with `df.apply(..., axis=1)`. Roughly how many times faster is vectorization?
2. Check `df["city"].memory_usage(deep=True)` before and after `.astype("category")`. How much memory did you save?
3. Save `df` as both CSV and Parquet, compare the file sizes with `os.path.getsize`, then time loading each back. Confirm the Parquet version keeps your `category` dtype but the CSV version doesn't.
