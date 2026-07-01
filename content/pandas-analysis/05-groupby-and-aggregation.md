# 05 — GroupBy and Aggregation

You've loaded, filtered, and cleaned. Now the interesting question: *what does the data actually say?* Most answers take the form "for each category, what's the total / average / count?" — revenue per item, average price per size, number of sales per day. Pandas answers all of these with one pattern: **groupby**, built on the idea of **split-apply-combine**.

Here's our cleaned coffee-shop data:

```python
import pandas as pd

sales = pd.DataFrame({
    "item": ["latte", "espresso", "muffin", "latte", "tea", "espresso"],
    "size": ["M", "S", "L", "L", "M", "M"],
    "price": [4.50, 3.00, 3.25, 5.00, 2.75, 3.50],
    "qty": [2, 1, 3, 1, 4, 2],
})
sales["revenue"] = sales["price"] * sales["qty"]
```

## Split-apply-combine

The idea has three steps:

1. **Split** the rows into groups by some key (e.g. group all the latte rows together, all the espresso rows together).
2. **Apply** a function to each group (e.g. sum the revenue within each group).
3. **Combine** the per-group results back into one tidy result.

Pandas does all three when you chain `.groupby()` with an aggregation. You rarely think about the middle machinery — you just say what to group by and what to compute.

## Your first groupby

Total revenue for each item:

```python
sales.groupby("item")["revenue"].sum()
```

Read it left to right: group the rows by `item`, look at the `revenue` column within each group, and `sum` it. The result is a Series indexed by item:

```
item
espresso    10.00
latte       14.00
muffin       9.75
tea         11.00
```

The two espresso rows (3.00 and 7.00 revenue) got added into a single 10.00. The two latte rows combined into 14.00. That's split-apply-combine in one line.

Swap in a different function to ask a different question:

```python
sales.groupby("item")["price"].mean()    # average price per item
sales.groupby("item")["qty"].sum()       # total units per item
sales.groupby("item").size()             # number of rows (sales) per item
```

Note `.size()` counts rows in each group and doesn't need a column — it's how you answer "how many sales of each item?"

## Grouping by multiple keys

Pass a list to group by more than one column. This splits into every combination that actually appears:

```python
sales.groupby(["item", "size"])["revenue"].sum()
```

You get a result indexed by item *and* size (a "MultiIndex"):

```
item      size
espresso  M       7.00
          S       3.00
latte     L       5.00
          M       9.00
muffin    L       9.75
tea       M      11.00
```

This is how you drill down: not just "how much latte revenue," but "how much *large* latte revenue versus *medium*."

## Several aggregations at once with agg

Often you want more than one summary per group. `.agg()` takes a list of functions:

```python
sales.groupby("item")["revenue"].agg(["sum", "mean", "count"])
```

```
          sum  mean  count
item
espresso  10.0   5.0      2
latte     14.0   7.0      2
muffin     9.75  9.75     1
tea       11.0  11.0      1
```

You can even apply different functions to different columns by passing a dictionary:

```python
sales.groupby("item").agg({
    "revenue": "sum",
    "price": "mean",
    "qty": "max",
})
```

Each column gets exactly the summary you asked for — total revenue, average price, and the largest single quantity, all per item.

## Naming your outputs

For readable results, name the output columns explicitly with "named aggregation":

```python
sales.groupby("item").agg(
    total_revenue=("revenue", "sum"),
    avg_price=("price", "mean"),
    num_sales=("item", "count"),
)
```

Each argument is `new_name=(column, function)`. The result has clean, self-describing column names instead of the defaults — worth the extra typing when you'll share or reuse the output.

## Getting a flat table back

By default, the grouping keys become the index. To turn them back into ordinary columns, add `reset_index()`:

```python
summary = sales.groupby("item")["revenue"].sum().reset_index()
print(summary)
```

```
       item  revenue
0  espresso    10.00
1     latte    14.00
2    muffin     9.75
3       tea    11.00
```

Now `item` is a normal column again, which is handy for merging (next lesson) or plotting. You can also pass `as_index=False` directly to `groupby` for the same effect.

## Sorting the results

Group results are usually more useful sorted. Chain `sort_values`:

```python
sales.groupby("item")["revenue"].sum().sort_values(ascending=False)
```

Now your biggest earner sits at the top — exactly what you want when the question is "what sells best?"

## A quick word on transform

`groupby` usually *shrinks* your data (one row per group). Occasionally you want a per-group statistic attached back to *every original row* — for example, each row's revenue as a share of its item's total. That's what `transform` does; it returns a result the same length as the input:

```python
item_total = sales.groupby("item")["revenue"].transform("sum")
sales["share_of_item"] = sales["revenue"] / item_total
```

You don't need this often as a beginner, but knowing it exists saves you from awkward manual joins.

## Key takeaways

- **Split-apply-combine**: group rows by a key, apply a function per group, combine the results.
- `df.groupby("col")["value"].sum()` is the core pattern; swap `sum` for `mean`, `max`, `count`, etc.
- `.size()` counts rows per group; group by a list of columns to drill down into combinations.
- Use `.agg([...])` for multiple summaries and named aggregation (`name=(col, func)`) for clean output columns.
- Add `reset_index()` to turn grouping keys back into columns, and `sort_values` to rank the results.

## Try it

Using the `sales` DataFrame:

1. Compute total `qty` sold per `item`. Which item moved the most units?
2. Group by `size` and find the average `price` for each size.
3. Use named aggregation to produce, per `item`, a table with `total_revenue` (sum of revenue) and `num_sales` (count of rows). Sort it so the top earner is first.
