# 07 — Plotting and EDA

You can now load, clean, filter, group, and reshape data. The final beginner skill is putting it all together into **exploratory data analysis** (EDA): the open-ended process of getting to know a dataset before you make claims about it or feed it to a model. Plots are central to EDA because your eyes catch patterns — and problems — that a table of numbers hides.

Pandas has plotting built in. Under the hood it uses Matplotlib, so you'll usually import that too:

```python
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.DataFrame({
    "item": ["latte", "espresso", "muffin", "latte", "tea", "espresso"],
    "size": ["M", "S", "L", "L", "M", "M"],
    "price": [4.50, 3.00, 3.25, 5.00, 2.75, 3.50],
    "qty": [2, 1, 3, 1, 4, 2],
})
sales["revenue"] = sales["price"] * sales["qty"]
```

## Plotting straight from a DataFrame

Any Series or DataFrame has a `.plot()` method. The `kind` argument picks the chart type. To call `plt.show()` displays the figure (in a notebook, plots often appear automatically).

**Bar charts** compare categories — perfect after a groupby:

```python
revenue_by_item = sales.groupby("item")["revenue"].sum()
revenue_by_item.plot(kind="bar")
plt.title("Revenue by item")
plt.ylabel("Revenue ($)")
plt.show()
```

You'll see four bars, one per item, instantly showing which sells best. Grouping produces the numbers; the bar chart makes the ranking obvious at a glance.

**Histograms** show the distribution of a single numeric column — how values spread out:

```python
sales["price"].plot(kind="hist", bins=5)
plt.title("Price distribution")
plt.show()
```

The histogram buckets prices into ranges and shows how many fall in each. This is how you spot skew, gaps, or a suspicious spike of identical values.

**Scatter plots** reveal the relationship between two numeric columns:

```python
sales.plot(kind="scatter", x="price", y="qty")
plt.title("Price vs quantity")
plt.show()
```

Each point is one row. If cheaper items tend to sell in larger quantities, you'd see the points trend down-right. Scatter plots are your first look at whether two variables move together.

**Line plots** (the default `kind`) suit ordered data like time series — revenue per day, for instance. With unordered categorical data, prefer bars.

## A repeatable EDA routine

EDA isn't a fixed recipe, but a good default sequence for any new dataset looks like this:

**1. Understand the shape and types.** Start where lesson 2 left off:

```python
print(sales.shape)
sales.info()
```

**2. Summarize the numbers.**

```python
print(sales.describe())
```

Look for impossible values, huge ranges, or a `min`/`max` that doesn't make sense.

**3. Look at each column on its own (univariate).** For numeric columns, a histogram. For categorical columns, counts:

```python
print(sales["item"].value_counts())
sales["item"].value_counts().plot(kind="bar")
plt.show()
```

**4. Look at relationships (bivariate).** How do two columns relate? Scatter plots for two numeric columns; grouped bars for a category versus a number:

```python
sales.groupby("size")["revenue"].mean().plot(kind="bar")
plt.title("Average revenue by size")
plt.show()
```

**5. Check correlations** among numeric columns to quantify what the scatter plots suggested:

```python
print(sales[["price", "qty", "revenue"]].corr())
```

`corr()` returns a table of correlation coefficients between -1 and 1. Values near +1 mean two columns rise together; near -1 means one rises as the other falls; near 0 means little linear relationship. It's a fast numeric companion to your scatter plots — though remember correlation isn't causation.

## What you're actually looking for

EDA has a purpose beyond making pretty pictures. You're hunting for:

- **Data quality problems** — the outlier that's really a typo, the column that's secretly all one value, the missing chunk you didn't notice with `info()`.
- **Distributions** — is a variable roughly balanced, or heavily skewed? Heavy skew changes how you'd model or summarize it.
- **Relationships** — which variables move together? These are the ones that'll matter if you build a model later.
- **Surprises** — anything that contradicts what you expected. Those are often the most valuable findings, because they force you to understand the data (or fix it) before you trust any conclusion.

The habit to build is *look before you leap*. Every wrong analysis and every misleading model traces back to something someone could have caught with ten minutes of plots and `describe()` at the start.

## Saving a figure

To keep a plot, save it before `show()`:

```python
revenue_by_item.plot(kind="bar")
plt.savefig("revenue_by_item.png", dpi=150, bbox_inches="tight")
```

`bbox_inches="tight"` trims excess whitespace; `dpi` controls resolution.

## Key takeaways

- Pandas plots directly with `.plot(kind=...)`: **bar** for categories, **hist** for a single variable's distribution, **scatter** for two-variable relationships, **line** for ordered/time data.
- Grouping produces the numbers; plots make the pattern obvious — pair `groupby` with a bar chart constantly.
- A solid EDA routine: shape and dtypes → `describe` → univariate views → bivariate views → `corr` on numeric columns.
- EDA exists to find data-quality problems, distributions, relationships, and surprises **before** you draw conclusions or build a model.
- Correlation measures linear association from -1 to 1, but never mistake it for causation.

## Try it

Using the `sales` DataFrame:

1. Make a bar chart of total `qty` sold per `item`. Which item's bar is tallest?
2. Plot a histogram of `price` with 4 bins. Are prices spread out or clustered?
3. Compute `corr()` on `price`, `qty`, and `revenue`. Which pair is most strongly correlated, and does that make intuitive sense given how `revenue` is defined?
