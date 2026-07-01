# 02 — Loading and Inspecting Data

Building DataFrames by hand is fine for learning, but real work starts with data that already exists — usually a file. In this lesson you'll load data and then run the small set of inspection commands that experienced analysts type almost reflexively before doing anything else.

## Loading a CSV

CSV (comma-separated values) is the most common data format you'll meet. Pandas reads it in one line:

```python
import pandas as pd

sales = pd.read_csv("sales.csv")
```

Suppose `sales.csv` contains our coffee-shop records:

```
item,size,price,qty
latte,M,4.50,2
espresso,S,3.00,1
muffin,L,3.25,3
latte,L,5.00,1
tea,M,2.75,4
```

After `read_csv`, `sales` is a DataFrame exactly like the one we built by hand last time — pandas figured out the column names from the header row and guessed a sensible data type for each column.

A few options you'll reach for often:

```python
pd.read_csv("sales.csv", sep=";")            # semicolon-separated file
pd.read_csv("sales.csv", index_col="item")   # use a column as the row index
pd.read_csv("sales.csv", nrows=100)          # read only the first 100 rows
```

## Loading JSON

Data from web APIs often arrives as JSON. If it's a list of records, `read_json` handles it directly:

```python
sales = pd.read_json("sales.json")
```

For a file like `[{"item": "latte", "price": 4.5}, {"item": "tea", "price": 2.75}]`, each object becomes a row and each key becomes a column. JSON can nest arbitrarily; when it does, you may need to flatten it, but for flat records this just works.

## First look: head and tail

Never assume a file loaded correctly — look at it. `head` shows the first rows (5 by default):

```python
print(sales.head())
print(sales.head(3))   # first 3 rows
print(sales.tail(2))   # last 2 rows
```

`head` is your sanity check: Are the column names right? Did the header get parsed, or is it sitting in row 0 as data? Are numbers actually numbers and not text? A three-second glance here saves hours later.

## The structural summary: info

`info()` is the single most useful command for understanding what you just loaded:

```python
sales.info()
```

Output looks roughly like:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   item    5 non-null      object
 1   size    5 non-null      object
 2   price   5 non-null      float64
 3   qty     5 non-null      int64
dtypes: float64(1), int64(1), object(2)
```

Read this carefully — it tells you three critical things:

- **How many rows** (5 entries).
- **The dtype of each column.** `object` almost always means text (strings). `float64` and `int64` are numbers. If a column you expected to be numeric shows up as `object`, something dirty is in it — a stray letter, a currency symbol — and math will fail until you clean it.
- **Non-Null Count.** If a column shows fewer non-null values than there are rows, it has missing data. Here everything is `5 non-null`, so nothing's missing yet.

## The numeric summary: describe

`describe()` computes summary statistics for the numeric columns:

```python
print(sales.describe())
```

```
          price       qty
count  5.000000  5.000000
mean   3.700000  2.200000
std    0.912ず...  1.303840
min    2.750000  1.000000
25%    3.000000  1.000000
50%    3.250000  2.000000
75%    4.500000  3.000000
max    5.000000  4.000000
```

Each column gets its count, mean, standard deviation, min, max, and the 25th/50th/75th percentiles. The 50% row is the median. This is a fast way to spot problems: a `min` of -1 on a price column, or a `max` in the millions where you expected tens, jumps right out.

To include text columns too, pass `include="all"` and you'll get extra rows like `unique` (number of distinct values) and `top` (most frequent).

## Quick counts and uniqueness

Two more inspection habits worth building. To count how often each value appears in a column:

```python
print(sales["item"].value_counts())
```

```
latte       2
espresso    1
muffin      1
tea         1
```

And to see the distinct values:

```python
print(sales["item"].unique())     # array of unique items
print(sales["item"].nunique())    # how many distinct items (4)
```

## A sensible loading routine

Put these together and you have a reliable first-contact routine for any new dataset:

```python
df = pd.read_csv("somefile.csv")
print(df.shape)          # how big?
df.info()                # dtypes and missing values
print(df.head())         # does it look right?
print(df.describe())     # any wild numbers?
```

Run these before you write a single line of analysis. They catch the great majority of "why are my results wrong" problems at the earliest and cheapest moment.

## Key takeaways

- `read_csv` and `read_json` load files into DataFrames in one line; useful options include `sep`, `index_col`, and `nrows`.
- `head` / `tail` let you eyeball the actual data.
- `info` reveals row count, dtypes, and missing values — an `object` dtype on a "numeric" column is a red flag.
- `describe` summarizes numeric columns and surfaces impossible values fast.
- `value_counts`, `unique`, and `nunique` explore the contents of a single column.

## Try it

Save the CSV text above as `sales.csv`, then:

1. Load it and confirm with `info()` that `price` is a float and `qty` is an int.
2. Use `describe()` to find the average price. Does it match what you'd compute by hand?
3. Run `value_counts()` on the `size` column. Which size appears most often?
