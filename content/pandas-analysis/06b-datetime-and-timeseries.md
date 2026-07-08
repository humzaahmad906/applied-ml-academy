# 06b — Datetime and Time Series

Almost every real dataset has a timestamp column: when a sale happened, when a user clicked, when a sensor fired. And almost every beginner treats those timestamps as text and then wonders why nothing works. The moment you want to ask "how many sales per week?" or "what's the 7-day average?" or feed "day of week" to a model, you need pandas to understand that a timestamp is a *point in time*, not a string. This lesson turns raw timestamps into something you can slice, aggregate, and build features from.

Here's a table of timestamped coffee-shop sales — the kind of event log you'd export from any point-of-sale system:

```python
import pandas as pd

sales = pd.DataFrame({
    "ts": ["2024-01-05 08:15", "2024-01-05 09:40", "2024-01-06 12:05",
           "2024-01-08 17:30", "2024-01-12 07:55", "2024-02-02 14:20"],
    "item": ["latte", "espresso", "muffin", "latte", "tea", "espresso"],
    "revenue": [4.50, 3.00, 3.25, 5.00, 2.75, 3.50],
})
```

## Why a timestamp string is useless for math

Look at what pandas thinks `ts` is:

```python
sales["ts"].dtype
```

```
dtype('O')
```

`O` means "object" — plain Python strings. To pandas, `"2024-01-05 08:15"` is just characters, no different from `"latte"`. You can't subtract two dates, you can't ask which came first in any meaningful way, and you can't extract the month. Sorting is worse than useless: `"2024-1-9"` would sort *after* `"2024-1-10"` because `9` > `1` character-by-character. A string that looks like a date is not a date.

## Parsing with pd.to_datetime

`pd.to_datetime` converts a string column into a real datetime dtype:

```python
sales["ts"] = pd.to_datetime(sales["ts"])
sales["ts"].dtype
```

```
dtype('<M8[ns]')
```

That `datetime64[ns]` dtype is the goal — now every value is a genuine `Timestamp`. Pandas is good at guessing common formats (ISO `2024-01-05`, `2024-01-05 08:15`, and many others parse with no help). For unambiguous, standard formats you can pass nothing extra.

When the format is unusual or ambiguous, tell pandas exactly how to read it with a `format` string. This is faster and removes guesswork:

```python
# "05/01/2024" — is that Jan 5 or May 1? Be explicit.
pd.to_datetime("05/01/2024", format="%d/%m/%Y")   # -> 2024-01-05
```

The format codes are the standard ones: `%Y` four-digit year, `%m` month, `%d` day, `%H` hour, `%M` minute. If some rows are malformed, `errors="coerce"` turns the bad ones into `NaT` (the datetime version of `NaN`) instead of crashing the whole parse:

```python
pd.to_datetime(["2024-01-05", "not a date"], errors="coerce")
# DatetimeIndex(['2024-01-05', 'NaT'], ...)
```

### Parsing at load time

Most often your data comes from a CSV. Rather than reading strings and converting afterward, parse the column as you load it with `parse_dates`:

```python
sales = pd.read_csv("sales.csv", parse_dates=["ts"])
```

Now `ts` arrives already typed as datetime — one less step, and no chance of forgetting.

## The .dt accessor — timestamps into features

Once a column is datetime, the `.dt` accessor unlocks every calendar part as its own column. This is where time series pays off for machine learning: a single timestamp hides a dozen useful signals.

```python
sales["year"]      = sales["ts"].dt.year
sales["month"]     = sales["ts"].dt.month
sales["day"]       = sales["ts"].dt.day
sales["hour"]      = sales["ts"].dt.hour
sales["dayofweek"] = sales["ts"].dt.dayofweek   # Monday=0 ... Sunday=6
```

```
                  ts      item  revenue  year  month  day  hour  dayofweek
0 2024-01-05 08:15:00     latte     4.50  2024      1    5     8          4
1 2024-01-05 09:40:00  espresso     3.00  2024      1    5     9          4
2 2024-01-06 12:05:00    muffin     3.25  2024      1    6    12          5
3 2024-01-08 17:30:00     latte     5.00  2024      1    8    17          0
4 2024-01-12 07:55:00       tea     2.75  2024      1   12     7          4
5 2024-02-02 14:20:00  espresso     3.50  2024      2    2    14          5
```

Think about what a model can now learn: coffee sales spike in the morning (`hour`), muffins sell on weekends (`dayofweek`), revenue climbs toward the holidays (`month`). None of that was reachable from the raw string. A few common `.dt` properties:

```python
sales["ts"].dt.dayofweek      # 0-6, Monday is 0
sales["ts"].dt.day_name()     # "Friday", "Saturday", ...
sales["ts"].dt.is_month_end   # True/False
sales["ts"].dt.quarter        # 1-4
```

A classic derived feature — is this a weekend?

```python
sales["is_weekend"] = sales["ts"].dt.dayofweek >= 5
```

That single boolean is often one of the most predictive columns in retail data, and you got it from a timestamp with one line.

## Making the timestamp the index

For anything that spans time — slicing date ranges, resampling, rolling windows — you want the timestamp to *be* the index. Use `set_index`:

```python
ts = sales.set_index("ts").sort_index()
```

Always `sort_index()` after: time-based slicing assumes the index is in order. With a datetime index, you can slice by partial date strings, which reads almost like English:

```python
ts.loc["2024-01"]              # every row in January 2024
ts.loc["2024-01-05"]           # everything on Jan 5
ts.loc["2024-01-05":"2024-01-08"]   # an inclusive date range
```

```
# ts.loc["2024-01"]
                      item  revenue  year  month  ...
ts
2024-01-05 08:15:00     latte     4.50  2024      1
2024-01-05 09:40:00  espresso     3.00  2024      1
2024-01-06 12:05:00    muffin     3.25  2024      1
2024-01-08 17:30:00     latte     5.00  2024      1
2024-01-12 07:55:00       tea     2.75  2024      1
```

No filtering with `>=` and `<=` on strings — just name the period you want. `"2024-01"` selects the whole month, `"2024"` the whole year.

## Resampling — event data into regular buckets

Our rows are irregular: several on Jan 5, none on Jan 7, one in February. Real analysis usually wants a *regular* grid — daily, weekly, monthly totals. `resample` is `groupby` for time: it buckets rows into fixed periods and aggregates each bucket.

```python
ts.resample("D")["revenue"].sum()     # total revenue per day
```

```
ts
2024-01-05    7.50
2024-01-06    3.25
2024-01-07    0.00
2024-01-08    5.00
2024-01-09    0.00
...
2024-02-02    3.50
```

Notice pandas *fills in* the empty days (Jan 7, Jan 9) with zeros — a huge convenience, because a gap in event data is real information ("no sales that day") that a plain groupby would silently drop.

The frequency string is the knob. The common ones:

```python
ts.resample("D")["revenue"].sum()      # daily
ts.resample("W")["revenue"].sum()      # weekly
ts.resample("ME")["revenue"].mean()    # monthly (month-END); "ME" since pandas 2.1
```

> Heads up: older tutorials use `"M"` for monthly. In modern pandas (2.1+) that's deprecated in favor of `"ME"` (month-end) and `"MS"` (month-start). Use `"ME"` and the deprecation warning goes away.

Swap the aggregation to ask a different question — `.sum()` for totals, `.mean()` for averages, `.count()` for how many events per bucket:

```python
ts.resample("W")["revenue"].agg(["sum", "mean", "count"])
```

Downsampling event logs to daily or weekly aggregates is the single most common time-series operation you'll do: it's how you turn "one row per transaction" into "one row per day" that you can chart or model.

## Rolling windows — moving averages

Resampling changes the *grid*. Rolling windows keep the grid but smooth across neighboring rows. A 7-day moving average of daily revenue evens out the day-to-day noise so a trend is visible:

```python
daily = ts.resample("D")["revenue"].sum()
daily.rolling(7).mean()
```

`rolling(7)` builds a sliding window of 7 consecutive rows; `.mean()` averages each window. The first 6 results are `NaN` because there aren't yet 7 rows behind them — that's expected, not a bug.

This is the pandas analog of SQL window functions (`AVG(...) OVER (...)`), and it's a feature-engineering workhorse. "Revenue over the trailing 7 days," "rolling 30-day active users," "3-period moving average" — all are `.rolling(n).agg(...)`:

```python
daily.rolling(7).mean()    # trailing 7-day average
daily.rolling(7).sum()     # trailing 7-day total
daily.rolling(3).max()     # highest single day in the last 3
```

## Time zones and the leakage trap

Two cautions before you ship time-series features.

**Time zones.** A bare timestamp has no zone — `08:15` in whose clock? If your data mixes zones, localize once and convert explicitly rather than letting off-by-hours bugs creep in:

```python
ts_utc = ts.tz_localize("UTC")            # stamp the naive index as UTC
ts_ny  = ts_utc.tz_convert("US/Eastern")  # then view it in another zone
```

Pick one canonical zone (UTC is the safe default for storage) and only convert for display.

**Leakage — the one that ruins models.** When you build features from time series, never use information from the future to predict the past. `rolling(7)` looks *backward*, which is safe. But a *centered* window (`rolling(7, center=True)`), or resampling in a way that peeks at later rows, leaks future data into a training row — the model looks brilliant in testing and fails in production. Rule of thumb: every time-based feature must be computable using only data available at that timestamp. If a feature needs tomorrow's number, it can't exist at prediction time.

## Key takeaways

- A date stored as a string is useless for math; `pd.to_datetime` (or `parse_dates=` in `read_csv`) converts it to a real `datetime64` dtype.
- The `.dt` accessor extracts calendar features — `.dt.year`, `.dt.month`, `.dt.hour`, `.dt.dayofweek` — turning one timestamp into many model inputs.
- `set_index` on a datetime column (then `sort_index()`) lets you slice by partial dates: `df.loc["2024-01"]`.
- `resample("D"/"W"/"ME")` is groupby-for-time; it buckets events into a regular grid and fills empty periods. Use `"ME"` for monthly in pandas 2.1+, not the deprecated `"M"`.
- `rolling(n).mean()` gives moving averages — the pandas version of SQL window functions, ideal for trailing features.
- Store in UTC and convert explicitly; and never build features from the future (`rolling` looks backward — keep it that way to avoid leakage).

## Try it

Using the `sales` DataFrame above:

1. Parse `ts` with `pd.to_datetime`, then add columns for `hour` and `day_name()`. In which hour does the shop see the most sales?
2. Set `ts` as the index, sort it, and select only the rows from January 2024 with `.loc`.
3. Resample daily revenue to a total per day, then compute a 3-day rolling mean of that daily series. Explain why the first two values are `NaN`.
