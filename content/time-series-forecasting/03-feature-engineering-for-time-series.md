# 03 — Feature Engineering for Time Series

Classical models (lesson 02) take a raw series and model its temporal structure internally. Machine-learning models don't — a gradient-boosting tree has no concept of "yesterday." To use the ML toolbox on a time series, *you* must hand the model the past explicitly, as columns. That translation — turning a one-column series into a feature table where each row's features describe its own recent history — is feature engineering for time series, and it is where most of the real work (and most of the real bugs) live. This lesson is the feature recipes and, above all, the leakage traps that are unique and unforgiving in the temporal setting.

## Lag features: handing the model the past

The foundational time-series feature is the **lag**: the value of the series some number of steps ago. `lag_1` is yesterday's value, `lag_7` is the value a week ago, `lag_365` is a year ago. In pandas this is one method, `shift`:

```python
import pandas as pd

df = pd.DataFrame({"sales": [102, 108, 115, 120, 118, 95, 88, 104, 110, 117]},
                  index=pd.date_range("2026-01-01", periods=10, freq="D"))

df["lag_1"] = df["sales"].shift(1)   # yesterday
df["lag_7"] = df["sales"].shift(7)   # same weekday last week
print(df[["sales", "lag_1", "lag_7"]].head(3))
# output:
#             sales  lag_1  lag_7
# 2026-01-01    102    NaN    NaN
# 2026-01-02    108  102.0    NaN
# 2026-01-03    115  108.0    NaN
```

`shift(1)` moves every value *down* one row, so the `lag_1` on any row holds the previous day's sales — information genuinely available at that point in time. The `NaN`s at the top are unavoidable: the first row has no yesterday. You drop those rows before training.

The direction of `shift` is the entire ballgame. `shift(1)` looks *backward* (correct — it uses the past). A negative shift, `shift(-1)`, pulls the *future* into the current row. Using a negative shift as a feature is the number-one leakage bug in time-series ML: you'd be feeding the model tomorrow's value to predict today, scoring beautifully in validation and failing completely in production. (A `shift(-1)` is legitimate for building your *target* in multi-step setups — lesson 04 — but never as a feature.)

Which lags to include? Let the series structure guide you: lag-1 for short-term momentum, lags at the seasonal period (`lag_7` for weekly, `lag_12` for monthly-with-yearly), and lag-1-of-the-season for year-over-year. The ACF from lesson 01 points you at the lags that actually carry signal.

## Rolling statistics: summarizing a window of the past

A single lag is a snapshot; a **rolling statistic** summarizes a whole recent window — its mean, standard deviation, min, or max. A 7-day rolling mean captures the recent level; a 7-day rolling std captures recent volatility. These features let a tree "see" trend and turbulence without your having to spell out every lag.

Here is where the subtlest leak in all of time-series ML hides. Consider the naive version:

```python
df["roll_mean_3"] = df["sales"].rolling(3).mean()
```

The window `[t-2, t-1, t]` for row *t* **includes `t` itself** — the very value you're trying to predict. The feature is contaminated with the answer. You must shift the rolling feature back by one so the window ends *before* the current row:

```python
# CORRECT: window ends yesterday, excludes today
df["roll_mean_3"] = df["sales"].shift(1).rolling(3).mean()
df["roll_std_7"]  = df["sales"].shift(1).rolling(7).std()
```

`shift(1)` first, `rolling` second. Read it as "as of yesterday, the mean of the three days ending yesterday." Every rolling feature you build on the target column needs this shift. Forget it and your model looks brilliant in testing and is worthless live — the classic silent time-series failure.

## Date and calendar features

The timestamp itself is a goldmine, because human and economic behavior is organized by the calendar. From a `DatetimeIndex` you can extract features with zero leakage risk — the calendar for a future date is known in advance, so these are always safe to use.

```python
idx = df.index
df["dayofweek"] = idx.dayofweek      # 0=Mon ... 6=Sun
df["month"]     = idx.month
df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
df["is_month_start"] = idx.is_month_start.astype(int)
```

Add domain calendars where they matter: public holidays (the `holidays` library, or a `is_holiday` flag), paydays, promotional periods, school terms. A retail model that knows December 24th is different from December 25th will crush one that doesn't.

One refinement: cyclical features like month and day-of-week are circular — December (12) is adjacent to January (1), and a tree treating them as 12 vs 1 misses that. A common trick is a **sine/cosine encoding** that wraps the cycle onto a circle so "distance" respects the wrap-around:

```python
import numpy as np
df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
```

Tree models often cope fine with the raw integer, but linear and neural models genuinely benefit from the cyclical encoding.

## The leakage traps, gathered in one place

Time-series leakage is more insidious than tabular leakage because the "obvious" pandas operations quietly look into the future. Commit these four traps to memory:

1. **The shuffle/random split** (lesson 01). Splitting randomly puts future rows in training. Always split by time.
2. **The unshifted rolling window.** `rolling(k).mean()` includes the current row. Always `shift(1)` before `rolling` on the target.
3. **The negative-shift feature.** `shift(-1)` as an input leaks the future. Features look backward only.
4. **Fitting transforms on the whole dataset.** Scaling, imputing, or encoding using statistics computed over *all* the data — including the test period — leaks test-set information into training. Fit your scaler on the *training slice only*, then apply it to the test slice. This mirrors the leakage lesson from ML Foundations, and it bites just as hard here.

There's a deeper, sneakier variant worth naming: **the point-in-time trap**. Suppose you join in an external feature like "total monthly revenue." If that number is only *finalized* at month-end but you attach it to every day of the month, then early-month rows are using information that didn't exist yet. The honest question for every feature is always: *"Would this exact value have been available at the moment I'm predicting?"* If not, it leaks. This is the same discipline as an as-of / point-in-time join in SQL — you join each row only to feature values whose timestamp is `<=` the row's timestamp, never the future ones. If you've done SQL window functions, note that `LAG(x) OVER (ORDER BY date)` is precisely `shift` and `AVG(x) OVER (... ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)` is precisely the correctly-shifted rolling mean — the framing that keeps you leak-free is identical across the two tools.

## A clean feature-building pattern

Putting it together into a small reusable function keeps the discipline consistent:

```python
def make_features(series, lags=(1, 7, 14), roll_windows=(7, 14)):
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in roll_windows:
        # shift(1) FIRST so the window excludes the current row
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(w).std()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    return df.dropna()      # drop the warm-up rows with NaN lags

features = make_features(df["sales"])
```

Every feature here is strictly backward-looking and every rolling stat is shifted — this table is safe to feed a regressor. Libraries automate exactly this: **`mlforecast`** (Nixtla) generates lag and rolling features for you with the shifting handled correctly, and **`sktime`** and **`darts`** offer transformer classes for the same. Use them in production — but build the table by hand once, as above, so you understand precisely what they're doing and can spot when a feature leaks.

## Key takeaways

- ML models can't see time; you must expose the past as columns — lag features (`shift(k)`) are the foundation.
- Rolling statistics summarize a recent window, but you **must `shift(1)` before `rolling`** or the window leaks the current value.
- Calendar features (day-of-week, month, holidays, weekend flags) are always safe because future dates are known; cyclical sin/cos encoding helps linear and neural models.
- Time-series leakage hides in ordinary operations: random splits, unshifted windows, negative shifts, and whole-dataset transforms. Fit transforms on the training slice only.
- The one question that catches every leak: "Would this value have actually been available at prediction time?" — the same point-in-time discipline as an as-of SQL join.

## Try it

Starting from a daily sales series:

1. Build `lag_1`, `lag_7`, and a **correctly shifted** 7-day rolling mean. Print the first 10 rows and confirm no row's rolling feature includes its own `y`.
2. Deliberately create the *buggy* rolling mean without the `shift(1)`, and compute its correlation with `y` versus the correct version's correlation with `y`. The buggy one will be suspiciously higher — explain why that higher number is a red flag, not good news.
3. Add `dayofweek` and an `is_weekend` flag. Group `y` by `is_weekend` and confirm the pattern the flag is meant to capture actually exists.
