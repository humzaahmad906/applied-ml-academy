# 01 — Time Series Fundamentals

Most of the machine learning you've seen so far assumes the rows of your dataset are interchangeable. A spam classifier doesn't care whether the email about a Nigerian prince came before or after the meeting invite — shuffle the rows, retrain, and you get the same model. Time series breaks that assumption completely. A time series is data indexed by time — daily sales, hourly server load, monthly electricity demand — and the *order* of the rows is not incidental. It *is* the signal. Today's sales depend on yesterday's. Rearrange the rows and you have destroyed the very thing you were trying to model.

That single fact — order carries information — reshapes everything: how you split data, which models help, and how you evaluate. This lesson builds the vocabulary and the instincts. Get these right and the rest of the course is downhill.

## What makes a time series special: autocorrelation

In ordinary tabular data, we hope rows are roughly independent. In a time series they are emphatically not. The temperature at 3pm is a great predictor of the temperature at 4pm. Today's stock price tells you a lot about tomorrow's. This dependence of a series on its own past is called **autocorrelation** — literally, correlation of the series with a time-shifted copy of itself.

We measure it with the **autocorrelation function (ACF)**. For a lag of *k*, the ACF is the correlation between the series and the same series shifted back by *k* steps. Lag-1 autocorrelation compares each value to the one immediately before it; lag-7 compares each value to the value a week earlier.

```python
import pandas as pd

# Daily sales, indexed by date
sales = pd.Series(
    [102, 108, 115, 120, 118, 95, 88, 104, 110, 117, 123, 121, 97, 90],
    index=pd.date_range("2026-01-01", periods=14, freq="D"),
)

print(sales.autocorr(lag=1))   # correlation with yesterday
# output: 0.71
print(sales.autocorr(lag=7))   # correlation with a week ago
# output: 0.94
```

The high lag-7 value is the fingerprint of a **weekly pattern**: each day resembles the same weekday last week far more than it resembles yesterday. Autocorrelation is not a nuisance to be scrubbed away — it is the structure we exploit to forecast. If a series had zero autocorrelation at every lag, the past would tell you nothing about the future, and forecasting would be hopeless.

## The four components: trend, seasonality, cycles, noise

A useful mental model is to imagine any time series as a sum (or product) of four ingredients.

**Trend** is the long-run direction — the slow drift up or down over the whole span. A SaaS company's user count trending upward year over year; a declining print-newspaper circulation. Trend is about the big arc, not the wiggles.

**Seasonality** is a pattern that repeats over a *fixed, known period*. Retail sales spike every December. Electricity demand peaks every afternoon. Restaurant traffic rises every weekend. The defining feature is the fixed calendar period: daily, weekly, yearly. If you know the period, you can line up the repeats.

**Cycles** are the confusing cousin of seasonality: patterns that rise and fall but *without a fixed period*. Economic boom-and-bust, multi-year commodity cycles. Because the length varies, you can't set a calendar clock to them, which makes them much harder to model than seasonality. Beginners routinely conflate the two — the test is simply "is the period fixed and known?" If yes, seasonality; if no, cycle.

**Noise** (or the residual) is everything left over — the irreducible randomness after trend, seasonality, and cycles are accounted for. A good model captures the structure and leaves behind noise that looks like noise. If your residuals still have visible pattern, you left signal on the table.

Statsmodels can pull these apart for you with a classic seasonal decomposition:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(sales, model="additive", period=7)
result.trend        # the slow-moving level
result.seasonal     # the repeating weekly shape
result.resid        # what's left — hopefully noise
```

`model="additive"` assumes the components add together (`y = trend + seasonal + noise`), which fits when the seasonal swing is a roughly constant size. Use `model="multiplicative"` when the seasonal swings *grow with the level* — sales that vary by 10% whether the baseline is 100 or 10,000. Decomposition is a diagnostic, not a forecaster, but it is the first thing to run when you meet a new series, because it tells you which ingredients you're dealing with.

## Stationarity: why models crave it

A series is **stationary** when its statistical properties don't change over time — roughly, constant mean, constant variance, and an autocorrelation structure that depends only on the lag, not on *when* you look. A stationary series looks the same in its first half as in its second half. Stock prices are not stationary (they wander); but *daily changes* in stock price often are much closer to stationary.

Why do we care? Many classical models — ARIMA above all — are built on the assumption of stationarity. Their math only holds when the series isn't drifting. Trend and seasonality both violate stationarity: a trend means the mean keeps changing; seasonality means the behavior depends on where you are in the cycle. So a huge part of classical forecasting is *transforming a non-stationary series into a stationary one*, modeling that, and transforming back.

The most common transform is **differencing**: replace each value with the difference from the previous value.

```python
diff = sales.diff().dropna()   # today minus yesterday
```

Differencing kills a linear trend. Seasonal differencing (subtracting the value one full season ago, `sales.diff(7)`) kills a fixed seasonal pattern. A log transform tames variance that grows with the level. You'll see all of these again in the ARIMA lesson.

To *test* for stationarity rather than eyeball it, use the **Augmented Dickey-Fuller (ADF) test**:

```python
from statsmodels.tsa.stattools import adfuller

stat, pvalue, *_ = adfuller(sales)
print(pvalue)
# output: 0.48   -> p > 0.05, cannot reject "non-stationary": treat as non-stationary
```

The ADF null hypothesis is "the series is non-stationary." A small p-value (< 0.05) lets you reject that and call it stationary. A large p-value, as here, means you have not shown stationarity — difference the series and test again.

## The golden rule: never shuffle, split by time

Here is the rule that separates people who understand time series from people who quietly ship broken models: **never shuffle a time series, and always split by time.**

In standard ML you split train and test randomly, because rows are interchangeable. Do that with a time series and you commit a subtle, catastrophic error: your training set now contains data from *after* your test set. The model gets to peek at the future to predict the past. This is a form of **data leakage** — the single most common way time-series projects fool themselves — and it produces gorgeous validation scores that evaporate the moment you deploy, because in production the future genuinely isn't available.

The correct split is chronological. Train on the earliest stretch, test on the most recent stretch, and never let a single future timestamp into training.

```python
# CORRECT: chronological split
cutoff = "2026-01-11"
train = sales[:cutoff]
test = sales[cutoff:]

# WRONG — never do this on a time series:
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(sales, shuffle=True)
```

That commented-out line is the trap. `train_test_split` shuffles by default, and on a time series it silently launders future information into your training set. Burn the chronological split into muscle memory now; the evaluation lesson (06) will formalize it into proper backtesting with rolling windows.

## A note on frequency and gaps

Real time series arrive at some **frequency** — daily, hourly, every five minutes — and pandas tracks this explicitly. Setting a proper `DatetimeIndex` with a known frequency unlocks the whole toolkit: resampling, rolling windows, and lag features (lesson 03).

```python
sales.index.freq          # 'D' for daily, once set
weekly = sales.resample("W").sum()   # roll daily up to weekly totals
```

Watch for missing timestamps. A daily series that skips weekends, or a sensor that dropped offline for an hour, leaves gaps that quietly break lag-based features and seasonal models, because "the value 7 steps ago" is no longer "a week ago." Reindexing to a complete date range and deciding how to fill (forward-fill, interpolate, or leave as NaN) is unglamorous but essential groundwork.

## Key takeaways

- A time series is data indexed by time, and the *order* of the rows is the signal — not incidental.
- Autocorrelation (a series correlated with its own past, measured by the ACF) is the structure that makes forecasting possible.
- Decompose any new series into trend, seasonality, cycles, and noise; seasonality has a fixed known period, cycles do not.
- Stationarity (stable mean and variance over time) is what classical models assume; differencing and log transforms are how you get there.
- The golden rule: never shuffle, always split chronologically. A random split leaks the future into training and inflates your scores.

## Try it

Take a daily series you have or synthesize one with a weekly pattern (e.g. add a `+15` bump every 7th day to a slow upward trend plus small random noise). Then:

1. Print the lag-1 and lag-7 autocorrelation. Which is larger, and what does that tell you about the dominant pattern?
2. Run `seasonal_decompose` with `period=7` and describe what the trend and seasonal components look like.
3. Run the ADF test, then difference the series once and run it again. Did the p-value drop below 0.05? Write one sentence explaining what differencing did to the trend.
