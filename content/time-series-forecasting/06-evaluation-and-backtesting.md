# 06 — Evaluation and Backtesting

You have models now — classical, gradient-boosted, deep. The question that decides whether any of them is worth deploying is: *how good is this forecast, really?* Get evaluation wrong and everything upstream is wasted, because you'll ship the model that *looked* best rather than the one that *is* best. Time-series evaluation has two halves, and both differ sharply from ordinary ML: **which metric** you compute, and — more importantly — **how you split the data to compute it.** This lesson covers both, and it ends on the honest way to say "I don't know" — prediction intervals.

## The metrics

All forecasting metrics compare predictions to actuals on a held-out period. They differ in units, how they treat scale, and how they handle outliers. Know four families.

**MAE — Mean Absolute Error.** The average absolute gap between forecast and actual. It's in the *same units as the data* (dollars, units sold), which makes it wonderfully interpretable — "we're off by 12 units a day on average." It treats all errors linearly, so it's robust to outliers.

**RMSE — Root Mean Squared Error.** Square the errors, average, square-root. Also in the data's units, but because it squares first, it **punishes large errors far more** than small ones. Use RMSE when a few big misses are much worse than many small ones (a stockout that empties the shelf); use MAE when every unit of error costs the same.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

**MAPE and sMAPE — percentage errors.** MAPE (Mean Absolute *Percentage* Error) expresses error as a percent of the actual, which makes it *scale-free* — you can compare a forecast of daily coffee sales against one of national electricity demand. That comparability is why executives love it. But MAPE has two nasty flaws: it **blows up (divides by zero) when actuals are zero or near-zero**, common in intermittent demand, and it **penalizes over- and under-forecasting asymmetrically** (a forecast can be at most 100% too low but unboundedly too high). **sMAPE** (symmetric MAPE) patches the asymmetry by dividing by the average of actual and forecast, but it has its own quirks. Treat percentage metrics with care around zeros.

**MASE — Mean Absolute Scaled Error.** The metric forecasting specialists actually trust, and the one most beginners have never heard of. MASE divides your model's MAE by the MAE of a **naive baseline** (usually "predict the last value," or the last *seasonal* value). The result is a ratio with a beautifully clear meaning:

- **MASE < 1**: your model beats the naive baseline. 
- **MASE = 1**: exactly as good as naive. 
- **MASE > 1**: *worse than doing nothing clever* — a genuine possibility that MAE alone would hide.

MASE is scale-free (so it aggregates across many series), it's well-defined when actuals hit zero, and it bakes the "did we beat the baseline?" question directly into the number. When in doubt, report MASE alongside MAE.

**Which to use?** Report at least two: an absolute metric in the data's units (MAE or RMSE) so the error is tangible, and a scale-free one (MASE preferred) so you know whether you beat naive and can compare across series. And decide the metric *before* modeling — picking it afterward, once you've seen which flatters your model, is the same self-deception warned about in the ML Foundations metrics lesson.

## Why a random test split lies

Here is the heart of the lesson, and the mistake that quietly ruins time-series projects. In ordinary ML you evaluate on a random hold-out. On a time series, a **random test split is a lie**, for two compounding reasons:

1. **It leaks the future into training.** (Lesson 01's golden rule.) Random selection puts some post-test timestamps into the training set, so the model is quietly trained on data from *after* the period it's being tested on. Scores look great and collapse in production.
2. **It doesn't measure what you actually do.** In production you always forecast *forward* from now into the unknown future. A random hold-out scattered through history never tests that. The only honest test asks the real question: *train on the past, forecast the genuinely-later future, compare.*

So the test set is always the **most recent** contiguous stretch, and training is everything before it. But a single train/test split has its own weakness: it evaluates on just one window, so your score depends heavily on whether that particular period happened to be easy or hard. The fix is backtesting.

## Backtesting: rolling and expanding windows

**Backtesting** evaluates a forecasting model the way you'd have used it historically: repeatedly, at many points in time, always training only on data prior to each forecast. It's *walk-forward validation*, and it's the time-series analog of cross-validation. Two flavors:

**Expanding window.** Start with an initial training period, forecast the next block, then *grow* the training set to include that block and forecast the next, and so on. Training data accumulates. This mirrors how you'd retrain in production as new data arrives, and uses all available history.

**Rolling (sliding) window.** Same walk-forward idea, but the training window is a *fixed size* that slides forward, dropping the oldest data as it adds new. Use this when old data is stale or the series' behavior drifts, so recent history is more representative than ancient history.

```
Expanding:                     Rolling (fixed width):
[train....][test]              [train..][test]
[train......][test]              [train..][test]
[train........][test]              [train..][test]
```

Each fold produces a forecast and a score; you average across folds for a stable estimate, and — crucially — you can look at the *spread* of scores to see how consistent the model is.

Scikit-learn's `TimeSeriesSplit` gives you expanding-window folds directly:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_tr, y_tr)
    scores.append(mean_absolute_error(y_te, model.predict(X_te)))

print(np.mean(scores), np.std(scores))   # average error and its stability
```

Notice `TimeSeriesSplit` never lets a test index precede its train indices — order is respected by construction. For richer backtesting (multiple horizons, gaps between train and test to simulate forecast lead time, per-fold refitting of classical models), the forecasting libraries have purpose-built tools: `sktime`'s `ExpandingWindowSplitter` / `SlidingWindowSplitter`, `darts`'s `historical_forecasts`, and Nixtla's `cross_validation` method. They automate the walk-forward loop and handle the retraining correctly.

One subtlety worth flagging: when your features include multi-step recursive forecasts (lesson 04), backtest the *whole horizon the way you'll use it* — forecast all H steps from each fold's cutoff and score the full path — not just one step ahead. A model that's great at one-step and terrible at seven-step will look fine under naive one-step evaluation and fail you in production.

## Prediction intervals: forecasting your uncertainty

A single number ("next week: 1,040 units") pretends to a certainty no forecast possesses. A **prediction interval** is honest: "next week: 1,040, with an 80% interval of 920–1,160." That range is often *more* valuable than the point forecast — an inventory planner sizes safety stock from the upper bound, not the mean.

How you get intervals depends on the model family:

- **Classical models** (ARIMA, ETS) produce them analytically from their noise assumptions — recall `get_forecast().conf_int()` from lesson 02. Nearly free.
- **ML models** need help: train with a **quantile loss** to predict, say, the 10th and 90th percentiles directly (LightGBM's `objective="quantile"`), giving an 80% interval from two extra models.
- **Deep and foundation models** are often *probabilistic* by design (Chronos, DeepAR-style models), emitting a full predictive distribution you can read any quantile from.
- **Conformal prediction** is a model-agnostic method that calibrates intervals from the model's own backtest residuals, with coverage guarantees and no distributional assumptions — increasingly the default for wrapping any point forecaster.

Whatever the source, *validate the interval*: over a backtest, an 80% interval should contain the actual value roughly 80% of the time. If it only covers 55%, your intervals are overconfident and dangerous; if 98%, they're uselessly wide. This **coverage check** is the analog of calibration for forecasting, and it's the difference between honest uncertainty and decorative error bars.

## Key takeaways

- Report at least two metrics: an absolute one in the data's units (MAE or RMSE) and a scale-free one — MASE is best because <1/=1/>1 tells you directly whether you beat the naive baseline.
- RMSE punishes big misses more than MAE; MAPE is scale-free but explodes near zeros and is asymmetric — use it cautiously.
- A random test split lies twice: it leaks the future into training, and it never tests the forward forecasting you actually do. The test set is always the most recent stretch.
- Backtesting is walk-forward validation — expanding windows (accumulate history, mirrors production retraining) or rolling windows (fixed size, for drifting series) — averaged over folds for a stable, honest estimate.
- Prediction intervals quantify uncertainty and are often more useful than the point forecast; always verify their coverage against a backtest.

## Try it

Using your best model from lesson 04 or 05:

1. Compute MAE, RMSE, and MASE on a chronological hold-out (build the naive baseline as "predict last week's value" for MASE). Is your MASE below 1? If not, your model loses to naive — say what that means.
2. Run a 5-fold `TimeSeriesSplit` backtest and report both the mean and the standard deviation of the error. Is the model consistent across folds, or does one period drag it down?
3. Train two LightGBM quantile models (`alpha=0.1` and `alpha=0.9`) to form an 80% interval, then over your backtest compute what fraction of actuals fell inside it. How close is the empirical coverage to 80%, and are your intervals over- or under-confident?
