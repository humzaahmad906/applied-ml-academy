# 07 — Project: Forecasting Retail Demand

Everything in this course converges here. You're going to forecast daily demand for a retail product — the single most common forecasting problem in industry, sitting under every inventory, staffing, and supply-chain decision. We'll go the way a professional actually goes: understand the data, build a *baseline first*, layer on a machine-learning model, and — the part that separates real work from Kaggle theater — evaluate it *honestly* with backtesting. By the end you'll have not just a model but a defensible answer to "should we trust this?"

The point of this project is not the specific model. It's the *discipline*: baseline before complexity, chronological everything, and honest backtested numbers.

## Step 0 — The problem and the data

Imagine daily unit sales for one product at one store, three years of history. Our job: forecast the next 28 days so the planner can order stock. The data is a `DatetimeIndex` and a `sales` column, and it has the structure lesson 01 taught us to expect — a mild upward trend, a strong weekly pattern (weekends busy), and an annual holiday bump.

```python
import pandas as pd

sales = pd.read_csv("store_sales.csv", parse_dates=["date"], index_col="date")
sales = sales.asfreq("D")               # enforce a complete daily index
sales["units"] = sales["units"].interpolate()   # fill the rare gap
```

That `asfreq("D")` is not cosmetic — it exposes missing days as NaN so lag features (lesson 03) stay aligned to real calendar distance. Before modeling, *look* at the data: plot it, run `seasonal_decompose(sales["units"], period=7)`, and confirm the weekly season and trend are really there. You forecast what you understand.

## Step 1 — Split by time, once and for all

Hold out the final 28 days as the true test set and never touch them until the very end. Everything before is for training and backtesting.

```python
horizon = 28
train = sales.iloc[:-horizon]
test  = sales.iloc[-horizon:]      # the last 28 days — locked away
```

No shuffling, no random split — the golden rule from lesson 01. This hold-out answers the only question that matters: forecasting forward from a real cutoff into a genuinely-unseen future.

## Step 2 — The baseline (do this first, always)

Before any model earns a line of code, build the baseline every later model must beat. For a weekly-seasonal series, the right naive baseline is **seasonal naive**: predict each day as the value from the same weekday last week.

```python
def seasonal_naive(history, horizon, season=7):
    last_season = history["units"].iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]

baseline_pred = seasonal_naive(train, horizon)
```

Then a classical model, which is often the *real* thing to beat — a Holt-Winters ETS (lesson 02) captures trend and weekly seasonality in three lines:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ets = ExponentialSmoothing(train["units"], trend="add",
                           seasonal="add", seasonal_periods=7).fit()
ets_pred = ets.forecast(horizon)
```

Score both against the naive baseline using MASE (lesson 06). If ETS can't beat seasonal-naive, that's a finding — maybe the series is nearly pure weekly repetition — and it means the bar for any ML model is high.

## Step 3 — Feature engineering

Now bring in lesson 03 to reframe forecasting as regression (lesson 04). Build a leak-free feature table: backward-looking lags, *shifted* rolling stats, and calendar features.

```python
import numpy as np

def make_features(df):
    out = pd.DataFrame(index=df.index)
    out["y"] = df["units"]
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = df["units"].shift(lag)
    for w in (7, 28):
        # shift(1) FIRST — the window must exclude the current day
        out[f"rmean_{w}"] = df["units"].shift(1).rolling(w).mean()
        out[f"rstd_{w}"]  = df["units"].shift(1).rolling(w).std()
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)
    return out

feat = make_features(sales).dropna()
```

Every feature is available at prediction time; every rolling stat is shifted so it can't peek at today (lesson 03's cardinal rule). Ask of each column the one question that kills leaks: *would this value have existed at the moment I predict?* Here, yes for all of them.

## Step 4 — Train the ML model

Split the feature table chronologically and fit a LightGBM regressor (lesson 04). The test period is the same final 28 days locked away in Step 1.

```python
import lightgbm as lgb

cutoff = feat.index[-horizon]
X = feat.drop(columns="y")
y = feat["y"]
X_tr, y_tr = X[X.index < cutoff], y[y.index < cutoff]
X_te, y_te = X[X.index >= cutoff], y[y.index >= cutoff]

model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31)
model.fit(X_tr, y_tr)
```

Because our horizon is 28 days, decide a multi-step strategy (lesson 04). The simplest honest choice here is a **direct** model per horizon or a recursive roll-forward; for the walkthrough we'll evaluate the model's one-step skill first, then backtest the full 28-day path in Step 5 — which is what actually matters.

If the series had a strong trend, recall that trees can't extrapolate (lesson 04): we'd model differences or detrend first. Our trend is mild and the lag/rolling features carry most of it, but on a steeper series that step is mandatory.

## Step 5 — Backtest honestly

A single test window can flatter or punish a model by luck. Backtest with a walk-forward loop (lesson 06): repeatedly cut off at earlier points, forecast the next 28 days, and score — always training only on the past.

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def mase(y_true, y_pred, y_train, season=7):
    naive_mae = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return mean_absolute_error(y_true, y_pred) / naive_mae

tscv = TimeSeriesSplit(n_splits=5, test_size=horizon)
mases = []
for tr_idx, te_idx in tscv.split(X):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = model.predict(X.iloc[te_idx])
    mases.append(mase(y.iloc[te_idx].values, pred, y.iloc[tr_idx].values))

print(f"LightGBM backtest MASE: {np.mean(mases):.3f} +/- {np.std(mases):.3f}")
# output: LightGBM backtest MASE: 0.78 +/- 0.06
```

Now compare *every* model — seasonal-naive, ETS, LightGBM — on the **same backtest folds**, reporting MAE (tangible units) and MASE (did we beat naive?), with the fold-to-fold spread. A results table like this is the deliverable:

```
model            MAE    MASE   notes
seasonal_naive   14.2   1.00   the bar
ETS              11.8   0.83   strong, three lines of code
LightGBM         11.1   0.78   best, but is 0.05 MASE worth the complexity?
```

The MASE of 0.78 means LightGBM beats naive by 22% — real skill. But notice ETS is close behind for a fraction of the effort and gives free prediction intervals. Whether the ML model's marginal gain justifies its maintenance cost is a *judgment call you now have the evidence to make* — and making that call, rather than reflexively shipping the fanciest model, is what a professional does.

## Step 6 — Intervals and the final forecast

Point forecasts alone are dishonest (lesson 06). Give the planner an interval by training LightGBM quantile models, and verify coverage on the backtest before trusting it.

```python
lo = lgb.LGBMRegressor(objective="quantile", alpha=0.1).fit(X_tr, y_tr)
hi = lgb.LGBMRegressor(objective="quantile", alpha=0.9).fit(X_tr, y_tr)
# 80% interval: [lo.predict(X_future), hi.predict(X_future)]
```

Then refit your chosen model on *all* data through today and produce the actual next-28-day forecast with its interval — the thing the business consumes.

## What you've built

Step back at what this workflow embodies. You started simple and only added complexity that *earned its place against a baseline*. You kept time sacred — chronological splits, backward-looking features, walk-forward backtesting — so no future ever leaked into a number you reported. And you delivered not a single fragile point forecast but a backtested, uncertainty-quantified answer with an honest account of how much better it is than doing nothing. That discipline transfers to any forecasting problem you'll meet: energy load, web traffic, call volume, cash flow. The models will change; the workflow won't.

## Key takeaways

- Baseline first, always: seasonal-naive and ETS set the bar every ML or deep model must clear before it earns deployment.
- Keep time sacred end to end — chronological hold-out, leak-free backward-looking features, and walk-forward backtesting on shared folds.
- Compare all models on the *same* backtest with both a tangible metric (MAE) and a baseline-relative one (MASE), and report the fold spread, not a single lucky number.
- The best model is a judgment call, not an automatic "pick the highest score" — weigh marginal accuracy against complexity, interpretability, and free intervals.
- Ship an interval, not just a point, and verify its coverage before you trust it.

## Try it

Take a real daily series (retail sales, energy demand, or website traffic from a public dataset) and run this whole pipeline end to end:

1. Build the seasonal-naive and ETS baselines and record their backtested MASE.
2. Engineer leak-free features and train a LightGBM model; backtest it on the same folds. Report the full results table (MAE, MASE, spread) for all three models.
3. Make the deployment decision *in writing*: which model would you ship, and why? Justify it with your backtest numbers, the complexity tradeoff, and whether you need the prediction intervals — exactly as you'd defend it to a skeptical manager.
