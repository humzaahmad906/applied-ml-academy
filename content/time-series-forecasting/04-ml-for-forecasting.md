# 04 — Machine Learning for Forecasting

Once you can turn a time series into a feature table (lesson 03), a striking possibility opens up: forecasting becomes *ordinary supervised regression*. Each row has features describing the recent past and a target — the next value. Feed that to any regressor you like — linear regression, random forest, or, most commonly in practice, a gradient-boosting machine — and you have a forecaster. This reframing is why gradient boosting has quietly become one of the most successful forecasting approaches in industry, winning Kaggle competitions and running production demand-planning systems worldwide. This lesson shows the reframing, the model of choice, and the one genuinely tricky part: forecasting more than one step ahead.

## Reframing forecasting as supervised regression

In standard supervised learning you have a feature matrix `X` and a target vector `y`, and you learn `y = f(X)`. Time-series forecasting fits that mold exactly once you build the features: `X` is the lag/rolling/calendar table, and `y` is the value you want to predict — typically the *next* observation.

```python
# From lesson 03's feature table:
#   features has columns lag_1, lag_7, roll_mean_7, dayofweek, month, and y
X = features.drop(columns="y")
y = features["y"]

# CHRONOLOGICAL split — never shuffle (lesson 01)
cutoff = "2026-06-01"
X_train, y_train = X[:cutoff], y[:cutoff]
X_test,  y_test  = X[cutoff:], y[cutoff:]
```

That's the whole trick. The temporal structure that ARIMA modeled internally is now encoded in the columns, and a general-purpose regressor learns the relationship. Two things carry over from lesson 01 and must never be dropped: the split is **chronological**, and every feature is **backward-looking**. Shuffle here and you've leaked the future; the reframing does not rescue you from the golden rule.

Why does this approach win so often? Because it inherits everything the ML toolbox is good at: it ingests **many features at once** (dozens of lags, rolling stats, calendar flags, *and* external drivers like price, weather, or promotions), it captures **nonlinear interactions** classical models can't, and it can be trained across **many related series at once** to share strength between them. That last point is the regime where ML pulls decisively ahead of per-series ARIMA.

## Gradient boosting: the practitioner's default

The regressor of choice is almost always a **gradient-boosting machine** — **LightGBM** or **XGBoost**. Gradient boosting builds an ensemble of shallow decision trees, each new tree correcting the errors of the ones before it. It handles nonlinearity and feature interactions automatically, is robust to feature scaling, tolerates missing values, and trains fast. LightGBM in particular is the common favorite for forecasting: it's quick and handles large feature tables gracefully.

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train)

preds = model.predict(X_test)
```

A few forecasting-specific notes. First, trees **cannot extrapolate beyond the range of the training data** — a tree's prediction is always some average of training targets it saw, so a pure lag-feature boosting model cannot forecast a value higher than anything in its training history. If your series has a strong upward trend, detrend it first (model the *differenced* series, or divide out a trend) so the boosting model works on the stationary residual. This is the same stationarity concern from lesson 01, now biting a tree model. Second, feed the model exogenous drivers when you have them — a demand model that knows next week's promotion price is far stronger than one working from lags alone. Third, gradient boosting gives point forecasts by default; for prediction intervals, train with a **quantile loss** (`objective="quantile", alpha=0.1` and `alpha=0.9` for a lower and upper band) — a topic lesson 06 returns to.

In production you rarely wire this by hand. **`mlforecast`** (Nixtla) wraps exactly this pattern — build lag/rolling features, fit a LightGBM/XGBoost, forecast — with the feature-shifting handled correctly and multi-series support built in. **`sktime`** and **`darts`** offer the same via reduction wrappers that "reduce" forecasting to regression. Knowing the manual version above means these libraries are never a black box to you.

## Multi-step forecasting: the real complication

Everything so far predicts *one step ahead* — given the past, forecast the next value. But you usually need a *horizon*: the next 7 days, the next 12 months. Forecasting multiple steps is where the interesting design decision lives, and there are two main strategies.

### Recursive (iterative) forecasting

Train a single one-step model. To forecast further, **feed its own prediction back in** as the lag feature for the next step, and repeat.

Predict day *t+1*. Now treat that prediction as if it were the real day *t+1* value, use it to build the `lag_1` feature for day *t+2*, predict *t+2*, and roll forward. One model, applied over and over.

```python
def recursive_forecast(model, history, horizon):
    preds = []
    series = history.copy()
    for _ in range(horizon):
        feats = make_features(series).iloc[[-1]].drop(columns="y")
        next_val = model.predict(feats)[0]
        preds.append(next_val)
        # append the prediction so it becomes a lag for the next step
        series = pd.concat([series, pd.Series([next_val],
                            index=[series.index[-1] + pd.Timedelta(days=1)])])
    return preds
```

- **Pro:** simple, one model, no matter how long the horizon.
- **Con:** errors compound. Step 2 is built on step 1's *prediction*, not the truth, so any error propagates and amplifies down the horizon. Recursive forecasts can drift badly on long horizons.

### Direct forecasting

Train a **separate model for each horizon step**. One model predicts *t+1*, another predicts *t+2*, another *t+3*, each learning to map today's features straight to that specific future step. To build the targets, shift the target *forward* by the horizon (the one legitimate use of a negative shift, and only for the target):

```python
# a direct model for horizon h uses y shifted forward by h as its target
y_h3 = features["y"].shift(-3)   # target = value 3 steps ahead
```

- **Pro:** no error compounding — each model predicts its step directly from real data.
- **Con:** you train and maintain *H* models for a horizon of *H*, which is costly, and the separate models can produce a slightly jagged, inconsistent forecast path since they don't share information.

### Which to use

The honest answer: try both and let backtesting (lesson 06) decide. Rules of thumb: **recursive** for short horizons and when you want one simple model; **direct** for longer horizons where error compounding hurts most. There are hybrids — **DirRec** combines them, and **multi-output** models predict the whole horizon vector at once (many deep models in lesson 05 do this natively, sidestepping the whole dilemma). The libraries expose these as options: `mlforecast` supports recursive and direct strategies with a parameter, so once you understand the tradeoff you can switch between them trivially.

## A note on validating these models

Because these are ML models, it's tempting to reach for scikit-learn's `cross_val_score` — but its default K-fold shuffles, which leaks the future. Time-series cross-validation must respect order, using `sklearn.model_selection.TimeSeriesSplit` or a proper rolling backtest. This matters enough that lesson 06 is devoted to it. For now, just internalize: the moment you're doing ML on a time series, your validation machinery needs the same time-aware discipline as your features.

## Key takeaways

- Build a lag/rolling/calendar feature table and forecasting becomes ordinary supervised regression: `X` = features, `y` = the next value.
- Gradient boosting (LightGBM/XGBoost) is the practitioner's default — it handles nonlinearity, many features, exogenous drivers, and multiple series at once.
- Trees can't extrapolate past the training range, so detrend a strongly trending series before fitting a boosting model.
- Multi-step forecasting: **recursive** feeds predictions back in (simple, one model, but errors compound) vs **direct** trains one model per horizon step (no compounding, but H models and a jagged path).
- Keep the golden rule everywhere: chronological split, backward-looking features, and time-aware cross-validation — never shuffle.

## Try it

Using your feature table from lesson 03:

1. Do a chronological train/test split and fit a `LGBMRegressor` to predict the next day. Report its test MAE and compare it to the 7-day moving-average baseline from lesson 02. Did the ML model earn its complexity?
2. Implement a 7-day **recursive** forecast and plot it against the actuals. Where along the horizon does the error start to grow, and why?
3. Build a **direct** target with `shift(-3)` and train a model for the 3-step-ahead forecast. Compare its 3-step error to the recursive model's 3-step error. Which strategy won on your series?
