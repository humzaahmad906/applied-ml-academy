# 02 — Classical Methods

Before neural networks, before gradient boosting, statisticians spent a century learning how to forecast with a handful of elegant, interpretable models. These classical methods — moving averages, exponential smoothing, and ARIMA — are not museum pieces. On a single univariate series with a few hundred observations, they routinely *beat* the fancy stuff, they run in milliseconds, and they hand you a forecast with honest uncertainty bands. Any forecasting practitioner who reaches for a transformer before trying these is doing it wrong. This lesson is the classical toolkit and, just as important, the judgment for when it's the right tool.

## Moving averages: the humblest baseline

The simplest forecast imaginable: predict the next value as the average of the last few. A **moving average** with window *k* smooths out noise by averaging the most recent *k* observations.

```python
import pandas as pd

sales = pd.Series(
    [102, 108, 115, 120, 118, 95, 88, 104, 110, 117, 123, 121, 97, 90],
    index=pd.date_range("2026-01-01", periods=14, freq="D"),
)

sales.rolling(window=3).mean().iloc[-1]   # forecast next value
# output: 102.67
```

A moving average is a *smoother*, not really a forecaster — it lags behind trends and it treats a value from *k* steps ago exactly as heavily as yesterday's value, which is rarely what you want. But it is the honest baseline every project should beat. If your deep model can't outperform a 7-day moving average, you have learned something valuable: probably that the series has little structure to exploit, or that your model has a bug.

## Exponential smoothing (ETS): recency-weighted forecasting

Exponential smoothing fixes the moving average's biggest flaw. Instead of weighting the last *k* values equally and ignoring everything older, it weights *all* past values, with weights that decay exponentially as you go back in time. Recent observations matter most; older ones fade smoothly rather than dropping off a cliff.

The simplest form, **simple exponential smoothing**, has one knob, the smoothing parameter α (between 0 and 1):

```
forecast(t+1) = α * actual(t) + (1 - α) * forecast(t)
```

A large α reacts fast to recent changes; a small α is sluggish and smooth. Simple exponential smoothing has no notion of trend or seasonality, so it forecasts a flat line — fine for a level series, useless otherwise.

The family is usually called **ETS**, for **E**rror, **T**rend, **S**easonality — the three components it can model. **Holt's method** adds a trend term; **Holt-Winters** adds seasonality on top. This gives you a model that can extrapolate a rising trend *and* repeat a yearly pattern, all from a handful of parameters.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    sales,
    trend="add",           # additive trend (Holt)
    seasonal="add",        # additive seasonality (Holt-Winters)
    seasonal_periods=7,    # weekly season
).fit()

model.forecast(3)   # forecast the next 3 days
# output:
# 2026-01-15    101.4
# 2026-01-16    107.9
# 2026-01-17    114.6
```

Use `trend="mul"` / `seasonal="mul"` when the swings grow with the level (recall additive-vs-multiplicative from lesson 01). ETS is a superb default: it's fast, it handles trend and seasonality directly, and for many business series it's genuinely hard to beat.

## ARIMA: the workhorse

**ARIMA** — AutoRegressive Integrated Moving Average — is the model that dominated forecasting for decades, and understanding its three letters teaches you how time-series models think. It is written `ARIMA(p, d, q)`.

**AR — the AutoRegressive part (order p).** Regress the series on its *own past values*. An AR(1) model says today is a weighted version of yesterday plus noise; AR(2) uses the last two days, and so on. This is autocorrelation (lesson 01) turned into an equation. *p* is how many past values you feed in.

**I — the Integrated part (order d).** This is **differencing**, exactly as in lesson 01. ARIMA assumes stationarity, so *d* is how many times you difference the series to remove trend before modeling. `d=1` handles a linear trend; `d=2` a quadratic one. After forecasting the differenced series, ARIMA integrates (cumulatively sums) back to the original scale — hence "integrated."

**MA — the Moving Average part (order q).** Confusingly, this is *not* the rolling mean from earlier. Here MA means regressing on the *past forecast errors* — the model corrects itself based on how wrong it was recently. *q* is how many past error terms to include.

Put together, `ARIMA(1,1,1)` says: difference once to get stationary, then predict using one past value and one past error. Add seasonality and you get **SARIMA**, with a second set of `(P, D, Q, s)` terms that do the same job at the seasonal lag `s` (e.g. `s=12` for monthly data with a yearly cycle).

### Choosing p and q with ACF and PACF

How do you pick *p* and *q*? The classic diagnostic is a pair of plots on the *differenced* (stationary) series: the **ACF** and the **PACF**.

- The **ACF** (autocorrelation function, from lesson 01) shows correlation at each lag, *including* indirect effects that ripple through intermediate lags.
- The **PACF** (partial autocorrelation function) shows the correlation at each lag *after removing* the effect of the shorter lags — the direct-only relationship.

The textbook reading: a PACF that cuts off sharply after lag *p* suggests an AR(p); an ACF that cuts off sharply after lag *q* suggests an MA(q).

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

diff = sales.diff().dropna()
plot_acf(diff)    # tail-off / cut-off tells you about q
plot_pacf(diff)   # cut-off after lag p suggests AR order p
```

In practice, few people read these plots by hand anymore. Automated search — `pmdarima.auto_arima` or Nixtla's `StatsForecast` `AutoARIMA` — fits many `(p,d,q)` combinations and picks the best by an information criterion (AIC/BIC, which reward fit while penalizing complexity). Still, knowing what the plots mean keeps auto-ARIMA from being a black box.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(sales, order=(1, 1, 1)).fit()
forecast = model.get_forecast(steps=3)
forecast.predicted_mean            # the point forecasts
forecast.conf_int()                # 95% prediction intervals — for free
```

Note that last line. Classical models give you **prediction intervals** almost for free, because they're built on explicit statistical assumptions about the noise. That honest quantification of uncertainty is a real advantage — many ML and deep models make you work much harder for it (lesson 06).

## Modern classical tooling

You rarely call statsmodels raw in production anymore. **Nixtla's `statsforecast`** reimplements ARIMA, ETS, and friends in a blazing-fast, parallelized package with a clean sklearn-style API, and it scales to forecasting thousands of series at once. **`sktime`** wraps the same models behind a unified interface with hyperparameter tuning. Reach for these when you move past a single series — but the model *concepts* are exactly the statsmodels ones above.

## When classical still wins

It's tempting to assume newer means better. For time series, that's often false. Classical methods win when:

- **The series is univariate and shortish.** With a few hundred observations and no exogenous inputs, ARIMA/ETS often beat deep models, which are starved for the large data they need.
- **You have many series to forecast fast.** Thousands of SKUs, each a short daily series — `statsforecast` fits them all in seconds; training a deep net per series is absurd.
- **You need calibrated uncertainty and interpretability.** Prediction intervals and named parameters beat a black box when a human must trust and act on the forecast.
- **You need a baseline.** Always. The famous **M-competitions** — large public forecasting benchmarks — repeatedly found simple methods competitive with or beating complex ones, a humbling and durable result. Every project should start with an ETS or ARIMA baseline before anything fancier earns its keep.

Deep learning earns its place on *many related series*, *rich covariates*, *long horizons*, or *cross-series patterns* — the subject of lesson 05. Until you're in that regime, the century-old models are frequently the right answer.

## Key takeaways

- Moving averages are smoothers and honest baselines, not real forecasters — but every project should beat one.
- Exponential smoothing (ETS) weights recent data more, and its Holt / Holt-Winters extensions add trend and seasonality; it's a strong, fast default.
- ARIMA(p,d,q) combines AutoRegression (past values), Integration (differencing for stationarity), and Moving-Average-of-errors; SARIMA adds a seasonal block.
- ACF/PACF plots guide the choice of q and p; in practice `auto_arima` or Nixtla `AutoARIMA` search it for you.
- Classical methods win on short univariate series, on huge fleets of series, and whenever calibrated intervals and interpretability matter — and they're always the baseline to beat.

## Try it

Using a daily series with a weekly pattern:

1. Fit a 7-day moving average and record its one-step forecast — this is your baseline.
2. Fit a Holt-Winters `ExponentialSmoothing` with `seasonal_periods=7` and forecast 7 days ahead. Does it capture the weekly shape the moving average flattened?
3. Fit an `ARIMA(1,1,1)` and print `get_forecast(7).conf_int()`. How wide are the intervals, and do they widen as you forecast further out? Explain in one sentence why they should.
