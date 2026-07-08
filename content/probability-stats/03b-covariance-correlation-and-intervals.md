# 03b — Covariance, Correlation, and Confidence Intervals

The last lesson summarized a *single* variable with its mean and variance. But data rarely comes one column at a time. You have height *and* weight, ad spend *and* revenue, a dozen features *and* a target. The natural next question is: when one number moves, does the other move with it? That relationship is what **covariance** and **correlation** capture. And once we can measure a quantity, we'll want to say how *sure* we are about it — which brings us to **confidence intervals**.

## Covariance: do two variables move together?

Variance asked how far a single variable strays from its own mean. Covariance asks something similar about a *pair*: when X is above its mean, does Y tend to be above its mean too?

The formula is a direct extension of variance. Instead of squaring one variable's distance from its mean, we multiply the two variables' distances together:

Cov(X, Y) = E[(X − μx)(Y − μy)]

Think through the sign. If X being high tends to come with Y being high (and X low with Y low), then both parentheses usually share a sign, their product is usually positive, and the average is positive. If X being high comes with Y being *low*, the product is usually negative, and covariance is negative. If there's no pattern, the positives and negatives cancel and covariance sits near zero.

(Notice that Cov(X, X) = E[(X − μx)²] is just the variance. Variance is the special case of covariance with itself.)

```python
import numpy as np

rng = np.random.default_rng(0)
height = rng.normal(170, 10, size=1000)
weight = 0.5 * height + rng.normal(0, 5, size=1000)  # weight rises with height

C = np.cov(height, weight)
print(C)
# ~ [[95.5  50.69]
#    [50.69 52.99]]
print("cov(height, weight):", round(C[0, 1], 2))   # ~50.69
```

`np.cov` returns a **covariance matrix**: the diagonal holds each variable's variance, and the off-diagonal holds the covariance between them. The positive off-diagonal confirms taller people tend to weigh more.

Here's the catch. That number, 50.69, has units of centimeters times kilograms. Is 50 a strong relationship or a weak one? You genuinely cannot tell. Covariance's magnitude is tangled up with the scales of both variables — measure height in millimeters instead and the number balloons, even though the relationship is identical. Sign is meaningful; magnitude is not comparable across different pairs.

## Correlation: covariance you can actually compare

To fix the scale problem, we divide the covariance by both standard deviations. This strips out the units entirely and squeezes the result into a fixed range from −1 to +1:

ρ = Cov(X, Y) / (σx · σy)

This is the **Pearson correlation coefficient**, the one people almost always mean when they say "correlation." Now the number interprets itself:

- **+1**: a perfect straight-line relationship going up.
- **−1**: a perfect straight-line relationship going down.
- **0**: no *linear* relationship.
- Values in between measure how tightly the points hug a straight line. ±0.9 is strong, ±0.3 is weak.

```python
import numpy as np

rng = np.random.default_rng(1)
x = rng.normal(0, 1, size=1000)
y_pos  =  2 * x + rng.normal(0, 0.5, size=1000)
y_neg  = -3 * x + rng.normal(0, 0.5, size=1000)
y_none = rng.normal(0, 1, size=1000)

print("positive:", round(np.corrcoef(x, y_pos)[0, 1], 3))   # ~0.969
print("negative:", round(np.corrcoef(x, y_neg)[0, 1], 3))   # ~-0.987
print("none:    ", round(np.corrcoef(x, y_none)[0, 1], 3))  # ~0.017
```

Notice that `y_neg` has a *steeper* slope than `y_pos` but a correlation of nearly the same magnitude. Correlation measures how *consistent* the linear relationship is, not how steep. Slope and correlation are different things.

One crucial limitation: correlation only sees *straight-line* relationships. A perfect U-shape (y = x²) can have a correlation near zero, because as much of it goes up as goes down. Zero correlation means no *linear* trend, not no relationship at all. Always plot your data.

### Correlation is not causation

This is the caution everyone quotes and few internalize. Ice cream sales correlate with drownings, but ice cream doesn't drown people — hot weather independently drives both. A variable lurking in the background (here, temperature) can manufacture a correlation between two things that have no direct link.

Correlation is a statement about patterns in observed data. Causation is a statement about what happens when you *intervene* and change one thing. The only reliable way to establish causation is a controlled experiment — randomly assigning who gets the treatment so lurking variables can't sneak in. That machinery (A/B tests, randomization, control groups) is the whole subject of the experimentation course; keep this distinction in mind whenever you're tempted to read a correlation as a cause.

## Why this matters for ML

Correlation is not a side topic in machine learning — it's woven through the daily workflow.

**Correlated features hurt linear models.** Suppose two features carry almost the same information — say "price in dollars" and "price in euros." A linear model wants to assign a coefficient to each, but it can't decide how to split the credit between two nearly identical columns. The coefficients become unstable: huge, opposite-signed, and wildly sensitive to tiny changes in the data. This is **multicollinearity**, and it makes a model's weights impossible to interpret and its predictions fragile.

```python
import numpy as np

rng = np.random.default_rng(2)
n = 500
f1 = rng.normal(0, 1, n)
f2 = f1 + rng.normal(0, 0.1, n)   # nearly a duplicate of f1
f3 = rng.normal(0, 1, n)          # independent

print(np.round(np.corrcoef([f1, f2, f3]), 2))
# ~ [[1.   1.   0.02]
#    [1.   1.   0.01]
#    [0.02 0.01 1.  ]]
```

That 1.00 between f1 and f2 is a red flag. In practice you'd drop one of them.

**Correlation heatmaps in EDA.** Early in any project, computing the full correlation matrix and viewing it as a color-coded **heatmap** is one of the fastest ways to understand a dataset. You scan for two things: features strongly correlated *with the target* (promising predictors) and features strongly correlated *with each other* (redundant, candidates for removal). This is a standard opening move in **feature selection** — trimming redundant inputs so the model is simpler, faster, and more stable.

## Confidence intervals: how sure are you about a number?

Every statistic you compute from data — a mean, an accuracy, a correlation — is an estimate from *one* sample. Run the experiment again with fresh data and you'd get a slightly different number. A **confidence interval (CI)** puts a range around your estimate to communicate that wobble: instead of "accuracy is 0.80," you report "accuracy is 0.80, 95% CI [0.79, 0.81]."

First, what a 95% CI does **not** mean: it does *not* say there's a 95% probability the true value is inside this particular interval. The true value is fixed; it's either in or out. The honest interpretation is about the *procedure*: if you repeated the whole sampling-and-interval process many times, about 95% of the intervals you build would contain the true value. Loosely and usefully, it's a range of plausible values for the quantity you're estimating.

The classic way to build a CI for a mean leans on the **standard error** — the standard deviation of the sample mean, which is σ/√n. The Central Limit Theorem (lesson 05) tells us the sample mean is approximately normal, so roughly 95% of the time it lands within about 1.96 standard errors of the truth. That gives the interval mean ± 1.96 × (s/√n). Notice the √n: quadruple your data and the interval only halves. Precision is expensive.

**Why a CI beats a bare p-value for comparing model runs.** When you compare model A against model B, a p-value collapses everything into a single yes/no verdict at some threshold. A confidence interval tells you *how much* better and *how uncertain* you are. "Model B is +2.0% accuracy, 95% CI [+0.1%, +3.9%]" says the gain is probably real but could be tiny. "+2.0%, CI [−1.5%, +5.5%]" says you genuinely can't tell yet — the interval straddles zero. That's far more actionable than "p = 0.04."

When the formula is awkward (or you don't trust the normality assumption), the **bootstrap** gives you a CI with almost no math. You resample your data *with replacement* many times, recompute the statistic on each resample, and read the interval straight off the percentiles of those results.

```python
import numpy as np

rng = np.random.default_rng(42)
scores = rng.normal(0.80, 0.05, size=200)   # per-example scores from one eval run

boot_means = np.array([
    rng.choice(scores, size=len(scores), replace=True).mean()
    for _ in range(10_000)
])
lo, hi = np.percentile(boot_means, [2.5, 97.5])

print("estimate:", round(scores.mean(), 3))       # ~0.798
print("95% CI:  ", round(lo, 3), round(hi, 3))     # ~0.792 0.804
```

The bootstrap interval here (~[0.792, 0.804]) matches what the standard-error formula gives, which is reassuring — but the bootstrap got there without assuming any particular distribution, and it works just as easily for medians, correlations, or any metric you like.

## Key takeaways

- Covariance measures whether two variables move together; its *sign* is meaningful but its magnitude depends on units, so it isn't comparable across pairs.
- Correlation is covariance normalized into [−1, +1], making strength directly interpretable; it captures only *linear* relationships, so always plot too.
- Correlation is not causation — lurking variables fake relationships; only controlled experiments establish cause (see the experimentation course).
- In ML, highly correlated features cause multicollinearity in linear models; correlation heatmaps are a core EDA and feature-selection tool.
- A confidence interval is a range of plausible values, not a probability about one interval; it's more informative than a bare p-value when comparing model runs, and the bootstrap builds one with almost no assumptions.

## Try it

Generate two related variables: `x = rng.normal(0, 1, 500)` and `y = x + rng.normal(0, scale, 500)`, and watch the correlation fall as you crank `scale` from 0.1 up to 5 — this shows noise washing out a relationship. Next, build `y = x**2` and confirm the correlation is near zero even though the relationship is perfect; plot it to see why. Finally, take a small sample of 30 numbers, bootstrap 10,000 resampled means, and compare the percentile CI you get against the standard-error formula mean ± 1.96 × (s/√n). How close are they? Now shrink the sample to 10 points and watch the interval widen.
