# 03 — Expectation and Variance

A distribution tells the full story of a random variable, but full stories are hard to compare. Often we want a couple of numbers that summarize it: where it's centered and how much it wobbles. Those two numbers are the **expectation** (the mean) and the **variance** (with its friendly cousin, the standard deviation). They're the workhorses of statistics and show up in nearly every ML metric.

## Expectation: the long-run average

The **expectation** of a random variable, written E[X], is its average value if you could repeat the experiment forever. It's also called the mean, often written μ.

For a discrete variable, you compute it by weighting each possible value by its probability and adding them up:

E[X] = Σ (value × probability of that value)

Take a fair die. Each face 1 through 6 has probability 1/6, so:

E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 21 / 6 = 3.5

Notice 3.5 is not a value the die can ever show. The expectation is a balance point, not a prediction of any single roll. Think of the distribution as weights on a ruler; the expectation is where the ruler balances.

```python
import numpy as np

rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=1_000_000)
print("simulated mean:", rolls.mean())   # ~3.5
```

A powerful property is **linearity of expectation**: the expected value of a sum is the sum of the expected values, always, even when the variables are related. E[X + Y] = E[X] + E[Y]. This is why the expected total of two dice is simply 3.5 + 3.5 = 7, with no messy calculation.

## Variance: how much things spread

The expectation tells you the center, but two distributions with the same center can look completely different. One might cluster tightly around it; another might swing wildly. **Variance** measures that spread.

Variance is the expected squared distance from the mean:

Var(X) = E[(X − μ)²]

We square the distances for two reasons: it makes everything positive (so distances above and below don't cancel), and it penalizes big deviations more than small ones. A point twice as far from the mean contributes four times as much.

The downside of squaring is that the units get squared too. If X is measured in dollars, variance is in dollars-squared, which is meaningless to interpret. So we usually take the square root to get back to sensible units. That square root is the **standard deviation**, written σ:

σ = √Var(X)

The standard deviation answers the intuitive question: "on a typical roll, how far off the mean am I?"

```python
import numpy as np

rng = np.random.default_rng(1)
rolls = rng.integers(1, 7, size=1_000_000)
print("variance:", rolls.var())          # ~2.92
print("std dev: ", rolls.std())           # ~1.71
```

## Reading the numbers together

Mean and standard deviation together give a quick sketch of any distribution. Consider two investment funds, both with an average annual return of 7%:

- Fund A: standard deviation 2%. Returns usually land between 5% and 9%. Steady.
- Fund B: standard deviation 15%. Returns swing from big gains to big losses. Volatile.

Same mean, very different experience. If you only looked at the average, you'd miss the entire risk story. This is a recurring lesson: **the mean alone can hide enormous differences.**

For roughly bell-shaped (normal) data, standard deviation has a handy rule of thumb: about 68% of values fall within one standard deviation of the mean, about 95% within two, and about 99.7% within three. So a value four standard deviations out is genuinely rare and worth investigating.

```python
import numpy as np

rng = np.random.default_rng(2)
x = rng.normal(loc=0, scale=1, size=1_000_000)
within_1 = np.mean(np.abs(x) < 1)
within_2 = np.mean(np.abs(x) < 2)
print(f"within 1 sd: {within_1:.3f}")     # ~0.683
print(f"within 2 sd: {within_2:.3f}")     # ~0.954
```

## A note on samples versus the truth

So far we've talked about the *true* mean and variance of a distribution. In practice we rarely know them; we only have a sample of data and estimate them. The **sample mean** is just the average of your data points. The **sample variance** is the average squared distance from the sample mean — with one quirk: we usually divide by (n − 1) instead of n. Dividing by n − 1 corrects a subtle bias, because using the sample mean (rather than the unknown true mean) slightly underestimates the spread. NumPy's `.var()` defaults to dividing by n; pass `ddof=1` for the n − 1 version. The gap shrinks as your sample grows, but for small samples it matters.

## Why this matters for ML

These ideas are everywhere in machine learning. **Mean squared error**, the most common regression loss, is literally an expectation of squared errors — variance in disguise. When we report a model's accuracy, we're estimating an expectation. The **bias-variance tradeoff**, one of the central concepts in ML, uses variance to describe how much a model's predictions bounce around when you retrain it on different data. Batch normalization standardizes activations using their mean and variance. You can't go far without these two quantities.

## Key takeaways

- Expectation E[X] is the long-run average, a balance point that need not be an attainable value.
- Linearity of expectation lets you add expected values freely, even for dependent variables.
- Variance measures spread as expected squared distance from the mean; standard deviation is its square root, in the original units.
- Equal means can hide very different spreads — always look at both.
- Use ddof=1 for sample variance from data; MSE and the bias-variance tradeoff are built directly on these concepts.

## Try it

Generate 100,000 samples from `rng.normal(loc=100, scale=15, size=...)` (a classic IQ-style distribution). Compute the sample mean and standard deviation and confirm they're close to 100 and 15. Then compute the fraction of samples above 130 (two standard deviations up). Does it match the roughly 2.5% you'd expect from the 95%-within-two-sd rule? Finally, try `ddof=0` versus `ddof=1` on a tiny sample of just 5 points and see how much the variance estimate changes.
