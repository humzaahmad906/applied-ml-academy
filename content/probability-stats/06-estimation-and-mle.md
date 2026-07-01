# 06 — Estimation and Maximum Likelihood

We've seen that data comes from distributions. But in real life we don't know the distribution's parameters — the true mean, the true success rate, the true spread. We have to *estimate* them from data. This module is about how estimation works, and about the single most important estimation principle in machine learning: **maximum likelihood**. If you understand MLE, you understand where most loss functions actually come from.

## The estimation problem

Suppose you have a coin and you don't know if it's fair. You flip it 20 times and get 13 heads. What's your best guess for p, the true probability of heads?

Your gut says 13/20 = 0.65, and your gut is right. But *why* is it right? What principle makes 0.65 a better guess than 0.6 or 0.7? Answering that question carefully leads straight to maximum likelihood.

An **estimator** is any recipe for turning data into a guess about a parameter. "Take the fraction of heads" is an estimator for p. "Take the sample average" is an estimator for a population mean. We want estimators that are, on average, correct (unbiased) and that don't bounce around too much (low variance) — the same mean-and-spread lens from earlier modules, now applied to our guesses themselves.

## Likelihood: how well does a parameter explain the data?

Here is the central idea. Pick a candidate value for the parameter — say, p = 0.5. Ask: *if that value were true, how probable would my observed data be?* That probability, viewed as a function of the parameter, is called the **likelihood**.

For our coin, the data is "13 heads in 20 flips." If p = 0.5, that outcome has a certain probability. If p = 0.65, it has a different (higher) probability. If p = 0.9, it's quite unlikely — 13 heads is too few for such a biased coin. We can compute the likelihood for every candidate p and see which one makes the data look most plausible.

```python
import numpy as np
from scipy.stats import binom

heads, flips = 13, 20
ps = np.linspace(0.01, 0.99, 99)
likelihoods = binom.pmf(heads, flips, ps)

best_p = ps[np.argmax(likelihoods)]
print("p that maximizes likelihood:", round(best_p, 2))   # ~0.65
```

The candidate p that makes the observed data most probable is exactly 0.65 — the fraction of heads. That's not a coincidence.

## Maximum likelihood estimation

**Maximum likelihood estimation** (MLE) is the principle: choose the parameter values that make your observed data as probable as possible. You're asking, "which settings of the world would most plausibly have produced what I saw?" and picking those.

Formally, you write down the likelihood as a function of the parameters, then find the parameter values that maximize it. For the coin, MLE gives p = (heads / total), the intuitive answer. For a normal distribution, MLE of the mean turns out to be the sample average, and MLE of the variance is the sample variance. MLE tends to recover the estimators you'd have guessed — which is reassuring, and also explains *why* those familiar formulas are the right ones.

## The log-likelihood trick

Likelihoods are products of many small probabilities (one per data point), and multiplying hundreds of numbers below 1 quickly underflows to zero and is awkward to differentiate. The standard fix is to maximize the **log-likelihood** instead. Because the logarithm is increasing, whatever maximizes the likelihood also maximizes its log — but logs turn products into sums, which are numerically stable and easy to work with.

log L = log(p₁ × p₂ × ... × pₙ) = log p₁ + log p₂ + ... + log pₙ

```python
import numpy as np

rng = np.random.default_rng(0)
data = rng.normal(loc=5.0, scale=2.0, size=1000)   # true mean 5

# Log-likelihood of the data for a candidate mean (fixed sd=2)
def log_likelihood(mu, sd=2.0):
    return np.sum(-0.5 * ((data - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi)))

mus = np.linspace(3, 7, 400)
lls = [log_likelihood(m) for m in mus]
print("MLE of mean:", round(mus[np.argmax(lls)], 3))   # ~5.0, matches data.mean()
print("sample mean:", round(data.mean(), 3))
```

The mean that maximizes the log-likelihood matches the sample average, just as the theory promises.

## From likelihood to loss functions

This is the punchline that ties the whole course together. **Minimizing a loss function is usually maximizing a likelihood in disguise.** Two examples you'll meet constantly:

- Assume your regression targets are the true values plus normal noise. Writing out the log-likelihood and simplifying, maximizing it is *exactly* the same as minimizing **mean squared error**. MSE is the maximum likelihood estimator under Gaussian noise.
- Assume your classification labels are Bernoulli outcomes. The log-likelihood, negated, is precisely the **cross-entropy loss**. Minimizing cross-entropy is maximum likelihood for a classifier.

So when a neural network trains by minimizing cross-entropy, it is doing maximum likelihood estimation: adjusting its parameters until the observed labels are as probable as the model can make them. Every gradient step nudges the parameters toward the values that best explain the training data. The loss function isn't arbitrary — it's the negative log-likelihood of a specific probabilistic assumption about your data.

## Key takeaways

- Estimation turns observed data into guesses about unknown parameters; good estimators are unbiased with low variance.
- The likelihood asks: how probable is my data if this parameter value were true?
- Maximum likelihood estimation picks the parameters that make the observed data most probable — and usually reproduces the intuitive formulas.
- Work with the log-likelihood: it turns fragile products into stable sums without changing the maximizer.
- MSE and cross-entropy are negative log-likelihoods, so training a model by minimizing loss is maximum likelihood in disguise.

## Try it

Simulate 200 flips of a coin with a true p of 0.35 using `rng.random(200) < 0.35`. Count the heads. Then, over a grid of candidate p values from 0.01 to 0.99, compute the log-likelihood of your observed head count using `scipy.stats.binom.logpmf`, and find the p that maximizes it. Confirm it matches heads/200. Bonus: repeat the whole experiment 1,000 times and look at the spread of your MLE estimates — that spread is the estimator's variance, and it should shrink if you raise the number of flips.
