# 02 — Random Variables and Distributions

In the last module we talked about events and their probabilities. Now we introduce a tool that makes probability far more powerful: the **random variable**. It lets us attach numbers to random outcomes, and once outcomes are numbers, we can add them, average them, and plot them. That is the bridge from coin flips to real data.

## What is a random variable?

A **random variable** is a number whose value depends on the outcome of a random process. When you roll a die, the result is a random variable that can be 1, 2, 3, 4, 5, or 6. When you measure the height of a randomly chosen person, that height is a random variable too.

We usually write random variables with capital letters like X. A specific value it takes is written with a lowercase letter, like x. So "P(X = 3)" means "the probability that the random variable X takes the value 3."

Random variables come in two flavors, and the distinction matters throughout ML.

- **Discrete** random variables take values from a countable set: 0, 1, 2, ... Examples: the number of heads in ten flips, the number of clicks on an ad, a class label.
- **Continuous** random variables take any value in a range: heights, temperatures, pixel intensities, model weights. Between any two values there are infinitely many others.

## Distributions: the full picture

A **distribution** describes how probability is spread across all the values a random variable can take. It's the complete story of the random variable.

For a discrete variable, the distribution is a list of probabilities, one per value, called a **probability mass function** (PMF). Each probability is between 0 and 1, and they all sum to 1. For a fair die the PMF assigns 1/6 to each face.

For a continuous variable we can't assign probability to single points (there are infinitely many, so each would be zero). Instead we use a **probability density function** (PDF). Density is not probability directly; probability is the *area under the curve* over a range. The total area under a PDF is always 1.

```python
import numpy as np

rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=100_000)
values, counts = np.unique(rolls, return_counts=True)
print(dict(zip(values, np.round(counts / counts.sum(), 3))))  # each ~0.167
```

## Common distributions you'll meet

A handful of distributions show up again and again. Recognizing them is like recognizing chords in music.

**Bernoulli.** The simplest: a single yes/no trial with probability p of success. A coin flip with p = 0.5, or "did the user click?" with p = 0.02. It takes value 1 with probability p and 0 with probability 1 − p.

**Binomial.** Count the successes in n independent Bernoulli trials, each with probability p. "How many heads in 10 flips?" is Binomial(n=10, p=0.5). Its values run from 0 to n, and the middle values are most likely.

**Uniform.** Every value in a range is equally likely. A fair die is discrete uniform; `rng.random()` gives a continuous uniform on [0, 1]. Useful as a baseline and for generating other distributions.

**Normal (Gaussian).** The famous bell curve, described by two numbers: its center (mean, μ) and its spread (standard deviation, σ). It's symmetric, and most of its probability sits within a few standard deviations of the mean. It appears everywhere — measurement noise, aggregated effects, model initializations — and the next-but-one module explains why it's so unavoidable.

**Poisson.** Counts of rare events over a fixed interval: emails per hour, defects per batch. It has a single parameter, the average rate.

```python
import numpy as np

rng = np.random.default_rng(1)

bernoulli = rng.random(5) < 0.3            # True/False with p=0.3
binomial  = rng.binomial(n=10, p=0.5, size=5)
normal    = rng.normal(loc=0, scale=1, size=5)
poisson   = rng.poisson(lam=3, size=5)

print("bernoulli:", bernoulli.astype(int))
print("binomial: ", binomial)
print("normal:   ", np.round(normal, 2))
print("poisson:  ", poisson)
```

## Reading a distribution

The single most useful skill is being able to look at a distribution and describe its shape. Three questions cover most cases:

1. **Where is it centered?** This is the typical value.
2. **How spread out is it?** Tight distributions are predictable; wide ones are uncertain.
3. **Is it symmetric or skewed?** Income, for instance, is right-skewed: most people cluster low, but a long tail of high earners pulls the average up.

You can build intuition by simulating and looking at a histogram. Here we compare a symmetric normal against a right-skewed distribution:

```python
import numpy as np

rng = np.random.default_rng(2)
symmetric = rng.normal(loc=50, scale=10, size=100_000)
skewed    = rng.exponential(scale=10, size=100_000)

for name, data in [("symmetric", symmetric), ("skewed", skewed)]:
    print(f"{name:9s} mean={data.mean():6.2f}  median={np.median(data):6.2f}")
```

For the symmetric data the mean and median nearly match. For the skewed data the mean sits noticeably above the median — a fingerprint of the long right tail.

## Why this matters for ML

Models make assumptions about distributions all the time. Linear regression assumes normally distributed noise. A classifier's output layer produces a distribution over classes. When you standardize features to mean 0 and standard deviation 1, you're reshaping their distribution to help training. Choosing the right distribution to model your data — counts with Poisson, binary outcomes with Bernoulli — is often the difference between a model that fits and one that fights the data.

## Key takeaways

- A random variable attaches a number to a random outcome; it's discrete or continuous.
- A distribution describes how probability spreads over all possible values (PMF for discrete, PDF for continuous).
- Bernoulli, Binomial, Uniform, Normal, and Poisson cover a huge range of real situations.
- Describe any distribution by its center, spread, and skew.
- ML models are built around distributional assumptions, so knowing the common shapes pays off constantly.

## Try it

Use `rng.binomial(n=20, p=0.5, size=100_000)` to simulate the number of heads in 20 coin flips, repeated 100,000 times. Print the fraction of runs that produced exactly 10 heads, and the fraction that produced 15 or more. Then increase n to 200 and rerun. As n grows, what shape does the histogram of head counts start to resemble? (Keep that observation in mind for the Central Limit Theorem module.)
