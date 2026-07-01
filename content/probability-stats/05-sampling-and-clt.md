# 05 — Sampling and the Central Limit Theorem

We almost never get to see an entire population. We can't survey every voter, weigh every apple, or measure every user. Instead we take a **sample** and use it to say something about the whole. This module explains how samples behave, why averages are so trustworthy, and one of the most beautiful results in all of statistics: the Central Limit Theorem.

## Populations and samples

The **population** is the complete set of things you care about — every possible measurement. Its true mean and variance are fixed numbers, usually unknown. A **sample** is a subset you actually observe, drawn at random from the population.

The core problem of statistics is inference: using the sample to estimate properties of the population. If you weigh 100 apples and their average weight is 150 grams, you'd guess the population average is around 150 grams. But you'd also expect to be a little off, because your particular 100 apples are just one of countless possible samples.

That "how far off might I be?" question is where things get interesting.

## The sampling distribution

Here's the mental shift that unlocks everything. Imagine repeating your study many times: draw a sample of 100 apples, compute the mean, write it down; draw another 100, compute the mean, write it down; and so on. Those means won't all be identical — each sample is a bit different — so they form their own distribution. This is the **sampling distribution of the mean**.

The sample mean is itself a random variable. It has its own center and its own spread, and understanding that spread is the key to knowing how much to trust a single estimate.

Two facts about the sampling distribution of the mean:

1. **It's centered on the true population mean.** On average, the sample mean gets it right. It's an unbiased estimate.
2. **Its spread shrinks as the sample grows.** Specifically, the standard deviation of the sample mean equals the population standard deviation divided by √n. This quantity is called the **standard error**.

That √n is why bigger samples help — but with diminishing returns. To halve your uncertainty you must *quadruple* your sample size.

```python
import numpy as np

rng = np.random.default_rng(0)
population_sd = 20

for n in [10, 40, 160, 640]:
    means = [rng.normal(150, population_sd, size=n).mean() for _ in range(10_000)]
    print(f"n={n:4d}  std error (observed): {np.std(means):.2f}  "
          f"predicted: {population_sd / np.sqrt(n):.2f}")
```

The observed spread of the sample means matches `population_sd / √n` closely, and it keeps shrinking as n grows.

## The Central Limit Theorem

Now the headline result. The **Central Limit Theorem** (CLT) says: no matter what shape the original population has, the sampling distribution of the mean becomes approximately **normal** (bell-shaped) as the sample size grows.

Read that again, because it's remarkable. The population can be lopsided, spiky, uniform, bimodal — almost anything. But average enough of its values together and the averages pile up into a smooth bell curve. The messiness of the individual data washes out.

Let's see it happen with a deliberately un-bell-shaped population: a coin-flip-like distribution that's only ever 0 or 1.

```python
import numpy as np

rng = np.random.default_rng(1)
# Population: 0 or 1, wildly non-normal
def sample_means(n, reps=50_000):
    data = rng.random((reps, n)) < 0.3   # each entry 0/1 with p=0.3
    return data.mean(axis=1)

for n in [1, 2, 5, 30, 100]:
    m = sample_means(n)
    print(f"n={n:3d}  mean of means={m.mean():.3f}  spread={m.std():.3f}")
```

With n = 1 the "means" are just the raw 0s and 1s — two spikes, nothing bell-shaped. By n = 30 the histogram of means is a tidy bell centered on 0.3. If you plotted these you'd watch a bimodal mess morph into a Gaussian purely by averaging.

A common rule of thumb is that n = 30 is "enough" for the CLT to kick in, though very skewed populations may need more, and near-normal ones need less. The theorem is asymptotic — it improves as n grows — so treat 30 as a soft guideline, not a magic threshold.

## Why the normal shows up everywhere

The CLT explains the mystery from the distributions module: why is the normal distribution so ubiquitous in nature and in ML? Because so many real quantities are *sums or averages* of many small, independent effects. A person's height is influenced by many genes and environmental factors added together. Measurement noise is many tiny errors combining. Each of those is an averaging process, and the CLT says averaging produces normality. The bell curve isn't a coincidence; it's a mathematical inevitability whenever things add up.

## Why this matters for ML

Sampling and the CLT sit underneath how we evaluate models. When you measure accuracy on a test set, that's a sample estimate, and it has a standard error — which is why two models differing by 0.2% might not be meaningfully different at all. Confidence intervals and the hypothesis tests in the next module all lean on the CLT to justify treating estimates as normally distributed. In training, **stochastic gradient descent** estimates the true gradient from a mini-batch; that estimate is a sample mean, its noise shrinks like √(batch size), and the CLT is exactly why larger batches give smoother, more normal gradient noise. Understanding sampling variability is what separates "my model improved" from "my measurement wiggled."

## Key takeaways

- A sample is a random subset used to infer properties of a full population.
- The sample mean is itself random; its distribution is the sampling distribution.
- The sample mean is centered on the truth, with spread (standard error) equal to σ/√n — so uncertainty shrinks slowly with sample size.
- The Central Limit Theorem: averages become approximately normal regardless of the population's shape, roughly by n = 30.
- This is why the normal distribution is everywhere, and why model metrics and SGD gradients have predictable, normal-ish noise.

## Try it

Pick a wildly non-normal population — try `rng.exponential(scale=1.0, size=n)`, which is heavily right-skewed. For each sample size n in {2, 10, 50, 200}, draw 50,000 samples, compute their means, and print the mean and standard deviation of those sample means. Confirm two things: the standard deviation shrinks like 1/√n, and the distribution of means looks more and more symmetric as n grows. If you can, plot a histogram at n = 2 and n = 200 to watch the bell emerge.
