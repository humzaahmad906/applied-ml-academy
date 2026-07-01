# 07 — Hypothesis Testing Basics

You changed the checkout button from blue to green, and sales went up 3%. Real improvement, or just luck? You retrained a model and accuracy rose from 91.2% to 91.5%. Meaningful, or noise? **Hypothesis testing** is the framework for answering exactly these questions: it tells you whether an observed effect is bigger than what random chance could plausibly produce. It's also one of the most misunderstood topics in statistics, so we'll be careful.

## The core idea

Hypothesis testing starts from a deliberately skeptical stance. You assume, for the sake of argument, that *nothing interesting is happening* — that any difference you saw is pure randomness. This skeptical assumption is the **null hypothesis**, written H₀. Its rival, the thing you actually suspect, is the **alternative hypothesis**, H₁.

For the green button: H₀ says "green and blue convert at the same rate; the 3% is noise." H₁ says "green really does convert differently."

The strategy is proof by contradiction, softened for a random world. You ask: *if the null hypothesis were true, how surprising would my data be?* If your data would be wildly unlikely under H₀, that's evidence against H₀. If your data is perfectly consistent with H₀, you have no reason to abandon it.

## The p-value

The tool that measures "how surprising" is the **p-value**. Here is its precise definition, which is worth memorizing because nearly every misuse comes from getting it wrong:

> The p-value is the probability of observing data at least as extreme as what you actually saw, *assuming the null hypothesis is true.*

A small p-value means your data would be rare if nothing were going on — so maybe something *is* going on. A large p-value means your data is unremarkable under H₀, giving you no reason to reject it.

By convention, people often use a threshold (called α, alpha) of 0.05. If p < 0.05, they "reject the null" and call the result **statistically significant**. But 0.05 is an arbitrary convention, not a law of nature, and treating it as a magic line causes real harm.

Let's make it concrete with a simulation. Suppose a coin gave 60 heads in 100 flips. Is it biased, or could a fair coin do that?

```python
import numpy as np

rng = np.random.default_rng(0)
observed = 60

# Simulate the null world: a truly fair coin, 100 flips, many times.
null_trials = rng.binomial(n=100, p=0.5, size=1_000_000)

# p-value: how often does a fair coin give a result at least as extreme?
p_value = np.mean(np.abs(null_trials - 50) >= abs(observed - 50))
print("p-value:", round(p_value, 4))   # ~0.057
```

The p-value comes out around 0.057 — just above 0.05. A fair coin produces 60-or-more-extreme results about 5.7% of the time, so 60 heads is suggestive but not conclusive. Notice how the simulation *is* the definition: we built the null world, then counted how often it matched or beat our data.

## What a p-value is not

This is where people go wrong, so read carefully. Each of these is a genuine, common misreading:

**A p-value is NOT the probability that the null hypothesis is true.** p = 0.03 does not mean "3% chance the coin is fair." It's a statement about the data given the hypothesis, not the hypothesis given the data. (Remember the Bayes module — reversing a conditional changes its meaning entirely.)

**A p-value is NOT the probability your result happened by chance.** It already *assumes* chance (the null) and measures how well your data fits that assumption.

**Statistical significance is NOT practical importance.** With a huge sample, a trivial, useless difference can be highly significant. A drug that extends life by 20 minutes could have p < 0.001 in a large enough trial. Significant means "probably not zero," not "big enough to matter."

**A non-significant result does NOT prove the null is true.** Failing to find evidence of an effect is not evidence there's no effect. Maybe your sample was just too small to detect it. Absence of evidence isn't evidence of absence.

## Two ways to be wrong

Because we're deciding under uncertainty, we can err in two directions:

- **Type I error (false positive):** rejecting H₀ when it's actually true. You conclude the button matters when it doesn't. The α threshold is exactly your tolerated Type I error rate — pick α = 0.05 and you'll wrongly reject a true null 5% of the time.
- **Type II error (false negative):** failing to reject H₀ when it's actually false. There's a real effect and you miss it. The chance of *catching* a real effect is called **power**, and bigger samples give you more of it.

There's a tension here: lowering α to avoid false positives makes false negatives more likely, and vice versa. Choosing the balance is a judgment call about which mistake is costlier in your situation.

## A caution on p-hacking

If you test twenty different button colors and one comes back significant at p < 0.05, you haven't found something real — you've found that testing many things guarantees a few false positives. Running many tests and reporting only the significant ones is called **p-hacking**, and it's how a lot of irreproducible results get published. If you run multiple tests, you must adjust for it (for example, by tightening your threshold), or your "discoveries" are just the tail of randomness dressed up as insight.

## Why this matters for ML

Every time you compare two models, tune a hyperparameter, or run an A/B test on a deployed system, you're implicitly doing hypothesis testing. A held-out accuracy of 91.5% versus 91.2% may be inside the noise band the CLT predicts, and calling it an improvement would be a Type I error. Understanding p-values keeps you honest: it stops you from chasing differences that are indistinguishable from luck, and it reminds you that with enough data even meaningless gaps become "significant." The disciplined question is always both "is it significant?" *and* "is it big enough to care about?"

## Key takeaways

- Hypothesis testing assumes a skeptical null (nothing happening) and asks whether the data is too surprising to fit it.
- A p-value is the probability of data at least as extreme as yours, *assuming the null is true* — nothing more.
- It is NOT the probability the null is true, and significance is NOT the same as practical importance.
- Type I errors are false positives (rate = α); Type II errors are false negatives (miss = low power); the two trade off.
- p-hacking (testing many things, reporting the winners) manufactures false positives; comparing ML models is hypothesis testing whether you call it that or not.

## Try it

Simulate an A/B test with *no real effect*: both variants convert at exactly 10%. Draw 1,000 users per variant with `rng.random(1000) < 0.10`, and compute the difference in conversion rates. Repeat this 10,000 times and look at the distribution of differences — it should center on zero but spread out by chance. Now count what fraction of these "no-effect" experiments show a difference large enough to look impressive (say, one variant beating the other by 2 percentage points or more). That fraction is how often pure noise would fool you — a visceral demonstration of why thresholds and sample sizes matter.
