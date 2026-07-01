# 04 — Bayes' Theorem

Bayes' theorem is a rule for updating your beliefs when new evidence arrives. It sounds abstract, but it captures something you do intuitively every day: you start with a hunch, you see some data, and you revise. What makes Bayes valuable is that it tells you *exactly* how much to revise, and the answer is often startlingly different from what intuition suggests.

## The setup

Recall conditional probability from the first module: P(A | B) is the probability of A given that B happened. Bayes' theorem connects P(A | B) to the reverse, P(B | A). That reversal is the whole trick, because often one direction is easy to know and the other is what you actually want.

The theorem states:

P(A | B) = P(B | A) × P(A) / P(B)

Each piece has a name worth learning:

- **P(A)** is the **prior**: your belief in A before seeing evidence.
- **P(B | A)** is the **likelihood**: how probable the evidence is if A is true.
- **P(A | B)** is the **posterior**: your updated belief in A after seeing evidence B.
- **P(B)** is a normalizing constant that makes the probabilities add up correctly.

In plain words: **posterior ∝ likelihood × prior.** Start with what you believed, scale it by how well the new evidence fits, and renormalize.

## Why the reversal matters

Here's the key insight people miss. P(evidence | disease) is not the same as P(disease | evidence). A test being positive when you're sick tells you the test works; it does not directly tell you how likely you are to be sick when the test is positive. Confusing these two is called the **base rate fallacy**, and Bayes' theorem is the cure.

## A worked example: the medical test

This is the classic example, and it will surprise you. Consider a disease and a test for it:

- **The prior.** 1% of people have the disease. So P(disease) = 0.01.
- **The likelihood (sensitivity).** If you have the disease, the test is positive 99% of the time. P(positive | disease) = 0.99.
- **The false positive rate.** If you don't have the disease, the test is still positive 5% of the time. P(positive | no disease) = 0.05.

You take the test and it comes back positive. What's the probability you actually have the disease?

Most people guess around 95%, reasoning "the test is 99% accurate." The real answer is far lower. Let's compute it.

First, the denominator P(positive) — the overall chance of a positive result — combines both ways a positive can happen:

P(positive) = P(pos | disease)·P(disease) + P(pos | no disease)·P(no disease)
P(positive) = 0.99 × 0.01 + 0.05 × 0.99 = 0.0099 + 0.0495 = 0.0594

Now apply Bayes:

P(disease | positive) = (0.99 × 0.01) / 0.0594 = 0.0099 / 0.0594 ≈ 0.167

Only about **17%**. Even after a positive result on a "99% accurate" test, you probably don't have the disease. Why? Because the disease is rare. In a population of 10,000, only 100 have it (99 test positive), while 9,900 don't — and 5% of those, 495 people, also test positive. So among all 594 positives, only 99 are truly sick. The rarity of the disease overwhelms the accuracy of the test.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1_000_000

has_disease = rng.random(n) < 0.01
# Test result depends on true status
positive = np.where(
    has_disease,
    rng.random(n) < 0.99,   # sensitivity
    rng.random(n) < 0.05,   # false positive rate
)

positives = has_disease[positive]           # true status of everyone who tested positive
print("P(disease | positive):", positives.mean())   # ~0.167
```

The simulation lands on 0.167, matching the hand calculation. Whenever the algebra feels slippery, simulating like this is a reliable sanity check.

## Updating again and again

Bayes really shines when evidence arrives in stages. Today's posterior becomes tomorrow's prior. Suppose after that first positive test (posterior ≈ 0.167), you take a second, independent test and it's also positive. Now your prior is 0.167, not 0.01:

P(disease) = 0.167
P(positive) = 0.99 × 0.167 + 0.05 × 0.833 ≈ 0.165 + 0.042 = 0.207
P(disease | second positive) = 0.165 / 0.207 ≈ 0.799

Two positive tests push you to about 80%. Each piece of evidence nudges the belief; they accumulate. This sequential updating is the heart of Bayesian reasoning, and it mirrors how a well-calibrated mind should change its views as data piles up.

## Why this matters for ML

Bayes' theorem underlies a whole family of models. **Naive Bayes classifiers** use it directly to turn P(words | spam) into P(spam | words) — exactly the reversal we did above. More broadly, the framing of "prior, likelihood, posterior" is how Bayesian machine learning treats model parameters: you have prior beliefs about weights, data provides a likelihood, and training produces a posterior. Even when a method isn't explicitly Bayesian, the base rate lesson is universal: a model's accuracy on a rare class can be deeply misleading if you ignore how rare the class is. That's why metrics like precision and recall exist.

## Key takeaways

- Bayes' theorem reverses conditional probabilities: posterior ∝ likelihood × prior.
- P(evidence | hypothesis) is not P(hypothesis | evidence); confusing them is the base rate fallacy.
- With a rare condition, even an accurate test yields mostly false positives — the medical example gives ~17%.
- Evidence compounds: each posterior becomes the next prior, so repeated tests sharpen belief.
- Naive Bayes and Bayesian ML apply this directly, and the base rate lesson explains why rare-class metrics need care.

## Try it

Redo the medical example but make the disease common: set the prior P(disease) to 0.30 instead of 0.01, keeping sensitivity 0.99 and false positive rate 0.05. Compute P(disease | positive) both by hand and by simulation. How much more informative is a positive test when the disease is common? Then, starting from your new posterior, apply a second positive test and see where two positives land you.
