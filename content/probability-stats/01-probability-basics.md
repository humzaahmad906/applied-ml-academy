# 01 — Probability Basics

Probability is the language we use to talk about uncertainty. Every machine learning model, every metric, and every loss function is quietly built on top of it. Before we can reason about why a model works, we need a shared vocabulary for chance. This module builds that vocabulary from scratch, with no assumptions beyond basic arithmetic.

## What is a probability?

A **probability** is a number between 0 and 1 that measures how likely something is. A 0 means "impossible," a 1 means "certain," and everything interesting lives in between. If you flip a fair coin, the probability of heads is 0.5. If you roll a fair six-sided die, the probability of rolling a 3 is 1/6, about 0.167.

The thing we assign a probability to is called an **event**. An event is just an outcome or a set of outcomes we care about. "The die shows an even number" is an event covering three outcomes: 2, 4, and 6. Its probability is 3/6 = 0.5.

A useful mental model: imagine repeating the experiment many, many times. The probability of an event is the fraction of times it happens in the long run. Flip a fair coin ten thousand times and you'll see close to five thousand heads.

```python
import numpy as np

rng = np.random.default_rng(0)
flips = rng.integers(0, 2, size=10_000)  # 0 = tails, 1 = heads
print("fraction heads:", flips.mean())   # ~0.5
```

## The rules of the game

Two rules cover most of what you need.

**Rule 1 — The complement.** The probability that an event does *not* happen is 1 minus the probability that it does. If rain tomorrow has probability 0.3, then no rain has probability 0.7. They must add to 1 because one of them is certain to occur.

**Rule 2 — Adding for "or".** If two events cannot both happen at the same time (they are *mutually exclusive*), the probability that either occurs is the sum of their probabilities. Rolling a 1 or a 2 on a die has probability 1/6 + 1/6 = 2/6.

If the events *can* overlap, you have to subtract the overlap so you don't count it twice:

P(A or B) = P(A) + P(B) − P(A and B)

Drawing a card that is a king or a heart: P(king) = 4/52, P(heart) = 13/52, but the king of hearts is in both, so P(king and heart) = 1/52. The total is (4 + 13 − 1) / 52 = 16/52.

## Conditional probability

Often we learn something partway through and want to update our estimate. **Conditional probability** answers the question: given that event B happened, how likely is event A? We write it P(A | B), read "the probability of A given B."

The formula is:

P(A | B) = P(A and B) / P(B)

The intuition: once you know B happened, you throw away every world where B did not happen. You're now living inside the slice where B is true, and you ask what fraction of *that* slice also has A.

Here's a concrete example. A box has 10 fruits: 6 apples and 4 oranges. Among the apples, 2 are ripe; among the oranges, 3 are ripe. You pick a fruit and someone tells you it's ripe. What's the probability it's an orange?

There are 5 ripe fruits total (2 apples + 3 oranges). Of those, 3 are oranges. So P(orange | ripe) = 3/5 = 0.6. Knowing "ripe" changed the answer from the base rate of 4/10 = 0.4 up to 0.6.

```python
import numpy as np

rng = np.random.default_rng(1)
# Encode each fruit as (is_orange, is_ripe)
fruits = ([(0, 1)] * 2 + [(0, 0)] * 4      # apples: 2 ripe, 4 not
        + [(1, 1)] * 3 + [(1, 0)] * 1)     # oranges: 3 ripe, 1 not
fruits = np.array(fruits)

draws = fruits[rng.integers(0, len(fruits), size=100_000)]
ripe = draws[draws[:, 1] == 1]             # keep only ripe draws
print("P(orange | ripe):", ripe[:, 0].mean())  # ~0.6
```

Notice how the simulation mirrors the formula: we *filter* down to the ripe draws, then measure the fraction that are oranges. That filtering step is exactly what conditioning means.

## Independence

Two events are **independent** when knowing one tells you nothing about the other. Formally, A and B are independent if:

P(A and B) = P(A) × P(B)

Equivalently, P(A | B) = P(A): conditioning on B leaves A's probability unchanged.

Coin flips are the classic example. The second flip doesn't care what the first flip did, so P(two heads) = 0.5 × 0.5 = 0.25. A common and costly mistake is assuming independence when it doesn't hold. If it rained yesterday, rain today is more likely; the events are dependent, and multiplying their individual probabilities would understate the chance of two rainy days.

Independence matters enormously in ML. Many models assume data points are independent so their probabilities multiply cleanly, which is what makes training tractable. When that assumption is wrong (correlated samples, leakage between train and test), models can look far better than they really are.

## Why this matters for ML

When a classifier outputs "0.87 probability this image is a cat," that number is a probability in exactly the sense above. Conditional probability is the heart of it: the model is estimating P(cat | pixels). A loss function like cross-entropy is a way of scoring how good those probability estimates are. Get comfortable with events, conditioning, and independence now, and the rest of the course will feel like variations on a theme you already know.

## Key takeaways

- A probability is a number in [0, 1]; think of it as a long-run frequency.
- Complements sum to 1; for "or" you add, subtracting any overlap.
- Conditional probability P(A | B) restricts attention to the worlds where B is true.
- Independence means P(A and B) = P(A) × P(B); assuming it wrongly is a common trap.
- ML classifiers output conditional probabilities, so these basics are foundational.

## Try it

A medical test is positive for 90% of people who have a disease and for 10% of people who don't. Suppose 5% of the population has the disease. Simulate 1,000,000 people in NumPy: assign each a disease status, then a test result based on the rates above. Filter to the people who tested positive, and compute the fraction of them who actually have the disease. Is it higher or lower than you expected? (You'll return to this exact setup when we reach Bayes' theorem.)
