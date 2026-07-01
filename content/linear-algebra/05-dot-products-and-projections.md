# 05 — Dot Products and Projections

You've already met the dot product — it's the little multiply-and-add that lives inside matrix multiplication. It deserves a lesson of its own, because it's the single most important operation for measuring *similarity* in machine learning. When a model asks "how aligned are these two things?", it computes a dot product.

## The dot product

To take the dot product of two vectors, multiply their matching components and add up the results. The answer is a single number:

```
[2, 3] · [4, 1] = (2×4) + (3×1) = 8 + 3 = 11
```

Both vectors must have the same number of components, since you're pairing them up one by one. The result is a scalar — not a vector — which is why it's sometimes called the "scalar product."

That's the whole computation. The interesting part is what the number *means*.

## What the dot product tells you: angle

The dot product secretly encodes the **angle** between two vectors. Here's the key relationship in words: the dot product equals the length of the first vector, times the length of the second, times the cosine of the angle between them.

```
a · b = |a| × |b| × cos(angle between them)
```

You don't need to compute angles by hand to use this. What matters is the *sign and size* of the result:

- **Positive** dot product → the vectors point in broadly the *same* direction (angle less than 90°).
- **Zero** dot product → the vectors are **perpendicular** (exactly 90°). This is a big deal: a zero dot product means the two vectors are unrelated in the geometric sense, or "orthogonal."
- **Negative** dot product → the vectors point in broadly *opposite* directions (angle more than 90°).

So the dot product is a directional agreement meter. Same way? Positive. At right angles? Zero. Opposite ways? Negative.

## Cosine similarity

There's a wrinkle: the dot product also grows when the vectors are simply *longer*, even if their directions haven't changed. If you want to measure *only* the direction agreement — ignoring length — you divide out the lengths. That gives **cosine similarity**:

```
cosine similarity = (a · b) / (|a| × |b|)
```

This always lands between -1 and +1:

- **+1** means the vectors point in exactly the same direction.
- **0** means they're perpendicular.
- **-1** means they point in exactly opposite directions.

Cosine similarity is *the* workhorse for comparing embeddings in ML. When a search system finds documents relevant to your query, or a recommender finds users like you, it's very often ranking by cosine similarity. Two word-vectors for "cat" and "kitten" will have high cosine similarity; "cat" and "spreadsheet" will have low similarity. Because it ignores length, it compares pure meaning-direction, which is usually what you want.

A quick worked example. Take `a = [1, 0]` (pointing right) and `b = [0, 1]` (pointing up):

```
a · b = (1×0) + (0×1) = 0
```

Zero — they're perpendicular, as the picture confirms. Now `a = [1, 0]` and `c = [2, 0]` (both pointing right):

```
a · c = (1×2) + (0×0) = 2   (positive → same direction)
cosine similarity = 2 / (1 × 2) = 1   (exactly aligned)
```

Even though `c` is longer, cosine similarity reports `1`: same direction, maximal similarity.

## Projection: casting a shadow

The dot product also answers a geometric question: **how much of one vector lies along another?** Picture shining a light straight down onto vector `a` from above vector `b`; the shadow `a` casts onto the line through `b` is the **projection** of `a` onto `b`.

The length of that shadow is:

```
projection length = (a · b) / |b|
```

Intuitively, projection strips a vector down to "just the part that agrees with this direction." If `a` points partly along `b` and partly sideways, the projection keeps only the along-`b` part and throws away the sideways part. When the projection is zero, `a` has no component along `b` at all — which is another way of saying they're perpendicular.

Worked example: project `a = [3, 4]` onto `b = [1, 0]` (the horizontal direction):

```
a · b = (3×1) + (4×0) = 3
|b| = 1
projection length = 3 / 1 = 3
```

The shadow of `[3, 4]` on the horizontal axis has length 3 — exactly its horizontal component, which makes perfect sense. Projection asks "how much of `a` points *this* way?" and the dot product hands you the answer.

## In code

```python
import numpy as np

a = np.array([2, 3])
b = np.array([4, 1])

# dot product
print(np.dot(a, b))          # 11

# cosine similarity
cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos)                   # ~0.83  (fairly aligned)

# projection length of a onto b
print(np.dot(a, b) / np.linalg.norm(b))   # ~2.67
```

## Why this matters for ML

Every neuron in a network computes a dot product between its inputs and its weights, then decides how strongly to fire — it's literally measuring "how well does this input match what I'm looking for?" Attention mechanisms in modern language models score how relevant each word is to each other word using dot products. Retrieval, recommendation, clustering, and similarity search all lean on the dot product and its normalized cousin, cosine similarity. Master this one operation and a huge amount of ML stops looking like magic.

## Key takeaways

- The **dot product** multiplies matching components and sums them into a single number.
- Its sign reveals direction: positive = same way, zero = perpendicular, negative = opposite.
- **Cosine similarity** divides out lengths to measure pure direction agreement, always between -1 and +1 — the standard tool for comparing embeddings.
- **Projection** uses the dot product to find how much of one vector lies along another — its "shadow."
- Neurons, attention, and search all run on dot products.

## Try it

With `a = [1, 2]` and `b = [3, 1]`, by hand where you can:

1. Compute `a · b`. Is it positive, zero, or negative — and what does that say about the angle?
2. Compute the L2 lengths of `a` and `b`.
3. Compute the cosine similarity between `a` and `b`.
4. Compute the projection length of `a` onto `b`.
5. Find any nonzero vector that is perpendicular to `a` (hint: aim for a dot product of zero).
6. Verify your answers with `np.dot` and `np.linalg.norm`.
