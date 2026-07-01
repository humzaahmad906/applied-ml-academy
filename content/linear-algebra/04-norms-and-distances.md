# 04 — Norms and Distances

So far we've treated vectors as arrows with a direction and a length, but we haven't put a number on that length. This lesson is about measuring: **how big is a vector**, and **how far apart are two vectors**. These questions sound abstract, but they're exactly what a model asks when it decides whether two things are similar, or how wrong a prediction is.

## The length of a vector

The size of a vector is called its **norm**. The most familiar one comes straight from geometry. Picture the 2-vector `[3, 4]` as an arrow. How long is that arrow? By the Pythagorean theorem, the length is the square root of the sum of the squared components:

```
length of [3, 4] = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

This is the **L2 norm**, also called the Euclidean norm — "as the crow flies" distance from the origin to the tip of the arrow. The recipe is always the same: square each component, add them up, take the square root. It works for any number of components:

```
length of [1, 2, 2] = sqrt(1 + 4 + 4) = sqrt(9) = 3
```

The L2 norm is the default meaning of "size" for a vector, and it's the one your geometric intuition already agrees with.

## A different way to measure: the L1 norm

The L2 norm isn't the only sensible way to measure size. The **L1 norm** adds up the *absolute values* of the components — no squaring, no square root:

```
L1 norm of [3, 4] = |3| + |4| = 7
L1 norm of [3, -4] = |3| + |-4| = 7
```

Notice L1 gives 7 while L2 gave 5 for the same vector. They're just different rulers. The nickname for L1 is the "taxicab" or "Manhattan" distance: if you can only travel along a grid of streets (no cutting diagonally through buildings), the distance from the origin to `[3, 4]` really is 3 blocks over plus 4 blocks up = 7 blocks. The L2 norm is the diagonal shortcut a bird could take; the L1 norm is the route a taxi must drive.

Both are legitimate measures of size. Which one you reach for depends on the job, and we'll see in a moment that ML uses both.

## Distance between two vectors

Once you can measure the length of a vector, measuring the distance between two vectors is easy: **subtract them, then take the norm of the difference.**

Take `a = [1, 2]` and `b = [4, 6]`. Their difference is:

```
a - b = [1-4, 2-6] = [-3, -4]
```

The L2 distance between them is the L2 norm of that difference:

```
sqrt((-3)² + (-4)²) = sqrt(9 + 16) = sqrt(25) = 5
```

So `a` and `b` are 5 units apart. This is the same "arrow from `b` to `a`" idea from the vectors lesson — the difference vector points from one to the other, and its length is how far apart they are. Use the L1 norm of the difference instead and you get the taxicab distance; use L2 and you get the straight-line distance.

## Measuring size and similarity in ML

These simple ideas do a lot of heavy lifting.

**Similarity.** When a model represents two items as vectors, "how similar are they?" often becomes "how small is the distance between them?" Two product-description vectors that are close together describe similar products. Nearest-neighbor search — the engine behind many recommendation and retrieval systems — is literally "find the vectors with the smallest distance to this one."

**Error.** When a model predicts a vector and you know the true answer, the distance between prediction and truth is the error. Squared L2 distance is the basis of the most common regression loss; L1 distance gives a loss that cares less about the occasional huge outlier.

**Keeping weights small.** Training often adds a penalty on the norm of the model's weights to discourage them from growing too large (a technique called regularization). Penalizing the L2 norm gently shrinks all weights; penalizing the L1 norm tends to push many weights all the way to zero, which is why L1 is prized when you want a model that ignores most of its inputs and keeps only a few. The *choice* of norm changes the behavior — that's why it's worth knowing both.

## In code

```python
import numpy as np

a = np.array([1, 2])
b = np.array([4, 6])

# L2 norm (length) of a
print(np.linalg.norm(a))          # sqrt(1 + 4) = 2.236...

# L1 norm of a
print(np.linalg.norm(a, ord=1))   # 1 + 2 = 3.0

# L2 distance between a and b
print(np.linalg.norm(a - b))      # 5.0

# L1 distance between a and b
print(np.linalg.norm(a - b, ord=1))  # 3 + 4 = 7.0
```

`np.linalg.norm` defaults to L2; pass `ord=1` for L1. Notice the pattern: distance is always `norm(a - b)`.

## Key takeaways

- A **norm** measures the size of a vector.
- The **L2 norm** squares the components, sums, and square-roots — it's the straight-line "arrow length" your geometry intuition expects.
- The **L1 norm** sums absolute values — the taxicab distance along a grid.
- **Distance** between two vectors is the norm of their difference: `norm(a - b)`.
- ML uses norms for similarity search, prediction error, and regularization; L1 and L2 behave differently, so the choice matters.

## Try it

With `a = [2, 3]` and `b = [5, 7]`, by hand:

1. Compute the L2 norm of `a`.
2. Compute the L1 norm of `a`.
3. Compute the difference `a - b`.
4. Compute both the L2 and L1 distance between `a` and `b`.
5. Confirm all four numbers using `np.linalg.norm`, remembering `ord=1` for L1.
6. Bonus: for what kind of vector do the L1 and L2 norms come out equal?
