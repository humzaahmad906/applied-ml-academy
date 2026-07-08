# 07b — Tensors and Broadcasting

So far every array you've built has been at most 2-D: a vector or a matrix. That's enough to reason about a single example flowing through a single layer. But real deep learning almost never works on one example at a time — it works on a *stack* of them at once, for speed. The moment you stack examples, your arrays grow a new dimension, and a whole set of new rules kicks in. Those rules are called **broadcasting**, and getting them wrong is the single most common source of shape bugs for beginners. This lesson takes you from matrices to tensors and shows you the rules that keep the shapes honest.

## From matrix to tensor

A **tensor** is just NumPy's word for an array with any number of dimensions. A vector is a 1-D tensor, a matrix is a 2-D tensor, and everything above that is a tensor too. The two habits you already have — check `.shape`, count the dimensions — carry over unchanged. NumPy adds `.ndim` to tell you the number of dimensions directly:

```python
import numpy as np

v = np.array([1, 2, 3])                  # 1-D
A = np.array([[1, 2], [3, 4]])           # 2-D
T = np.zeros((2, 3, 4))                  # 3-D

print(v.ndim, v.shape)   # 1 (3,)
print(A.ndim, A.shape)   # 2 (2, 2)
print(T.ndim, T.shape)   # 3 (2, 3, 4)
```

The reason you'll see 3-D and 4-D arrays constantly is the **batch dimension** — an extra leading axis that says "this many examples at once." It's almost always the *first* axis. Here are the shapes you'll actually meet downstream:

- `(batch, features)` — a plain batch of feature vectors, e.g. `(32, 10)`: 32 examples, 10 features each.
- `(batch, seq, dim)` — a batch of sequences, e.g. `(32, 128, 512)`: 32 sentences, 128 tokens each, 512 numbers per token. This is the shape a Transformer lives in.
- `(batch, channels, H, W)` — a batch of images, e.g. `(32, 3, 224, 224)`: 32 RGB images of 224×224 pixels.

In every case the leading dimension is "how many," and the trailing dimensions are "the actual data for one example." Keep that picture in mind — it's the key to reading these shapes at a glance.

## Broadcasting rules

Broadcasting is how NumPy lets you combine arrays of *different* shapes without writing a loop. The rules are mechanical, and once you know them you can predict every result.

NumPy compares shapes **from the trailing (rightmost) dimension leftward**. For each pair of dimensions, they're compatible if:

1. they are **equal**, or
2. one of them is **1** (that axis gets stretched to match the other), or
3. one of them is **missing** (a shorter shape is treated as having leading 1s).

If any pair fails all three, NumPy raises an error. Nothing is ever copied in memory — the size-1 axis is *virtually* repeated, which is why broadcasting is both cheap and fast.

The classic example is adding a bias vector to a batch:

```python
X = np.ones((4, 3))          # 4 examples, 3 features
b = np.array([10, 20, 30])   # one bias per feature, shape (3,)

print((X + b))
# [[11. 21. 31.]
#  [11. 21. 31.]
#  [11. 21. 31.]
#  [11. 21. 31.]]
print((X + b).shape)         # (4, 3)
```

Line up the shapes: `(4, 3)` and `(3,)`. The trailing dims are `3` and `3` — equal, fine. The next dim left is `4` versus *missing*, treated as `1` — so `b` is stretched down all 4 rows. The `(N, 3) + (3,)` pattern is everywhere: adding a per-feature bias to a batch.

Size-1 axes let you broadcast in *two* directions at once. This builds an outer-sum grid:

```python
col = np.array([[1], [2], [3]])   # shape (3, 1)
row = np.array([[10, 20]])        # shape (1, 2)

print(col + row)
# [[11 21]
#  [12 22]
#  [13 23]]
print((col + row).shape)          # (3, 2)
```

Here `(3, 1)` and `(1, 2)` meet: trailing dims `1` and `2` → the `1` stretches to `2`; next dims `3` and `1` → the `1` stretches to `3`. Both operands expand and you get a full `(3, 2)` grid.

When it **fails**, the error tells you exactly where:

```python
X = np.ones((4, 3))
b = np.array([10, 20])       # shape (2,) — wrong size!

X + b
# ValueError: operands could not be broadcast together with shapes (4,3) (2,)
```

Read it literally: it lists both shapes. The trailing dims are `3` and `2` — not equal, and neither is `1`, so the rule fails. The fix is to make that last axis either `3` or `1`. Whenever you see "could not be broadcast together," print the two shapes and check them trailing-first; the mismatched axis jumps out.

## Batched matrix multiplication

Once you have a batch dimension, you want to matrix-multiply *every* example without a loop. `np.matmul` (and its `@` operator) does exactly this: it treats **all leading dimensions as batch** and multiplies only the **last two** axes using the ordinary row-times-column rule you already know.

```python
B, n, k, m = 8, 2, 3, 4
X = np.random.rand(B, n, k)   # a batch of 8 matrices, each (2, 3)
W = np.random.rand(B, k, m)   # a batch of 8 matrices, each (3, 4)

out = X @ W                   # multiply each (2,3) by its (3,4)
print(out.shape)              # (8, 2, 4)
```

The batch axis of length 8 is carried straight through. Within each of the 8 slots, `(2, 3) @ (3, 4)` follows the inner-dimension rule (the `3`s cancel) to give `(2, 4)`. The final shape is `(8, 2, 4)`.

Broadcasting applies to the batch dimensions too. A single weight matrix (no batch axis) multiplies against a whole batch — the pattern behind every linear layer:

```python
X = np.random.rand(8, 2, 3)   # batch of 8 inputs
W = np.random.rand(3, 4)      # ONE shared weight matrix

out = X @ W                   # W is broadcast across the batch
print(out.shape)              # (8, 2, 4)
```

`W` has no batch axis, so it's reused for all 8 examples — which is exactly what you want, since a layer's weights are shared across the batch. As always, only the last two dimensions have to satisfy the matmul rule; everything to the left just has to broadcast.

## einsum: naming your axes

For anything fancier than a plain matmul, `np.einsum` lets you spell out the operation by **labeling every axis with a letter** and stating what the output axes are. Repeated labels that vanish from the output are summed over; labels that stay are kept.

Here's a batched dot product — for each example in the batch, dot two vectors:

```python
a = np.random.rand(8, 5)      # 8 vectors of length 5
b = np.random.rand(8, 5)

dots = np.einsum('bi,bi->b', a, b)
print(dots.shape)             # (8,)
```

Read the string: `b` is the batch label (kept, because it's on the right of `->`), and `i` appears in both inputs but *not* the output, so it's summed — that's the dot product. One line, and you can see the whole operation.

The reason einsum earns its place is readability on the operations that would otherwise be a puzzle of transposes. Attention scores are the poster child — for each item in the batch, multiply a set of query vectors by a set of key vectors:

```python
Q = np.random.rand(8, 10, 64)   # batch=8, 10 queries, dim 64
K = np.random.rand(8, 12, 64)   # batch=8, 12 keys,    dim 64

scores = np.einsum('bik,bjk->bij', Q, K)
print(scores.shape)             # (8, 10, 12)
```

The label `k` (the 64-dim feature axis) appears in both inputs and not the output, so it's summed — that's the dot product between each query and each key. `b` is the batch, `i` indexes queries, `j` indexes keys, and you get a `(8, 10, 12)` score for every query-key pair. Doing this with `@` would require transposing `K`'s last two axes first; with einsum you just name the axes and the intent is obvious.

## Why this matters

Every tensor you'll touch in PyTorch, JAX, or TensorFlow is exactly this object — an N-dimensional array with a shape, a batch dimension out front, and the same broadcasting and batched-matmul rules underneath. The frameworks add autograd and GPUs on top, but `x.shape`, broadcasting a bias, and `@` batching over leading dims behave identically to what you just ran in NumPy.

And here's the payoff: once you're working in 3-D and 4-D, the overwhelming majority of beginner bugs are **shape and broadcast bugs**, not math bugs. A model that "trains but the loss is nonsense" is very often a silent broadcast that lined up the wrong axes. The habit from lesson 07 — print `.shape` the instant something looks off — is your single best defense. Add `.ndim` to it, line the shapes up trailing-first, and you can diagnose almost any shape error in seconds.

## Key takeaways

- A **tensor** is an array with any number of dimensions; use `.shape` and `.ndim` to inspect it. Real DL shapes carry a leading **batch dimension**: `(batch, features)`, `(batch, seq, dim)`, `(batch, channels, H, W)`.
- **Broadcasting** aligns shapes from the trailing dimension leftward; axes are compatible when they're equal, one is `1` (it stretches), or one is missing (treated as `1`).
- The `(N, 3) + (3,)` bias add and the `(N, 1) + (1, M)` grid are the two patterns you'll see most; a "could not be broadcast together" error names both shapes — read it trailing-first.
- `np.matmul` / `@` treats all leading dims as **batch** and multiplies only the last two axes; a single weight matrix broadcasts across a whole batch.
- `np.einsum` labels every axis with a letter — repeated labels missing from the output are summed. It makes batched dots and attention-style products (`'bik,bjk->bij'`) readable.

## Try it

In a Python session:

1. Build a `(32, 10)` array of ones and add a `(10,)` bias; confirm the result is `(32, 10)` and every row matches the bias.
2. Add a `(32, 1)` column to a `(1, 5)` row and predict the output shape before running it.
3. Deliberately add a `(4, 3)` array to a `(2,)` array and read the broadcast error message.
4. Make `X` of shape `(16, 2, 3)` and `W` of shape `(3, 4)`; predict the shape of `X @ W`, then verify.
5. Rewrite the batched dot `np.einsum('bi,bi->b', a, b)` as `(a * b).sum(axis=1)` and confirm the two give the same numbers.
6. Compute attention-style scores with `np.einsum('bik,bjk->bij', Q, K)` and check the output shape is `(batch, n_queries, n_keys)`.
