# 06b — Rank, Orthogonality, and Outer Products

The last lesson gave you eigenvalues and the SVD as *pictures* — the grain of a transformation, the rotate-stretch-rotate story, the "keep what counts" theme. This lesson fills in three supporting ideas that make those pictures precise and that you'll meet constantly in machine learning: **rank** (how much a matrix really does), **orthogonality** (directions that don't overlap, and the well-behaved matrices built from them), and the **outer product** (the smallest possible matrix, and a building block hiding inside attention and gradients). Same spirit as before — intuition first, a little numpy to make it concrete.

## Linear independence and span

Start with a bag of vectors. Their **span** is everywhere you can reach by scaling them and adding them together. Two arrows pointing different ways in a plane span the whole plane — mix them in the right amounts and you can land on any point. But two arrows pointing *the same way* only span a single line; the second one tells you nothing new, because it's just a stretched copy of the first.

That's the whole idea of **linear independence**. A set of vectors is independent when none of them is a combination of the others — each one adds a genuinely new direction. The moment one vector can be built from the rest, it's *dependent*, and it contributes nothing to the span. Independence is about *non-redundancy*: how many of these arrows are actually pulling their weight.

Keep this picture, because rank is just this idea counted up.

## Rank: how many independent directions

The **rank** of a matrix is the number of linearly independent directions it contains — equivalently, the dimension of the space its columns can span. Think of a matrix as a stack of column vectors. If all the columns point in genuinely different directions, the rank is as high as it can be and we call the matrix **full rank**. If some columns are redundant — combinations of the others — the rank drops. That's a **low-rank** matrix: it looks big, but it only does a little.

Tie this back to the SVD. The singular values were ranked largest-to-smallest, and the small ones "barely mattered." The rank is exactly *how many singular values are meaningfully nonzero*. A 1000×1000 matrix with only 20 real singular values is a rank-20 matrix wearing a big coat — almost all of its apparent size is redundancy.

Why does low rank matter so much? Because a low-rank matrix can be stored and computed cheaply. Instead of holding all `m × n` numbers, you can factor it into two skinny matrices and hold only `r × (m + n)` of them, where `r` is the rank. When `r` is small, that's a massive saving — which is precisely the compression, denoising, and recommendation payoff from the SVD lesson.

Here's the payoff that connects directly to modern ML: **LoRA** (Low-Rank Adaptation), the standard way to fine-tune large models. Fine-tuning a giant weight matrix `W` normally means updating all of its millions of entries. LoRA's bet is that the *change* you need — the update `ΔW` — has low intrinsic rank: adapting a model to a new task nudges it along a few directions, not all of them. So instead of learning a full `ΔW`, LoRA learns two skinny matrices `B` and `A` and sets `ΔW = B A`, a low-rank update added on top of the frozen original. A `d × k` weight matrix that would need `d × k` trainable numbers now needs only `r × (d + k)` — often a fraction of a percent. Same low-rank idea from the SVD lesson, now steering how you'll fine-tune models in the fine-tuning chapter ahead.

## Orthogonality: directions that don't overlap

Two vectors are **orthogonal** when they're at right angles — geometrically perpendicular, and algebraically their **dot product is zero**. The dot product measures how much two vectors point the same way; zero means they share no common direction at all. Orthogonal directions are the cleanest kind of independent: not just non-redundant, but completely non-overlapping.

A set of vectors is **orthonormal** when they're all mutually orthogonal *and* each has length exactly 1. This is the gold standard for a coordinate system — a set of perpendicular unit axes, like the familiar x, y, z, just possibly rotated. Stack orthonormal vectors as the columns of a matrix and you get an **orthonormal matrix** (often just called orthogonal), and these matrices have a special, beautiful property: **they preserve length.**

Multiply any vector by an orthonormal matrix and its length comes out unchanged — the transformation can rotate or reflect the space, but it never stretches or squishes it. A rotation is the canonical example: spin a cloud of arrows around the origin and every arrow keeps its length; only the directions change. (This is exactly the "rotate" part of the SVD's rotate-stretch-rotate — the `U` and `V` factors are orthonormal, and all the stretching lives in the middle.)

Length preservation is why orthonormal matrices are the workhorses of *stable* computation. A transformation that preserves length can't blow small numbers up to infinity or crush them to zero — it can't amplify error. That's gold in deep learning, where signals pass through many layers in a row. It's why **orthogonal weight initialization** is a favored trick: start a layer's weights as an orthonormal matrix and the signal (and the gradient flowing back) keeps its scale as it propagates, instead of exploding or vanishing. The same property underlies numerically stable algorithms throughout scientific computing.

## The outer product: a column times a row

You've seen the dot product: a row vector times a column vector, giving a single number. The **outer product** is the mirror image — a *column* vector times a *row* vector — and it gives back a whole **matrix**. If `u` has length `m` and `v` has length `n`, then `u vᵀ` is an `m × n` matrix whose `(i, j)` entry is simply `u[i] * v[j]`.

The crucial fact: **an outer product always has rank 1.** Every column of `u vᵀ` is just `u` scaled by a different entry of `v`, so all the columns point the same way — one direction, one independent column, rank 1. It's the simplest nontrivial matrix there is, the atom that everything else is built from. In fact the SVD says *any* matrix is a sum of rank-1 outer products, ranked by their singular values, which is the deep reason low-rank approximation works: keep the first few outer products, drop the rest.

Outer products aren't just a curiosity; they show up in the hot path of real models:

- **Attention scores.** The heart of a transformer computes how much each token should attend to every other token by multiplying a matrix of queries against a matrix of keys. Every single score in that matrix is a dot product of one query with one key — and the score matrix as a whole is built from these outer-product-style interactions between the query and key vectors.
- **Gradients of a linear layer.** When you backpropagate through a layer `y = W x`, the gradient of the loss with respect to the weight matrix `W` is an **outer product**: the incoming gradient (a column) times the layer's input (a row). Every weight-update step in training a linear layer is, at its core, an outer product. Rank-1 updates are quietly running your whole training loop.

## A peek at the code

```python
import numpy as np

# rank: how many independent directions?
A = np.array([[1, 2, 3],
              [2, 4, 6],     # row 2 is 2x row 1 -> redundant
              [1, 0, 1]])
print(np.linalg.matrix_rank(A))   # output: 2  (only 2 independent rows)

# orthogonality: dot product zero means perpendicular
u = np.array([1, 0])
v = np.array([0, 1])
print(np.dot(u, v))               # output: 0  (right angle)

# an orthonormal (rotation) matrix preserves length
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
x = np.array([3.0, 4.0])
print(np.linalg.norm(x))          # output: 5.0
print(np.linalg.norm(R @ x))      # output: 5.0  (rotated, same length)

# outer product: column times row = a rank-1 matrix
a = np.array([1, 2, 3])
b = np.array([10, 20])
M = np.outer(a, b)
print(M)                          # output: [[10 20] [20 40] [30 60]]
print(np.linalg.matrix_rank(M))   # output: 1  (always rank 1)
```

Notice the pieces clicking together: the redundant matrix has rank below its size, a rotation leaves the length `5` untouched, and the outer product is rank 1 no matter how big it looks.

## Key takeaways

- **Linear independence** means each vector adds a genuinely new direction; the **span** is everywhere you can reach by combining them.
- **Rank** counts the independent directions in a matrix — equivalently, its meaningfully-nonzero singular values. **Low-rank** matrices look big but do little, which is why they compress. **LoRA** fine-tunes huge models with a cheap low-rank update `ΔW = B A`.
- **Orthogonal** vectors have a **dot product of zero** (perpendicular); **orthonormal** matrices are built from perpendicular unit vectors and **preserve length**, making rotations, stable transforms, and orthogonal initialization possible.
- The **outer product** (column × row) is a **rank-1 matrix** — the atom of the SVD, the shape of attention interactions, and the exact form of a linear layer's weight gradient.

## Try it

1. In your own words, why do two arrows pointing the same direction span only a line, not a plane?
2. A 500×500 matrix has just 10 nonzero singular values. What is its rank, and roughly how many numbers do you need to store it as two skinny factors instead of all 250,000 entries?
3. Explain LoRA's bet in one sentence: why learn `B` and `A` instead of a full weight update `ΔW`?
4. Why can multiplying by an orthonormal matrix never make a vector longer or shorter? Connect this to why it helps signals survive many layers.
5. In code, take any nonzero column vector `u` and row vector `v`, form `np.outer(u, v)`, and confirm with `np.linalg.matrix_rank` that the result is rank 1 — then explain why it must be.
