# 06 — Eigenvalues and SVD (Intuition)

This lesson is different. Eigenvalues and the SVD (singular value decomposition) are usually where beginners hit a wall of formulas. We're going to skip the wall. The goal here is *intuition*: what these ideas mean, why anyone cares, and where they quietly show up in machine learning. You'll leave able to recognize them and explain them, and you can pick up the mechanics later once the pictures are solid.

## A matrix is a transformation

Recall the big idea from matrix multiplication: a matrix is a machine that takes a vector in and gives a transformed vector out. It rotates, stretches, squishes, and shears the space. Feed in a whole cloud of arrows and the matrix moves them all — some grow, some shrink, most also change direction.

Eigenvalues and the SVD are both ways of answering: **what does this transformation actually do, underneath?** They pull a matrix apart into its essential moves.

## Eigenvectors: the directions that don't turn

When a matrix transforms space, most vectors come out pointing somewhere new. But for special matrices there are certain directions that *don't rotate at all* — a vector pointing that way comes out still pointing the same way, only stretched (or shrunk). Those special directions are the **eigenvectors**, and the amount of stretch along each one is its **eigenvalue**.

Picture stretching a photo horizontally to make everything twice as wide. A vector pointing straight right stays pointing straight right — it just doubles in length. A vector pointing straight up stays pointing straight up, unchanged. Those two directions are the eigenvectors of that stretch; their eigenvalues are 2 (doubled) and 1 (unchanged). A vector pointing diagonally, though, *does* get tilted, because it's a mix of a doubled direction and an unchanged one — so it isn't an eigenvector.

The intuition to keep: **eigenvectors are the "grain" of a transformation** — the natural axes along which it simply scales things. The eigenvalues tell you how much scaling happens along each grain line. If an eigenvalue is large, the transformation stretches hard in that direction; if it's near zero, it flattens things in that direction; if it's negative, it flips them.

Why does ML care? Because if you have a big cloud of data, you can ask which directions it spreads out along the most. Those directions are eigenvectors of a matrix built from the data, and the eigenvalues rank them by importance. That's the engine of **Principal Component Analysis (PCA)** — a classic technique for finding the few directions that capture most of the variation in a dataset, so you can compress or visualize high-dimensional data.

## SVD: every matrix as rotate-stretch-rotate

Eigenvectors are cleanest for square, well-behaved matrices. The **singular value decomposition** is the more general, more powerful cousin, and it works for *any* matrix — even a rectangular one. Its message is beautifully simple:

> Any matrix, no matter how complicated it looks, does its job in three steps: **rotate, then stretch along the axes, then rotate again.**

That's it. Whatever tangle of numbers a matrix contains, the SVD says the transformation is really just a rotation, followed by a pure stretching along perpendicular axes, followed by another rotation. The amounts of stretching are called the **singular values**, and — this is the useful part — they come out ranked from largest to smallest. The big singular values are the directions where the matrix does most of its work; the tiny ones barely matter.

The picture: imagine the transformation acting on a circle of arrows. The SVD tells you the circle becomes an ellipse. The singular values are the lengths of the ellipse's axes — how far the transformation stretched things in each principal direction. Big axis, important direction; nearly-flat axis, nearly-ignorable direction.

## Why SVD matters: keeping what counts

Because singular values are ranked, you can *throw away the small ones* and keep a close approximation of the original matrix using far less information. This is the idea behind **low-rank approximation**, and it shows up everywhere:

- **Compression.** An image stored as a matrix can be approximated by keeping only its largest singular values, shrinking the file while keeping it recognizable.
- **Recommendation systems.** A giant, mostly-empty table of users-versus-items gets factored by SVD into a few hidden "taste" directions, which is how a system guesses ratings you never gave.
- **Noise reduction.** Real signal tends to live in the large singular values; random noise scatters into the small ones. Dropping the small ones cleans the data.
- **Understanding models.** Looking at the singular values of a trained weight matrix tells you whether a layer is doing a lot of independent work or effectively collapsing everything into a few directions.

The common thread: the SVD finds the *few directions that matter most* and lets you ignore the rest. In a field drowning in high-dimensional data, "keep what counts, drop the rest" is priceless.

## How they relate

Eigen-decomposition and SVD are close relatives. Both break a matrix into "directions plus amounts." Eigenvectors/eigenvalues describe how a square matrix scales along its natural axes. Singular vectors/values generalize this to any matrix and always give clean, perpendicular axes ranked by importance. In practice, when people say "the important directions in this data," they usually mean the top eigenvectors or top singular vectors — the two ideas blur together, and that's fine at this level.

## A peek at the code

```python
import numpy as np

A = np.array([[2, 0],
              [0, 1]])

# eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(A)
print(vals)   # [2. 1.]  -> stretch by 2 in one direction, unchanged in the other

# singular value decomposition of any matrix
U, S, Vt = np.linalg.svd(A)
print(S)      # [2. 1.]  -> the singular values, largest first
```

Notice for this simple stretch matrix, the eigenvalues and singular values are both `[2, 1]` — the "double one axis, leave the other" story we told in pictures, confirmed in numbers.

## Key takeaways

- A matrix is a transformation; eigen-decomposition and SVD reveal what it really does.
- **Eigenvectors** are directions that don't rotate under the transformation — only scale — and **eigenvalues** are how much they scale. They're the "grain" of a matrix and the basis of PCA.
- **SVD** says *any* matrix is a rotate–stretch–rotate; the **singular values**, ranked largest-first, measure how much stretching happens in each key direction.
- Keeping only the big singular values gives a **low-rank approximation** — the core of compression, recommendation, and denoising.
- Both ideas share one theme: find the few directions that matter, ignore the rest.

## Try it

No heavy computation — this is about intuition:

1. In your own words, explain what makes a vector an *eigenvector* of a transformation.
2. A stretch triples everything horizontally and halves everything vertically. Name the two eigenvector directions and their eigenvalues.
3. Explain the "rotate, stretch, rotate" story of the SVD to a friend in two sentences.
4. If a 1000×1000 image matrix has only its top 20 singular values kept, what have you gained and what have you lost?
5. In code, run `np.linalg.svd` on `[[3, 0], [0, 2]]` and confirm the singular values match your intuition about the stretch.
