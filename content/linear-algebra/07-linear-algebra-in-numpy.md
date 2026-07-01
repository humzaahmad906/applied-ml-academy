# 07 — Linear Algebra in NumPy

You've built the ideas by hand: vectors, matrices, multiplication, norms, dot products, and the intuition behind eigenvalues and SVD. Doing them on paper cements the concepts, but in real machine learning you'll do all of it in code. This lesson brings everything together in NumPy — the standard Python library for numerical work — so you can compute confidently and, just as importantly, *catch your own mistakes*.

## Setting up: arrays

NumPy's core object is the **array**, which represents vectors and matrices alike. You build one from a Python list:

```python
import numpy as np

v = np.array([3, 2])              # a vector (1D array)
A = np.array([[1, 2, 3],          # a 2x3 matrix (2D array)
              [4, 5, 6]])
```

The most important habit in all of NumPy is checking `.shape`. It tells you the dimensions, and mismatched shapes cause the majority of beginner bugs:

```python
print(v.shape)   # (2,)     -> a vector with 2 components
print(A.shape)   # (2, 3)   -> 2 rows, 3 columns
```

Whenever an operation fails or a result looks wrong, print the shapes first. It's the fastest debugging move you have.

## Vector operations

Addition and scaling work exactly as you learned, and NumPy applies them element by element automatically:

```python
a = np.array([3, 2])
b = np.array([1, 4])

print(a + b)     # [4 6]   element-wise addition
print(a - b)     # [2 -2]
print(2 * a)     # [6 4]   scaling
print(2 * a + b) # [7 8]   a linear combination
```

No loops needed — NumPy does the whole array at once, which is both cleaner to write and dramatically faster than a Python `for` loop.

## Matrix operations

Transpose, addition, and scalar multiplication carry straight over:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.T)         # transpose -> shape (3, 2)
print(A.T.shape)   # (3, 2)

B = np.array([[1, 1, 1],
              [2, 2, 2]])
print(A + B)       # element-wise, requires matching shapes
print(3 * A)       # scale every entry
```

If you try to add mismatched shapes, NumPy raises a `ValueError`. That error is a *feature* — it's telling you the math doesn't line up, exactly like it wouldn't on paper.

## Matrix multiplication: use @

This is the one to get right. NumPy uses the `@` operator for true matrix multiplication (the row-times-column rule). Do **not** use `*` for this — `*` does element-wise multiplication, which is a completely different operation and a classic silent bug.

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A @ B)       # matrix multiplication -> [[19 22], [43 50]]
print(A * B)       # element-wise (NOT the same!) -> [[5 12], [21 32]]
```

Compare those two outputs carefully — they're different, and confusing them will quietly wreck a model. When you mean "matrix multiply," reach for `@`.

The shape rule from before is enforced automatically. Left columns must equal right rows:

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)
W = np.array([[1, 0],
              [0, 1],
              [1, 1]])          # shape (3, 2)

print((X @ W).shape)   # (2, 2)  -> inner 3s cancel, outer 2 and 2 remain
```

If the inner dimensions don't match, NumPy raises an error naming the mismatched shapes — read it, and you'll usually spot the fix (often a transpose).

## Norms and distances

`np.linalg.norm` handles vector sizes and distances. It defaults to L2; pass `ord=1` for L1:

```python
a = np.array([3, 4])
b = np.array([0, 0])

print(np.linalg.norm(a))          # 5.0   L2 length
print(np.linalg.norm(a, ord=1))   # 7.0   L1 length

# distance is the norm of the difference
c = np.array([1, 2])
d = np.array([4, 6])
print(np.linalg.norm(c - d))      # 5.0   L2 distance
```

## Dot products, similarity, projection

```python
a = np.array([2, 3])
b = np.array([4, 1])

print(np.dot(a, b))    # 11   (a @ b works too for vectors)

# cosine similarity
cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos)             # ~0.83

# projection length of a onto b
print(np.dot(a, b) / np.linalg.norm(b))   # ~2.67
```

For two vectors, `np.dot(a, b)` and `a @ b` give the same scalar — use whichever reads more clearly to you.

## Eigenvalues and SVD

The decompositions live in `np.linalg`:

```python
A = np.array([[2.0, 0.0],
              [0.0, 3.0]])

# eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(A)
print(vals)            # [2. 3.]  -> stretch factors along each axis

# singular value decomposition (works for any shape)
U, S, Vt = np.linalg.svd(A)
print(S)               # [3. 2.]  -> singular values, largest first
```

`S` comes back sorted largest-first, which is exactly what you want when deciding which directions matter and which to drop.

## A worked mini-pipeline

Let's tie it together — two data points, a weight matrix, and a similarity check, the shape of a real forward pass:

```python
import numpy as np

# two examples, three features each -> shape (2, 3)
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

# weights turning 3 features into 2 outputs -> shape (3, 2)
W = np.array([[0.1, 0.0],
              [0.0, 0.2],
              [0.5, 0.5]])

out = X @ W                       # (2, 3) @ (3, 2) -> (2, 2)
print("output:", out)
print("output shape:", out.shape)

# how similar are the two output rows?
r0, r1 = out[0], out[1]
cos = np.dot(r0, r1) / (np.linalg.norm(r0) * np.linalg.norm(r1))
print("cosine similarity of outputs:", cos)
```

Every operation you learned by hand appears here: matrix multiplication for the layer, norms and a dot product for the similarity. This is what the paper math looks like when it grows up.

## Key takeaways

- `np.array` builds vectors and matrices; check `.shape` constantly — it's your first debugging tool.
- Element-wise `+`, `-`, and scalar `*` work just as on paper and need matching shapes.
- Use `@` for matrix multiplication and `*` for element-wise — confusing them is a classic silent bug.
- `np.linalg.norm` gives lengths and distances (`ord=1` for L1); `np.dot` gives dot products for similarity and projection.
- `np.linalg.eig` and `np.linalg.svd` give eigenvalues and singular values, the latter ranked largest-first.

## Try it

In a Python session:

1. Build `X` with shape `(3, 4)` and confirm `X.shape` and `X.T.shape`.
2. Build a compatible `W` so that `X @ W` is legal, and predict the output shape before running it.
3. Deliberately create a shape mismatch and read the error message NumPy gives you.
4. Take two rows of your output and compute their cosine similarity.
5. Run `np.linalg.svd` on any 3×3 matrix and confirm the singular values come out in descending order.
6. Prove to yourself that `A @ B` and `A * B` differ by printing both for a small pair of matrices.
