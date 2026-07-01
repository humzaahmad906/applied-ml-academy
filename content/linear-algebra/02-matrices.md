# 02 — Matrices and Operations

A vector is a list of numbers. A **matrix** is a grid of numbers — rows and columns lined up in a rectangle. If vectors are the words of machine learning, matrices are the sentences. A single layer of a neural network is a matrix. A batch of images is a matrix. Learning to read a matrix's shape and manipulate it is the next step after vectors.

## What a matrix looks like

Here is a matrix with 2 rows and 3 columns:

```
A = [ 1  2  3 ]
    [ 4  5  6 ]
```

We describe its size as its **shape**, written rows-by-columns. This one is a 2×3 matrix ("two by three"). The number in row `i`, column `j` is called an **entry** or **element**. In `A`, the entry in row 2, column 3 is `6`.

Shape is the thing you will check constantly, so build the habit now: always know how many rows and how many columns you have. Rows first, columns second — every time.

You can think of a matrix in a few useful ways:

- **As a stack of rows.** Each row is a vector. `A` is two 3-vectors stacked on top of each other.
- **As a set of columns.** Each column is a vector. `A` is three 2-vectors sitting side by side.
- **As a table of data.** Rows are examples (say, three customers), columns are features (age, income, visits). Machine learning datasets are almost always matrices in exactly this sense.

Some special cases have names. A matrix with the same number of rows and columns is **square**. A matrix with a single column is really just a vector (a "column vector"). A single row is a "row vector." These aren't different objects — they're matrices with a shape that makes them look familiar.

## Transpose: flipping rows and columns

The **transpose** of a matrix swaps its rows and columns. The first row becomes the first column, the second row becomes the second column, and so on. We write it with a small `T`:

```
A = [ 1  2  3 ]        A^T = [ 1  4 ]
    [ 4  5  6 ]              [ 2  5 ]
                            [ 3  6 ]
```

`A` was 2×3; its transpose `A^T` is 3×2. The shape flips too. Transposing twice gets you back where you started.

Why care? Two reasons. First, transpose is how you line up shapes so operations fit together — you'll see this constantly when we get to matrix multiplication. Second, it's how you switch between "rows are examples" and "columns are examples" views of your data, which happens all the time in real code.

## Adding matrices

Matrix addition works just like vector addition: add entry by entry. The two matrices must have the *exact same shape*.

```
[ 1  2 ] + [ 5  6 ] = [ 1+5  2+6 ] = [ 6  8 ]
[ 3  4 ]   [ 7  8 ]   [ 3+7  4+8 ]   [ 10 12 ]
```

If the shapes don't match — say a 2×2 plus a 2×3 — the operation is simply undefined. There's no sensible way to add a grid to a grid of a different size. This "shapes must match" rule is one of the most common sources of bugs for beginners, and one of the most common error messages you'll see in real ML code.

## Scalar multiplication

Multiplying a matrix by a single number multiplies *every* entry by that number:

```
3 * [ 1  2 ] = [ 3  6 ]
    [ 4  5 ]   [ 12 15 ]
```

This is exactly the scaling you learned for vectors, applied to the whole grid at once. It stretches all the values uniformly. Combined with addition, it lets you form weighted blends of matrices — for instance, `0.9 * old + 0.1 * new`, a pattern that appears whenever a model updates a running average of its weights.

## A quick worked example

Say a dataset holds two people's scores on two tests:

```
scores = [ 80  90 ]   (person 1)
         [ 70  60 ]   (person 2)
```

The teacher adds a 5-point bonus to everyone and then curves by scaling to 1.1×. Step one, add a bonus matrix; step two, scale:

```
scores + [ 5  5 ] = [ 85  95 ]
         [ 5  5 ]   [ 75  65 ]

1.1 * [ 85  95 ] = [ 93.5  104.5 ]
      [ 75  65 ]   [ 82.5  71.5  ]
```

Every operation here is entry-wise, and every one required the shapes to line up.

## Why this matters for ML

A neural network layer stores its knowledge as a matrix of **weights**. Your data arrives as a matrix. Training nudges the weight matrix a little at each step — that nudge is a scaled matrix added to the current one. Before any of the heavier machinery makes sense, you need to be fluent in reading shapes, transposing to line things up, and combining matrices with addition and scaling.

## Key takeaways

- A matrix is a grid of numbers with a **shape** written rows×columns; always know both numbers.
- You can read a matrix as stacked row-vectors, side-by-side column-vectors, or a data table of examples and features.
- **Transpose** swaps rows and columns and flips the shape; it's how you line things up.
- Addition is entry-wise and requires identical shapes; scalar multiplication scales every entry.
- These entry-wise operations are the everyday tools for storing data and nudging model weights.

## Try it

Given `A = [[2, 0, 1], [3, 1, 4]]` (2×3) and `B = [[1, 5, 2], [0, 2, 6]]` (2×3):

1. State the shape of `A` and of `A` transposed.
2. Write out `A^T` fully.
3. Compute `A + B`.
4. Compute `2 * A`.
5. Try to compute `A + A^T`. What goes wrong, and why? What does that tell you about when addition is allowed?
