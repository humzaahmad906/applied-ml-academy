# 03 — Matrix Multiplication

Matrix multiplication is the operation that makes machine learning *go*. Every prediction a neural network makes is, underneath, a chain of matrix multiplications. It looks strange the first time — it is *not* entry-by-entry like addition — but once the rule clicks, it stays clicked. Take your time here; this is the most important lesson in the course.

## The rule

To multiply two matrices, you slide *rows* of the first across *columns* of the second, and for each row-column pair you compute a **dot product**: multiply matching numbers and add them up.

Let's do the smallest interesting example:

```
A = [ 1  2 ]     B = [ 5  6 ]
    [ 3  4 ]         [ 7  8 ]
```

The result `C = A × B` is 2×2. Each entry of `C` comes from one row of `A` and one column of `B`:

- Row 1 of A `[1, 2]` with column 1 of B `[5, 7]`: `1×5 + 2×7 = 5 + 14 = 19`
- Row 1 of A `[1, 2]` with column 2 of B `[6, 8]`: `1×6 + 2×8 = 6 + 16 = 22`
- Row 2 of A `[3, 4]` with column 1 of B `[5, 7]`: `3×5 + 4×7 = 15 + 28 = 43`
- Row 2 of A `[3, 4]` with column 2 of B `[6, 8]`: `3×6 + 4×8 = 18 + 32 = 50`

```
C = [ 19  22 ]
    [ 43  50 ]
```

The entry in row `i`, column `j` of the result is always "row `i` of the left matrix, met with column `j` of the right matrix." That single sentence *is* matrix multiplication.

## Shapes must match

Here's the rule that trips up every beginner: the number of **columns in the left matrix** must equal the number of **rows in the right matrix**. If it doesn't, the operation is undefined.

Line the shapes up next to each other:

```
(2 × 3) × (3 × 4)  →  works, result is (2 × 4)
        \___/
      these must match
```

The two inner numbers must be equal. When they are, they cancel, and the two *outer* numbers give you the shape of the result. So a 2×3 times a 3×4 produces a 2×4. A 2×3 times a 2×4 is illegal, because 3 ≠ 2.

This is why the transpose from earlier is so handy: often two matrices *almost* fit, and transposing one flips its shape so the inner numbers line up.

A consequence worth remembering: **order matters**. `A × B` is generally *not* the same as `B × A`. Sometimes `B × A` isn't even a legal shape. This is very different from ordinary number multiplication, where `3 × 5 = 5 × 3`. With matrices, swapping the order changes — or breaks — the result.

## Why it works this way

The definition seems arbitrary until you see what a matrix *does*. A matrix is a machine that transforms vectors: feed a vector in, get a transformed vector out. Multiplying a matrix by a vector rotates, stretches, or reshapes that vector.

Now suppose you want to apply transformation `A`, and then transformation `B`, one after the other. Matrix multiplication `B × A` builds a *single* matrix that does both steps at once. That's the whole point: the row-times-column rule is exactly what's needed so that "multiply the matrices" equals "do one transformation, then the other." Chaining transformations is what a deep network is — each layer transforms the data, and stacking layers is stacking matrix multiplications.

## Cost intuition

Matrix multiplication is not free, and knowing roughly how expensive it is will make you a better ML practitioner. To produce each entry of the result, you do a dot product: a handful of multiplications plus additions. Multiply an `m×n` matrix by an `n×p` matrix and you produce `m × p` entries, each costing about `n` multiply-adds. So the total work scales like `m × n × p`.

The practical takeaway: cost grows *fast* as matrices get bigger. Double all three dimensions and the work grows roughly eightfold. This is exactly why training large models needs powerful hardware — a single forward pass can be billions of multiply-adds, and GPUs exist largely to grind through matrix multiplications in parallel. You don't need to count operations by hand, but carry the instinct that "bigger matrices means much more compute."

## A quick worked example

A tiny layer has a 1×3 input (three features) and a 3×2 weight matrix (turning 3 features into 2 outputs):

```
x = [ 2  1  3 ]        W = [ 1  0 ]
                           [ 0  2 ]
                           [ 1  1 ]

x × W = [ 2×1 + 1×0 + 3×1 ,  2×0 + 1×2 + 3×1 ]
      = [ 5 ,  5 ]
```

A 1×3 times a 3×2 gives a 1×2. Three features went in, two numbers came out — a single neural-network layer in miniature.

## Key takeaways

- Each result entry is the **dot product** of a row from the left matrix and a column from the right.
- Shapes must match: left columns = right rows. The inner numbers cancel; the outer numbers give the result's shape.
- Order matters — `A × B` usually differs from `B × A`, and may not even be legal.
- Multiplication chains transformations: it's *why* stacking network layers works.
- Cost scales like `m × n × p`; bigger matrices mean dramatically more compute, which is why ML needs fast hardware.

## Try it

With `A = [[1, 2, 0], [0, 1, 3]]` (2×3) and `B = [[2, 1], [0, 1], [4, 0]]` (3×2):

1. Confirm the shapes are compatible for `A × B`, and predict the result's shape before computing.
2. Compute `A × B` by hand, showing each row-column dot product.
3. Is `B × A` a legal operation? If so, what shape would it be?
4. Compute `B × A` and confirm it differs from `A × B` — proof that order matters.
5. Roughly how many multiply-adds did `A × B` take? Use the `m × n × p` rule.
