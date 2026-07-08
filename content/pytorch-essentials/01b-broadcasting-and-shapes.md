# 01b — Broadcasting and Shape Debugging

Lesson 01 covered how to reshape a tensor. This lesson covers the two skills that consume most of a beginner's debugging time and that interviewers love to probe: **broadcasting** (how PyTorch combines tensors of different shapes) and **reading shape-mismatch errors** (how to diagnose the `RuntimeError` you will inevitably hit). Get comfortable here and most of your PyTorch bugs turn from mysteries into ten-second fixes.

## Shapes are everything

Before any operation, ask one question: *what shape is this tensor, and what shape does the next step expect?* Almost every PyTorch bug is a disagreement between those two answers.

A tensor's shape is a tuple describing its size along each axis. Two properties are your constant companions:

```python
import torch

x = torch.randn(32, 128)
print(x.shape)   # torch.Size([32, 128])
print(x.ndim)    # 2  (number of dimensions / axes)
```

By convention the **first dimension is almost always the batch dimension** — how many independent examples you are processing at once. A few shapes you will see over and over:

```python
tabular = torch.randn(32, 10)        # (B, features)         — 32 rows, 10 features each
sequence = torch.randn(32, 50, 128)  # (B, seq_len, d_model) — text/time-series
images = torch.randn(32, 3, 224, 224) # (B, C, H, W)          — batch of RGB images
```

When you read `(32, 50, 128)`, train yourself to say "batch of 32, sequence length 50, embedding size 128" rather than "some three-dimensional thing." Naming the axes is half the battle.

## Broadcasting rules

Broadcasting lets you combine tensors of *different but compatible* shapes without manually copying data. PyTorch follows the exact same rules as NumPy. There are two:

1. **Align shapes from the right.** Compare dimensions starting from the last axis and working left.
2. **Each dimension pair must either be equal, or one of them must be 1.** A size-1 dimension is *stretched* (virtually repeated) to match the other. Missing leading dimensions are treated as 1.

The classic use is **adding a bias vector to a batch**:

```python
x = torch.randn(32, 128)      # (B, d)
b = torch.randn(128)          # (d,)
y = x + b                     # b is treated as (1, 128), stretched to (32, 128)
print(y.shape)                # torch.Size([32, 128])
```

Aligned from the right, `128` matches `128`; `x` has a leading `32` and `b` has nothing there, so `b` is broadcast across all 32 rows. This is exactly how a linear layer adds its bias.

A more deliberate example — building an outer product from a column and a row:

```python
col = torch.arange(3).reshape(3, 1)   # (3, 1)
row = torch.arange(4).reshape(1, 4)   # (1, 4)
grid = col + row
print(grid.shape)                     # torch.Size([3, 4])
```

Here the `1`s do the work in both directions: `col` stretches across 4 columns, `row` stretches across 3 rows.

### The `(B, 1) * (1, N)` trap

Broadcasting is happy to produce a bigger tensor than you intended, silently. Suppose you have per-sample scores and per-class weights and you *meant* an elementwise product:

```python
scores = torch.randn(32, 1)    # (B, 1) — one score per sample
weights = torch.randn(1, 10)   # (1, 10) — one weight per class
out = scores * weights
print(out.shape)               # torch.Size([32, 10])  — NOT (32,) or (10,)!
```

No error is raised. You wanted a vector and got a 32x10 matrix, and the bug only surfaces three lines later as a *different* shape mismatch. The lesson: whenever a size-1 dimension meets a size-N dimension, confirm the expansion is what you actually want.

## Reading a shape-mismatch traceback

When broadcasting *can't* reconcile two shapes, you get PyTorch's most common runtime error:

```python
a = torch.randn(32, 128)
b = torch.randn(32, 64)
c = a + b
# RuntimeError: The size of tensor a (128) must match the size of
# tensor b (64) at non-singleton dimension 1
```

Read this literally, it tells you everything:

- **"tensor a (128)" vs "tensor b (64)"** — the two conflicting sizes.
- **"at non-singleton dimension 1"** — the axis (index 1) where they clash. "Non-singleton" means neither is 1, so broadcasting can't save you.

Diagnosis routine: print both shapes and line them up from the right.

```python
print(a.shape, b.shape)   # torch.Size([32, 128]) torch.Size([32, 64])
#   a:  32  128
#   b:  32   64
#            ^ dimension 1 disagrees, and neither is 1 -> error
```

Now you decide: did you mean these to match (a real bug upstream), or does one need reshaping? A frequent real cause is forgetting the batch dimension — passing `(128,)` where the model expects `(1, 128)`. `unsqueeze` fixes that (see below).

## reshape vs view vs permute vs transpose

These four all rearrange axes, but they are not interchangeable.

**`view` and `reshape`** change the shape while keeping element order:

```python
t = torch.arange(12)
print(t.view(3, 4).shape)     # torch.Size([3, 4])
print(t.reshape(2, 6).shape)  # torch.Size([2, 6])
print(t.view(3, -1).shape)    # torch.Size([3, 4])  — -1 infers the size
```

**`transpose`** swaps two axes; **`permute`** reorders all of them:

```python
x = torch.randn(32, 3, 224, 224)      # (B, C, H, W)
print(x.transpose(1, 2).shape)        # torch.Size([32, 224, 3, 224]) — swap C and H
print(x.permute(0, 2, 3, 1).shape)    # torch.Size([32, 224, 224, 3]) — to (B, H, W, C)
```

### The contiguity gotcha

`transpose` and `permute` do not move data — they return a view with rearranged strides, so the tensor is no longer laid out contiguously in memory. `view` requires contiguous memory, so calling it after a transpose fails:

```python
x = torch.randn(2, 3)
xt = x.transpose(0, 1)        # (3, 2), non-contiguous
xt.view(6)
# RuntimeError: view size is not compatible with input tensor's size and stride
#   (at least one dimension spans across two contiguous subspaces).
#   Use .reshape(...) instead.
```

Two fixes: call `.contiguous()` first, or just use `reshape`, which copies when it has to.

```python
print(xt.contiguous().view(6).shape)  # torch.Size([6])
print(xt.reshape(6).shape)            # torch.Size([6])  — safe default
```

Rule of thumb: reach for `reshape` unless you have a specific reason to want `view`'s no-copy guarantee.

### Adding and removing axes

`unsqueeze` inserts a size-1 axis; `squeeze` removes size-1 axes. These are your tools for making shapes broadcast-compatible.

```python
v = torch.randn(128)
print(v.unsqueeze(0).shape)   # torch.Size([1, 128]) — add a batch dimension
print(v.unsqueeze(1).shape)   # torch.Size([128, 1]) — make it a column

t = torch.randn(1, 128, 1)
print(t.squeeze().shape)      # torch.Size([128]) — drop all size-1 axes
print(t.squeeze(0).shape)     # torch.Size([128, 1]) — drop only axis 0
```

## Batched matrix multiplication

`@` (equivalently `torch.matmul`) does more than 2D matrix multiply — it *batches* over leading dimensions, broadcasting them if needed. Only the last two axes participate in the actual matmul.

```python
A = torch.randn(32, 50, 128)   # (B, seq, d)
W = torch.randn(128, 64)       # (d, out)
Y = A @ W                      # W broadcast across the batch
print(Y.shape)                 # torch.Size([32, 50, 64])
```

When *both* operands carry a batch dimension, use `@`/`matmul` or the explicit `torch.bmm`. `bmm` is strict: it requires exactly 3D inputs with matching batch sizes and no broadcasting.

```python
P = torch.randn(32, 50, 128)
Q = torch.randn(32, 128, 64)
print(torch.bmm(P, Q).shape)   # torch.Size([32, 50, 64])
print((P @ Q).shape)           # torch.Size([32, 50, 64]) — same result
```

The inner dimensions must meet (`128` and `128` here). If they don't, you get a matmul-specific error naming the mismatched sizes, e.g. `mat1 and mat2 shapes cannot be multiplied (50x128 and 64x50)`.

## A debugging habit that pays for itself

Two cheap practices eliminate the majority of shape bugs:

**Print `.shape` at every step** while a pipeline is misbehaving:

```python
def forward(x):
    print("input   ", x.shape)
    x = x.reshape(x.shape[0], -1)
    print("flattened", x.shape)
    x = x @ W
    print("after mm ", x.shape)
    return x
```

**Assert the shapes you expect** so a wrong shape fails loudly *at its source* rather than three layers downstream:

```python
def attention(q, k, v):
    B, T, D = q.shape
    assert k.shape == (B, T, D), f"k has wrong shape: {k.shape}"
    assert v.shape == (B, T, D), f"v has wrong shape: {v.shape}"
    ...
```

An assert that fires on line 2 is worth an hour of tracing a `RuntimeError` that surfaced on line 40.

## Key takeaways

- The **first dimension is the batch**; learn to name the axes of common shapes like `(B, seq, d)` and `(B, C, H, W)`.
- Broadcasting aligns shapes **from the right**; each dimension pair must be **equal or one must be 1**, and size-1 dimensions stretch.
- A size-1 axis meeting a size-N axis expands silently — the `(B, 1) * (1, N)` trap makes a matrix when you wanted a vector.
- Read `RuntimeError: The size of tensor a (X) must match tensor b (Y) at non-singleton dimension D` literally: sizes `X`/`Y` clash at axis `D`. Print both shapes and align them.
- `transpose`/`permute` return non-contiguous views, so `view` may fail afterward — use `reshape` (or `.contiguous()`) as the safe default; use `unsqueeze`/`squeeze` to add/remove size-1 axes.
- `@`/`matmul` batch over leading dims; `torch.bmm` is the strict 3D-only version.
- When in doubt, **print `.shape` at each step and assert the shapes you expect.**

## Try it

Start with `x = torch.randn(8, 3, 32, 32)` — a batch of 8 RGB 32x32 images.

1. Print its `.shape` and `.ndim`. Name each axis out loud.
2. Flatten each image into a vector using `reshape` so the result is `(8, 3072)`. Confirm `3 * 32 * 32 == 3072`.
3. Add a bias vector `b = torch.randn(3072)` to it via broadcasting and confirm the shape is unchanged. What shape does `b` need to be for this to work — and what happens if you use `b = torch.randn(8)` instead? (Read the error.)
4. Permute the original `x` to channels-last `(8, 32, 32, 3)` with `permute`, then try to `view` it as `(8, 3072)`. Observe the contiguity error, then fix it two ways.
5. Multiply the flattened `(8, 3072)` batch by a weight matrix `W = torch.randn(3072, 10)` using `@` and confirm you get `(8, 10)` class scores.
