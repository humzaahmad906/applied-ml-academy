# 01 — Tensors

Tensors are the core data structure in PyTorch. If you have used NumPy arrays, you already have most of the mental model: a tensor is a multi-dimensional array of numbers. What makes PyTorch tensors special is that they can live on a GPU and they can track the operations performed on them so gradients can be computed automatically. In this lesson we focus on the array part — creating tensors, understanding their shapes and dtypes, doing math with them, and reshaping.

## Creating tensors

The most direct way to make a tensor is from a Python list.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x)          # tensor([1., 2., 3.])
print(x.shape)    # torch.Size([3])
```

You can build nested lists for higher dimensions:

```python
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(m.shape)    # torch.Size([2, 3])  -> 2 rows, 3 columns
```

Often you want a tensor of a known shape filled with a constant or random values. These factory functions take the shape as arguments:

```python
zeros = torch.zeros(2, 3)          # all zeros, shape (2, 3)
ones = torch.ones(4)               # [1., 1., 1., 1.]
rand = torch.rand(2, 2)            # uniform random in [0, 1)
randn = torch.randn(2, 2)          # standard normal (mean 0, std 1)
seq = torch.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
lin = torch.linspace(0, 1, 5)      # 5 evenly spaced points from 0 to 1
```

To match the shape of an existing tensor, the `_like` variants are handy:

```python
y = torch.zeros_like(rand)         # zeros with the same shape and dtype as rand
```

## Shapes and dimensions

The **shape** describes how many elements exist along each axis. A vector has one axis, a matrix has two, and an image batch might have four (batch, channels, height, width). The number of axes is the **rank** or number of dimensions.

```python
t = torch.randn(8, 3, 32, 32)   # a batch of 8 RGB 32x32 images
print(t.ndim)       # 4
print(t.shape)      # torch.Size([8, 3, 32, 32])
print(t.numel())    # 24576  (total number of elements)
```

Getting comfortable reading shapes is the single most useful skill in PyTorch. Most bugs are shape mismatches, and printing `.shape` is your best debugging tool.

## Data types (dtypes)

Every tensor has a dtype. The two you will use most are `torch.float32` (the default for floating point) and `torch.int64` (the default for integer tensors and the standard type for class labels).

```python
a = torch.tensor([1.0, 2.0])
print(a.dtype)                 # torch.float32

b = torch.tensor([1, 2, 3])
print(b.dtype)                 # torch.int64

c = a.to(torch.float16)        # cast to half precision
d = b.float()                  # shortcut to float32
```

Mixing dtypes carelessly causes errors. If you feed integer data into a model expecting floats, cast it first with `.float()`.

## Operations

Arithmetic works elementwise, and PyTorch supports operator overloading so the code reads naturally.

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([10.0, 20.0, 30.0])

print(x + y)        # tensor([11., 22., 33.])
print(x * y)        # tensor([10., 40., 90.]) elementwise product
print(x ** 2)       # tensor([1., 4., 9.])
print(x.sum())      # tensor(6.)
print(x.mean())     # tensor(2.)
```

For matrix multiplication use `@` or `torch.matmul`, not `*`:

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B           # shape (2, 4)
```

**Broadcasting** lets tensors of different but compatible shapes combine without manual copying. A scalar or a smaller tensor is stretched to match the larger one:

```python
m = torch.ones(2, 3)
row = torch.tensor([1.0, 2.0, 3.0])   # shape (3,)
print(m + row)      # row is added to every row of m
```

The rule: align shapes from the right; each pair of dimensions must be equal, or one of them must be 1.

## Reshaping

You will constantly reorganize the same data into different shapes. `view` and `reshape` change the shape without changing the data:

```python
t = torch.arange(12)         # shape (12,)
a = t.view(3, 4)             # shape (3, 4)
b = t.reshape(2, 6)          # shape (2, 6)
c = t.view(3, -1)            # -1 means "infer this dimension" -> (3, 4)
```

`view` requires the tensor to be contiguous in memory; `reshape` is more forgiving and will copy if needed, so `reshape` is a safe default.

Other useful shape tools:

```python
t = torch.randn(1, 3, 1)
print(t.squeeze().shape)        # (3,)  removes dimensions of size 1
print(t.unsqueeze(0).shape)     # (1, 1, 3, 1) adds a dim at position 0
x = torch.randn(2, 3)
print(x.transpose(0, 1).shape)  # (3, 2) swaps two dimensions
print(x.permute(1, 0).shape)    # (3, 2) reorders all dimensions
```

## Indexing and NumPy bridge

Indexing and slicing work like NumPy:

```python
t = torch.arange(12).reshape(3, 4)
print(t[0])         # first row
print(t[:, 1])      # second column
print(t[1, 2])      # single element (still a tensor)
print(t[t > 5])     # boolean mask selects matching elements
```

You can convert between NumPy and PyTorch cheaply:

```python
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)     # shares memory with arr
back = t.numpy()              # back to a NumPy array
```

## Key takeaways

- A tensor is a multi-dimensional array; its **shape**, **dtype**, and later its device fully describe it.
- Create tensors with `torch.tensor`, `zeros`, `ones`, `rand`, `randn`, and `arange`.
- Math is elementwise; use `@` for matrix multiplication and rely on **broadcasting** for compatible shapes.
- Reshape with `view`/`reshape` (use `-1` to infer a dimension) and adjust dimensions with `squeeze`, `unsqueeze`, `transpose`, and `permute`.
- When something breaks, print `.shape` first.

## Try it

Create a tensor `x = torch.arange(24, dtype=torch.float32)`. Reshape it into a `(2, 3, 4)` tensor. Then:

1. Print its shape and total number of elements.
2. Compute the mean over the last dimension (hint: `x.mean(dim=-1)`) and print the resulting shape.
3. Reshape it back to a 2D tensor with 6 rows using `-1` for the columns.
4. Multiply it by a broadcasted row vector of length equal to your column count and confirm the shape is unchanged.
