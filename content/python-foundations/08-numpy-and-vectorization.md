# 08 — NumPy and Vectorized Math

This is where Python turns into a serious tool for data and machine learning. NumPy is a library for working with arrays of numbers, and it is the foundation that nearly every data and ML library on this platform is built on. It brings two things you will lean on constantly: a fast, compact array type, and the ability to do math on whole collections of numbers at once. This final foundation lesson introduces both, and explains why they matter so much.

## Getting NumPy

NumPy is not part of Python itself, so you install it once:

```
pip install numpy
```

By strong convention, everyone imports it under the short name `np`:

```python
import numpy as np
```

You will see `np` in essentially every piece of numerical Python you ever read, so it is worth adopting from the start.

## Arrays

The heart of NumPy is the array: an ordered grid of numbers, all of the same type. You create one from a list:

```python
import numpy as np

a = np.array([1, 2, 3, 4])
print(a)          # [1 2 3 4]
print(a.shape)    # (4,)
print(a.dtype)    # int64
```

An array knows its shape (how many items along each dimension) and its dtype (the type of number it holds). Arrays can have more than one dimension, which is how you represent tables, images, and more:

```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(grid.shape)   # (2, 3)  -> 2 rows, 3 columns
```

NumPy also offers quick ways to build common arrays:

```python
print(np.zeros(3))          # [0. 0. 0.]
print(np.ones((2, 2)))      # a 2x2 array of ones
print(np.arange(0, 10, 2))  # [0 2 4 6 8]
print(np.linspace(0, 1, 5)) # [0.   0.25 0.5  0.75 1.  ]
```

## Why loops are slow

Suppose you want to add two lists of a million numbers element by element. In plain Python you would write a loop:

```python
result = []
for x, y in zip(list_a, list_b):
    result.append(x + y)
```

This works, but it is slow. Python checks the type of every value, dispatches the addition, and manages the list, a million times over, all in comparatively slow interpreted code. For the scale of data in machine learning, this quickly becomes painful.

NumPy solves this with vectorization: you express the operation on the whole array at once, and NumPy runs the loop internally in fast, compiled code operating on tightly packed memory. The same addition becomes a single expression:

```python
result = array_a + array_b
```

This is not just shorter; it is often tens or hundreds of times faster. The lesson to carry forward is a mindset shift: in NumPy, prefer operating on whole arrays over writing your own loops.

## Vectorized operations

Arithmetic on arrays applies element by element, automatically:

```python
a = np.array([1, 2, 3, 4])

print(a + 10)     # [11 12 13 14]
print(a * 2)      # [2 4 6 8]
print(a ** 2)     # [1 4 9 16]

b = np.array([10, 20, 30, 40])
print(a + b)      # [11 22 33 44]
```

NumPy also provides fast whole-array summaries, called aggregations:

```python
a = np.array([3, 1, 4, 1, 5, 9])

print(a.sum())    # 23
print(a.mean())   # 3.833...
print(a.max())    # 9
print(a.min())    # 1
```

And you can select parts of an array with conditions, which returns just the matching elements. This is called boolean indexing and it replaces many loops:

```python
a = np.array([3, 1, 4, 1, 5, 9])
print(a[a > 3])   # [4 5 9]
```

## Broadcasting

Broadcasting is NumPy's rule for combining arrays of different shapes. You have already used its simplest form: when you wrote `a + 10`, the single number 10 was stretched to match every element of the array. Broadcasting generalizes this so that compatible shapes can be combined without writing loops:

```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
row = np.array([10, 20, 30])

print(grid + row)
# [[11 22 33]
#  [14 25 36]]
```

Here the one-dimensional `row` was applied to each row of the two-dimensional `grid`. NumPy lined the shapes up and stretched the smaller one to fit. Broadcasting is what lets you, for example, subtract the average of each column from a whole table in a single line, a routine step in preparing data for machine learning:

```python
data = np.array([[1.0, 2.0],
                 [3.0, 4.0],
                 [5.0, 6.0]])
column_means = data.mean(axis=0)   # average down each column
centered = data - column_means
print(centered)
# [[-2. -2.]
#  [ 0.  0.]
#  [ 2.  2.]]
```

The `axis=0` argument tells NumPy to work down the columns; `axis=1` would work across the rows. Choosing the right axis is a small but important skill you will use constantly.

## Why this matters

Almost every tool you will meet later, for handling tables of data, training models, and processing images or text, is built on NumPy arrays or on structures that behave like them. The habits from this lesson, thinking in whole arrays instead of loops, watching shapes, and letting broadcasting do the work, are the habits that make the rest of the journey smooth. You now have the complete foundation the other courses assume.

## Key takeaways

- NumPy is the numerical foundation for data and ML in Python; import it as `import numpy as np`.
- An array is a grid of same-typed numbers with a `shape` and a `dtype`; arrays can be multi-dimensional.
- Python loops over big numeric data are slow; vectorized array operations run in fast compiled code.
- Arithmetic and comparisons apply element by element, and aggregations like `sum` and `mean` summarize whole arrays.
- Boolean indexing (`a[a > 3]`) selects matching elements without a loop.
- Broadcasting stretches compatible shapes to combine them; use `axis` to work down columns or across rows.

## Try it

Create a NumPy array of the numbers 1 through 10 and, in single expressions, compute their sum, mean, and the array of their squares. Use boolean indexing to pull out only the values greater than 5. Then build a two-dimensional array of exam scores with a few students as rows and three subjects as columns, and use broadcasting with `axis=0` to subtract each subject's average from every score, producing a table centered on the subject means. Finally, for a sense of scale, time a plain Python loop against the equivalent vectorized operation over a large array and see the difference for yourself.
