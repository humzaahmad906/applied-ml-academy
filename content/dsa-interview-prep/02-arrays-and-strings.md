# 02 — Arrays and Strings

Arrays and strings are the raw material of most coding-interview problems. In Python those are the `list` and the `str`, and to use them well under pressure you need to know two things: what operations are cheap versus expensive, and the handful of manipulation patterns that show up again and again. This lesson covers both, with the complexity of every operation stated up front so you never accidentally hide an O(n) cost inside what looks like a quick line.

## How a Python list is laid out

A Python list is a dynamic array: a contiguous block of references that grows automatically. Because the references sit next to each other in memory, reaching any element by index is a single hop — O(1) — regardless of how big the list is.

```python
nums = [10, 20, 30, 40, 50]
print(nums[0])    # 10    O(1)
print(nums[-1])   # 50    O(1), negative index counts from the end
print(len(nums))  # 5     O(1), the length is stored, not counted
```

Appending to the end is amortized O(1), as covered in lesson 01. The expensive operations are the ones that touch the *front* or *middle*, because every element after the touched position has to shift:

```python
nums = [10, 20, 30]
nums.append(40)       # O(1) amortized  -> [10, 20, 30, 40]
nums.pop()            # O(1)            -> [10, 20, 30]
nums.insert(0, 5)     # O(n) — shifts everything right -> [5, 10, 20, 30]
nums.pop(0)           # O(n) — shifts everything left  -> [10, 20, 30]
print(nums)           # [10, 20, 30]
```

The rule to burn in: **operate on the end of a list, never the front.** If a problem seems to need front operations, that is a signal to reach for `collections.deque` (lesson 05) instead.

Membership testing on a list is also a trap. `x in nums` scans element by element, so it is O(n). If you find yourself doing membership checks in a loop, you have an accidental O(n²), and the fix is a set (lesson 03).

```python
nums = [10, 20, 30]
print(20 in nums)     # True   but this is O(n), not O(1)
```

## Strings are immutable

A Python string behaves like an array of characters for reading, but you cannot change it in place — strings are immutable. Indexing and slicing read fine, but any "edit" actually builds a brand-new string.

```python
s = "hello"
print(s[0])       # h      O(1)
print(s[-1])      # o      O(1)
# s[0] = "H"      # TypeError: strings do not support item assignment
```

This immutability has a sharp consequence: building a string by repeated concatenation in a loop is O(n²), because each `+=` copies the entire string built so far.

```python
# SLOW — each += copies the whole accumulated string -> O(n^2)
out = ""
for ch in "abcde":
    out += ch

# FAST — collect pieces, join once at the end -> O(n)
parts = []
for ch in "abcde":
    parts.append(ch)
result = "".join(parts)
print(result)     # abcde
```

`"".join(iterable)` is the idiomatic, O(n) way to assemble a string from parts. Reach for it every time you would otherwise concatenate in a loop.

## Slicing, and what it costs

Slicing is one of Python's most loved features, but in an interview you must remember it copies. A slice `nums[a:b]` builds a new list (or string) containing those elements, so it costs O(b − a) time and space.

```python
nums = [0, 1, 2, 3, 4, 5]
print(nums[1:4])    # [1, 2, 3]     O(k) where k = 3
print(nums[:3])     # [0, 1, 2]     from the start
print(nums[3:])     # [3, 4, 5]     to the end
print(nums[::-1])   # [5, 4, 3, 2, 1, 0]   reversed copy, O(n)
print(nums[::2])    # [0, 2, 4]     every second element
```

`nums[::-1]` is the shortest way to reverse, but it allocates a full copy. When an interviewer asks you to reverse "in place" with O(1) extra space, you must not use it — use the two-pointer swap below.

## In-place operations

An in-place operation modifies the existing list without allocating a new one, achieving O(1) extra space. The workhorse technique is swapping elements with two indices moving from the ends toward the middle. Reversing in place is the classic:

```python
def reverse_in_place(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        nums[lo], nums[hi] = nums[hi], nums[lo]   # swap, O(1)
        lo += 1
        hi -= 1
    return nums          # O(n) time, O(1) extra space

print(reverse_in_place([1, 2, 3, 4, 5]))   # [5, 4, 3, 2, 1]
```

Python's tuple-swap `a, b = b, a` needs no temporary variable, which keeps swap-heavy code clean. This same "two indices from the ends" idea generalizes into the two-pointer pattern you will meet in lesson 04.

## Common array patterns

**Prefix sums** turn repeated range-sum queries from O(n) each into O(1) each, after an O(n) preprocessing pass. You build a running total once, then any range sum is a subtraction.

```python
def prefix_sums(nums):
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)   # running total
    return prefix          # prefix[i] = sum of first i elements

p = prefix_sums([2, 4, 6, 8])   # [0, 2, 6, 12, 20]
# sum of nums[1:3] (that is 4 + 6) = p[3] - p[1]
print(p[3] - p[1])              # 10   O(1) per query
```

**Building a frequency map in one pass** is O(n) and underlies a huge number of string problems (anagram checks, most-common-character, and more). Lesson 03 gives you `Counter`, but the manual version shows the idea:

```python
def char_counts(s):
    counts = {}
    for ch in s:                       # O(n)
        counts[ch] = counts.get(ch, 0) + 1
    return counts

print(char_counts("banana"))   # {'b': 1, 'a': 3, 'n': 2}
```

**Scanning with a running best** solves "maximum subarray sum" (Kadane's algorithm) in one O(n) pass. At each element you decide whether to extend the current run or start fresh:

```python
def max_subarray(nums):
    best = current = nums[0]
    for x in nums[1:]:                 # O(n)
        current = max(x, current + x)  # extend or restart
        best = max(best, current)
    return best

print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))   # 6  (from [4,-1,2,1])
```

## Two-dimensional lists

Matrices and grids come up constantly — image data, dynamic-programming tables, board games — and Python represents them as a list of lists. Build one with a comprehension, never with `*`, because `*` copies the same inner list reference and mutating one row mutates them all.

```python
rows, cols = 2, 3
grid = [[0] * cols for _ in range(rows)]   # correct: independent rows
grid[0][1] = 9
print(grid)          # [[0, 9, 0], [0, 0, 0]]

# BROKEN — every row is the SAME list object:
bad = [[0] * cols] * rows
bad[0][1] = 9
print(bad)           # [[0, 9, 0], [0, 9, 0]]   the 9 leaked into both rows
```

Access is `grid[row][col]`, and iterating over neighbors (up/down/left/right) is the backbone of grid-as-graph problems in lesson 06:

```python
def neighbors(r, c, n_rows, n_cols):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:   # stay in bounds
            yield nr, nc

print(list(neighbors(0, 0, 2, 3)))   # [(1, 0), (0, 1)]
```

Transposing a matrix (swap rows and columns) is a one-liner with `zip`, which is worth knowing when a problem is easier along the other axis:

```python
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = [list(row) for row in zip(*matrix)]
print(transposed)   # [[1, 4], [2, 5], [3, 6]]
```

## Useful string methods

A quick reference for the methods that carry their weight in interviews. Each returns a new value because strings are immutable:

```python
s = "  Hello, World  "
print(s.strip())            # "Hello, World"   remove surrounding whitespace
print(s.strip().lower())    # "hello, world"
print("a,b,c".split(","))   # ['a', 'b', 'c']  string -> list
print("-".join(["a","b"]))  # "a-b"            list -> string
print("hello".find("l"))    # 2                first index, or -1
print("hello".replace("l", "L"))   # "heLLo"
print("Hi123".isalpha())    # False
print("banana".count("a"))  # 3
```

Knowing `split` and `join` cold matters because parsing input and formatting output eat interview time you would rather spend on the algorithm.

## Key takeaways

- A Python list is a dynamic array: index access and length are O(1); appending and popping the *end* are O(1); inserting or popping the *front* or *middle* is O(n).
- `x in list` is O(n) — a membership check inside a loop is an accidental O(n²); use a set instead.
- Strings are immutable; building one with `+=` in a loop is O(n²), so collect parts and use `"".join(...)` for O(n).
- Slicing copies: `nums[a:b]` is O(b − a) in time and space, and `nums[::-1]` allocates a full reversed copy.
- Reverse or rearrange in place with two indices swapping from the ends for O(1) extra space.
- Learn the staple patterns: prefix sums for O(1) range queries, one-pass frequency maps, and running-best scans like Kadane's.

## Try it

1. **Two-sum check, the slow way first.** Write a function that returns `True` if any two numbers in a list add up to a target. Do it with nested loops, state its complexity, and note where the O(n²) comes from. (You will make this O(n) in lesson 03 — keep this version to compare.)

2. **Palindrome in place.** Write `is_palindrome(s)` that returns whether a string reads the same forwards and backwards, using two pointers from the ends and O(1) extra space — no slicing, no reversed copy. Test it on `"racecar"`, `"hello"`, and `""`.

3. **Fix the slow builder.** The function below is O(n²). Explain why, then rewrite it to be O(n) using `join`.

   ```python
   def repeat_upper(words):
       out = ""
       for w in words:
           out += w.upper() + " "
       return out.strip()
   ```
