# 01 — Big-O and Complexity

Before you write a single line of interview code, you need a shared language for talking about how expensive that code is. That language is Big-O notation. When an interviewer asks "what's the time complexity?" they are not testing whether you memorized a table. They are checking whether you can predict how your solution behaves as the input grows, because a function that is fast on ten items and unusable on ten million is a real risk in a production ML pipeline. This lesson gives you that language and, more importantly, teaches you how to read it off your own code.

## What Big-O actually measures

Big-O describes how the amount of work a function does grows as its input grows. It deliberately ignores constants and small terms, because those wash out at scale. If one function does `3n + 100` operations and another does `n` operations, both are O(n): as `n` gets large, the difference stops mattering. What matters is the *shape* of the growth, not the exact count.

We care about the shape because hardware speed is roughly fixed, but input size is not. Doubling the input on an O(n) function doubles the time. Doubling it on an O(n²) function quadruples the time. That gap is the difference between a batch job that finishes overnight and one that never finishes.

## The common complexity classes

Here are the classes you will name in almost every interview, from best to worst:

```python
# O(1)      constant   — work does not depend on input size
# O(log n)  logarithmic — halve the problem each step (binary search)
# O(n)      linear     — touch each item once
# O(n log n) linearithmic — sort, or divide-and-conquer with linear merge
# O(n^2)    quadratic  — nested loop over the same input
# O(2^n)    exponential — try every subset (naive recursion)
# O(n!)     factorial  — try every ordering (naive permutations)
```

To feel the difference, imagine `n = 1000`. An O(n) function does about 1,000 operations. O(n log n) does about 10,000. O(n²) does 1,000,000. O(2ⁿ) does a number with 300 digits — it will not finish before the sun burns out. This is why turning an O(n²) solution into O(n) is the single most common "aha" an interviewer is waiting for.

```python
n = 1000
print(n)                 # 1000       O(n)
print(n * 10)            # 10000      ~O(n log n)
print(n * n)             # 1000000    O(n^2)
# 2 ** 1000 has 302 digits — do not print it in an interview
```

## Reading complexity off a loop

The mechanical rule: count how many times the innermost operation runs as a function of the input size, then keep only the fastest-growing term.

A single loop over `n` items is O(n):

```python
def total(nums):
    s = 0
    for x in nums:        # runs n times
        s += x            # O(1) each
    return s              # overall O(n)

print(total([1, 2, 3, 4]))   # 10
```

A loop nested inside another loop, both over the same input, is O(n²):

```python
def has_duplicate_slow(nums):
    for i in range(len(nums)):        # n iterations
        for j in range(i + 1, len(nums)):  # up to n iterations each
            if nums[i] == nums[j]:
                return True
    return False                      # overall O(n^2)

print(has_duplicate_slow([1, 2, 3, 2]))   # True
```

Two loops that are sequential, not nested, add rather than multiply. O(n) followed by O(n) is O(2n), which simplifies to O(n):

```python
def stats(nums):
    total = 0
    for x in nums:        # O(n)
        total += x
    biggest = nums[0]
    for x in nums:        # O(n)
        if x > biggest:
            biggest = x
    return total, biggest  # O(n) + O(n) = O(n)

print(stats([3, 1, 4, 1, 5]))   # (14, 5)
```

A loop whose counter is repeatedly halved (or doubled) is O(log n), because the number of steps to reach `n` by doubling is `log₂ n`:

```python
def count_halvings(n):
    steps = 0
    while n > 1:          # n -> n/2 -> n/4 -> ... -> 1
        n //= 2
        steps += 1
    return steps          # O(log n)

print(count_halvings(1000))   # 9   (2^9 = 512, 2^10 = 1024)
```

Binary search is the canonical O(log n) algorithm: each comparison throws away half the remaining candidates.

```python
def binary_search(sorted_nums, target):
    lo, hi = 0, len(sorted_nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2      # halve the range each step
        if sorted_nums[mid] == target:
            return mid
        elif sorted_nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1                     # overall O(log n)

print(binary_search([1, 3, 5, 7, 9, 11], 7))   # 3
print(binary_search([1, 3, 5, 7, 9, 11], 8))   # -1
```

## Reading complexity off recursion

For recursion, ask two questions: how many calls are made, and how much work does each call do? Multiply them.

A recursion that makes one call and shrinks the problem by one each time is O(n):

```python
def factorial(n):
    if n <= 1:            # base case
        return 1
    return n * factorial(n - 1)   # one call, n levels deep

print(factorial(5))       # 120   O(n) time, O(n) call-stack space
```

Naive Fibonacci makes *two* recursive calls per level and recomputes the same values over and over. The call tree roughly doubles at each level, giving O(2ⁿ):

```python
def fib_slow(n):
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)   # two calls -> O(2^n)

print(fib_slow(10))   # 55   (fib_slow(35) already feels slow)
```

Lesson 07 shows how memoization collapses that exponential tree back down to O(n) — a preview of why complexity analysis pays off directly.

## Space complexity

Time is only half the story. Space complexity counts the *extra* memory your algorithm allocates as the input grows, not counting the input itself. Building a new list of size `n` is O(n) space. Using a handful of variables is O(1) space, often called "in-place."

```python
def squares(nums):
    return [x * x for x in nums]   # allocates a new list -> O(n) space

def sum_in_place(nums):
    s = 0                          # one variable -> O(1) extra space
    for x in nums:
        s += x
    return s
```

Recursion also costs space: each pending call sits on the call stack. `factorial(n)` uses O(n) stack space even though it allocates no data structures. Interviewers frequently ask you to trade time for space or vice versa, so always be ready to state both.

## Amortized analysis

Some operations are usually cheap but occasionally expensive, and the honest way to describe them is the *average* cost over many operations. This is amortized analysis. The textbook example is Python's `list.append`. Most appends just drop a value into a pre-allocated slot in O(1). Occasionally the list is full, so Python allocates a bigger backing array and copies everything over — an O(n) event. But because it roughly *doubles* the capacity each time, those expensive copies happen rarely enough that the cost, spread across all appends, averages to O(1) per append.

```python
result = []
for i in range(1000):
    result.append(i)   # amortized O(1) each -> O(n) total, not O(n^2)
print(len(result))     # 1000
```

The takeaway: `append` in a loop is O(n) overall, which is exactly what you want. This is why building a list with `append` is fine, but be aware that operations like `list.insert(0, x)` (insert at the front) are genuinely O(n) *every* time, because every other element must shift.

## Why interviewers care

An interviewer watching you reach for a nested loop will often just wait, then ask "can you do better?" What they are really asking is: do you know that O(n²) has an O(n) alternative here, and can you find it? The whole toolkit in this course — hashing, two pointers, sliding windows, the right data structure — exists to knock an exponent off the complexity. Stating the complexity of your first idea out loud, then improving it, is exactly the thought process they want to see.

## Key takeaways

- Big-O describes how work grows with input size, ignoring constants and lower-order terms; the *shape* is what matters at scale.
- Memorize the ladder: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!).
- Read loops by counting innermost operations: nested loops multiply, sequential loops add, halving loops are O(log n).
- Read recursion as (number of calls) × (work per call); two-way recursion without memoization is often O(2ⁿ).
- Space complexity counts extra memory, including the recursion call stack; "in-place" means O(1) extra space.
- Amortized analysis gives the honest average cost: `list.append` is amortized O(1), but `list.insert(0, x)` is O(n) every time.
- Interviewers use complexity as the yardstick for "can you do better?" — state it for your first idea, then improve it.

## Try it

1. **Classify these functions.** For each, state the time and space complexity and justify it in one sentence: (a) a function that prints the first element of a list; (b) a function with two nested loops that both run to `n`; (c) a function that repeatedly divides `n` by 3 until it reaches 1; (d) a function that builds and returns a list of the squares of `0..n`. Predict before checking.

2. **Spot the hidden cost.** The following looks linear but is not. Explain its true complexity and rewrite it to be O(n). *Hint: what is the cost of `x in seen` when `seen` is a list?*

   ```python
   def uniques(nums):
       seen = []
       for x in nums:
           if x not in seen:   # what does this cost?
               seen.append(x)
       return seen
   ```

3. **Amortized reasoning.** Explain in your own words why appending `n` items to a Python list is O(n) total and not O(n²), then contrast it with calling `list.insert(0, x)` `n` times. What is the total complexity of the second case, and why?
