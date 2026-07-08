# 07 — Recursion and Dynamic Programming

Dynamic programming (DP) is the topic candidates dread most, but it is nothing more than recursion with the wasteful repetition removed. If you can write a correct recursive solution, you are two mechanical steps away from an efficient DP one. This lesson walks that exact path — recursion, then memoization, then tabulation — on problems small enough to hold in your head, then applies it to the classics interviewers actually ask: knapsack, edit distance, and longest increasing subsequence. The throughline: find the recurrence, then stop recomputing.

## Recursion: solve a problem in terms of smaller versions of itself

A recursive function calls itself on a smaller input until it hits a base case simple enough to answer directly. Two ingredients are mandatory: a **base case** that stops the recursion, and a **recursive case** that makes progress toward it. Miss the base case and you recurse forever.

```python
def factorial(n):
    if n <= 1:                 # base case
        return 1
    return n * factorial(n - 1)   # recursive case, smaller input

print(factorial(5))   # 120
```

The trouble starts when the same subproblem gets solved many times. Naive Fibonacci is the poster child: computing `fib(5)` recomputes `fib(3)` twice, `fib(2)` three times, and so on. The call tree branches, giving O(2ⁿ) — unusable past `n` around 35.

```python
def fib_slow(n):
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)   # recomputes everything

print(fib_slow(10))   # 55   but fib_slow(40) is painfully slow
```

The insight of DP: the number of *distinct* subproblems here is only `n` (`fib(0)` through `fib(n)`). We are just solving each many times. Cache the answers and the exponential collapses to linear.

## Step one: memoization (top-down DP)

Memoization means storing each subproblem's answer the first time you compute it and returning the stored value on later calls. You keep the recursive structure and add a cache. In Python the cleanest way is the `functools.lru_cache` decorator, which wraps any function so identical-argument calls are served from memory.

```python
from functools import lru_cache

@lru_cache(maxsize=None)          # cache every distinct call
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(10))    # 55
print(fib(100))   # 354224848179261915075   instant — O(n) with the cache
```

That one decorator turns O(2ⁿ) into O(n) time (each of the `n` subproblems computed once) and O(n) space (the cache). `lru_cache` requires the function's arguments to be hashable — the same immutability rule from lesson 03 — so pass tuples, not lists. When you want the cache logic explicit, a plain dict does the same job:

```python
def fib_memo(n, cache=None):
    if cache is None:
        cache = {}
    if n < 2:
        return n
    if n not in cache:
        cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)
    return cache[n]

print(fib_memo(50))   # 12586269025
```

Memoization is called *top-down* because you start from the problem you want and recurse down to base cases, caching along the way.

## Step two: tabulation (bottom-up DP)

Tabulation flips the direction: start from the base cases and iteratively fill a table upward until you reach the answer. There is no recursion, so no call-stack overhead and no risk of hitting Python's recursion limit. The recurrence is identical; only the control flow changes.

```python
def fib_table(n):
    if n < 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]   # build up from the base cases
    return dp[n]                         # O(n) time, O(n) space

print(fib_table(10))   # 55
```

Often you notice each entry depends only on the last couple, so you can throw away the full table and keep O(1) space — a common interview follow-up ("can you reduce the space?").

```python
def fib_optimized(n):
    prev, curr = 0, 1
    for _ in range(n):
        prev, curr = curr, prev + curr   # only keep the last two
    return prev                          # O(n) time, O(1) space

print(fib_optimized(10))   # 55
```

Memoization and tabulation compute the same thing with the same complexity; pick memoization when the recurrence is easier to see recursively, tabulation when you want to avoid recursion overhead or optimize space.

## The recipe for any DP problem

1. **Define the subproblem** in words: what does `dp[i]` (or `dp[i][j]`) mean?
2. **Write the recurrence**: how does a subproblem's answer combine smaller ones?
3. **State the base cases**.
4. **Choose a direction**: memoized recursion or a bottom-up table.
5. **Read off the answer** and, if asked, optimize the space.

Getting steps 1 and 2 right is the whole game. The rest is mechanical.

## Classic 1: 0/1 knapsack

You have items each with a weight and value, and a capacity. Maximize total value without exceeding capacity, taking each item at most once. Subproblem: `dp[i][c]` is the best value using the first `i` items with capacity `c`. For each item you choose the better of skipping it or taking it (if it fits).

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]                 # skip item i
            if w <= c:                              # or take it, if it fits
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)
    return dp[n][capacity]     # O(n * capacity) time and space

print(knapsack([1, 3, 4], [15, 20, 30], 4))   # 35  (items of weight 1 + 3)
```

Note this is O(n × capacity), not O(2ⁿ) as brute-forcing every subset would be — that gap is the payoff of DP.

## Classic 2: edit distance

The minimum number of single-character insertions, deletions, or substitutions to turn one string into another. Subproblem: `dp[i][j]` is the edit distance between the first `i` characters of `a` and the first `j` of `b`. If the current characters match, no edit is needed; otherwise take the cheapest of the three operations.

```python
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i           # delete all of a's first i chars
    for j in range(n + 1):
        dp[0][j] = j           # insert all of b's first j chars
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]        # match, no cost
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],   # delete
                                   dp[i][j - 1],   # insert
                                   dp[i - 1][j - 1])  # substitute
    return dp[m][n]            # O(m * n)

print(edit_distance("kitten", "sitting"))   # 3
```

Edit distance is not just a puzzle: it underlies diff tools, spell-checkers, and sequence-alignment metrics you may meet evaluating text or bioinformatics models.

## Classic 3: longest increasing subsequence

The length of the longest subsequence (not necessarily contiguous) whose values strictly increase. Subproblem: `dp[i]` is the length of the longest increasing subsequence *ending at* index `i`. Each element extends the best compatible earlier one.

```python
def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)       # each element alone is a subsequence of length 1
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:            # can extend
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)             # O(n^2); an O(n log n) version exists with binary search

print(length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]))   # 4  ([2, 3, 7, 101])
```

Mentioning that an O(n log n) solution exists (via patience sorting / binary search) is exactly the kind of "I know the better bound is out there" remark interviewers reward, even if you code the O(n²) version.

## Key takeaways

- Recursion needs a base case and a recursive case that shrinks the input; without a base case it never stops.
- Naive branching recursion (like Fibonacci) is O(2ⁿ) because it recomputes the same subproblems; DP removes that waste.
- Memoization (top-down) keeps the recursion and caches results; `functools.lru_cache` does it in one decorator, and requires hashable arguments.
- Tabulation (bottom-up) fills a table from base cases with a loop — no recursion overhead, and often lets you shrink space to O(1).
- The DP recipe: define the subproblem, write the recurrence, state base cases, pick a direction, read off the answer.
- Know the classics: 0/1 knapsack is O(n × capacity); edit distance and LIS are O(m × n) / O(n²), each far below the brute-force exponential.

## Try it

1. **Climbing stairs.** You can climb 1 or 2 steps at a time; how many distinct ways to reach step `n`? Solve it three ways — naive recursion, `@lru_cache` memoization, and an O(1)-space bottom-up loop — and confirm all three agree for `n = 10` (expect 89). Notice it is Fibonacci in disguise.

2. **Coin change.** Given coin denominations and an amount, return the fewest coins that sum to the amount (-1 if impossible). Define `dp[a]` as the fewest coins for amount `a`, and build up from `dp[0] = 0`. Test on `([1, 2, 5], 11)` (expect 3, from 5+5+1). State the complexity.

3. **Longest common subsequence.** Given two strings, return the length of their longest common subsequence. Set up a 2-D table like edit distance: when characters match, extend the diagonal; otherwise take the max of dropping one character from either string. Test on `("abcde", "ace")` (expect 3).
