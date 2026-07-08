# 04 — Two Pointers and Sliding Window

Two patterns solve a startling share of array and string interview problems, and they are close cousins. The two-pointer pattern uses two indices moving through a sequence to avoid a nested loop. The sliding-window pattern is a specialized version where the two pointers bound a contiguous stretch you grow and shrink. Both take a natural O(n²) brute force down to O(n). This lesson teaches you to recognize when each applies and drills the two problems interviewers reach for most: pair-sum on a sorted array, and longest substring without repeats.

## The core idea: replace a nested loop with coordinated indices

The brute-force instinct for "find something about a pair of elements" is two nested loops, which is O(n²). The two-pointer insight is that if the data has some structure — usually *sortedness* — you can move two indices intelligently and touch each element only a constant number of times, getting O(n).

There are two common arrangements. In the **converging** arrangement, one pointer starts at the left, the other at the right, and they move toward each other. In the **fast/slow** arrangement, both start at the left and move at different speeds (you will see this on linked lists in lesson 05).

## Two pointers, converging: pair sum on a sorted array

Suppose you want two numbers in a *sorted* array that add up to a target. Put a pointer at each end and look at their sum. If the sum is too small, the only way to increase it is to move the left pointer right (to a bigger value). If it is too big, move the right pointer left. Each move eliminates a value that cannot possibly work, so you never backtrack.

```python
def pair_sum_sorted(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target:
            return [lo, hi]      # found it
        elif s < target:
            lo += 1              # need a bigger sum -> raise the low end
        else:
            hi -= 1              # need a smaller sum -> lower the high end
    return None                  # O(n) time, O(1) space

print(pair_sum_sorted([1, 2, 4, 7, 11, 15], 15))   # [1, 4]  (4 + 11)
print(pair_sum_sorted([1, 2, 3], 7))               # None
```

Why is this correct? At every step, one of the two pointers is at a value that cannot be part of any valid pair given what we have already ruled out, so discarding it loses no solution. The array must be sorted for the "too small / too big" logic to hold — if it is not, sort first (O(n log n)) or use the hash-based Two Sum from lesson 03.

The same converging pattern removes duplicates from a sorted array in place, checks palindromes (lesson 02), and merges two sorted lists. It also generalizes: the classic Three Sum problem fixes one element and runs this two-pointer scan on the rest, turning a naive O(n³) into O(n²).

```python
def remove_duplicates_sorted(nums):
    if not nums:
        return 0
    write = 1                        # next slot for a new unique value
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1
    return write                     # length of the deduped prefix

data = [1, 1, 2, 2, 2, 3]
n = remove_duplicates_sorted(data)
print(n, data[:n])                   # 3 [1, 2, 3]
```

That is a two-pointer variant where a slow "write" pointer trails a fast "read" pointer — a template for compacting arrays in place with O(1) extra space.

## Sliding window: the pointers bound a range

A sliding window is two pointers marking the start and end of a contiguous sub-range. You extend the window by advancing the right pointer, and when the window violates some condition, you shrink it by advancing the left pointer. Because each pointer only ever moves forward, and each moves at most `n` times, the whole scan is O(n) even though the window is constantly resizing.

Windows come in two flavors. A **fixed-size** window keeps a constant width — useful for "maximum sum of any `k` consecutive elements." A **variable-size** window grows and shrinks to satisfy a constraint — useful for "longest / shortest sub-range with property X."

### Fixed window: maximum sum of k consecutive elements

Rather than re-summing every window from scratch (O(n·k)), you slide: add the new element entering on the right, subtract the one leaving on the left. Each step is O(1).

```python
def max_sum_window(nums, k):
    window = sum(nums[:k])           # first window, O(k)
    best = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i - k]   # add new, drop old — O(1)
        best = max(best, window)
    return best                      # O(n) total

print(max_sum_window([2, 1, 5, 1, 3, 2], 3))   # 9   (5 + 1 + 3)
```

### Variable window: longest substring without repeating characters

This is the interview classic. You grow the window to the right, tracking the characters currently inside with a set. The moment the incoming character is already in the window, you shrink from the left until the duplicate is gone, then continue. The answer is the largest width the window ever reached.

```python
def longest_unique_substring(s):
    seen = set()
    left = 0
    best = 0
    for right, ch in enumerate(s):
        while ch in seen:            # duplicate — shrink from the left
            seen.remove(s[left])
            left += 1
        seen.add(ch)                 # now the window is valid again
        best = max(best, right - left + 1)
    return best                      # O(n): each char added and removed once

print(longest_unique_substring("abcabcbb"))   # 3   ("abc")
print(longest_unique_substring("bbbbb"))       # 1   ("b")
print(longest_unique_substring("pwwkew"))      # 3   ("wke")
```

The subtle point that makes this O(n) and not O(n²): although there is a `while` loop inside a `for` loop, the `left` pointer only ever advances, and it can advance at most `n` times *total* across the entire run. Every character is added to the set once and removed at most once. Being able to explain this to an interviewer is worth as much as writing the code.

## Choosing between them

Reach for **two converging pointers** when the data is sorted (or you can sort it) and you are looking for a pair, a triple, or want to partition/compact in place. The tell is "sorted" plus "pair/sum/palindrome."

Reach for a **sliding window** when the problem asks about a *contiguous* subarray or substring and some property of it — longest, shortest, sum, count of distinct, contains all of. The tells are "contiguous," "consecutive," "substring/subarray," and a constraint you can check as the window grows.

If the subrange does not need to be contiguous, or the data is unsorted and you are matching values, hashing (lesson 03) is usually the better tool. Many problems admit more than one approach; naming the trade-off out loud is exactly the signal interviewers want.

```python
# quick decision guide
# sorted array + find a pair/triple/partition   -> two converging pointers
# contiguous subarray/substring + a constraint  -> sliding window
# unsorted + "have I seen a matching value?"     -> hash set / dict
```

## Key takeaways

- Two pointers and sliding windows both replace an O(n²) nested loop with an O(n) single pass by moving indices that never backtrack.
- Converging pointers (one at each end) solve sorted pair-sum, palindromes, in-place dedup, and merges; the data usually must be sorted.
- A fast read / slow write pointer pair compacts arrays in place with O(1) extra space.
- A sliding window is two pointers bounding a contiguous range you grow on the right and shrink on the left.
- Fixed-size windows update in O(1) by adding the entering element and subtracting the leaving one; variable windows grow and shrink to satisfy a constraint.
- The window stays O(n) because each pointer only moves forward, so every element enters and leaves the window at most once — be ready to explain this.
- Choose by the tells: "sorted + pair" → converging pointers; "contiguous subrange + constraint" → sliding window; "unsorted value match" → hashing.

## Try it

1. **Reverse-check with two pointers.** Write `is_palindrome(s)` using converging pointers (revisit lesson 02 if needed), then extend it to `is_palindrome_alnum(s)` that ignores non-alphanumeric characters and case, so `"A man, a plan, a canal: Panama"` returns `True`. Keep it O(n) time, O(1) space.

2. **Smallest subarray ≥ target.** Given an array of positive integers and a target, return the length of the shortest contiguous subarray whose sum is at least the target (0 if none). Use a variable-size window that grows to reach the target, then shrinks to minimize. Test on `([2,3,1,2,4,3], 7)` (expect 2, from `[4,3]`).

3. **Longest substring with at most two distinct characters.** Given a string, return the length of the longest substring containing at most two distinct characters. Use a sliding window plus a dict counting characters currently inside; shrink from the left whenever a third distinct character appears. Test on `"eceba"` (expect 3, from `"ece"`).
