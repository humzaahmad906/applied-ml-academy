# 03 — Hashing

If there is one idea that separates a quadratic interview answer from a linear one more often than any other, it is hashing. A hash table trades a little memory for near-constant-time lookups, and that single trade turns a nested-loop scan into a single pass over and over again. In Python you get hash tables for free as the `dict` and the `set`, plus two purpose-built helpers in `collections`. This lesson explains how hashing works just deeply enough to reason about it, then drills the patterns that use it.

## What a hash table buys you

A hash table stores items in a way that lets you find, insert, and delete them in O(1) *average* time. It does this by running each key through a hash function to compute where the value lives, so it can jump straight there instead of scanning. Compare that to a list, where finding an item means checking elements one by one — O(n).

```python
seen_list = [10, 20, 30, 40]
print(30 in seen_list)   # True   but O(n) — scans the list

seen_set = {10, 20, 30, 40}
print(30 in seen_set)    # True   O(1) average — jumps straight there
```

That "O(1) average" carries a caveat worth stating in an interview: worst-case lookups can degrade to O(n) if many keys collide into the same slot, but for the general-purpose data you meet in problems, average O(1) is the honest expectation. The other cost is memory — a set or dict uses more space than a bare list — which is exactly the space-for-time trade lesson 01 described.

## dict and set essentials

A `dict` maps keys to values; a `set` stores just keys (membership, no values). Both require their elements to be hashable, which in practice means immutable: numbers, strings, and tuples work as keys; lists and dicts do not.

```python
ages = {"ada": 30, "alan": 41}
print(ages["ada"])            # 30       O(1) lookup
print(ages.get("grace", 0))   # 0        safe lookup with default
ages["grace"] = 28            # O(1) insert
print("alan" in ages)         # True     O(1) key membership

tags = {"python", "ml"}
tags.add("data")              # O(1)
tags.discard("go")            # O(1), no error if absent
print(len(tags))              # 3

# key must be hashable:
# bad = {[1, 2]: "x"}         # TypeError: unhashable type: 'list'
good = {(1, 2): "x"}          # a tuple works fine
print(good[(1, 2)])           # x
```

## The moment O(n²) becomes O(n)

Here is the single most important transformation in this whole course. The naive way to check whether a collection contains a duplicate, or whether two numbers sum to a target, uses a nested loop — O(n²). Swapping the inner scan for a set lookup makes it O(n).

Watch the "seen set" pattern turn duplicate detection from quadratic to linear:

```python
# SLOW — O(n^2): inner loop rescans for each element
def has_dup_slow(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False

# FAST — O(n): remember what we have seen in a set
def has_dup_fast(nums):
    seen = set()
    for x in nums:              # O(n) total
        if x in seen:           # O(1) lookup
            return True
        seen.add(x)             # O(1) insert
    return False

print(has_dup_fast([1, 2, 3, 2]))   # True
print(has_dup_fast([1, 2, 3, 4]))   # False
```

The same idea powers the **complement pattern**, the canonical solution to Two Sum. Instead of trying every pair, you ask for each number: "have I already seen the value that would complete the target?" You look that complement up in O(1).

```python
def two_sum(nums, target):
    seen = {}                       # value -> index
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:      # O(1) — did we see the partner already?
            return [seen[complement], i]
        seen[x] = i
    return None                     # overall O(n) time, O(n) space

print(two_sum([2, 7, 11, 15], 9))   # [0, 1]   (2 + 7)
print(two_sum([3, 2, 4], 6))        # [1, 2]   (2 + 4)
```

Notice the shape: one pass, one dict, O(1) work per element. Whenever a problem asks "is there a pair / does something match / have I seen this before," your first instinct should be a hash lookup.

## Counter: frequency maps for free

`collections.Counter` is a dict subclass built for counting. It replaces the manual `counts.get(ch, 0) + 1` idiom from lesson 02 with a single call, and adds handy methods on top.

```python
from collections import Counter

freq = Counter("banana")
print(freq)                 # Counter({'a': 3, 'n': 2, 'b': 1})
print(freq["a"])            # 3
print(freq["z"])            # 0   missing key returns 0, no error
print(freq.most_common(2))  # [('a', 3), ('n', 2)]
```

Counting is O(n) over the input. `most_common(k)` is the go-to for "top-K frequent elements" style questions. Counters also compare directly, which gives a one-line anagram check:

```python
def is_anagram(a, b):
    return Counter(a) == Counter(b)   # O(n), same letters, same counts

print(is_anagram("listen", "silent"))   # True
print(is_anagram("rat", "car"))         # False
```

## defaultdict: grouping without boilerplate

`collections.defaultdict` supplies a default value the first time you touch a missing key, which removes the "check if the key exists, if not create it" dance. It shines when grouping items into lists.

```python
from collections import defaultdict

def group_by_length(words):
    groups = defaultdict(list)      # missing key -> new empty list
    for w in words:
        groups[len(w)].append(w)    # no need to initialize the list first
    return dict(groups)

print(group_by_length(["hi", "cat", "dog", "a", "ok"]))
# {2: ['hi', 'ok'], 3: ['cat', 'dog'], 1: ['a']}
```

The classic "group anagrams together" problem is the same pattern, keyed by the sorted letters of each word:

```python
def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        key = "".join(sorted(w))    # anagrams share a sorted signature
        groups[key].append(w)
    return list(groups.values())    # O(n * k log k), k = word length

print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

Use `defaultdict(int)` for counting when you want more control than `Counter`, `defaultdict(list)` for grouping, and `defaultdict(set)` for grouping into unique buckets.

## Tuples as keys: coordinates and composite lookups

Because tuples are hashable, they make excellent keys when a single value is not enough to identify something. Grid problems use `(row, col)` tuples as keys in a `visited` set — an O(1) way to remember which cells you have already touched, which you will lean on for the grid traversals in lesson 06.

```python
visited = set()
visited.add((0, 0))
visited.add((1, 2))
print((0, 0) in visited)   # True    O(1)
print((3, 3) in visited)   # False
```

The same idea keys a cache or lookup by a *combination* of fields — for instance, counting how often each `(user, action)` pair occurs:

```python
from collections import Counter

events = [("ada", "click"), ("ada", "buy"), ("ada", "click")]
counts = Counter(events)
print(counts[("ada", "click")])   # 2
```

This tuple-key trick is also exactly why the memoization in lesson 07 can cache multi-argument recursive calls: the arguments form a hashable tuple.

## Insertion order and iteration

One modern-Python fact worth stating in an interview: since Python 3.7, a `dict` remembers the order keys were inserted. That means a plain dict doubles as an ordered map — you get "first seen" ordering for free without reaching for anything special. This is why the first-unique-character pattern (below in Try it) works cleanly with an ordinary dict.

```python
order = {}
for ch in "loveleetcode":
    order[ch] = order.get(ch, 0) + 1
# iteration follows first-seen order: l, o, v, e, ...
print(next(ch for ch, n in order.items() if n == 1))   # v  (first unique)
```

## When a set is exactly the tool

Sets also express set algebra directly, which occasionally is the whole answer — finding common elements between two collections, for instance, is an intersection.

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(a & b)     # {3, 4}         intersection — in both
print(a | b)     # {1,2,3,4,5,6}  union — in either
print(a - b)     # {1, 2}         difference — in a, not b
```

These operations run in time proportional to the smaller set, far better than the nested loop a beginner might write to find common elements.

## Key takeaways

- A hash table (dict/set) gives average O(1) insert, lookup, and delete by hashing keys to locations, versus O(n) scanning in a list.
- Keys must be hashable (immutable): numbers, strings, tuples work; lists and dicts do not.
- The "seen set" pattern turns duplicate detection and similar checks from O(n²) to O(n) in a single pass.
- The complement pattern solves Two Sum in O(n): for each item, look up the value that would complete the target.
- `Counter` builds frequency maps in one line and gives `most_common(k)` and direct equality for anagram checks.
- `defaultdict(list)` groups items without initialization boilerplate; keying by a canonical signature (like sorted letters) groups anagrams.
- Set algebra (`&`, `|`, `-`) expresses intersection, union, and difference far more cheaply than nested loops.

## Try it

1. **First unique character.** Given a string, return the index of the first character that appears exactly once, or -1 if there is none. Use a `Counter` in one pass to count, then a second pass to find the first with count 1. State the complexity. Test on `"leetcode"` (expect 0) and `"aabb"` (expect -1).

2. **Contains-nearby-duplicate.** Given a list and an integer `k`, return `True` if there are two equal values whose indices differ by at most `k`. Use a set (or dict of latest index) so the whole thing stays O(n). *Hint: you only need to remember values within the last `k` positions.*

3. **Intersection of two lists.** Write a function returning the unique values present in both input lists. First write it with nested loops and state the complexity, then rewrite it using set intersection and compare. Which would you say in an interview, and why?
