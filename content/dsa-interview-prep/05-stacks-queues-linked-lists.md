# 05 — Stacks, Queues, and Linked Lists

These three structures share a theme: they are all about *order of access*. A stack serves the most recently added item first; a queue serves the oldest; a linked list threads elements together with pointers so you can splice in the middle cheaply. Interviewers love them because the right choice often collapses a messy problem into a few clean lines — and the wrong choice (say, using a list as a queue) quietly turns your solution O(n²). This lesson shows the idiomatic Python for each, plus two patterns worth memorizing: the monotonic stack and linked-list pointer surgery.

## Stacks: last in, first out

A stack is LIFO — last in, first out, like a stack of plates. You only touch the top. In Python a plain `list` *is* a stack: `append` pushes onto the top and `pop` removes from the top, both O(1).

```python
stack = []
stack.append(1)      # push -> [1]
stack.append(2)      # push -> [1, 2]
stack.append(3)      # push -> [1, 2, 3]
print(stack.pop())   # 3    pop the top, O(1)
print(stack[-1])     # 2    peek without removing
print(len(stack))    # 2
```

Stacks are the natural fit whenever you need to remember things to come back to in reverse order: matching brackets, undo history, or converting recursion into iteration. The bracket-matching problem is the canonical example — push each opener, and every closer must match the most recent unmatched opener, which is exactly the top of the stack.

```python
def valid_parentheses(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)              # opener -> remember it
        else:
            if not stack or stack.pop() != pairs[ch]:
                return False              # closer with no/ wrong match
    return not stack                      # leftover openers -> invalid

print(valid_parentheses("([]{})"))   # True
print(valid_parentheses("([)]"))     # False
print(valid_parentheses("((("))      # False
```

## Queues: first in, first out — and why not a list

A queue is FIFO — first in, first out, like a checkout line. You add at the back and remove from the front. Here is the trap: a Python list can add at the back in O(1), but removing from the front with `pop(0)` is O(n), because every remaining element shifts left (lesson 02). Use a list as a queue in a loop and you have an accidental O(n²).

The right tool is `collections.deque` (double-ended queue), which supports O(1) operations at *both* ends.

```python
from collections import deque

q = deque()
q.append(1)          # enqueue at the back -> deque([1])
q.append(2)          # -> deque([1, 2])
print(q.popleft())   # 1   dequeue from the front, O(1)
print(q.popleft())   # 2

# deque also does the stack ends, and appendleft in O(1):
d = deque([2, 3])
d.appendleft(1)      # -> deque([1, 2, 3])   O(1), unlike list.insert(0, x)
d.append(4)          # -> deque([1, 2, 3, 4])
print(list(d))       # [1, 2, 3, 4]
```

`deque` is the workhorse behind breadth-first search (lesson 06). Any time a problem processes items in arrival order, or needs cheap operations at both ends, reach for it rather than a list.

## The monotonic stack

A monotonic stack keeps its elements in sorted order (increasing or decreasing) by popping any element that would break the order *before* pushing the new one. It sounds niche, but it answers a whole family of "next greater / next smaller element" questions in a single O(n) pass instead of O(n²).

The classic: for each element, find the next element to its right that is larger. Naively that is a nested loop. With a monotonic (decreasing) stack of *indices*, each element is pushed and popped at most once, so the total work is O(n).

```python
def next_greater(nums):
    result = [-1] * len(nums)     # default: no greater element to the right
    stack = []                    # holds indices, values decreasing
    for i, x in enumerate(nums):
        # x is greater than the values at these indices -> resolve them
        while stack and nums[stack[-1]] < x:
            idx = stack.pop()
            result[idx] = x
        stack.append(i)
    return result                 # O(n): each index pushed/popped once

print(next_greater([2, 1, 2, 4, 3]))   # [4, 2, 4, -1, -1]
```

The key insight to explain: although there is a `while` inside a `for`, every index is pushed exactly once and popped at most once, so the two loops together do O(n) work — the same amortized argument as the sliding window. Monotonic stacks also solve "largest rectangle in histogram" and "daily temperatures," so recognizing the pattern pays off repeatedly.

## Linked lists

A linked list stores each value in a node that also holds a pointer to the next node. There is no index-based random access — to reach the fifth element you follow four pointers — but inserting or deleting a node is O(1) once you have a pointer to the spot, because you just re-thread pointers instead of shifting elements. Interview linked-list problems are pure pointer manipulation, and drawing the pointers on paper beats trying to hold them in your head.

```python
class ListNode:
    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt

# build 1 -> 2 -> 3
head = ListNode(1, ListNode(2, ListNode(3)))

def to_list(node):                # helper for printing
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

print(to_list(head))   # [1, 2, 3]
```

### Reversing a linked list

The most-asked linked-list problem. Walk the list re-pointing each node's `next` to the node behind it, carrying three pointers: `prev`, `curr`, and a saved `nxt` so you do not lose the rest of the list.

```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next      # save the rest before we overwrite
        curr.next = prev     # flip the pointer backward
        prev = curr          # advance prev
        curr = nxt           # advance curr
    return prev              # new head; O(n) time, O(1) space

print(to_list(reverse_list(head)))   # [3, 2, 1]
```

### Cycle detection: fast and slow pointers

Does a linked list loop back on itself? Floyd's algorithm uses two pointers, one stepping once and one stepping twice per iteration. If there is a cycle, the fast pointer eventually laps the slow one and they meet; if there is no cycle, the fast pointer runs off the end. It uses O(1) extra space — no set of visited nodes needed.

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # one step
        fast = fast.next.next     # two steps
        if slow is fast:          # they met -> cycle
            return True
    return False                  # fast hit the end -> no cycle

# 1 -> 2 -> 3 -> back to 2 (a cycle)
a, b, c = ListNode(1), ListNode(2), ListNode(3)
a.next, b.next, c.next = b, c, b
print(has_cycle(a))               # True
print(has_cycle(head))            # False (the reversed 3->2->1 above)
```

The fast/slow idea generalizes: the same two-speed trick finds the *middle* of a list (slow lands in the middle when fast reaches the end) and the start of a cycle, all in one pass with constant space.

## Key takeaways

- A stack is LIFO; a Python `list` is a stack with `append` (push) and `pop` (pop), both O(1). Use it for bracket matching, undo, and derecursion.
- A queue is FIFO; never use `list.pop(0)` (O(n)) — use `collections.deque` with `append`/`popleft`, both O(1) at each end.
- A monotonic stack answers "next greater/smaller element" families in O(n) because each element is pushed and popped at most once.
- A linked list trades O(1) index access for O(1) insertion/deletion given a node pointer; problems are pointer re-threading, so draw them.
- Reverse a list by walking it with `prev`/`curr`/`nxt` pointers, flipping each `next` backward — O(n) time, O(1) space.
- Floyd's fast/slow pointers detect a cycle (and find the middle) in O(1) extra space; the fast pointer moves twice per slow step.

## Try it

1. **Min stack.** Design a stack that also returns its current minimum in O(1). *Hint: keep a second stack that tracks the minimum so far at each level.* Support `push`, `pop`, `top`, and `get_min`, all O(1).

2. **Merge two sorted linked lists.** Given two sorted linked lists, merge them into one sorted list by splicing nodes (no new values). Use a dummy head node to simplify the edge cases, and two pointers walking the inputs. What is the complexity?

3. **Daily temperatures.** Given a list of daily temperatures, return a list where each entry is how many days you must wait for a warmer temperature (0 if none comes). Solve it with a monotonic stack of indices in O(n). Test on `[73, 74, 75, 71, 69, 72, 76, 73]` (expect `[1, 1, 4, 2, 1, 1, 0, 0]`).
