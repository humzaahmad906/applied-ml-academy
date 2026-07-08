# 08 — Interview Strategy

You can know every pattern in this course and still fail a coding round by working silently, jumping to code, or forgetting to test. The coding interview is a communication exercise wearing an algorithms costume. The interviewer is deciding whether they want you on their team, and that judgment rests as much on *how* you think out loud as on whether you reach the optimal solution. This lesson gives you a repeatable method for the 45 minutes, tells you what ML-engineer rounds specifically test, and shows the difference in transcript form.

## Why a method matters

Under pressure, the failure mode is predictable: you read the problem, spot something familiar, and start typing. Ten minutes later you realize you misread a constraint, your approach does not handle a case, and you have no time to recover. A method prevents that by front-loading the thinking the interviewer most wants to see — clarifying and planning — before any code exists to distract from it. It also gives you something to fall back on when your mind goes blank, which it will.

The framework worth memorizing is **UMPIRE**: Understand, Match, Plan, Implement, Review, Evaluate. Several other acronyms (REACTO, the "clarify-plan-code-test" loop) describe the same arc. The name does not matter; the discipline does.

## U — Understand: clarify before you commit

Never start solving the problem you *assume* was asked. Restate it in your own words and ask about the parts that change the solution. Good clarifying questions cover input size (which decides what complexity is acceptable), value ranges (negatives? duplicates? empty?), and the expected output shape.

Concrete questions to have ready:

- "Can the input be empty or have a single element?"
- "Are the values sorted? Can they be negative? Can there be duplicates?"
- "How large can the input get?" — this quietly tells you the target complexity. If `n` can be a million, O(n²) is off the table and you should aim for O(n) or O(n log n).
- "Should I optimize for time or memory if I have to choose?"

Write down one or two concrete examples, including an edge case, and confirm the expected output with the interviewer. This catches misunderstandings while they are still cheap to fix.

## M — Match: name the pattern

Map the problem to a category you know. This is where the previous seven lessons pay off. Say the mapping out loud — it shows structured thinking even if you later change your mind.

```python
# a running pattern-matching cheat sheet
# "pair/sum in a SORTED array"            -> two pointers          (04)
# "have I seen a matching value?"          -> hash set / dict       (03)
# "contiguous subarray/substring + limit"  -> sliding window        (04)
# "next greater/smaller element"           -> monotonic stack       (05)
# "process in arrival order / shortest path"-> BFS with a deque     (05, 06)
# "explore all paths / tree recursion"     -> DFS                   (06)
# "dependencies / ordering"                -> topological sort      (06)
# "count ways / min cost / optimal over choices" -> dynamic programming (07)
# "top-K / K-th largest"                   -> heap (heapq)          (below)
```

## P — Plan: design out loud before typing

State your approach in plain language and, critically, its complexity, *before* writing code. "I'll use a hash map to store values I've seen; for each element I check whether its complement is already there. That's one pass, O(n) time and O(n) space." Now the interviewer can redirect you if the approach is wrong — a five-second correction instead of fifteen wasted minutes. If you see a brute force and a better solution, name both and the trade-off, then confirm which to implement.

## I — Implement: code cleanly, narrate as you go

Now write it, talking through each part. Use clear names (`left`/`right`, not `i`/`j`, when it aids reading), handle the edge cases you identified in step U, and keep talking so the interviewer follows your reasoning. If you get stuck, say what you are stuck on — silence reads as being lost, whereas "I'm deciding whether to track indices or values here" reads as thinking.

## R — Review: read your own code

Before declaring done, walk your code line by line as if debugging someone else's. Check the boundaries: empty input, a single element, the first and last iterations, integer overflow (rare in Python, but mention it), and off-by-one errors in ranges and indices. Interviewers notice when you find your own bug — it signals you will catch bugs before they reach production.

## E — Evaluate: trace an example and state complexity

Run through a concrete example by hand, tracking the variables, and confirm the output matches. Then state the final time and space complexity and whether it can be improved. This is also the moment to mention the follow-up you would make with more time ("with a larger input I'd switch to the O(n log n) approach").

## The heap: the one structure this course held back

One tool worth adding to your kit before interviews is the heap, via Python's `heapq`. A heap gives you the smallest (or largest) element in O(1) and insertion/removal in O(log n), which makes it the right answer for "top-K" and "K-th largest" questions. `heapq` implements a *min*-heap on a plain list.

```python
import heapq

def k_largest(nums, k):
    # keep a min-heap of the k largest seen so far
    heap = nums[:k]
    heapq.heapify(heap)                 # O(k)
    for x in nums[k:]:
        if x > heap[0]:                 # bigger than the smallest kept?
            heapq.heapreplace(heap, x)  # pop smallest, push x — O(log k)
    return sorted(heap, reverse=True)   # overall O(n log k)

print(k_largest([3, 1, 5, 12, 2, 11], 3))   # [12, 11, 5]
```

This beats sorting the whole array (O(n log n)) when `k` is small, and stating that trade-off is exactly the kind of remark that lands well.

## What ML-engineer coding rounds actually test

ML-engineer loops usually include a general coding round that is indistinguishable from a software-engineer one — the patterns in this course, full stop. Do not expect the algorithm round to involve gradients or models; that is what the ML-specific rounds are for. But there are ML-flavored twists worth anticipating:

- **Array and matrix manipulation** without NumPy, to prove you understand what the vectorized call does underneath. Reversing, transposing, and reshaping by hand show up.
- **Streaming / online computations** — a running mean, a reservoir sample, a moving average over a window — because production ML processes unbounded data. The sliding window (lesson 04) is directly relevant.
- **Top-K and sampling**, which map onto heaps and are ubiquitous in retrieval, ranking, and recommendation.
- **Graph problems**, because computation graphs, feature dependencies, and pipeline DAGs are graphs (lesson 06). Framing your topological-sort answer in those terms signals seniority.
- **Clean, tested, readable code.** ML engineers ship code others maintain, so interviewers weight clarity and correctness over cleverness.

A tiny worked example of the streaming flavor — a running mean that never stores the whole stream:

```python
class RunningMean:
    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def add(self, x):                    # O(1) per element, O(1) memory
        self.n += 1
        self.mean += (x - self.mean) / self.n   # numerically stable update
        return self.mean

rm = RunningMean()
for v in [10, 20, 30]:
    print(round(rm.add(v), 2))   # 10.0  15.0  20.0
```

## Communication habits that pass rounds

- **Think out loud.** A correct answer arrived at silently scores worse than a slightly imperfect one arrived at transparently.
- **Take hints.** Interviewers nudge on purpose; resisting a hint reads as inflexibility. "Good point, that lets me drop the extra pass" is the right response.
- **State assumptions explicitly** rather than acting on them silently — the same habit that serves you in real engineering.
- **Manage time.** If you are stuck, get *a* working solution (even brute force) on the board, then optimize. A correct O(n²) beats an unfinished O(n).
- **Be honest about complexity.** Do not claim O(n) for something that is O(n log n). Interviewers check, and a wrong claim costs more than the extra log factor.

## Key takeaways

- The coding interview grades communication as much as correctness; work out loud and involve the interviewer at every step.
- Use a method — UMPIRE (Understand, Match, Plan, Implement, Review, Evaluate) — to front-load clarifying and planning before any code.
- Clarify input size, ranges, duplicates, and empty cases first; the size ceiling silently tells you the target complexity.
- Match the problem to a pattern and state your plan *and its complexity* before typing, so a wrong approach is corrected in seconds.
- Review your own code against edge cases and trace a concrete example before declaring done.
- Add a heap (`heapq`) to your kit for top-K / K-th-largest in O(n log k).
- ML-engineer coding rounds test the same DSA patterns, with twists toward matrix work, streaming/online computation, top-K, and graphs; prize clean, tested, readable code.

## Try it

1. **Full UMPIRE dry run.** Take Two Sum (lesson 03) and write out each UMPIRE step explicitly as comments: the clarifying questions you'd ask, the pattern you match, the plan and its complexity, the code, your review notes, and a traced example. Practicing the narration until it is automatic is the point.

2. **K-th largest, two ways.** Return the K-th largest element in an unsorted list, first by sorting (state the complexity), then with a size-K min-heap using `heapq`. Explain when the heap wins. Test on `([3,2,1,5,6,4], 2)` (expect 5).

3. **Design a moving average.** Implement a class `MovingAverage(size)` with a method `next(val)` returning the mean of the last `size` values seen. Use a `deque` bounded to `size` so each call is O(1). This is the streaming pattern ML rounds favor — test it on a stream of `[1, 10, 3, 5]` with `size=3`.
