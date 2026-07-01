# 02 — Derivatives

A **derivative** is one of the most useful ideas in all of math, and it boils down to a single question: *how fast is the output changing as I nudge the input?* That's it. If you can picture a slope, you already have most of the intuition.

## Slope, the everyday version

Think about a road. If you climb 3 meters upward over 10 meters of horizontal distance, the slope is `3/10 = 0.3`. Steeper road, bigger number. Downhill road, negative number. Flat road, zero.

Slope is always **rise over run**: how much the output changes divided by how much the input changed.

```
slope = (change in output) / (change in input)
```

For a straight line this is easy because the slope is the same everywhere. For `f(x) = 2x + 1`, every step of `1` to the right raises the output by `2`. The slope is just `2`, forever.

## The problem with curves

Curves are trickier. On `f(x) = x^2`, the slope keeps changing. Near the bottom it's gentle; far out on the sides it's steep. So asking "what is the slope of `x^2`?" has no single answer — it depends on *where* you are.

The derivative fixes this by measuring the slope **at a single point**. Here's the trick: pick your point, then pick a second point a tiny distance `h` away, and measure the slope of the line between them.

```
slope ≈ (f(x + h) - f(x)) / h
```

As we shrink `h` toward zero, that line hugs the curve tighter and tighter until it becomes the slope *at* your point. That "shrink `h` toward zero" step is exactly the limit idea. The result is the **derivative**.

## Notation

You'll see the derivative written a few ways. They all mean the same thing:

- `f'(x)` — read "f prime of x."
- `df/dx` — read "d f d x." Think of it as "the tiny change in `f` per tiny change in `x`."

Both say: *the slope of `f` at the point `x`.*

## A worked example, numerically

Let's find the slope of `f(x) = x^2` at `x = 3` without any rules — just by shrinking `h`.

| h      | (f(3+h) - f(3)) / h |
|--------|---------------------|
| 1      | 7.0                 |
| 0.1    | 6.1                 |
| 0.01   | 6.01                |
| 0.001  | 6.001               |

The numbers are marching toward `6`. So the slope of `x^2` at `x = 3` is `6`.

```python
def f(x):
    return x**2

x = 3.0
for h in [1, 0.1, 0.01, 0.001]:
    slope = (f(x + h) - f(x)) / h
    print(h, slope)
# 1 7.0
# 0.1 6.1...
# 0.01 6.01...
# 0.001 6.001...
```

This "nudge the input and measure" approach is called a **numeric derivative**, and it's a great way to check your work.

## The simple rules

Shrinking `h` by hand every time is tedious. Luckily, patterns emerge, and a few rules cover most functions you'll meet.

**Power rule.** For `f(x) = x^n`, the derivative is `n * x^(n-1)`. Bring the exponent down front, then subtract one from it.

- `f(x) = x^2` → `f'(x) = 2x`. At `x = 3` that's `2(3) = 6`. Matches our table exactly.
- `f(x) = x^3` → `f'(x) = 3x^2`.
- `f(x) = x` → `f'(x) = 1` (the slope of a plain line through the origin).

**Constant rule.** A constant never changes, so its slope is `0`. If `f(x) = 5`, then `f'(x) = 0`.

**Constant multiple.** A number out front just tags along. If `f(x) = 4x^2`, then `f'(x) = 4 * 2x = 8x`.

**Sum rule.** Derivatives of added-together pieces are just the derivatives of each piece, added. If `f(x) = x^2 + 3x`, then `f'(x) = 2x + 3`.

Putting it together for `f(x) = x^2 + 3x + 5`:

```
f'(x) = 2x + 3 + 0 = 2x + 3
```

At `x = 1`, the slope is `2(1) + 3 = 5`.

## Why the derivative is the whole game

Here's the payoff. The derivative tells you two things at once:

- **The sign** tells you direction. Positive slope means the function is going up as you move right; negative means it's going down.
- **The size** tells you steepness. A big number means the output changes fast; a small number means it changes slowly.

When a model wants to reduce its error, it asks the derivative "which way is downhill, and how steep?" A positive slope says "go left to go down." A negative slope says "go right to go down." A slope of zero says "you're at the bottom — stop." That's the seed of gradient descent, and it grows directly from the idea you just learned.

## Key takeaways

- A derivative is the **slope of a function at a single point** — rise over run as the run shrinks to nothing.
- It's built on a limit: nudge the input by `h`, measure the slope, shrink `h` toward zero.
- Write it as `f'(x)` or `df/dx`; both mean "slope at `x`."
- The **power rule** (`x^n` → `n·x^(n-1)`), constant rule, constant-multiple rule, and sum rule handle most everyday functions.
- Sign says direction (up/down); magnitude says steepness. Together they point downhill.

## Try it

For `f(x) = 3x^2 + 2x`:

1. Use the rules to find `f'(x)`.
2. Evaluate `f'(x)` at `x = 0` and `x = 2`. Which point is steeper?
3. Check your answer at `x = 2` with a numeric derivative: compute `(f(2 + h) - f(2)) / h` for `h = 0.001` and confirm it's close to your rule-based answer.
