# 03 — The Chain Rule

The chain rule is the single most important tool for understanding how neural networks learn. It sounds intimidating, but the idea is something you already understand from everyday life: when things are connected in a chain, effects multiply along the chain.

## Functions inside functions

So far we've had one function at a time. But often the output of one function becomes the input of another. We call this **composing** functions.

Say you have two functions:

- `g(x) = x + 1`
- `f(u) = u^2`

Now feed the output of `g` into `f`:

```
f(g(x)) = (x + 1)^2
```

You do `g` first (add one), then `f` (square it). At `x = 3`: `g(3) = 4`, then `f(4) = 16`. This nesting is called a **composite function**, and it's everywhere in machine learning, where the output of one layer feeds the next.

## The gears intuition

Picture two connected gears. Turn the first gear, and it turns the second. If the first gear turns twice as fast as your hand, and the second turns three times as fast as the first, then the second turns `2 × 3 = 6` times as fast as your hand. Rates multiply through the chain.

The chain rule says exactly this for derivatives: *to find how the final output responds to the input, multiply the rates along the chain.*

## The rule

For a composite `f(g(x))`, the derivative is:

```
d/dx f(g(x)) = f'(g(x)) · g'(x)
```

In words: the derivative of the outer function (evaluated at the inner function's output) times the derivative of the inner function. Outer's rate times inner's rate.

Using the `df/du` notation makes the "rates multiply" idea even clearer. If `u = g(x)`:

```
df/dx = (df/du) · (du/dx)
```

Notice how it reads like the `du` cancels — that's a helpful memory aid, though it's really the gears multiplying.

## Worked example 1

Find the derivative of `f(x) = (x + 1)^2`.

Break it into pieces:

- Inner: `u = g(x) = x + 1`, so `g'(x) = 1`.
- Outer: `f(u) = u^2`, so `f'(u) = 2u`.

Multiply:

```
df/dx = 2u · 1 = 2(x + 1)
```

Let's sanity-check at `x = 3` with a numeric derivative:

```python
def f(x):
    return (x + 1)**2

x, h = 3.0, 0.001
print((f(x + h) - f(x)) / h)   # ≈ 8.0
```

Our rule gives `2(3 + 1) = 8`. They match.

## Worked example 2

Find the derivative of `f(x) = (3x^2 + 1)^3`.

- Inner: `u = 3x^2 + 1`, so `du/dx = 6x`.
- Outer: `u^3`, so `df/du = 3u^2`.

Multiply:

```
df/dx = 3u^2 · 6x = 18x(3x^2 + 1)^2
```

Notice the pattern: peel the outer layer (bring the power down, drop it by one), then multiply by the derivative of what's inside. That "multiply by the derivative of the inside" step is the chain rule earning its keep.

## Worked example 3: a longer chain

Chains can be longer than two links, and the rule extends the obvious way — just keep multiplying. Suppose:

```
y = f(g(h(x)))
```

Then:

```
dy/dx = f'(g(h(x))) · g'(h(x)) · h'(x)
```

Every link contributes one factor. A concrete version: if `h(x) = 2x`, `g(u) = u + 1`, `f(v) = v^2`, the local rates are `2`, `1`, and `2v`. At `x = 1`: `h = 2`, `g = 3`, and the outer rate is `2(3) = 6`. Multiply the chain: `6 · 1 · 2 = 12`. Every layer's rate stacks up by multiplication.

## Why this is the heart of learning

A neural network is a giant composite function: input goes into layer 1, its output feeds layer 2, and so on until a prediction pops out the end. When the network is wrong, we want to know how each layer's knobs contributed to the error.

The chain rule is the only tool that lets us trace an effect *backward* through all those nested layers, multiplying local rates as we go. That backward trace is what training is made of. If you're solid on "rates multiply along the chain," you already understand the engine that trains every deep network — the rest is bookkeeping.

## Key takeaways

- **Composing** functions means feeding one function's output into another: `f(g(x))`.
- The **chain rule**: `d/dx f(g(x)) = f'(g(x)) · g'(x)` — outer rate times inner rate.
- In `df/du` form it reads `df/dx = (df/du)(du/dx)`, like the pieces multiply through.
- For longer chains, keep multiplying one factor per link.
- Peel from the outside in: differentiate the outer layer, then multiply by the derivative of the inside.
- This is the mechanism that lets us trace error backward through a network's layers.

## Try it

Find the derivative of each, then check one with a numeric derivative:

1. `f(x) = (2x + 5)^2`
2. `f(x) = (x^2 + x)^4`
3. For `f(x) = (2x + 5)^2`, compute the numeric derivative at `x = 1` using `h = 0.001` and confirm it matches your rule-based answer of `2(2x + 5) · 2 = 4(2x + 5)`.
