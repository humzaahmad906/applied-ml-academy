# 04 — Partial Derivatives and Gradients

Up to now our functions took a single input. But real models have many knobs — sometimes millions. We need a way to measure slope when there's more than one input. That's what **partial derivatives** and the **gradient** are for. Good news: they're just the derivative you already know, applied one input at a time.

## Functions of several inputs

Consider a function with two inputs:

```
f(x, y) = x^2 + y^2
```

Picture this as a landscape. The two inputs `x` and `y` are your position on a flat map (east-west and north-south), and the output `f(x, y)` is the *height* of the ground at that spot. This particular function is a big bowl: lowest at the center `(0, 0)` and rising as you walk away in any direction.

The question "what's the slope?" now has a catch. Slope in *which direction*? Walking east feels different from walking north. So we measure them separately.

## Partial derivatives: one direction at a time

A **partial derivative** measures the slope if you move along just one input while holding the others frozen.

To find the partial with respect to `x`, treat `y` as a fixed constant and differentiate normally. We write it with a curly `∂` instead of `d`:

```
∂f/∂x  — slope as x changes, y held still
∂f/∂y  — slope as y changes, x held still
```

For `f(x, y) = x^2 + y^2`:

- `∂f/∂x`: treat `y^2` as a constant (its slope is 0), so `∂f/∂x = 2x`.
- `∂f/∂y`: treat `x^2` as a constant, so `∂f/∂y = 2y`.

That's the whole trick: **freeze the other variables and differentiate as usual.**

## A worked example

Take `f(x, y) = 3x^2 + xy`.

Partial with respect to `x` (freeze `y`):

- `3x^2` → `6x`.
- `xy` → treat `y` as a constant multiplier, so the slope in `x` is `y`.
- Result: `∂f/∂x = 6x + y`.

Partial with respect to `y` (freeze `x`):

- `3x^2` → `0` (no `y` in it, so it's constant).
- `xy` → treat `x` as constant, so the slope in `y` is `x`.
- Result: `∂f/∂y = x`.

Evaluate at the point `(x, y) = (1, 2)`:

- `∂f/∂x = 6(1) + 2 = 8`
- `∂f/∂y = 1`

So at that spot, the ground rises steeply (rate 8) as you head in the `x` direction, and gently (rate 1) as you head in the `y` direction.

## The gradient vector

Collect all the partial derivatives into a list, and you get the **gradient**, written `∇f` ("grad f"). For our example:

```
∇f = [∂f/∂x, ∂f/∂y] = [6x + y, x]
```

At `(1, 2)` the gradient is `[8, 1]`.

The gradient is more than just bookkeeping. It has a beautiful geometric meaning:

**The gradient points in the direction of steepest ascent, and its length tells you how steep.**

In the landscape picture, stand anywhere on the hillside and the gradient is an arrow pointing straight *uphill* — the fastest way up. A longer arrow means a steeper hill. At `(1, 2)`, the arrow `[8, 1]` points mostly in the `x` direction, which makes sense: that's where the slope was biggest.

```python
def grad(x, y):
    return (6*x + y, x)   # ∂f/∂x, ∂f/∂y

print(grad(1, 2))   # (8, 1)
```

## Checking a partial numerically

The same "nudge and measure" idea works. To check `∂f/∂x`, nudge only `x`:

```python
def f(x, y):
    return 3*x**2 + x*y

x, y, h = 1.0, 2.0, 0.001
dfdx = (f(x + h, y) - f(x, y)) / h
dfdy = (f(x, y + h) - f(x, y)) / h
print(dfdx, dfdy)   # ≈ 8.0, 1.0
```

Nudge `x` alone, hold `y` still — that's the partial with respect to `x` in action.

## Why this matters for learning

A model's error depends on all its knobs at once, so the error is a function of many inputs — a landscape in a very high-dimensional space. To reduce the error, the model needs to know which way is downhill, and *how much each knob* is responsible for the slope.

The gradient answers both. It's a list with one number per knob, and each number says "nudging this knob changes the error at this rate." Since the gradient points uphill (steepest ascent), going the *opposite* direction takes you downhill fastest. That single fact is the engine of training: compute the gradient, then step against it. That's the idea we build on next.

## Key takeaways

- A **partial derivative** is the slope along one input while all others are held constant — differentiate as usual, freezing the rest.
- Write partials with `∂`: `∂f/∂x`, `∂f/∂y`.
- The **gradient** `∇f` collects all the partials into one vector, one entry per input.
- Geometrically, the gradient points in the direction of **steepest ascent**, and its length is the steepness.
- Going *opposite* the gradient is the fastest way downhill — the key to reducing error.

## Try it

For `f(x, y) = x^2 + 4y^2`:

1. Find `∂f/∂x` and `∂f/∂y` by hand.
2. Write the gradient `∇f` as a vector, then evaluate it at `(2, 1)`.
3. Which direction is steeper at that point — `x` or `y`? Confirm with a numeric check by nudging each input by `h = 0.001`.
4. The bowl's lowest point is `(0, 0)`. What is the gradient there, and why does that value make sense for a minimum?
