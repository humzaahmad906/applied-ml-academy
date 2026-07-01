# 01 — Functions and Limits

Before we can talk about how models learn, we need one big idea: a **function**. Almost everything in machine learning is a function of some kind, so getting comfortable here pays off for the rest of the course.

## What is a function?

A function is a rule that takes an input and gives back exactly one output. You feed it a number, it hands you a number back.

We usually write it like this:

```
f(x) = 2x + 1
```

Read that as "f of x equals two x plus one." The `x` is the input, and `f(x)` is the output. If you put in `x = 3`, you get:

```
f(3) = 2(3) + 1 = 7
```

The key word is *exactly one output*. For any input you give, a function commits to a single answer. A vending machine is a good mental model: press B4, get one specific snack, every time.

Some functions you'll see over and over:

- **Linear:** `f(x) = 2x + 1` — a straight line.
- **Quadratic:** `f(x) = x^2` — a U-shaped curve (a parabola).
- **Squaring the error:** `f(x) = (x - 4)^2` — a U-shape shifted so its lowest point sits at `x = 4`.

That last one matters a lot later. When a model is "wrong by some amount," we often square that amount, and the function above is exactly the shape we get.

## Functions as graphs

The friendliest way to understand a function is to draw it. Put the input `x` on the horizontal axis and the output `f(x)` on the vertical axis. Every input-output pair becomes a point, and all the points together trace a curve.

For `f(x) = x^2`, plugging in a few values:

| x  | f(x) |
|----|------|
| -2 | 4    |
| -1 | 1    |
| 0  | 0    |
| 1  | 1    |
| 2  | 4    |

Plot those and you get a smooth valley with its bottom at `(0, 0)`. When we later talk about a model "finding the bottom of the valley," this is literally the picture in our heads.

```python
import numpy as np

def f(x):
    return x**2

xs = np.linspace(-2, 2, 5)
print(list(zip(xs, f(xs))))
# [(-2.0, 4.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]
```

## The idea of a limit

Here is the one genuinely new idea. A **limit** answers the question: *as the input gets closer and closer to some value, what does the output head toward?*

Notice we said "head toward," not "equal." A limit is about the trend as you sneak up on a point, not necessarily the value at the point itself.

Take `f(x) = x^2` again and ask: as `x` gets close to `2`, where does `f(x)` go?

| x     | f(x)   |
|-------|--------|
| 1.9   | 3.61   |
| 1.99  | 3.9601 |
| 1.999 | 3.996  |
| 2.001 | 4.004  |
| 2.01  | 4.0401 |

From both sides, the output is closing in on `4`. We say "the limit of `f(x)` as `x` approaches `2` is `4`." Here it happens to match `f(2) = 4`, and for the smooth curves we care about, it usually will.

So why bother with the "head toward" language instead of just plugging in? Because the most important idea in the next lesson — the derivative — is built on a division that would be `0/0` if we plugged in directly. The limit lets us describe what a quantity is *approaching* even when we can't evaluate it right at the point. We sneak up on the answer instead of landing on it.

You don't need to compute limits by hand in this course. You just need the mental picture: **zoom in on a point and watch where the curve is going.**

## Why this matters for learning

A machine learning model is a function with knobs. You feed in data, it produces a prediction, and an **error function** measures how wrong that prediction is. That error function has a shape — often a valley, like `x^2`. Training a model means finding the input (the knob settings) that sits at the bottom of the valley, where the error is smallest.

To find the bottom, we need to know which way is downhill at any point. That means measuring the *slope* of the curve, and slope is defined using a limit. So the humble idea of "watch where the curve heads as you zoom in" is the foundation of every training loop you'll ever write.

## Key takeaways

- A **function** is a rule: one input in, exactly one output out.
- Graphing a function turns algebra into a picture you can reason about.
- Quadratics like `(x - c)^2` make valley shapes — the same shape as a model's error.
- A **limit** describes what an output heads toward as the input closes in on a value, even if you can't evaluate it directly at that point.
- Limits are the bridge to slopes, and slopes are how models figure out which way is downhill.

## Try it

Consider the function `f(x) = (x - 4)^2`.

1. Build a small table of outputs for `x = 2, 3, 4, 5, 6`. Where is the lowest output, and at what `x`?
2. Sketch the curve. Is it a valley or a hill? Where is the bottom?
3. Estimate the limit of `f(x)` as `x` approaches `3` by trying `x = 2.9, 2.99, 3.01, 3.1`. What value do the outputs close in on?

If you found the bottom sits at `x = 4`, you've just found the answer that a model training on this error function would eventually settle at.
