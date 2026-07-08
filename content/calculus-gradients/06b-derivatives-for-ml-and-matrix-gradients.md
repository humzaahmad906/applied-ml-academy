# 06b — Derivatives for ML and Matrix Gradients

So far the rules we've practiced have been about polynomials: powers, sums, the odd product. That's a fine warm-up, but it isn't quite the calculus that real machine learning runs on. The functions inside actual losses and activations are `exp`, `log`, and `sigmoid` — and the quantities we differentiate aren't single numbers but whole vectors and matrices of weights. This lesson closes that gap. By the end you'll know the handful of derivatives ML actually uses, and you'll have the shape intuition that lets the chain rule from the backprop lesson scale up to real networks.

## Why these particular functions?

Open the hood of almost any classifier and you'll find the same three functions. `exp` and `log` show up because probabilities and losses are built from them: models often output a raw score and then exponentiate it to make it positive, and losses take the logarithm of a probability to turn "how likely was the right answer?" into a number we can minimize. `sigmoid` shows up as the squashing function that turns any real number into something between 0 and 1 — perfect for "probability that this email is spam."

Because these functions sit inside the network, training has to differentiate through them. If we only knew how to differentiate `x^2`, we'd be stuck the moment a `log` appeared. So let's collect their derivatives.

## The derivatives of exp and log

Two clean facts, worth memorizing:

```
d/dx e^x   = e^x      (the exponential is its own derivative)
d/dx ln(x) = 1/x      (the natural log)
```

The first is the famous one: `e^x` is the only function (up to scaling) that equals its own rate of change. The second is its mirror image — the natural log grows quickly near zero and flattens out, and its slope is exactly `1/x`.

These rarely appear bare; they're usually wrapped inside something else, so we reach for the chain rule from the previous lessons. A couple of small examples:

```
d/dx e^(3x)   = e^(3x) · 3        (outer rate e^(3x), inner rate 3)
d/dx ln(2x)   = (1/(2x)) · 2 = 1/x
```

Here's why `ln` matters so much in ML. When a model predicts a probability `p` for the correct class, we score it with the **log-likelihood** `ln(p)` — high when `p` is near 1, plunging toward negative infinity as `p` heads to 0. Losses like **cross-entropy** are just `-ln(p)`: a big penalty for being confidently wrong. Because `d/dp [-ln(p)] = -1/p`, the gradient is gentle when `p` is close to 1 (we were nearly right) and enormous when `p` is close to 0 (we were badly wrong). The log is what makes the loss punish confident mistakes so sharply.

## Sigmoid and its beautiful derivative

The sigmoid function squashes any real input into `(0, 1)`:

```
σ(x) = 1 / (1 + e^(-x))
```

Its derivative is one of the tidiest results in all of ML:

```
σ'(x) = σ(x) · (1 - σ(x))
```

That's remarkable — to get the slope, you don't need to re-run any exponentials. If you already computed the output `σ(x)` during the forward pass, the derivative is just `output · (1 - output)`. This is exactly the kind of value-reuse that makes backpropagation efficient.

There's a catch hiding in that formula, though. The product `σ·(1-σ)` is largest at `x = 0`, where `σ = 0.5` and the derivative equals `0.25`. Out at the tails — when `x` is very positive or very negative — `σ` sits near 1 or near 0, so `σ·(1-σ)` collapses toward **0**. A near-zero derivative means almost no gradient signal flows back through that unit. Stack several sigmoids in a deep network and these small numbers multiply together (remember: rates multiply along a chain), and the signal can shrink to nothing. That's the **vanishing-gradient** problem, and it's a big reason modern networks reach for other activations like ReLU. For now just hold onto the picture: sigmoid learns fastest in the middle and barely learns at the extremes.

## The beautiful result: softmax + cross-entropy

Here is the payoff that makes classification training so stable. For multi-class problems we use **softmax** to turn a vector of raw scores into probabilities `p`, and **cross-entropy** to compare `p` against the true one-hot label `y`. Each piece has a messy-looking derivative on its own. But when you combine them and work out the gradient of the loss with respect to the raw scores `z`, an avalanche of terms cancels and you're left with:

```
dL/dz = p - y      (predicted probabilities minus true labels)
```

That's the whole gradient. No exponentials, no logs, no fractions — just "prediction minus truth." The intuition is direct: for each class, take how much probability the model assigned and subtract how much it should have. If the model over-predicted a class, the gradient is positive and its score gets pushed down; if it under-predicted, the gradient is negative and its score gets pushed up. The `log` inside cross-entropy and the `exp` inside softmax are designed to cancel each other exactly, and this is no accident — the pairing was chosen to produce this clean result.

Why does this matter in practice? A gradient of `p - y` is always bounded between `-1` and `1` and never blows up or silently dies, so the training signal stays well-behaved no matter how wrong the model currently is. That stability is a large part of why softmax + cross-entropy is the default final layer for classification everywhere.

## From scalars to vectors and matrices: track the shapes

Everything in the backprop lesson used single numbers. Real networks pass around vectors and matrices, but — and this is the key reassurance — **the chain rule still holds unchanged**. Rates still multiply along the chain. The only new skill is bookkeeping: keeping track of the **shapes** of the things you're multiplying.

The single most useful rule for a beginner:

> The gradient of the loss with respect to a weight matrix `W` has the **same shape as `W`**.

This is what makes gradient descent work at all. The update `W = W - learning_rate · dL/dW` requires `dL/dW` to line up element-for-element with `W`, so of course it must have the same shape. Whenever you compute a gradient, sanity-check it against this rule first.

Take a linear layer, the workhorse of every network:

```
y = W x         (x is the input vector, W the weight matrix, y the output vector)
```

Suppose backprop has handed us `dL/dy`, the gradient signal arriving from later in the network (same shape as `y`). The gradient with respect to the weights turns out to be the **outer product** of that incoming signal with the input:

```
dL/dW = (dL/dy) outer x
```

You can check this with shapes alone, no heavy formalism needed. If `x` has 3 numbers and `y` has 2, then `W` is a 2×3 matrix. The signal `dL/dy` has 2 numbers and `x` has 3, so their outer product is 2×3 — exactly the shape of `W`. It matches. The intuition is the scalar rule `dL/dw = (dL/dy)·x` from the last lesson, just written for every weight at once: each weight's gradient is its output's signal times the input it multiplied.

That's genuinely all the "matrix calculus" you need to follow the deep learning course. You don't have to derive Jacobians by hand — the framework does that. You just need to trust that the chain rule scales up and to check that every gradient matches the shape of the thing it updates.

## Tying back to backprop

In lesson 06 the "gradient signal" was a single running number multiplied by local rates on its way back to the input. Nothing about that changes here — we've just filled in what the local rates actually are for the functions ML uses (`e^x`, `1/x`, `σ(1-σ)`, and the clean `p - y`), and we've noted that the signal is now a vector or matrix whose shape has to match at every step. Same idea, richer pieces. When you meet backprop again in the deep learning course, it'll be this exact machinery running over real layers.

## A tiny numpy check

Let's confirm the sigmoid derivative the honest way — compare the analytic formula `σ(1-σ)` against a numerical estimate of the slope:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = 0.7
s = sigmoid(x)

analytic = s * (1 - s)                                  # σ(x)·(1-σ(x))

eps = 1e-6
numerical = (sigmoid(x + eps) - sigmoid(x - eps)) / (2 * eps)

print(analytic)    # 0.22171287329310904
print(numerical)   # 0.2217128733...  (matches to ~6 decimals)
```

The two agree, which is exactly the reassurance you want: the clean formula really is the slope.

## Key takeaways

- ML runs on `exp`, `log`, and `sigmoid`, not just polynomials — so we need their derivatives.
- `d/dx e^x = e^x` and `d/dx ln(x) = 1/x`; the `log` in cross-entropy is what makes the loss punish confident mistakes.
- Sigmoid has the tidy derivative `σ'(x) = σ(x)·(1 - σ(x))`, reusing the forward output; it shrinks to 0 at the tails, hinting at **vanishing gradients**.
- Softmax + cross-entropy has the beautiful gradient `p - y` (**predicted minus true**), which is bounded and stable — a key reason classification training is well-behaved.
- Moving from scalars to matrices, the chain rule is unchanged; you just **track shapes**. The gradient of the loss w.r.t. `W` has the **same shape as `W`**.
- For a linear layer `y = Wx`, `dL/dW = (dL/dy) outer x` — check it with shapes, no Jacobians required.

## Try it

1. Use the chain rule to find `d/dx e^(-2x)` and `d/dx ln(5x)`. (Answers: `-2·e^(-2x)` and `1/x`.)
2. Compute `σ(x)·(1-σ(x))` at `x = 0`, `x = 3`, and `x = -3`. Which is largest? What does the pattern tell you about where sigmoid learns fastest?
3. A linear layer has input `x` of length 4 and output `y` of length 2. What shape is `W`? What shape must `dL/dW` be, and why?
4. Adapt the numpy snippet above to check the derivative of `ln(x)` at `x = 2`: compare `1/x` against the two-sided numerical estimate. Do they match?
