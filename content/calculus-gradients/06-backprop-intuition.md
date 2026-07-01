# 06 — Backpropagation, Intuitively

This is where it all comes together. **Backpropagation** is the method that computes the gradient for a neural network so gradient descent can do its job. It has a fearsome reputation, but it's really just the chain rule applied carefully, working backward through the network. If you're comfortable with "rates multiply along a chain," you're most of the way there.

## A network is a chain of functions

Recall that a neural network is a big composite function. Data enters, passes through a series of transformations (layers), and a prediction comes out the end. Schematically:

```
input → layer 1 → layer 2 → ... → prediction → error
```

Each layer has knobs (called **weights**) that shape its transformation. Training means adjusting every weight to shrink the error. To do that with gradient descent, we need the gradient: *how much does each weight affect the error?*

A network can have millions of weights. Computing each partial derivative from scratch would be hopeless. Backpropagation is the clever bookkeeping that gets them all in one efficient sweep.

## The forward pass

First we run the network forward: feed in the data, let each layer compute its output, and get a prediction. Then we measure how wrong it was with an error function. This is called the **forward pass**, and along the way we remember the intermediate values each layer produced — we'll need them in a moment.

Let's use a tiny toy "network" to see the mechanics. Suppose:

```
a = 2 * x         (layer 1: multiply by weight w1 = 2)
b = a + 1         (layer 2: add weight w2 = 1)
error = b^2       (how wrong we are)
```

With input `x = 3`: `a = 6`, `b = 7`, `error = 49`. That's the forward pass — compute left to right and stash each value.

## The backward pass

Now we want to know how the error responds to each piece. We ask, one link at a time, using local derivatives:

- How does `error` change with `b`? Since `error = b^2`, that rate is `2b = 14`.
- How does `b` change with `a`? Since `b = a + 1`, that rate is `1`.
- How does `a` change with `x`? Since `a = 2x`, that rate is `2`.

The chain rule says: to get how the error responds to `x`, multiply the rates along the chain:

```
d(error)/dx = 14 · 1 · 2 = 28
```

The magic is the *order* we compute this. We start at the error and multiply our way **backward** toward the input, carrying a running number as we go. That running number is often called the "gradient signal" flowing back through the network:

- Start at the error: signal = `1`.
- Through `error = b^2`: multiply by `2b = 14`. Signal is now `14`.
- Through `b = a + 1`: multiply by `1`. Signal stays `14`.
- Through `a = 2x`: multiply by `2`. Signal is now `28`.

Each layer receives the signal from the layer *after* it, multiplies by its own local rate, and passes the result to the layer *before* it. That's the whole idea — hence "back-propagation," the signal propagating backward.

```python
# forward pass
x = 3.0
a = 2 * x        # 6
b = a + 1        # 7
error = b**2     # 49

# backward pass: multiply local rates from the error back to x
grad = 1.0
grad = grad * (2 * b)   # d(error)/db = 14
grad = grad * 1         # db/da = 1
grad = grad * 2         # da/dx = 2
print(grad)             # 28.0
```

## Why go backward instead of forward?

You might ask why we multiply from the error back to the input rather than the other way. The reason is efficiency. There's one error at the end but many weights spread across the network. By starting from that single error and reusing the running signal as it flows back, every weight's gradient falls out of one backward sweep. Going forward from each weight separately would repeat enormous amounts of work. Backpropagation shares the effort — that's why it made training deep networks practical.

## Closing the loop

Once backpropagation hands us the gradient — one number per weight, each saying "nudging me changes the error at this rate" — we hand it straight to gradient descent:

```
new_weight = weight - learning_rate * gradient_for_that_weight
```

Then we do it all again: forward pass to get a prediction and error, backward pass to get the gradient, update every weight a small step downhill. Repeat this loop over many batches of data, and the network's weights gradually settle into values that make good predictions. That cycle — forward, backward, update — is literally what "training a neural network" means.

## Key takeaways

- A neural network is a **composite function**; its weights are the knobs we tune.
- The **forward pass** runs data through the layers to a prediction and an error, remembering intermediate values.
- The **backward pass** applies the chain rule from the error back toward the input, multiplying local rates to build a "gradient signal."
- Each layer takes the signal from the layer after it, multiplies by its local derivative, and passes it back — that's backpropagation.
- Going backward is efficient because one sweep produces every weight's gradient by reusing shared work.
- Feed those gradients to gradient descent, repeat forward-backward-update, and the network **learns**.

## Try it

Use the toy network `a = 3x`, `b = a - 2`, `error = b^2`, with input `x = 4`.

1. Do the forward pass by hand: find `a`, `b`, and `error`.
2. Write the three local rates: `d(error)/db`, `db/da`, `da/dx`.
3. Multiply them backward to get `d(error)/dx`.
4. Check your answer with a numeric derivative: compute the error at `x = 4` and at `x = 4.001`, then take `(error_new - error_old) / 0.001`. Does it match your backprop result?
