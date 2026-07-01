# 04 — Backpropagation

The forward pass produces a prediction. Almost always, that prediction is wrong — at least at first. Backpropagation is how the network figures out *which weights to blame* and *in which direction to nudge each one* to make the prediction a little less wrong. It's the engine of learning in a neural network, and it rests on one idea from calculus you already have intuition for: the chain rule.

## The goal, stated plainly

We have a **loss** — a single number measuring how wrong the prediction is (more on losses next module). Every weight and bias in the network affects that loss. We want, for each weight `w`, the quantity `∂loss/∂w`: "if I nudge this weight up a tiny bit, how much does the loss change?" That's the **gradient**. Once we have it for every weight, we simply push each weight in the direction that lowers the loss. Backpropagation is an efficient algorithm for computing all those gradients at once.

## The chain rule is the whole trick

Recall from calculus: if `y` depends on `u`, and `u` depends on `x`, then

```
dy/dx = (dy/du) * (du/dx)
```

To find how `x` affects `y`, you multiply the effects along the chain. A neural network is one long chain of functions: input → linear → activation → linear → activation → ... → loss. To find how an early weight affects the final loss, you multiply the local effects all the way back down the chain. That's it. That's backpropagation — the chain rule applied systematically, from the loss backward to every parameter.

## Why "back"?

The forward pass runs left to right. Backprop runs right to left. Here's the reason: the gradient at a layer depends on the gradient of everything downstream of it (closer to the loss). So we compute the gradient at the output first, then hand it backward to the previous layer, which uses it to compute its own gradient, and so on. Each layer receives an "incoming gradient" from its right, does a small local computation, and passes a "new gradient" to its left. Information flows backward.

## A worked example, one neuron deep

Let's do the smallest case in full so the mechanics are concrete. One input, one weight, one bias, sigmoid activation, and a squared-error loss against a target `t`.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass — save every intermediate
x = 2.0
w = 0.5
b = 0.1
t = 1.0                    # the true target

z = w * x + b              # pre-activation:  1.1
a = sigmoid(z)             # prediction:      ≈ 0.750
loss = 0.5 * (a - t)**2    # squared error:   ≈ 0.0312
```

Now walk the chain rule backward from `loss` to `w`. We need `∂loss/∂w`, and we get there by multiplying local derivatives:

```
∂loss/∂w = (∂loss/∂a) * (∂a/∂z) * (∂z/∂w)
```

Each piece is easy on its own:

```python
dloss_da = (a - t)         # derivative of 0.5*(a-t)^2  → (a - t)  ≈ -0.250
da_dz    = a * (1 - a)     # derivative of sigmoid       ≈ 0.187
dz_dw    = x               # derivative of (w*x + b) wrt w  = 2.0

dloss_dw = dloss_da * da_dz * dz_dw   # ≈ -0.250 * 0.187 * 2.0 ≈ -0.0937
dloss_db = dloss_da * da_dz * 1.0     # bias: ∂z/∂b = 1        ≈ -0.0468
```

The gradient for `w` is about `-0.094`. Negative means: increasing `w` *decreases* the loss. So to reduce the loss we nudge `w` up. That nudge is what an optimizer does (next module). The point here is that backprop just handed us the exact direction, computed as a product of three simple local derivatives.

## The recurring pattern

Notice the middle quantity `dloss_da * da_dz`. Call it `delta` — the gradient of the loss with respect to the pre-activation `z`. It shows up as the reusable building block:

```python
delta = dloss_da * da_dz    # gradient at z
grad_w = delta * dz_dw      # = delta * x
grad_b = delta * 1.0        # = delta
```

In a full network, each layer computes its own `delta` from the `delta` of the layer downstream, using the same three ingredients every time:

1. The **incoming gradient** from the layer to its right.
2. The **local derivative of its activation** (e.g. `a*(1-a)` for sigmoid, `1 if z>0 else 0` for ReLU).
3. The **local derivative of its linear step**, which is just the layer's inputs (for weights) and 1 (for the bias).

Multiply the first two to get this layer's `delta`; combine `delta` with the inputs to get the weight gradients; and pass `delta` weighted by `W` further back to become the next layer's incoming gradient. Repeat until you reach the first layer.

## Two-layer sketch

For our 2→3→1 network from the forward-pass module, the backward pass looks like this (matrix form):

```python
# Given forward pass saved: x, z1, h (=relu(z1)), z2, a (=sigmoid(z2)), target t

# Output layer
delta2 = (a - t) * a * (1 - a)        # gradient at z2
grad_W2 = np.outer(delta2, h)          # weight grads for layer 2
grad_b2 = delta2

# Hidden layer — propagate delta2 back through W2, then through ReLU
delta1 = (W2.T @ delta2) * (z1 > 0)    # gradient at z1
grad_W1 = np.outer(delta1, x)          # weight grads for layer 1
grad_b1 = delta1
```

Read `delta1`'s line carefully: `W2.T @ delta2` carries the downstream gradient back across the output layer's weights, and `(z1 > 0)` is ReLU's local derivative. The same three ingredients, one layer earlier. Stack more layers and you just repeat this step.

## Why it's efficient

A naive approach would recompute the effect of each weight from scratch — enormously wasteful in a deep net. Backprop is clever because it *reuses* the downstream gradient (`delta`) at every layer. Each intermediate result is computed exactly once and passed along. That reuse is why we could save the `z`s and `a`s during the forward pass: they're the ingredients the backward pass needs, and keeping them turns an intractable computation into a single efficient sweep.

## Key takeaways

- **Backpropagation** computes `∂loss/∂w` for every weight — the direction to nudge each one to lower the loss.
- It's the **chain rule** applied systematically from the loss backward to every parameter.
- It runs **right to left**: each layer takes an incoming gradient, computes its local `delta`, and passes a gradient further back.
- The repeating recipe: incoming gradient × activation derivative = `delta`; combine `delta` with the layer's inputs for weight gradients.
- Saving the forward pass's intermediate values (`z`, `a`) is what makes the backward pass efficient — each is computed once and reused.

## Try it

Take the single-neuron example and add a second neuron in front of it (so: input → neuron A → neuron B → loss, both with sigmoid). Pick numbers, run the forward pass saving all intermediates, then compute `∂loss/∂w_A` by hand using the chain rule across both neurons. Verify your gradient numerically: nudge `w_A` by a tiny `ε = 1e-5`, recompute the loss, and check that `(loss_new - loss_old) / ε` matches your analytic gradient. They should agree to several decimal places.
