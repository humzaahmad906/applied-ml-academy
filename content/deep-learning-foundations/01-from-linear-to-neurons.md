# 01 — From Linear Models to Neurons

You already know a linear model. Given some inputs, you multiply each by a weight, add them up, add a bias, and out comes a number. Logistic regression, linear regression, a simple classifier — they all live inside that same tidy equation. So why do we need anything more? This module answers that question and, in doing so, quietly introduces the neuron: the smallest unit of a neural network.

## The linear model, one more time

Suppose you have two inputs, `x1` and `x2`. A linear model computes:

```python
z = w1 * x1 + w2 * x2 + b
```

The weights `w1, w2` decide how much each input matters. The bias `b` shifts the result up or down. That's it. In vector form, with `x = [x1, x2]` and `w = [w1, w2]`:

```python
import numpy as np

def linear(x, w, b):
    return np.dot(w, x) + b

x = np.array([2.0, -1.0])
w = np.array([0.5, 1.5])
b = 0.1
print(linear(x, w, b))  # 0.5*2 + 1.5*(-1) + 0.1 = -0.4
```

This is powerful and fast. It's also fundamentally limited: no matter how you tune `w` and `b`, the decision boundary it can draw is a straight line (or a flat plane in higher dimensions). If your data can be split by a line, you're done. If it can't, you're stuck.

## Where lines fail: the XOR problem

Here is the classic example that broke the early hype around simple models. Consider four points and the labels we want:

| x1 | x2 | label |
|----|----|-------|
| 0  | 0  | 0     |
| 0  | 1  | 1     |
| 1  | 0  | 1     |
| 1  | 1  | 0     |

This is XOR: the output is 1 when exactly one input is 1. Try to draw a single straight line that puts the two `1`s on one side and the two `0`s on the other. You can't. The `1`s sit on opposite corners; so do the `0`s. No line separates them. A linear model will always get at least one point wrong on XOR.

This isn't a quirk of XOR. Real data — images, audio, language — is full of relationships that no straight line can capture. We need models that can bend.

## Two ideas that fix everything

The fix comes from two simple moves.

**Idea 1: Stack layers.** Instead of going straight from inputs to output, insert an intermediate layer. The first layer transforms the inputs into a new set of values (a "hidden" representation), and the second layer works on those. You're composing functions: the output of one becomes the input of the next.

**Idea 2: Add a nonlinearity.** Here's the catch — stacking *linear* layers alone gains you nothing. A linear function of a linear function is still linear. Do the algebra: if `layer2(layer1(x))` is just matrix multiplies, the whole thing collapses back into one big matrix multiply, i.e. one linear model in disguise.

```python
# Two linear layers with NO nonlinearity between them
h = W1 @ x + b1      # first layer
y = W2 @ h + b2      # second layer
# Substitute: y = W2 @ (W1 @ x + b1) + b2
#               = (W2 @ W1) @ x + (W2 @ b1 + b2)
# That's just one linear layer with weights (W2 @ W1). No gain.
```

To actually gain expressive power, you insert a nonlinear function between the layers — something that bends. A common choice squashes negatives to zero (ReLU), or squashes everything into a smooth S-curve (sigmoid). We cover these in the next module. The point for now: the nonlinearity is what makes stacking worthwhile.

## Meet the neuron

A single **neuron** is a linear model followed by a nonlinearity. That's the entire definition:

```python
def neuron(x, w, b, activation):
    z = np.dot(w, x) + b   # the linear part (a "pre-activation")
    return activation(z)   # the bend
```

The `z` is called the pre-activation. Passing it through `activation` gives the neuron's output. Wire many neurons side by side and you get a **layer**. Stack layers and you get a **network**. Every neuron is doing the same humble thing you already understand — a weighted sum plus a bias — but the nonlinearity and the stacking together let the network carve curved, complex decision boundaries.

## Why this solves XOR

With one hidden layer of two neurons, a network can solve XOR. Roughly: the first neuron can learn to detect "at least one input is on," the second can learn "both inputs are on," and the output neuron combines them into "exactly one is on." Each neuron is still simple; the *composition* is what expresses the curve a single line never could. The network learns those intermediate detectors on its own during training — you don't program them by hand.

## A mental model

Think of each layer as re-drawing your data in new coordinates. The raw inputs might be tangled, but after a layer bends and reshapes them, they can become easier to separate. Deep networks just repeat this: fold, reshape, fold again, until the final layer sees data simple enough to split with a line. Depth buys you more folds; nonlinearity is what lets each fold actually bend.

## Key takeaways

- A linear model computes a weighted sum plus a bias; it can only draw straight decision boundaries.
- Some problems (like XOR) are not linearly separable — no single line works.
- Stacking linear layers alone gains nothing: linear composed with linear is still linear.
- A **nonlinearity** between layers is what unlocks expressive power.
- A **neuron** = linear step + nonlinearity. Layers of neurons stacked into a network can bend to fit complex data.

## Try it

Implement a single neuron in numpy that takes two inputs and applies a step nonlinearity: output 1 if `z >= 0`, else 0. By hand, pick weights and a bias so the neuron computes the **AND** function (output 1 only when both inputs are 1). Test it on all four input pairs `(0,0), (0,1), (1,0), (1,1)`. Then try to find weights for **XOR** with a single neuron — convince yourself it's impossible, and note what a second layer would need to add.
