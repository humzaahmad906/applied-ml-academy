# 02 — Activation Functions

In the last module we learned that a neuron is a linear step followed by a nonlinearity, and that the nonlinearity is what makes a deep network more than an expensive line. That nonlinearity has a name: the **activation function**. This module looks at the three you'll meet most often — sigmoid, tanh, and ReLU — what each one does, and when to reach for which.

## Why we need the bend, restated

An activation function takes the neuron's pre-activation `z` (the weighted sum plus bias) and reshapes it. Without it, layers collapse into one linear layer. With it, each layer can bend its output, and stacked bends can approximate astonishingly complex functions. The activation is applied element-wise: every value gets the same function, independently.

## Sigmoid

The sigmoid squashes any real number into the range `(0, 1)`:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Large positive `z` maps near 1; large negative `z` maps near 0; `z = 0` maps to exactly 0.5. That smooth S-shape makes sigmoid feel like a soft on/off switch, which is why it's a natural fit for the *output* of a binary classifier — you can read the result as a probability.

Its weakness shows up when you use it deep inside a network. Look at the curve's tails: for very large or very small `z`, the output flattens out. A flat curve has a near-zero slope, and slope (the derivative) is exactly what training needs to update weights. When the slope vanishes, learning stalls. This is the **vanishing gradient** problem, and it's why sigmoid fell out of favor for hidden layers.

```python
# sigmoid's derivative — note it peaks at 0.25 and dies in the tails
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)   # max value 0.25, at z = 0
```

Even at its best (`z = 0`), the gradient is only 0.25. Multiply several of those together across layers and the signal shrinks fast.

## Tanh

Tanh is sigmoid's close cousin, squashing into `(-1, 1)` instead of `(0, 1)`:

```python
def tanh(z):
    return np.tanh(z)
```

The key difference is that tanh is **zero-centered**: its outputs are balanced around 0, whereas sigmoid outputs are always positive. Zero-centered activations tend to make optimization a little smoother, because the inputs to the next layer aren't all pushed in the same direction. Tanh was the go-to hidden-layer activation for years. But it shares sigmoid's flaw — flat tails, vanishing gradients — so it too struggles in very deep networks.

## ReLU

ReLU (Rectified Linear Unit) is almost embarrassingly simple:

```python
def relu(z):
    return np.maximum(0, z)
```

If the input is positive, pass it through unchanged. If it's negative, output zero. That's the whole function. Yet it powers most modern deep networks, for three reasons:

1. **No vanishing gradient on the positive side.** For any positive `z`, the slope is exactly 1 — the gradient flows through untouched, no matter how deep the network.
2. **Cheap.** A single comparison. No exponentials, unlike sigmoid and tanh.
3. **Sparsity.** Because negatives become zero, many neurons output 0 for a given input, which can make representations cleaner and computation lighter.

Its gradient is trivial:

```python
def relu_grad(z):
    return (z > 0).astype(float)   # 1 where positive, 0 where negative
```

ReLU has one failure mode worth knowing: the **dying ReLU**. If a neuron's weights push it to always output negative pre-activations, its gradient is always 0, and it never updates again — it's dead. In practice this is manageable, and variants exist to prevent it. **Leaky ReLU** is the common fix: instead of flattening negatives to zero, it lets a small slope through.

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
```

Now negative inputs still produce a tiny (nonzero) gradient, so a neuron can recover.

## Choosing one

A practical default:

- **Hidden layers:** start with ReLU. It's fast, deep-friendly, and the standard baseline. Reach for Leaky ReLU if you suspect dying neurons.
- **Binary classification output:** sigmoid, so you get a probability in `(0, 1)`.
- **Multi-class output:** softmax (a generalization of sigmoid across several classes that produces probabilities summing to 1).
- **tanh:** useful in some recurrent networks and when you specifically want zero-centered, bounded outputs.

Here's a quick comparison you can run:

```python
z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("sigmoid:", np.round(sigmoid(z), 3))
print("tanh:   ", np.round(tanh(z), 3))
print("relu:   ", relu(z))
# sigmoid: [0.119 0.378 0.5   0.622 0.881]
# tanh:    [-0.964 -0.462 0.    0.462 0.964]
# relu:    [0.  0.  0.  0.5 2. ]
```

Notice how sigmoid and tanh compress the extremes while ReLU leaves positive values completely alone.

## The bigger picture

Activation functions are one of the few genuinely simple pieces of a neural network, but the choice has real consequences for how well gradients flow during training — which is the whole subject of backpropagation, coming up shortly. When a deep network "won't learn," a saturating activation squashing gradients to zero is one of the first suspects.

## Key takeaways

- The activation function is the nonlinearity that makes stacked layers expressive; it's applied element-wise.
- **Sigmoid** maps to `(0,1)`, great for probabilities, but its flat tails cause vanishing gradients — avoid it in hidden layers.
- **Tanh** maps to `(-1,1)` and is zero-centered, but shares the vanishing-gradient issue.
- **ReLU** passes positives through and zeros negatives; it's cheap, deep-friendly, and the standard hidden-layer default. Watch for dying neurons; Leaky ReLU fixes them.
- Match the output activation to the task: sigmoid for binary, softmax for multi-class.

## Try it

Plot all three activations and their gradients over the range `z ∈ [-6, 6]` using numpy (compute the values; sketch or print them). For each function, find the range of `z` where the gradient is essentially zero. Then explain in one sentence why a network built entirely from sigmoid hidden layers would train more slowly than one built from ReLU.
