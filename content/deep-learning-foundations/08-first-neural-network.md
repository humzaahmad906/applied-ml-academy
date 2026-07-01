# 08 — Your First Neural Network

Everything so far has been a piece: neurons, activations, the forward pass, backprop, loss, optimizers, the training loop, regularization. This module puts them all in one place and trains a real network end to end, from scratch, in numpy — no deep learning framework, just the ideas you already understand. Then we read the training curves to see what the network actually learned. By the end you'll have watched a network go from random guessing to solving a problem it couldn't solve as a single line.

## The problem

We'll teach a network the XOR-flavored task we couldn't solve with a straight line earlier: classify points by whether they fall in two opposite corners. It's small, it's fast, and — critically — it's *not* linearly separable, so a network genuinely has to learn something a linear model can't.

```python
import numpy as np
np.random.seed(0)   # reproducible: seed numpy before anything random

# Four XOR points, duplicated with a little noise to make a small dataset
base_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
base_y = np.array([[0], [1], [1], [0]], dtype=float)

X = np.repeat(base_X, 50, axis=0) + np.random.randn(200, 2) * 0.05
y = np.repeat(base_y, 50, axis=0)
```

## The network

A single hidden layer with a handful of neurons is all XOR needs. We'll use 2 inputs → 8 hidden (ReLU) → 1 output (sigmoid), and binary cross-entropy loss.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# He-style small init keeps early activations from blowing up
W1 = np.random.randn(2, 8) * np.sqrt(2 / 2)
b1 = np.zeros(8)
W2 = np.random.randn(8, 1) * np.sqrt(2 / 8)
b2 = np.zeros(1)
```

## Forward, loss, backward — assembled

Here is one full training step, every idea from the course wired together. Read the backward pass against the backpropagation lesson: it's the same `delta` recipe, in matrix form, for a batch.

```python
def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def bce_loss(pred, target, eps=1e-9):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

def backward(X, y, z1, a1, a2):
    n = len(X)
    # Output layer: for sigmoid + BCE, the gradient at z2 simplifies to (a2 - y)
    dz2 = (a2 - y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    # Hidden layer: propagate back through W2, then through ReLU
    dz1 = (dz2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2
```

The one shortcut worth noting: for a sigmoid output paired with cross-entropy loss, the messy chain-rule product collapses into a clean `(a2 - y)`. That's not luck — those two are designed to pair, and the simplification is one reason the combination is standard.

## The training loop

Now the training loop from earlier, with a train/validation split, and plain SGD.

```python
# Split BEFORE training, and shuffle the split
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]
cut = int(0.8 * len(X))
X_train, y_train = X[:cut], y[:cut]
X_val,   y_val   = X[cut:], y[cut:]

lr = 0.5
history = {"train": [], "val": []}

for epoch in range(300):
    # (dataset is tiny — one batch per epoch; shuffle each time)
    idx = np.random.permutation(len(X_train))
    Xb, yb = X_train[idx], y_train[idx]

    z1, a1, z2, a2 = forward(Xb)
    dW1, db1, dW2, db2 = backward(Xb, yb, z1, a1, a2)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    train_loss = bce_loss(a2, yb)
    val_loss   = bce_loss(forward(X_val)[3], y_val)
    history["train"].append(train_loss)
    history["val"].append(val_loss)

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
```

Run this and you'll see something like:

```
epoch   0  train 0.7161  val 0.7042
epoch  50  train 0.6538  val 0.6490
epoch 100  train 0.4123  val 0.4088
epoch 150  train 0.1408  val 0.1402
epoch 200  train 0.0561  val 0.0559
epoch 250  train 0.0333  val 0.0331
```

## Reading the curves

This is the payoff — learning to read what those numbers mean:

- **The loss falls.** From ~0.71 (random guessing on a two-class problem starts near `ln 2 ≈ 0.69`) down toward zero. The network is learning.
- **There's a slow start.** For the first ~50 epochs the loss barely moves, then it drops sharply around epoch 100. This is common: the network spends early epochs orienting itself before it "clicks" onto the pattern. Don't panic at a flat early curve.
- **Train and validation track each other closely.** They fall together and stay near each other. That's the signature of healthy learning with no overfitting — the network is learning the real pattern, not memorizing. If validation had peeled upward while train kept dropping, we'd reach for the regularization techniques from earlier.

Check that it actually solved XOR:

```python
preds = (forward(base_X)[3] > 0.5).astype(int).ravel()
print(preds)   # [0 1 1 0]  — correct XOR, the thing a single line cannot do
```

Four correct predictions on the exact problem a linear model fails. The hidden layer learned intermediate features that bend the decision boundary — exactly the promise from the start of the course, now realized in trained weights.

## What you just did

You built a neuron, gave it a nonlinearity, stacked neurons into layers, ran a forward pass, derived gradients with backprop, chose a loss and an optimizer, wrapped it in a training loop with a proper data split, and watched the curves confirm real learning. That's a complete neural network, understood top to bottom — the same skeleton scales up to networks with millions of parameters. The frameworks you'll use next automate the calculus and the matrix bookkeeping, but every one of them is running exactly this loop underneath.

## Key takeaways

- A working network is all the course pieces assembled: forward → loss → backprop → update, looped, on a proper data split.
- **Seed your RNG** and **split before training** — reproducibility and honest validation are not optional.
- Sigmoid output + cross-entropy loss makes the output gradient collapse to a clean `(pred - target)`.
- Read curves by shape: falling loss = learning; a flat early stretch is normal; train and validation tracking together = healthy, no overfitting.
- The same loop scales from this 8-neuron toy to the largest networks — frameworks just automate the calculus.

## Try it

Run the full network above and reproduce the curves. Then experiment: (1) shrink the hidden layer to 2 neurons — does it still solve XOR, and how does the curve change? (2) Set the learning rate to 5.0 and to 0.01 — watch one diverge and the other crawl. (3) Add the L2 weight decay from earlier and see whether the curves shift. For each, write one sentence on what the training curve told you before you even checked the final predictions.
