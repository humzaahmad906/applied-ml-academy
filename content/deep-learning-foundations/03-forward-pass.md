# 03 — The Forward Pass

We have neurons, and we have activations. Now let's watch a whole network actually compute something. The **forward pass** is the process of feeding inputs in one end and reading a prediction out the other — layer by layer, each one transforming the data before handing it on. If you understand the forward pass, you understand what a network *is* when it makes a prediction. (How it learns comes later; for now we just run it.)

## From one neuron to a layer

A single neuron takes a vector of inputs, computes a weighted sum plus a bias, and applies an activation. A **layer** is just many neurons operating on the same inputs in parallel. Each neuron has its own weights and its own bias.

Rather than loop over neurons one at a time, we stack their weights into a matrix and do it all with one matrix multiply. Say a layer has 3 neurons and takes 2 inputs. Its weight matrix `W` has shape `(3, 2)` — one row per neuron — and its bias vector `b` has shape `(3,)`:

```python
import numpy as np

def layer(x, W, b, activation):
    z = W @ x + b        # shape (3,): one pre-activation per neuron
    return activation(z) # shape (3,): one output per neuron
```

That single line `W @ x + b` computes all three neurons at once. This is why deep learning loves matrix multiplication: it's the natural language of "many neurons, same inputs."

## Stacking layers

A network chains layers together. The outputs of layer 1 become the inputs to layer 2, and so on. Let's build a tiny two-layer network: 2 inputs → a hidden layer of 3 neurons (ReLU) → 1 output neuron (sigmoid, so we get a probability).

```python
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, params):
    W1, b1, W2, b2 = params
    h = relu(W1 @ x + b1)     # hidden layer:  (3,)
    y = sigmoid(W2 @ h + b2)  # output layer:  (1,)
    return y
```

Read it top to bottom and you're reading the forward pass. Input `x` enters, gets transformed into a hidden representation `h`, and `h` gets transformed into the final output `y`. Nothing more mysterious than function composition.

## A concrete run

Let's put numbers in and turn the crank by hand, so nothing is hidden.

```python
x = np.array([1.0, 2.0])              # two inputs

W1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])           # (3, 2): 3 hidden neurons
b1 = np.array([0.0, 0.0, 0.0])

W2 = np.array([[0.7, 0.8, 0.9]])      # (1, 3): 1 output neuron
b2 = np.array([0.1])

# Layer 1
z1 = W1 @ x + b1        # [0.1*1+0.2*2, 0.3*1+0.4*2, 0.5*1+0.6*2]
                        # = [0.5, 1.1, 1.7]
h  = relu(z1)           # all positive → unchanged: [0.5, 1.1, 1.7]

# Layer 2
z2 = W2 @ h + b2        # 0.7*0.5 + 0.8*1.1 + 0.9*1.7 + 0.1
                        # = 0.35 + 0.88 + 1.53 + 0.1 = 2.86
y  = sigmoid(z2)        # ≈ 0.946

print(y)  # [0.946...]
```

The network says "0.946" — if this were a classifier, that's a strong vote for class 1. Every number came from the same two operations repeated: a matrix multiply plus bias, then an activation.

## Notation you'll see everywhere

People write the forward pass compactly. For a layer `l`:

```
z[l] = W[l] @ a[l-1] + b[l]     # pre-activation
a[l] = activation(z[l])         # activation (the layer's output)
```

Here `a[l-1]` is the previous layer's output, and `a[0]` is just the input `x`. The two symbols to keep straight:

- **`z`** — the pre-activation, the raw weighted sum before the bend.
- **`a`** — the activation, what the layer actually passes forward.

We keep track of both because backpropagation (next module) needs them. During a forward pass alone you only care about the final `a`; but saving the intermediate `z`s and `a`s is what makes learning possible later.

## Batching: many inputs at once

In practice you rarely push one example through at a time. You stack a **batch** of examples into a matrix `X` of shape `(batch_size, num_features)` and process them together. The math barely changes — you just arrange the matrix multiply so each row is one example:

```python
def forward_batch(X, W1, b1, W2, b2):
    # X: (batch_size, 2)
    H = relu(X @ W1.T + b1)      # (batch_size, 3)
    Y = sigmoid(H @ W2.T + b2)   # (batch_size, 1)
    return Y
```

The whole batch flows through in the same two matrix multiplies. This is why GPUs, which are extremely good at large matrix multiplications, make neural networks fast: a batch of a thousand examples is one big matrix operation, not a thousand little ones.

## What the forward pass gives you

The output is the network's current best guess, given its current weights. Right after initialization those weights are random, so the guess is garbage — 0.946 might as well be a coin flip. The forward pass doesn't know or care whether the answer is right; it just computes. Comparing that output to the true answer, and using the gap to improve the weights, is the job of the loss function and backpropagation. But every bit of learning starts here, with a forward pass producing a prediction to critique.

## Key takeaways

- The **forward pass** feeds inputs through the network layer by layer to produce a prediction.
- A layer is many neurons in parallel; stacking their weights into a matrix lets one `W @ x + b` compute the whole layer.
- Each layer's output becomes the next layer's input — it's function composition.
- Track two quantities per layer: `z` (pre-activation) and `a` (activation); both matter for learning later.
- **Batching** processes many examples in one matrix multiply, which is what makes networks fast on GPUs.

## Try it

Extend the two-layer example above to a three-layer network: 2 inputs → 4 hidden (ReLU) → 3 hidden (ReLU) → 1 output (sigmoid). Initialize the weight matrices with small random numbers (`np.random.randn(...) * 0.1`) and the biases with zeros. Run a forward pass on `x = [1.0, -1.0]` and print the shape of `z` and `a` at every layer. Confirm the shapes chain correctly — each layer's output length must match the next layer's expected input length.
