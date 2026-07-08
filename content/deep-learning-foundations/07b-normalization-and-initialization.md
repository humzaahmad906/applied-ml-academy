# 07b — Normalization and Initialization

Regularization fought overfitting — the gap between training and validation. This lesson tackles a problem that shows up *earlier*: a network that won't train well in the first place. Before a deep network can learn anything useful, the numbers flowing through it — the activations and the gradients — have to stay in a sane range. When they don't, training stalls or blows up before overfitting is even on the table. Two techniques keep those numbers in check: careful **weight initialization** at the start, and **normalization layers** during every forward pass. Both stabilize training, and almost every modern network uses them.

## The core problem: signals that drift

Picture a signal — a vector of activations — entering the first layer of a deep network. Each layer multiplies it by a weight matrix and applies an activation function. Do that a few dozen times and something subtle happens: the *scale* of the numbers drifts. If each layer tends to shrink the signal a little, after 30 layers it has shrunk toward zero. If each layer amplifies it, after 30 layers it has exploded toward huge values. The distribution of activations at layer 30 looks nothing like the distribution at layer 1.

This drift is deadly during backpropagation. Gradients are chained products of these per-layer factors, so the same shrinking or amplifying compounds backward. Shrinking gives you **vanishing gradients** — early layers get updates so tiny they never learn. Amplifying gives you **exploding gradients** — updates so large the weights swing wildly and the loss becomes `NaN`.

The classic name for this drift is **internal covariate shift**: every layer's input distribution keeps changing as training proceeds, so each layer is forever chasing a moving target. That framing motivated normalization historically. The more modern view is humbler and more empirical — normalization mostly helps because it **smooths the loss landscape**, making the surface the optimizer walks over less jagged, so larger, more stable steps are safe. You don't need to pick a side; the practical payoff is the same. Keep the signals well-scaled and training just works better.

## Weight initialization: the starting point matters

Before training even begins, you have to fill the weight matrices with numbers. The obvious-seeming choices all fail.

**All zeros fails** because of symmetry. If every weight in a layer is identical, every neuron in that layer computes the exact same thing, receives the exact same gradient, and updates identically. They stay clones forever — a 500-neuron layer has the expressive power of one neuron. You must break symmetry with random values.

**Random but badly scaled also fails.** Too big, and activations explode layer by layer. Too small, and they vanish. The goal is a scale that *preserves the variance* of the signal as it passes through each layer — activations that leave a layer with roughly the same spread they entered with. That's the variance-preserving intuition, and two schemes formalize it:

- **Xavier / Glorot** — for `tanh` and `sigmoid`. Draws weights with variance `1 / n_in` (or `2 / (n_in + n_out)`). It assumes the activation is roughly linear near zero, which `tanh` and `sigmoid` satisfy.
- **He / Kaiming** — for `ReLU` and its relatives. ReLU zeros every negative input, cutting the signal's variance roughly in half at each layer. He initialization compensates by *doubling* the weight variance to `2 / n_in`, cancelling the halving so signals survive through deep stacks.

Use the wrong one and it bites: Xavier under ReLU assumes symmetric activations and lets variance decay exponentially — signals can vanish by layer 10.

Here's the drift, made concrete. We push a signal through 10 ReLU layers and watch its variance under bad init versus He init:

```python
import numpy as np
rng = np.random.default_rng(0)

def run(scale_fn, layers=10, width=512):
    x = rng.standard_normal((1000, width))
    for _ in range(layers):
        W = rng.standard_normal((width, width)) * scale_fn(width)
        x = np.maximum(0, x @ W)          # ReLU
        print(f"variance: {x.var():.4f}")

print("Too small (0.01):")
run(lambda n: 0.01)                        # variance collapses toward 0
print("\nHe init (sqrt(2/n)):")
run(lambda n: np.sqrt(2 / n))              # variance stays roughly stable
```

With the tiny fixed scale, the variance nosedives toward zero within a few layers — those are vanishing activations, and the gradients vanish with them. With He init, the variance hovers in a stable range all the way down. Same network, same data; the only difference is the number you multiplied the initial weights by.

## Normalization layers: fixing drift during training

Good initialization sets you up at step zero, but weights change as training proceeds, and the drift creeps back. Normalization layers fix it *continuously*, on every forward pass, by re-standardizing activations to zero mean and unit variance and then applying a learned scale and shift so the network can still represent any distribution it needs.

### BatchNorm

**BatchNorm** normalizes each feature *across the batch* — for a given feature, it computes the mean and variance over all examples in the current mini-batch and standardizes with them. It was a breakthrough for deep convolutional networks and can dramatically speed up their training.

Two catches. First, it behaves differently in **train vs. eval**: during training it uses the current batch's statistics, but at inference you may have a single example and no batch to average over, so BatchNorm keeps a running average of the statistics seen during training and uses those at eval time. Forgetting to switch modes (`model.eval()` in PyTorch) is a classic bug. Second, it's **awkward for small batches** — with a batch of 2, the batch statistics are noisy garbage — and awkward for **sequences**, where varying lengths and cross-example coupling make batch statistics ill-defined. That's why BatchNorm is now less common than it once was, dominant mainly in vision CNNs.

### LayerNorm

**LayerNorm** sidesteps all of that by normalizing *across the features of a single example* instead of across the batch. Each example is standardized using its own mean and variance over its feature dimension — no dependence on the batch at all. That means:

- It works with **batch size 1**, and behaves identically in train and eval (no running statistics to track).
- It's natural for **sequences**, where each token's feature vector is normalized on its own.

This is exactly why **transformers use LayerNorm**. It's now the default normalization for transformers and MLPs.

### RMSNorm — the modern simplification

The current frontier trims LayerNorm further. **RMSNorm** drops the mean-centering step and rescales by only the root-mean-square of the features, keeping just a learned scale (no shift). It's cheaper, trains comparably, and is what most recent large language models — LLaMA, Mistral, Qwen — actually use.

## Where each piece goes, and the practical default

In a transformer or MLP block, a normalization layer typically sits right before (pre-norm) the main computation of each block, with the initialization scheme chosen to match the activation function. The defaults that will serve you almost always:

- **LayerNorm** for transformers and MLPs; **BatchNorm** mainly for vision CNNs.
- **He / Kaiming init** whenever your activation is **ReLU** (the common case); **Xavier / Glorot** for `tanh` / `sigmoid`.

A tiny PyTorch layer wiring both together:

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)          # normalize per example, across features
        self.fc = nn.Linear(dim, dim)
        # He / Kaiming init explicitly, matched to the ReLU that follows
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return torch.relu(self.fc(self.norm(x)))   # pre-norm, then linear, then ReLU
```

Note biases are initialized to zero — symmetry breaking only needs to happen in the weights, and the normalization layer handles the centering.

## Why this matters

In the next lesson you'll build and train a real multi-layer network, and it will actually converge. That's not automatic. Stack enough layers with careless initialization and no normalization, and the network either sits frozen (vanishing gradients) or diverges to `NaN` (exploding gradients) — it never gets far enough to overfit, so none of the regularization from lesson 07 would even apply. Initialization gives the optimizer a sane starting point; normalization keeps it sane as the weights move. Together they're the reason deep networks are trainable at all, and the reason lesson 08's network learns instead of stalling.

## Key takeaways

- As signals flow through many layers their scale **drifts**, causing vanishing or exploding gradients; the fixes are good initialization and normalization layers.
- **All-zero init fails** (symmetry — all neurons stay identical); the goal of random init is to **preserve activation variance** across layers.
- **He / Kaiming init** for ReLU (variance `2 / n_in`, compensating for ReLU's halving); **Xavier / Glorot** for tanh/sigmoid.
- **BatchNorm** normalizes across the batch — great for CNNs but awkward for small batches and sequences, and behaves differently in train vs. eval.
- **LayerNorm** normalizes across features per example, works at batch size 1, and is the default for transformers/MLPs; **RMSNorm** is the modern, cheaper simplification used by recent LLMs.
- Practical defaults: **LayerNorm + He init with ReLU**.

## Try it

Extend the numpy demo above to compare initializations *and* activations. First, run the 10-layer stack with `tanh` instead of ReLU under both He and Xavier init, and watch which one keeps the variance stable — you should see the opposite winner from the ReLU case. Then, in PyTorch, build a 20-layer MLP two ways: once with default `nn.Linear` weights and no normalization, once with explicit `kaiming_normal_` init and an `nn.LayerNorm` before each layer. Feed a batch through both and print the gradient norm of the *first* layer after one backward pass. Notice how the unnormalized, badly-initialized network's early-layer gradient is orders of magnitude smaller — that's the vanishing gradient you just learned to prevent.
