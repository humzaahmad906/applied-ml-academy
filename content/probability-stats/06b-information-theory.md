# 06b — Information Theory

Every classifier you will ever train is scored by a loss that came straight out of information theory. Before you can understand why cross-entropy is *the* loss for classification, you need four ideas that stack on top of each other: how surprising an event is, how surprising a whole distribution is on average, what happens when you measure surprise with the wrong beliefs, and how far apart two distributions really are. Each one is a small step, and by the end you'll see why minimizing cross-entropy is the same as making your model's predicted probabilities match reality.

## Surprise: how shocked should you be?

Start with a single event. If something almost certain happens, you learn almost nothing. If something you thought impossible happens, you learn a lot. **Self-information** (also called surprise) captures this: the surprise of an event with probability p is

surprise(p) = −log p

Why the logarithm and not, say, 1/p? Because we want surprise to *add up*. If you flip two independent fair coins, the chance of two heads is 1/2 × 1/2 = 1/4. Intuitively the surprise of "two heads" should be the surprise of one head plus the surprise of another. Probabilities multiply, but logs turn multiplication into addition, so −log(1/4) = −log(1/2) + −log(1/2). The log is the only function that converts "independent events multiply" into "surprise adds."

The minus sign is just bookkeeping: probabilities are at most 1, so their logs are negative, and negating gives a positive surprise. A certain event (p = 1) has surprise −log 1 = 0. Nothing learned.

The base of the log sets the units. Base 2 gives **bits**: one bit is the surprise of a fair coin flip. The natural log (base e) gives **nats**, which is what ML code uses because it plays nicely with calculus. They differ only by a constant factor, so the choice never changes which model wins — it just rescales the numbers.

```python
import numpy as np

# surprise of a fair coin flip, in bits and nats
print("bits:", -np.log2(0.5))   # 1.0
print("nats:", -np.log(0.5))    # ~0.693

# a rare event is far more surprising
print("p=0.01, bits:", -np.log2(0.01))  # ~6.64
```

## Entropy: surprise you expect on average

Surprise is about one event. **Entropy** is the average surprise of a whole distribution — how uncertain you are before the outcome arrives. It's the expected value of surprise, weighting each outcome's surprise by how often it occurs:

H(p) = Σ p(x) · (−log p(x)) = −Σ p(x) log p(x)

Entropy is largest when a distribution is most spread out and smallest when it's concentrated. A fair coin has entropy 1 bit — maximum uncertainty for two outcomes. A rigged coin that lands heads 99% of the time has low entropy: you can almost always guess right, so there's little uncertainty to resolve. A two-headed coin has entropy 0: no surprise ever.

Here is the fair-die version. Six equally likely faces, each with surprise −log₂(1/6):

H = 6 × (1/6) × log₂(6) = log₂(6) ≈ 2.585 bits

That's the most uncertain a six-sided distribution can be. **The uniform distribution always maximizes entropy** — when every outcome is equally likely, you have the least possible information about what will happen. Any lopsidedness lowers it.

```python
import numpy as np

def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    return -np.sum(p * np.log2(p))

fair_die = np.full(6, 1/6)
loaded_die = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
print("fair:", entropy_bits(fair_die))     # ~2.585
print("loaded:", entropy_bits(loaded_die)) # ~2.161
```

The loaded die is more predictable, so its entropy is lower.

## Cross-entropy: paying for the wrong code

Now the key move. Suppose the true distribution of outcomes is p, but you built your understanding of the world around a *different* distribution q — your model's predictions. **Cross-entropy** measures the average surprise you actually experience when outcomes follow p but you compute surprise using q:

H(p, q) = −Σ p(x) log q(x)

Notice which distribution sits where. The real frequencies p do the weighting (reality decides how often each outcome shows up), but the surprise −log q comes from your beliefs. If q matches p perfectly, cross-entropy equals entropy — you're perfectly calibrated. If q is wrong, you're systematically surprised more than necessary.

The classic intuition is codes. Entropy is the shortest average message length if you design your code around the true frequencies p. Cross-entropy is the average length you get if you designed your code around q instead — using the wrong codebook. You waste bits every time reality disagrees with your assumptions. A worked case: reality is a fair coin, p = [0.5, 0.5], but your model believes q = [0.9, 0.1].

```python
import numpy as np

p = np.array([0.5, 0.5])
q = np.array([0.9, 0.1])
cross_entropy = -np.sum(p * np.log2(q))
print("cross-entropy:", cross_entropy)  # ~1.74 bits
# vs entropy of p, which is exactly 1.0 bit — the wrong beliefs cost ~0.74 extra bits
```

## KL divergence: the gap you're wasting

Cross-entropy is always at least as big as entropy, and the *extra* amount — the bits wasted by using the wrong distribution — is the **Kullback–Leibler divergence**:

KL(p ‖ q) = H(p, q) − H(p) = Σ p(x) log( p(x) / q(x) )

Rearrange that and you get the single most important identity in this lesson:

**H(p, q) = H(p) + KL(p ‖ q)**

Cross-entropy equals the true entropy plus the KL gap. KL divergence has two properties worth memorizing. First, it is **always ≥ 0**, and it equals 0 exactly when q = p. You can never do better than knowing the truth. Second, it is **not symmetric**: KL(p ‖ q) ≠ KL(q ‖ p) in general, so it is not a distance. It measures how far q is from p, asymmetrically, from p's point of view. In the coin example above, the wasted 0.74 bits *is* the KL divergence.

```python
import numpy as np

p = np.array([0.5, 0.5])
q = np.array([0.9, 0.1])
kl = np.sum(p * np.log2(p / q))
print("KL(p||q):", kl)   # ~0.74  -> matches cross-entropy(1.74) - entropy(1.0)
print("KL(q||p):", np.sum(q * np.log2(q / p)))  # ~0.53, different -> not symmetric
```

## Why this matters for ML

Here is the payoff. When you train a classifier, the "true" distribution p for a given example is what the label tells you. With a one-hot label — class 3 is correct, everything else wrong — p puts probability 1 on the true class and 0 elsewhere. Your model outputs q, a predicted probability for each class. Training minimizes the cross-entropy H(p, q) between them.

Because H(p, q) = H(p) + KL(p ‖ q), and the label's entropy H(p) is a fixed constant you can't change, **minimizing cross-entropy is exactly minimizing the KL divergence** from your predictions to the true label distribution. You are pushing q toward p as measured by KL. This is also identical to **maximizing the likelihood** of the data under your model — three descriptions of the same optimization.

And with a one-hot label, the sum collapses beautifully. Every term is p(x)·(−log q(x)), but p is zero everywhere except the true class, where it's 1. So all terms vanish except one:

loss = −log q(true class)

The entire loss is just the negative log of the probability your model assigned to the correct answer. Confident and right → tiny loss. Confident and wrong → huge loss.

```python
import numpy as np

logits = np.array([2.0, 0.5, 1.0, 3.0])   # raw model outputs, one per class
true_class = 3

# softmax turns logits into probabilities
exp = np.exp(logits - logits.max())        # subtract max for numerical stability
probs = exp / exp.sum()
loss = -np.log(probs[true_class])
print("probs:", probs.round(3))            # ~[0.24 0.05 0.09 0.62]
print("loss:", loss)                       # ~0.483
```

In practice you don't write this by hand. **`torch.nn.CrossEntropyLoss` takes raw logits directly** — do *not* apply a softmax first. It fuses log-softmax and the negative-log-likelihood step into one numerically stable operation, avoiding the underflow you'd hit computing tiny probabilities yourself. (If you *do* have log-probabilities already, `nn.NLLLoss` is the second half on its own.) It also accepts a `label_smoothing` argument (default 0.0): a small value like 0.1 nudges the one-hot target toward a slightly softer distribution, which discourages the model from becoming wildly overconfident and often improves calibration. That single loss function, sitting on top of these four ideas, trains nearly every classifier and language model you'll meet.

## Key takeaways

- Self-information −log p measures the surprise of one event; logs make independent surprises add, and base 2 gives bits, base e gives nats.
- Entropy H(p) = −Σ p log p is expected surprise; the uniform distribution maximizes it, a certain outcome has entropy 0.
- Cross-entropy H(p, q) = −Σ p log q is the average surprise when reality is p but you score with q — the cost of the wrong codebook.
- KL divergence is the gap H(p, q) − H(p): always ≥ 0, zero only when q = p, and not symmetric, so it's not a true distance.
- Minimizing cross-entropy = minimizing KL to the true labels = maximizing likelihood; with one-hot labels it collapses to −log(predicted prob of the true class).
- `nn.CrossEntropyLoss` expects logits, applies log-softmax internally, and offers `label_smoothing`; never softmax your outputs before feeding it.

## Try it

Take the logits `[1.0, 2.0, 0.5, 0.1]` and a true class of 1. Compute the softmax probabilities with numpy (remember to subtract the max first), then compute the cross-entropy loss as −log of the true class probability. Now nudge the logit for class 1 upward to `3.0` and recompute — watch the loss drop as the model grows more confident in the right answer. Finally, build a full true distribution `p = [0, 1, 0, 0]` and confirm that −Σ p log q gives you the exact same number as the one-line collapse. If you have PyTorch installed, feed the same raw logits into `torch.nn.CrossEntropyLoss()` with target `torch.tensor([1])` and check that it matches your by-hand value.
