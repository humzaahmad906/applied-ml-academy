# 07 — Regularization

In the last module we saw the danger sign: training loss keeps falling while validation loss starts climbing. That's **overfitting** — the network is memorizing the training data instead of learning patterns that generalize. Regularization is the family of techniques that fight this. The goal isn't to make the network fit the training data better; it's to make it fit *unseen* data better, which sometimes means deliberately handicapping it during training.

## The core problem: memorizing vs. learning

A big neural network has enough capacity to memorize its entire training set — to essentially build a lookup table. When it does, training loss goes to near zero, and the network looks brilliant on data it has seen. But on new data it's useless, because a lookup table has no idea what to do with an input it's never encountered. Real learning means capturing the underlying pattern, not the specific examples.

The tension is fundamental: enough capacity to learn the pattern, but not so much freedom that it just memorizes. Regularization tips the balance toward general patterns. We'll cover the three techniques you'll use most: weight decay, dropout, and early stopping.

## Weight decay (L2 regularization)

The intuition: large weights let a network draw wild, jagged decision boundaries that snake around individual training points — a hallmark of memorizing. Small, smooth weights force gentler boundaries that generalize better. Weight decay discourages large weights by adding a penalty to the loss proportional to the sum of squared weights:

```python
# Add lambda * sum(w^2) to the loss.  Lambda controls the strength.
def loss_with_l2(base_loss, params, lam):
    l2 = sum(np.sum(w**2) for w in params)
    return base_loss + lam * l2
```

Because the optimizer minimizes total loss, it now has an incentive to keep weights small unless a large weight genuinely helps fit the data. In practice this shows up as a tiny extra term in the update — every step, weights are nudged slightly toward zero (hence "decay"):

```python
# Equivalent effect directly in the update step
w = w - lr * grad - lr * lam * w   # the last term shrinks w each step
```

The strength `lam` (often `weight_decay` in libraries) is a knob: too small and it does nothing; too large and it crushes the weights so hard the network can't learn. Typical values are small, like `1e-4` to `1e-2`. Note we usually don't decay the biases — only the weights.

## Dropout

Dropout is a wonderfully simple idea that feels almost too crude to work. During training, on each forward pass, randomly "drop" (set to zero) a fraction of the neurons in a layer. Different neurons drop out each pass:

```python
def dropout(a, rate, training):
    if not training:
        return a                       # use the full network at test time
    mask = (np.random.rand(*a.shape) > rate) / (1 - rate)
    return a * mask                    # zero some neurons, scale the rest
```

Why does this help? Because the network can't rely on any single neuron always being present, it's forced to spread the work across many neurons and learn redundant, robust features. No neuron gets to become a fragile single point of failure. It's a bit like training a whole ensemble of slightly different networks and averaging them, at almost no extra cost.

Two details matter. First, dropout is only active **during training** — at test time you use the full network with every neuron present. Second, the `/ (1 - rate)` scaling (called inverted dropout) keeps the expected magnitude of the outputs the same whether or not dropout is on, so the network sees consistent input scales in both modes. A `rate` of 0.2 to 0.5 is common; higher rates regularize harder.

## Early stopping

The simplest and often most effective technique needs no math at all. You're already watching the validation loss each epoch. Early stopping says: **stop training when the validation loss stops improving**, even if the training loss would keep falling. The moment validation loss turns upward is the moment overfitting begins, so you halt there and keep the best model you saw.

```python
best_val = float("inf")
patience, wait = 5, 0

for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = evaluate(X_val, y_val)

    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(params)   # remember the best-so-far weights
        wait = 0
    else:
        wait += 1
        if wait >= patience:      # no improvement for `patience` epochs
            print("early stopping")
            break
```

The `patience` parameter gives training a few epochs of grace before quitting, because validation loss is noisy and can wobble upward briefly before continuing down. When it does stop, you restore the checkpoint from the best epoch — not the last one. Early stopping is essentially free and worth using almost always.

## How they combine

These aren't either/or. A typical setup uses all three together: a modest weight decay, dropout in the larger hidden layers, and early stopping as the final safety net watching the validation curve. Each attacks overfitting from a different angle, and they stack well.

There's also a quieter, broader truth: **more and better data is the best regularizer of all.** A network can't memorize a set it's too large and varied to memorize, and data augmentation (creating modified copies of your examples) is a way to manufacture more variety. When you can get more data, that often beats any amount of clever regularization tuning.

## A word of caution

Regularization is a balance, not a "more is better" dial. Over-regularize — huge weight decay, aggressive dropout, stopping too early — and you swing into **underfitting**, where the network is too constrained to even capture the real pattern. Both curves stay high and flat. The right amount is the amount that minimizes *validation* loss, which you find by watching the curves and adjusting. Start light, add regularization only when you actually see overfitting.

## Key takeaways

- **Overfitting** is memorizing training data instead of learning general patterns; regularization fights it to improve performance on unseen data.
- **Weight decay (L2)** penalizes large weights, favoring smoother boundaries that generalize; tune the strength `lam`.
- **Dropout** randomly zeros neurons during training so the network learns robust, redundant features; disable it at test time.
- **Early stopping** halts when validation loss stops improving and keeps the best checkpoint — nearly free, almost always worth it.
- Combine techniques, don't over-regularize (that causes underfitting), and remember more/varied data is the best regularizer.

## Try it

Take your trained two-layer network and deliberately overfit it: use a tiny training set (say 20 examples), a large hidden layer, and train for many epochs until validation loss clearly rises while training loss keeps dropping — plot both curves to confirm. Then add each technique one at a time: (1) weight decay with `lam = 1e-2`, (2) dropout at rate 0.3 on the hidden layer, (3) early stopping with patience 5. Note how each changes the gap between the training and validation curves.
