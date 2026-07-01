# 06 — Training Loops and Batching

You now have all four pieces: forward pass, loss, backprop, optimizer. Training a network is just running those four in a loop, over and over, until the network is good enough. This module assembles them into a **training loop** and explains the vocabulary that surrounds it — epochs, mini-batches, learning rate schedules — and the practical choices that decide whether training goes smoothly or falls apart.

## The loop, in one glance

Here is the entire skeleton of neural network training. Everything else is detail hung on this frame:

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in get_batches(X_train, y_train, batch_size):
        preds = forward(batch_x, params)         # 1. forward pass
        loss  = loss_fn(preds, batch_y)          # 2. measure loss
        grads = backward(loss, params)           # 3. backprop
        params = optimizer_step(params, grads)   # 4. update weights
```

Read it as a sentence: for each pass over the data, break the data into batches, and for each batch predict, measure, blame, and adjust. That's training. The art is in the surrounding decisions.

## Epochs, batches, and iterations

Three words get used constantly, and mixing them up causes confusion:

- **Epoch** — one full pass through the entire training set. If you have 10,000 examples, one epoch means the network has seen all 10,000 once.
- **Mini-batch (batch)** — a small subset of examples processed together in a single forward/backward pass. Common sizes are 32, 64, 128, 256.
- **Iteration (step)** — one weight update, i.e. one batch processed. If you have 10,000 examples and a batch size of 100, one epoch is 100 iterations.

So training for 20 epochs at batch size 100 on 10,000 examples means `20 * 100 = 2,000` weight updates total.

## Why mini-batches?

You could compute the gradient over the whole dataset before each update (**batch gradient descent**) or over a single example at a time (**pure SGD**). Mini-batches sit in the sweet spot between them, and there are three good reasons they win:

1. **Efficiency.** A batch is one big matrix multiply — exactly what GPUs are built for. Processing 128 examples together is far faster than 128 separate passes.
2. **Gradient quality.** A single example gives a noisy, jumpy gradient. A batch averages over many examples, giving a steadier estimate — but not so smooth that it loses the helpful noise that lets SGD escape bad spots.
3. **Memory.** The whole dataset often won't fit in memory (or GPU memory) at once. A batch does.

Batch size is a real knob. Larger batches give smoother gradients and use hardware better, but each step costs more and very large batches can generalize slightly worse. Smaller batches are noisier but sometimes find better solutions. Start around 32–128 and adjust.

```python
def get_batches(X, y, batch_size):
    n = len(X)
    idx = np.random.permutation(n)       # shuffle every epoch!
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
```

Note the shuffle. Reshuffling the data each epoch stops the network from learning the *order* of examples and keeps successive batches varied. Skipping the shuffle is a classic silent bug — training still runs, it just learns worse.

## The learning rate, revisited

The learning rate controls how big each update step is, and it interacts with everything. Too high and the loss spikes or oscillates wildly; too low and training crawls or stalls in a mediocre spot. Because the ideal step size often changes over training — big steps early to make fast progress, small steps late to settle precisely — people use a **learning rate schedule** that shrinks `lr` as training proceeds:

```python
# Simple step decay: cut the learning rate by 10x every 10 epochs
def lr_schedule(base_lr, epoch):
    return base_lr * (0.1 ** (epoch // 10))
```

Other common schedules decay smoothly (cosine) or drop when progress stalls. For a first network, a constant learning rate with Adam is fine; reach for a schedule when you're squeezing out the last bit of performance.

## Watch the training curve

The single most useful habit is to log the loss and watch it over time. Track loss on the **training set** and, crucially, on a held-out **validation set** the network never trains on:

```python
for epoch in range(num_epochs):
    train_loss = run_epoch(X_train, y_train, params, train=True)
    val_loss   = run_epoch(X_val, y_val, params, train=False)
    print(f"epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}")
```

Reading these two curves together tells you almost everything about how training is going:

- **Both falling, close together:** healthy learning. Keep going.
- **Both flat and high:** underfitting — the model isn't learning. Train longer, raise the learning rate, or use a bigger network.
- **Train falling, validation rising:** overfitting — the model is memorizing training data instead of learning general patterns. Time for regularization (next module) or early stopping.
- **Loss spiking to NaN:** the learning rate is too high, or gradients are exploding. Lower `lr`.

The validation curve is what you actually care about — it estimates how the model will do on data it hasn't seen. Training loss can always be driven to near-zero by memorizing; that's not the goal.

## A note on data splits

Always split your data *before* you start: a **training set** to learn from, a **validation set** to tune and watch, and ideally a **test set** you touch only once at the very end to get an honest final number. If you make decisions based on the test set, it stops being an honest estimate. Keep it sealed.

## Key takeaways

- Training is the four-step loop — forward, loss, backprop, update — run repeatedly over batches of data.
- **Epoch** = one pass over all data; **batch** = a subset processed together; **iteration** = one weight update.
- **Mini-batches** win on efficiency, gradient quality, and memory; batch size (start 32–128) is a real tuning knob.
- **Shuffle** the data each epoch — forgetting to is a silent bug.
- Watch **training vs. validation loss** together to diagnose underfitting, overfitting, or an exploding learning rate.
- Split data before training and keep the **test set** sealed until the very end.

## Try it

Write a complete training loop around the two-layer network from earlier modules (2→3→1). Generate a small synthetic dataset — say, points where the label is 1 if `x1 + x2 > 0` — split it 70/15/15 into train/val/test, and train for 50 epochs with batch size 16. Log train and validation loss each epoch. Then deliberately break it two ways: (1) remove the per-epoch shuffle, and (2) set the learning rate 100x too high. Describe what each change does to your loss curves.
