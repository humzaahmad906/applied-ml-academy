# 05 — The Training Loop

This is the lesson everything has been building toward. Tensors hold your data, autograd computes gradients, `nn.Module` defines your model, and `DataLoader` feeds it batches. The training loop is where these pieces come together into the process that actually learns. It is a short, repetitive pattern — once you know it, you can read almost any PyTorch training script.

## The five steps

Every training iteration performs the same five steps on a batch of data:

1. **Forward** — run the batch through the model to get predictions.
2. **Loss** — measure how wrong the predictions are.
3. **Zero grad** — clear gradients left over from the last iteration.
4. **Backward** — compute gradients of the loss with respect to every parameter.
5. **Step** — nudge each parameter downhill using its gradient.

In code, the heart of the loop is just five lines:

```python
optimizer.zero_grad()          # 3
preds = model(batch_x)         # 1
loss = loss_fn(preds, batch_y) # 2
loss.backward()                # 4
optimizer.step()               # 5
```

The ordering flexes slightly — you can compute the forward pass before or after zeroing — but `zero_grad` must come before `backward`, and `backward` must come before `step`. Everything else is loops and logging.

## Loss functions and optimizers

A **loss function** turns predictions and targets into a single scalar to minimize. PyTorch provides the common ones in `torch.nn`. For regression use `nn.MSELoss`; for multi-class classification use `nn.CrossEntropyLoss`, which expects raw logits (not softmaxed) and integer class labels.

An **optimizer** reads the gradients autograd computed and updates the parameters. You give it `model.parameters()` and a learning rate. `torch.optim.Adam` is a strong default; `torch.optim.SGD` is the classic choice.

```python
import torch.nn as nn
import torch.optim as optim

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

The optimizer holds references to the same parameter tensors as the model, so when `optimizer.step()` runs, it modifies the model in place.

## A complete training loop

Here is a full, runnable example that trains a small regression model. It ties together the dataset, loader, model, loss, and optimizer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# synthetic data: y = 3x + 2 with noise
x = torch.randn(1000, 1)
y = 3 * x + 2 + 0.1 * torch.randn(1000, 1)
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    print(f"epoch {epoch:2d}  loss {epoch_loss:.4f}")
```

Run this and you will watch the loss fall each epoch as the model learns. Two details worth noting: `loss.item()` pulls the scalar value out as a plain Python float (calling it detaches from the graph, so it will not leak memory), and multiplying by the batch size then dividing by the dataset size gives a correct average even when the last batch is smaller.

## Epochs and batches

One pass over the entire dataset is an **epoch**. Within an epoch you iterate over batches. So the loop is nested: an outer loop over epochs and an inner loop over the batches a `DataLoader` produces. Parameters update once per batch, so a single epoch produces many updates.

## Why zero_grad is not optional

Recall from the autograd lesson that gradients **accumulate**. If you forget `optimizer.zero_grad()`, each `backward()` adds to the gradients from previous batches, and your updates use a corrupted, ever-growing gradient. The symptom is a loss that behaves erratically or blows up. This is the single most common training bug, so make zeroing a reflex.

## Adding a validation loop

Training loss alone can mislead — a model can memorize the training set while failing on new data. After each epoch, evaluate on held-out data with gradients turned off and the model in eval mode.

```python
model.eval()
val_loss = 0.0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = model(batch_x)
        val_loss += loss_fn(preds, batch_y).item() * batch_x.size(0)
val_loss /= len(val_loader.dataset)
print(f"val loss {val_loss:.4f}")
```

Two things change for evaluation: `model.eval()` puts dropout and batchnorm into inference behavior, and `torch.no_grad()` skips graph building for speed and memory. Remember to switch back with `model.train()` before the next training epoch.

## A classification variant

For classification, swap the loss and read off accuracy. `nn.CrossEntropyLoss` takes logits of shape `(batch, num_classes)` and integer labels of shape `(batch,)`.

```python
loss_fn = nn.CrossEntropyLoss()

logits = model(batch_x)               # shape (batch, num_classes)
loss = loss_fn(logits, batch_y)       # batch_y is int64 class indices

preds = logits.argmax(dim=1)          # predicted class per example
accuracy = (preds == batch_y).float().mean()
```

Do not apply a softmax before `CrossEntropyLoss` — it applies log-softmax internally, and doing it twice hurts training.

## Key takeaways

- Every iteration is five steps: forward, loss, zero grad, backward, step.
- Pick a loss (`MSELoss` for regression, `CrossEntropyLoss` for classification) and an optimizer (`Adam` is a solid default) built from `model.parameters()`.
- The loop is nested: epochs on the outside, batches on the inside; parameters update once per batch.
- Always `optimizer.zero_grad()` before `backward()` — gradients accumulate otherwise.
- Evaluate with `model.eval()` inside `torch.no_grad()`, then return to `model.train()`.

## Try it

Train a classifier end to end:

1. Generate `x = torch.randn(800, 4)` and labels `y = (x.sum(dim=1) > 0).long()` (a binary task).
2. Build a `TensorDataset` and `DataLoader`, and a 3-layer MLP ending in 2 output units.
3. Use `nn.CrossEntropyLoss` and `optim.Adam`, and train for 20 epochs, printing the loss each epoch.
4. After training, compute and print the accuracy on the full dataset using `argmax`. Aim to get well above 90%.
