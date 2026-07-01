# 06 — Saving and Loading

Training a model can take minutes or days. Once you have a trained model you want to save it to disk so you can reuse it later — for inference, for sharing, or for resuming training. PyTorch makes this straightforward, but there is a right way and a fragile way. This lesson covers the recommended approach based on `state_dict`, how to build checkpoints that let you resume training, and how to load a model correctly for inference.

## What is a state_dict?

A model's learnable state lives in its parameters and a few buffers (like batchnorm running statistics). A **state_dict** is a plain Python dictionary mapping each parameter's name to its tensor. It is the canonical, portable representation of a model's learned values.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

for name, tensor in model.state_dict().items():
    print(name, tuple(tensor.shape))
# 0.weight (8, 4)
# 0.bias (8,)
# 2.weight (2, 8)
# 2.bias (2,)
```

The state_dict contains only numbers and names — no Python code, no class definitions.

## Saving and loading the state_dict

The recommended pattern is to save the state_dict, not the whole model object. To reload, you first recreate the model architecture in code, then load the saved weights into it.

```python
# save
torch.save(model.state_dict(), "model.pt")

# load — recreate the architecture first, then fill in the weights
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

Why not just `torch.save(model, ...)` the entire object? That approach pickles the model class and file paths along with the weights, so it breaks if you rename or move your code. Saving the state_dict decouples the weights from the code and is far more robust. The cost is that you must have the model class available to recreate the architecture — which you always do, since it is your own code.

`load_state_dict` checks that the keys and shapes match your model. If they do not, it raises an error naming the mismatched keys. Do not silence these errors with `strict=False` unless you genuinely intend to load a partial set of weights and know exactly which keys are missing.

## Checkpoints for resuming training

To resume training later, the weights alone are not enough. Optimizers like Adam keep internal state (momentum, variance estimates), and you want to remember which epoch you stopped at. A **checkpoint** is a dictionary bundling everything needed to continue.

```python
checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "loss": loss.item(),
}
torch.save(checkpoint, "checkpoint.pt")
```

To resume, recreate the model and optimizer, then restore each piece:

```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"] + 1

model.train()      # continue training from start_epoch
```

Restoring the optimizer state matters: if you skip it, Adam's momentum resets to zero and your loss may spike at resume. During long training runs it is common to save a checkpoint every few epochs, and to keep a separate "best" checkpoint whenever validation loss improves.

## Loading for inference

When you only want to make predictions, do two things after loading: put the model in eval mode and disable gradient tracking.

```python
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
model.load_state_dict(torch.load("model.pt"))
model.eval()

x = torch.randn(1, 4)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    prediction = probs.argmax(dim=1)
print(prediction)
```

`model.eval()` switches dropout off and makes batchnorm use its stored running statistics, so predictions are deterministic. `torch.no_grad()` (or `torch.inference_mode()`) skips building the autograd graph, saving memory and time. Forgetting `eval()` is a classic mistake — a model with dropout will give slightly different, noisier answers each call if left in train mode.

## Loading onto a different device

If you trained on a GPU but want to load on a CPU-only machine, tell `torch.load` where to place the tensors with `map_location`:

```python
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
```

This avoids errors from a checkpoint that references a GPU that is not present. You will see more about devices in the next lesson.

## A note on file format

There is no required extension; `.pt` and `.pth` are conventional. The saved file is a serialized dictionary of tensors. Keep your model-definition code alongside your saved weights — the weights are meaningless without the architecture that gives them shape.

## Key takeaways

- Save the **state_dict** (`torch.save(model.state_dict(), path)`), not the whole model object — it is portable and robust to code changes.
- To load, recreate the architecture, then call `model.load_state_dict(torch.load(path))`.
- For resuming training, save a **checkpoint** bundling model state, optimizer state, and the epoch.
- For inference, call `model.eval()` and wrap prediction in `torch.no_grad()`.
- Use `map_location` to load a checkpoint onto a different device than it was saved from.

## Try it

Practice the full save/load cycle:

1. Define and briefly train any small model on synthetic data (reuse the training loop from the previous lesson).
2. Save a checkpoint dictionary containing the model state, optimizer state, and final epoch.
3. In a fresh model and optimizer, load the checkpoint and confirm training resumes without a loss spike.
4. Separately, save just the state_dict, reload it into a new model, switch to `eval()`, run one input through `torch.no_grad()`, and verify the output matches the original model's output on the same input.
