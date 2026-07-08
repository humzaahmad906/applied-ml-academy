# 04b — Real Datasets, Schedulers, and Safe Loading

Lesson 04 built datasets out of tensors you generated yourself. That is the right way to learn the `Dataset`/`DataLoader` split, but real projects start from data on disk: image folders, downloaded benchmarks, files someone else produced. This lesson closes three gaps. First, it wires a genuine dataset — MNIST from `torchvision` — through transforms into a `DataLoader`, so the pipeline from lesson 04 finally has real pixels flowing through it. Second, it introduces **learning-rate schedulers**, the objects that change the learning rate as training progresses and that slot into the loop from lesson 05. Third, it covers **safe checkpoint loading**, a security issue that lesson 06 touches and that recent PyTorch releases changed the defaults around.

## A real dataset: torchvision

`torchvision.datasets` ships ready-made wrappers for common benchmarks (MNIST, CIFAR-10, FashionMNIST, and more). Each one is an ordinary `Dataset` — it implements `__len__` and `__getitem__` exactly as your `ToyDataset` did — so everything you learned in lesson 04 applies unchanged. The difference is that these classes know how to download the files, unpack them, and hand you one image plus its label at a time.

```python
from torchvision import datasets

train_data = datasets.MNIST(
    root="./data",   # where files live (downloaded here if missing)
    train=True,      # the training split; train=False gives the test split
    download=True,
)

print(len(train_data))   # output: 60000
img, label = train_data[0]
print(type(img), label)  # output: <class 'PIL.Image.Image'> 5
```

Notice the problem: `train_data[0]` returns a **PIL image**, not a tensor. A model built from `nn.Module` layers cannot consume a PIL image. This is exactly what transforms are for.

## Transforms: from images to model-ready tensors

A transform is a function applied to each example inside `__getitem__`. You pass it once when you build the dataset, and every fetched image runs through it. Two transforms cover most first pipelines: `ToTensor`, which converts a PIL image (pixels 0–255) to a float tensor scaled to `[0, 1]` with shape `(channels, height, width)`; and `Normalize`, which shifts and scales each channel to have roughly zero mean and unit variance. You chain them with `Compose`.

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                     # PIL image -> float tensor in [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # (mean,) and (std,) for MNIST's one channel
])

train_data = datasets.MNIST(root="./data", train=True, download=True,
                            transform=transform)

img, label = train_data[0]
print(img.shape, img.dtype)          # output: torch.Size([1, 28, 28]) torch.float32
print(round(img.mean().item(), 3))   # output: ~0.0  (roughly centered)
```

The magic numbers `0.1307` and `0.3081` are MNIST's precomputed pixel mean and standard deviation. Normalizing to zero-centered inputs helps optimization: gradients behave better when features share a scale. For a three-channel dataset like CIFAR-10 you pass three means and three stds, one per RGB channel. The pattern is identical — swap `datasets.MNIST` for `datasets.CIFAR10` and give `Normalize` three-tuples.

## Wiring it into a DataLoader

Because a `torchvision` dataset is just a `Dataset`, the loader code is exactly what you wrote in lesson 04. Nothing new to learn here — that is the payoff of the clean interface.

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)   # output: torch.Size([64, 1, 28, 28]) torch.Size([64])
```

The loader stacked 64 transformed images into a single `(64, 1, 28, 28)` batch, ready to feed a model. Keep `shuffle=True` for training and build a separate `train=False` loader with `shuffle=False` for evaluation, following the rule from lesson 04.

## Learning-rate schedulers

In lesson 05 you set one learning rate on the optimizer and left it fixed for all of training. That works, but a constant rate is a compromise: large enough to make early progress is often too large to settle into a good minimum later. A **scheduler** solves this by changing the learning rate over time — usually large early, smaller near the end.

A scheduler is an object that wraps your optimizer. Each time you call `scheduler.step()`, it recomputes the learning rate and writes it into the optimizer. PyTorch offers several policies in `torch.optim.lr_scheduler`:

- **`StepLR`** drops the rate by a fixed factor every `step_size` epochs. Simple and predictable.
- **`CosineAnnealingLR`** decreases the rate smoothly along a cosine curve from its start value down to near zero over `T_max` steps. A strong, popular default.
- **`OneCycleLR`** ramps the rate *up* to a peak in the first part of training, then back down — the "1cycle" policy, which often trains faster.

```python
import torch
import torch.nn as nn

model = nn.Linear(784, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(15):
    # ... training happens here ...
    scheduler.step()                        # advance the schedule once per epoch
    lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch}: lr = {lr}")
# output: lr starts at 0.1, halves to 0.05 at epoch 5, 0.025 at epoch 10, ...
```

### Where step() goes in the loop

This is the detail people get wrong. Most schedulers — `StepLR`, `CosineAnnealingLR` — are **per-epoch**: call `scheduler.step()` once, after the training pass over the whole dataset. It goes *after* `optimizer.step()`, never inside the batch loop.

```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()        # update weights (per batch)
    scheduler.step()            # update the learning rate (per epoch)
```

`OneCycleLR` is the exception: it is designed to step **once per batch**, so its `scheduler.step()` goes *inside* the inner loop right after `optimizer.step()`. It also needs to know the total number of steps up front:

```python
steps = len(train_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=steps)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()        # OneCycleLR steps every batch
```

**Warmup** — starting at a tiny rate and rising over the first few hundred steps before the main schedule takes over — stabilizes training for large models and adaptive optimizers like AdamW. PyTorch has no single "warmup" class; you build it with `LinearLR` for the ramp and chain it into a decay schedule using `SequentialLR`. That composition is worth knowing exists, but a plain `CosineAnnealingLR` is a fine starting point for your first models.

## Safe checkpoint loading

Lesson 06 showed you `torch.save` and `torch.load` for state dicts. There is a security wrinkle that changed recently and that you must understand before loading any file you did not create yourself.

Under the hood, `torch.save` uses Python's `pickle`, and unpickling can **execute arbitrary code**. A malicious checkpoint downloaded from the internet can run commands on your machine the moment you load it — a genuine remote-code-execution risk, not a theoretical one (see CVE-2025-32434). To limit this, `torch.load` has a `weights_only` argument. When `True`, it refuses to run arbitrary code and only reconstructs tensors and a small allowlist of safe types.

The important change: **as of PyTorch 2.6, `weights_only=True` is the default.** Older code and tutorials assumed `False`, so you will meet checkpoints that fail to load under the new default, and warnings nudging you to set it explicitly.

```python
# Safe: only tensors are reconstructed, no code runs. The 2.6+ default.
state = torch.load("model.pt", weights_only=True)
model.load_state_dict(state)

# Unsafe: runs pickle. Only ever do this for files YOU produced and trust.
state = torch.load("model.pt", weights_only=False)
```

Keep `weights_only=True` for anything you download. If a trusted checkpoint of yours needs a custom class to load, allowlist it explicitly with `torch.serialization.add_safe_globals([...])` rather than falling back to `weights_only=False`.

### safetensors: the safer format

The cleaner fix is to not use pickle at all. **safetensors** stores only a JSON header plus raw tensor bytes — there is no code to execute, so an untrusted file cannot attack you by construction. It is also fast, thanks to zero-copy loading, and has become the default weight format across the Hugging Face ecosystem.

```python
from safetensors.torch import save_file, load_file

save_file(model.state_dict(), "model.safetensors")

state = load_file("model.safetensors")   # no pickle, no code execution
model.load_state_dict(state)
```

One caveat: safetensors stores **tensors only**. Optimizer state, epoch counters, and Python objects still need `torch.save`, so a full training checkpoint often mixes both — weights in safetensors, the rest in a small pickle you control.

## Key takeaways

- `torchvision.datasets` are ordinary `Dataset` objects, so the lesson-04 `DataLoader` code works unchanged.
- Apply `transforms.ToTensor()` then `transforms.Normalize(mean, std)` via `Compose` to turn PIL images into zero-centered model-ready tensors.
- A scheduler wraps the optimizer and changes the learning rate over time; large early, small late is the usual shape.
- Call `scheduler.step()` **once per epoch** for `StepLR`/`CosineAnnealingLR`, but **once per batch** for `OneCycleLR`.
- `torch.load` defaults to `weights_only=True` since PyTorch 2.6 because pickle can execute code — keep it on for any file you did not create.
- Prefer **safetensors** for distributing weights: no code execution, fast loading, and the ecosystem default.

## Try it

Build a real end-to-end setup:

1. Load MNIST with a `Compose` of `ToTensor` and `Normalize((0.1307,), (0.3081,))`, and wrap it in a `DataLoader` with `batch_size=64, shuffle=True`. Print the shape of one batch and confirm it is `(64, 1, 28, 28)`.
2. Create an `SGD` optimizer at `lr=0.1` and a `CosineAnnealingLR` with `T_max=10`. Loop for 10 epochs calling only `scheduler.step()` each epoch, and print `scheduler.get_last_lr()[0]` — watch it curve down toward zero.
3. Save a model's `state_dict` twice: once with `torch.save` and once with `safetensors.torch.save_file`. Load each back (`weights_only=True` for the torch file) and confirm the reloaded weights match the originals.
