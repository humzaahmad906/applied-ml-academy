# 04 — Datasets and DataLoaders

Real training data rarely fits neatly into a single tensor you can pass to a model. You have thousands or millions of examples, you want to feed them in small batches, shuffle them each epoch, and maybe load them from disk on demand. PyTorch splits this cleanly into two pieces: a `Dataset` that knows how to fetch one example, and a `DataLoader` that turns a dataset into batches. This separation keeps your data logic tidy and your training loop simple.

## The Dataset interface

A `Dataset` answers two questions: how many examples are there, and how do I get example number `i`? You implement this by subclassing `torch.utils.data.Dataset` and defining `__len__` and `__getitem__`.

```python
import torch
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, n=100):
        # a simple regression task: y = 2x + 1 with noise
        self.x = torch.randn(n, 1)
        self.y = 2 * self.x + 1 + 0.1 * torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

data = ToyDataset()
print(len(data))          # 100
features, target = data[0]
print(features.shape, target.shape)   # torch.Size([1]) torch.Size([1])
```

`__getitem__` returns a single example, typically as a tuple of `(features, label)`. It does not need to know anything about batching — that is the loader's job. This is also where you would load an image from disk, read a row from a file, or apply per-example transforms.

## The DataLoader

A `DataLoader` wraps a dataset and hands you batches. You tell it the batch size and whether to shuffle, and then you iterate over it.

```python
from torch.utils.data import DataLoader

loader = DataLoader(data, batch_size=16, shuffle=True)

for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)   # torch.Size([16, 1]) torch.Size([16, 1])
    break
```

Each iteration gives you a batch. The loader called `__getitem__` 16 times and stacked the results into tensors with a new leading batch dimension. This stacking is called **collation**, and the default collate function handles tuples of tensors automatically.

## Batching

Processing examples in batches instead of one at a time is central to deep learning. It uses hardware efficiently (a GPU can multiply many examples at once) and it gives the gradient estimate less noise than a single example while staying cheaper than the full dataset. Common batch sizes are powers of two like 32, 64, or 128, chosen to balance speed against memory.

The last batch may be smaller than `batch_size` if the dataset size does not divide evenly. If your model requires a fixed batch size, pass `drop_last=True` to discard that partial batch:

```python
loader = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)
```

## Shuffling

Setting `shuffle=True` reorders the examples at the start of every epoch. This matters: if your data is sorted (all class-0 examples, then all class-1 examples), a model trained without shuffling sees long runs of one class and learns poorly. Shuffling breaks any accidental ordering and gives more varied, representative batches.

A crucial rule: **shuffle the training set, but not the validation or test set.** Evaluation does not care about order, and keeping it fixed makes results reproducible.

```python
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
```

## Using built-in tensors quickly

If your data already sits in tensors, you do not need a custom class. `TensorDataset` wraps them for you:

```python
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(200, 4)
y = torch.randint(0, 3, (200,))     # 200 integer labels in {0, 1, 2}

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

xb, yb = next(iter(loader))
print(xb.shape, yb.shape)           # torch.Size([32, 4]) torch.Size([32])
```

## Faster loading with workers

Loading and preprocessing data can become a bottleneck, leaving an expensive GPU idle. The `num_workers` argument spawns background processes that prepare batches in parallel while the model trains on the current batch.

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # background loading processes
    pin_memory=True,      # faster host-to-GPU transfer
)
```

Start with `num_workers=0` while debugging (errors are easier to read), then raise it once things work. `pin_memory=True` speeds up moving batches to a GPU and is a good default when training on one.

## Splitting data

You almost always want separate training and validation sets. `random_split` divides a dataset for you:

```python
from torch.utils.data import random_split

n_val = int(0.2 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
```

Split before any augmentation so that transformed copies of a training example never leak into validation.

## Key takeaways

- A `Dataset` implements `__len__` and `__getitem__` to return one example at a time.
- A `DataLoader` turns a dataset into shuffled, collated batches you iterate over.
- Batching improves hardware efficiency and gradient quality; pick a size that fits memory.
- Shuffle the **training** loader, never the validation/test loader.
- Use `TensorDataset` for in-memory tensors, `num_workers`/`pin_memory` for throughput, and `random_split` to create validation sets.

## Try it

Build a classification pipeline from scratch:

1. Create `x = torch.randn(500, 8)` and integer labels `y = torch.randint(0, 4, (500,))`.
2. Wrap them in a `TensorDataset` and split into 80% train / 20% validation with `random_split`.
3. Make a training `DataLoader` with `batch_size=32, shuffle=True` and a validation loader with `shuffle=False`.
4. Iterate one epoch over the training loader, printing each batch's feature and label shapes, and confirm the total number of examples seen equals your training split size.
