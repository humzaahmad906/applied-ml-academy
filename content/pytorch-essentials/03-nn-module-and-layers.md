# 03 — nn.Module and Layers

You could build a neural network purely out of raw tensors and manual matrix multiplications, but you would spend all your time tracking parameters by hand. PyTorch gives you `torch.nn`, a library of ready-made building blocks. The central idea is `nn.Module`: a base class that holds parameters, organizes layers, and defines how data flows through the network. Every model you build, and every layer inside it, is an `nn.Module`.

## Layers are modules

A **layer** is a small module that transforms an input tensor into an output tensor and owns some learnable parameters. The most common is `nn.Linear`, which applies `y = x @ W.T + b`.

```python
import torch
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)
x = torch.randn(3, 4)          # batch of 3 examples, 4 features each
out = layer(x)
print(out.shape)               # torch.Size([3, 2])
```

Calling `layer(x)` runs the layer's forward computation. The layer created a weight matrix of shape `(2, 4)` and a bias of shape `(2,)` for you, both with `requires_grad=True` so autograd tracks them.

```python
for name, p in layer.named_parameters():
    print(name, p.shape)
# weight torch.Size([2, 4])
# bias torch.Size([2])
```

PyTorch ships many layer types: `nn.Conv2d` for images, `nn.Embedding` for tokens, `nn.LayerNorm` and `nn.BatchNorm1d` for normalization, `nn.Dropout` for regularization, and activation modules like `nn.ReLU` and `nn.Sigmoid`.

## Building a model

To make a full model, subclass `nn.Module`. You do two things: define the layers in `__init__`, and describe the data flow in `forward`.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()                     # always call this first
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

model = MLP(in_dim=10, hidden_dim=32, out_dim=3)
```

Two rules matter here. First, always call `super().__init__()` before assigning layers — this is what wires up the parameter bookkeeping. Second, assign layers as attributes (`self.fc1 = ...`). When you do, `nn.Module` automatically registers their parameters so they show up in `model.parameters()` and move with the model to a GPU.

## Calling forward — never call it directly

To run the model, call the model object itself, not `model.forward(x)`:

```python
x = torch.randn(5, 10)         # batch of 5
logits = model(x)              # calls forward under the hood
print(logits.shape)            # torch.Size([5, 3])
```

`model(x)` invokes `__call__`, which runs some setup (like hooks and train/eval bookkeeping) and then calls your `forward`. Skipping it by calling `forward` directly bypasses that machinery, so always use `model(x)`.

Notice the model naturally handles a **batch**: the first dimension is the batch size, and every layer processes all examples at once.

## Inspecting parameters

`model.parameters()` returns every learnable tensor in the model — this is exactly what you hand to an optimizer later.

```python
total = sum(p.numel() for p in model.parameters())
print(f"{total} parameters")

for name, p in model.named_parameters():
    print(name, tuple(p.shape))
# fc1.weight (32, 10)
# fc1.bias (32,)
# fc2.weight (3, 32)
# fc2.bias (3,)
```

Printing the model gives a readable summary of its structure:

```python
print(model)
# MLP(
#   (fc1): Linear(in_features=10, out_features=32, bias=True)
#   (act): ReLU()
#   (fc2): Linear(in_features=32, out_features=3, bias=True)
# )
```

## nn.Sequential for simple stacks

When your model is just a straight chain of layers with no branching, `nn.Sequential` saves you from writing a class:

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)
out = model(torch.randn(5, 10))    # shape (5, 3)
```

Reach for a custom `nn.Module` subclass when you need control flow, multiple inputs or outputs, skip connections, or anything beyond a single straight line.

## Nesting modules

Because layers and models are both modules, you can nest them freely. A block can be a module, and a bigger model can hold several blocks. Parameters register recursively no matter how deep the nesting.

```python
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.linear(x))

class Net(nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

net = Net(dim=16, n_blocks=3)
print(sum(p.numel() for p in net.parameters()), "parameters")
```

Use `nn.ModuleList` (or `nn.ModuleDict`) rather than a plain Python list when storing sub-modules — a plain list will not register the parameters and they will silently be left out of training.

## train and eval modes

Some layers behave differently during training versus inference — `nn.Dropout` drops activations only during training, and `nn.BatchNorm` updates running statistics only during training. Switch modes with `model.train()` and `model.eval()`. You will call these in the training loop.

```python
model.train()    # dropout active, batchnorm updating
model.eval()     # deterministic inference behavior
```

## Key takeaways

- Everything — layers and full models — is an `nn.Module`.
- Define layers in `__init__` (after `super().__init__()`) and the data flow in `forward`.
- Run the model with `model(x)`, never `model.forward(x)`.
- Assigning layers as attributes (or using `nn.ModuleList`/`Sequential`) auto-registers their parameters into `model.parameters()`.
- Use `model.train()` and `model.eval()` to toggle behavior of dropout and normalization layers.

## Try it

Build a small classifier for 28×28 grayscale images flattened to 784 features, with 10 output classes:

1. Subclass `nn.Module` with two `nn.Linear` layers (784 → 128, then 128 → 10) and a `nn.ReLU` between them.
2. Create a fake batch `x = torch.randn(16, 784)` and run it through the model.
3. Confirm the output shape is `(16, 10)`.
4. Print the total number of parameters, then rewrite the same model using `nn.Sequential` and confirm the parameter count matches.
