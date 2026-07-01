# 02 — Autograd

Training a neural network means adjusting its parameters to reduce a loss, and doing that requires gradients: how much the loss changes when each parameter changes. Computing these by hand for a deep model would be hopeless. PyTorch's **autograd** engine does it for you automatically. This lesson explains how autograd tracks operations, how to trigger gradient computation with `backward()`, and how to turn tracking off when you do not need it.

## The computation graph

When you perform operations on tensors that require gradients, PyTorch quietly builds a **computation graph** behind the scenes. Each tensor remembers the operation that produced it and the inputs to that operation. When you later call `backward()`, autograd walks this graph in reverse, applying the chain rule to compute a gradient for every tensor that requested one.

You opt a tensor into this tracking with `requires_grad=True`.

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1        # y = x^2 + 2x + 1
print(y)                       # tensor(16., grad_fn=<AddBackward0>)
```

Notice the `grad_fn` in the output — that is the graph node recording how `y` was built. Tensors created directly by you are called **leaf** tensors; intermediate results carry a `grad_fn`.

## backward() and grad

Calling `backward()` on a scalar computes the derivative of that scalar with respect to every leaf tensor with `requires_grad=True`. The result lands in each tensor's `.grad` attribute.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)     # tensor(8.)
```

The derivative of `x^2 + 2x + 1` is `2x + 2`, which at `x = 3` equals `8`. Autograd matched the calculus exactly, and you never wrote a derivative.

`backward()` must be called on a scalar (a single number). Loss values are scalars, so this is the normal case in training. If you have a non-scalar output, you either reduce it to a scalar first (for example with `.sum()` or `.mean()`) or pass a gradient argument.

Gradients also flow through multiple inputs:

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
loss = a * b + b ** 2
loss.backward()
print(a.grad)     # tensor(4.)   d(loss)/da = b = 4
print(b.grad)     # tensor(10.)  d(loss)/db = a + 2b = 2 + 8 = 10
```

## Gradients accumulate

A subtle but critical detail: `.grad` **accumulates**. Every call to `backward()` adds to the existing gradient rather than replacing it. If you compute gradients in a loop without clearing them, they pile up and your updates go wrong.

```python
x = torch.tensor(1.0, requires_grad=True)

for _ in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)     # 2., then 4., then 6. — accumulating!
```

That is why training loops call `optimizer.zero_grad()` (or `x.grad = None`) before each backward pass. It is not optional; forgetting it is one of the most common PyTorch bugs. You will see this in the training loop lesson.

```python
x = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    if x.grad is not None:
        x.grad.zero_()
    y = x ** 2
    y.backward()
    print(x.grad)     # 2. every time — correct
```

## Turning tracking off with no_grad

Tracking operations costs memory and time. During inference — when you only want predictions and never call `backward()` — you should disable it with `torch.no_grad()`. This makes the code faster and prevents accidental graph building.

```python
x = torch.tensor(3.0, requires_grad=True)

with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)    # False — no graph was built
```

You will wrap evaluation and prediction code in `with torch.no_grad():`. There is also `torch.inference_mode()`, an even more optimized variant intended purely for inference; you can treat it as a stronger `no_grad`.

To permanently stop tracking a tensor — for example to use a trained parameter as fixed data — use `.detach()`, which returns a copy sharing the same values but cut off from the graph:

```python
p = torch.tensor([1.0, 2.0], requires_grad=True)
frozen = p.detach()
print(frozen.requires_grad)    # False
```

## Why this matters for training

Every parameter in a PyTorch model is a tensor with `requires_grad=True`. When you compute a loss and call `loss.backward()`, autograd fills in `.grad` for all of them. An optimizer then reads those gradients and nudges each parameter downhill. That entire mechanism — the thing that makes learning possible — is autograd. Understanding it removes most of the mystery from the training loop you will build shortly.

A quick end-to-end taste of the pattern:

```python
w = torch.tensor(0.0, requires_grad=True)
target = 5.0
lr = 0.1

for step in range(20):
    pred = w * 2.0            # a trivial "model"
    loss = (pred - target) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad     # gradient descent step
    w.grad.zero_()

print(w.item())              # approaches 2.5, where pred == target
```

## Key takeaways

- Set `requires_grad=True` to make a tensor part of the autograd graph.
- Operations build a graph; `backward()` walks it in reverse to fill each tensor's `.grad`.
- `backward()` runs on a scalar (typically your loss).
- Gradients **accumulate** — clear them each iteration with `zero_grad()` or `grad.zero_()`.
- Use `torch.no_grad()` (or `inference_mode()`) for evaluation, and `.detach()` to cut a tensor out of the graph.

## Try it

Fit a line by hand. Create `w = torch.tensor(0.0, requires_grad=True)` and `b = torch.tensor(0.0, requires_grad=True)`. Given inputs `x = torch.tensor([1., 2., 3., 4.])` and targets `y = torch.tensor([3., 5., 7., 9.])` (which follow `y = 2x + 1`):

1. In a loop, compute `pred = w * x + b` and `loss = ((pred - y) ** 2).mean()`.
2. Call `loss.backward()`, then update `w` and `b` with gradient descent inside `torch.no_grad()`.
3. Zero both gradients each iteration.
4. After enough steps, print `w` and `b` and confirm they approach 2 and 1.
