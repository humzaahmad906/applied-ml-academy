# 07 — Devices and GPUs

Everything so far has run on the CPU. That is fine for learning and small experiments, but real deep learning leans on accelerators — NVIDIA GPUs (CUDA) or Apple Silicon (MPS) — that perform the massive matrix multiplications in a network far faster. PyTorch makes moving computation to these devices remarkably easy: you move your tensors and your model to a device, and the same code runs there. This lesson covers how devices work, how to move things, the rules you must follow, and when the speedup is actually worth it.

## What a device is

Every tensor lives on a **device** — a place where its data is stored and its math is executed. The default is `"cpu"`. A GPU is a separate device, referred to as `"cuda"` (NVIDIA) or `"mps"` (Apple Silicon). Operations require all their inputs to be on the same device; you cannot add a CPU tensor to a GPU tensor.

## Detecting the available device

Write code that works everywhere by picking the best available device at runtime rather than hard-coding one:

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device}")
```

This pattern is standard at the top of a training script. From here on, everything moves to `device`, and the script runs unchanged on a GPU box, a Mac, or a plain CPU.

## Moving tensors and models with .to()

The `.to(device)` method moves a tensor or model to a device. For tensors it returns a new tensor on that device; for models it moves the parameters in place.

```python
x = torch.randn(3, 4)
x = x.to(device)            # tensor now on the chosen device
print(x.device)

model = torch.nn.Linear(4, 2)
model = model.to(device)    # moves all parameters and buffers
```

You can also create a tensor directly on a device, which avoids a copy:

```python
y = torch.zeros(3, 4, device=device)
```

For a model, `model.to(device)` moves it in place, so `model = model.to(device)` and `model.to(device)` are equivalent — but tensors are not moved in place, so you must reassign: `x = x.to(device)`.

## The golden rule: model and data on the same device

The one rule you cannot break: the model and the data you feed it must be on the same device. Inside a training loop this means moving each batch to the device before the forward pass.

```python
model = model.to(device)

for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()
    preds = model(batch_x)
    loss = loss_fn(preds, batch_y)
    loss.backward()
    optimizer.step()
```

If you forget to move a batch, you get a runtime error like "Expected all tensors to be on the same device." When you see that, check that both the model and every input tensor went through `.to(device)`. Note that you move the model once, before the loop, but you move each batch inside the loop because the loader produces fresh CPU tensors every iteration.

## Getting results back to the CPU

Numbers you want to log, plot, or convert to NumPy must come back to the CPU first. A GPU tensor cannot be handed to NumPy directly.

```python
loss_value = loss.item()               # .item() works from any device
preds_cpu = preds.detach().cpu()       # move back before .numpy()
preds_np = preds_cpu.numpy()
```

`.item()` on a scalar works regardless of device. For larger tensors, call `.cpu()` (and `.detach()` if it is still attached to the graph) before `.numpy()`.

## Loading checkpoints across devices

A checkpoint saved on a GPU stores its tensors' device. When loading somewhere without that GPU, redirect with `map_location`, then move the model to the local device:

```python
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state)
model = model.to(device)
```

## When a GPU actually helps

A GPU is not automatically faster. Moving data to and from it has overhead, and for tiny models or tiny batches that overhead can outweigh the compute savings — a small model may even run slower on a GPU than on a CPU.

GPUs win when there is enough parallel arithmetic to keep them busy:

- **Large models** with many parameters and big matrix multiplications.
- **Large batches**, so each transfer does a lot of work.
- **Convolutions and transformers**, which are dominated by the dense linear algebra GPUs excel at.

Practical tips: move data as few times as possible (transfers are the usual bottleneck), use a reasonably large batch size to keep the device saturated, and set `pin_memory=True` in your `DataLoader` when training on CUDA to speed host-to-device copies. If your GPU utilization is low, the culprit is usually data loading, not the model — raise `num_workers` before blaming the hardware.

Half precision can further speed up large models on capable hardware. On modern NVIDIA GPUs, mixed precision with `torch.autocast` computes many operations in `bfloat16` or `float16` while keeping master weights in `float32`, cutting memory use and boosting throughput. It is a next step once the basics here are comfortable.

## Key takeaways

- Every tensor and model has a **device**; the default is CPU, accelerators are `"cuda"` or `"mps"`.
- Detect the device at runtime so the same script runs anywhere.
- Move with `.to(device)` — reassign for tensors (`x = x.to(device)`), in place for models.
- The golden rule: **model and data must be on the same device**; move each batch inside the loop.
- Bring tensors back with `.cpu()` before converting to NumPy; use `map_location` when loading checkpoints across devices.
- GPUs help most with large models and batches — for tiny workloads the transfer overhead can make them slower.

## Try it

Make a training loop device-agnostic:

1. Take the classifier training loop from the earlier lesson.
2. Add the device-detection block at the top and move the model to the device once.
3. Inside the loop, move each `batch_x` and `batch_y` to the device before the forward pass.
4. Log the per-epoch loss using `.item()`, and after training move the predictions back to the CPU and convert to NumPy.
5. Confirm the script runs unchanged whether `device` resolves to CPU, CUDA, or MPS.
