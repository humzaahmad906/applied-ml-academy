# 07b — torch.compile and Mixed Precision

The previous lesson got your model onto a GPU. That is table stakes. Two techniques then decide whether you are using that GPU well or leaving most of it idle: **mixed precision**, which does the heavy math in a 16-bit format to halve memory traffic and light up the tensor cores, and **`torch.compile`**, which fuses many small operations into a few big kernels so the GPU spends its time computing instead of waiting. Every serious 2026 training run on NVIDIA hardware uses both. This lesson shows what they are, the exact current APIs, and how to combine them without breaking correctness.

## Why default training is slow

A plain PyTorch loop is usually **memory-bound**, not compute-bound. Modern GPUs can do arithmetic far faster than they can move numbers in and out of memory, so the bottleneck is bandwidth and the overhead of launching thousands of tiny kernels — one per operation — from Python. Two problems, two fixes:

- **Numbers are too big.** `float32` tensors move 4 bytes per element. Dropping to a 16-bit format halves that, and matrix multiplications in 16-bit run on dedicated **tensor cores** that are several times faster than the general-purpose FP32 units. That is mixed precision.
- **Too many small kernels.** Each elementwise op (add, multiply, activation) is a separate GPU launch with its own memory round-trip. `torch.compile` fuses chains of these into single kernels, cutting both launch overhead and memory traffic.

## Mixed precision: fp16, bf16, and autocast

Mixed precision keeps a `float32` master copy of the weights but runs most of the forward pass in a 16-bit type. There are two:

- **`float16`** — 5 exponent bits, 10 mantissa bits. Precise, but a narrow dynamic range: small gradients underflow to zero during backprop.
- **`bfloat16`** — 8 exponent bits, 7 mantissa bits. Same exponent range as `float32` (so almost nothing underflows), at the cost of precision. Requires Ampere-class hardware or newer (A100, RTX 30-series, Ada, Hopper).

`torch.autocast` is a context manager. You wrap **only the forward pass and the loss computation** in it; PyTorch then automatically runs each operation in the type best suited to it — matmuls and convolutions in 16-bit on the tensor cores, reductions and normalizations kept in `float32` for stability. The backward pass runs outside the context and inherits the right types automatically.

```python
import torch

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    preds = model(batch_x)
    loss = loss_fn(preds, batch_y)
```

Note the API: `torch.autocast(device_type=..., dtype=...)`. The old `torch.cuda.amp.autocast(...)` form is **deprecated as of PyTorch 2.4** — use the device-agnostic path above.

### GradScaler: only for fp16

`float16`'s narrow range means gradients can underflow to zero before the optimizer ever sees them. The fix is **loss scaling**: multiply the loss by a large factor before `.backward()` (pushing gradients up into the representable range), then divide it back out before the optimizer step. `torch.amp.GradScaler` does this automatically and adapts the scale factor over time.

Here is a full training step with fp16 and a scaler:

```python
import torch

scaler = torch.amp.GradScaler("cuda")   # note: torch.amp, not torch.cuda.amp

for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)

    scaler.scale(loss).backward()   # scale up, then backprop
    scaler.step(optimizer)          # unscale, then step (skips step if inf/nan)
    scaler.update()                 # adjust scale factor for next iteration
```

The three scaler calls replace the usual `loss.backward()` / `optimizer.step()`. `scaler.scale(loss).backward()` backprops the scaled loss, `scaler.step(optimizer)` unscales the gradients and applies them (silently skipping the step if it detects inf/nan), and `scaler.update()` grows or shrinks the scale for next time.

**`bfloat16` usually needs no scaler at all.** Its dynamic range matches `float32`, so gradients don't underflow and loss scaling is unnecessary. That makes the bf16 loop simpler — just autocast, then a normal backward and step:

```python
for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)

    loss.backward()
    optimizer.step()
```

### The rule: bf16 by default

On any modern NVIDIA GPU (Ampere or newer), **use `bfloat16` by default.** It is numerically robust, skips the scaler, and avoids a whole class of "loss went to NaN" debugging sessions. Reach for `float16` only under real memory pressure or on older hardware (like a T4 or a GTX card) that lacks bf16 support — and when you do, bring the `GradScaler` with you. If you ever want a switch that handles both, note that `GradScaler` is a no-op when its underlying type doesn't need scaling, so you can construct one unconditionally and it will simply do nothing for bf16.

## torch.compile: fuse the kernels

`torch.compile` traces your model, hands the trace to a backend compiler (TorchInductor by default), and produces fused, optimized GPU kernels. The interface is one line:

```python
model = model.to(device)
model = torch.compile(model)
```

That's it — the returned object is a drop-in replacement you train and call exactly as before. There is no separate build step; compilation happens lazily.

### The first-iteration cost

Compilation is **just-in-time**: the first call (or first few, if input shapes vary) is *slow* — often several seconds to a minute — because that is when the graph is captured and the kernels are built and cached. Every call after that runs the fast compiled version. So the first training step of a compiled model will look alarmingly slow; this is expected. Time your throughput after warmup, never on step one, or you will conclude compile made things worse.

### Modes

`torch.compile(model, mode=...)` trades compile time for runtime speed:

- **`"default"`** — balanced; fast to compile, solid speedups. Start here.
- **`"reduce-overhead"`** — uses CUDA graphs to cut per-launch Python overhead. Best for small models or small batches where launch overhead dominates; costs a little extra memory.
- **`"max-autotune"`** — benchmarks many kernel variants and picks the fastest. Longest compile time, best steady-state performance. Worth it for long runs where a slow compile amortizes over thousands of steps.

```python
model = torch.compile(model, mode="max-autotune")
```

### Graph breaks

TorchDynamo captures your `forward` into a graph. When it hits something it can't trace — a data-dependent Python branch, a `.item()` call, printing a tensor, an unsupported library call — it inserts a **graph break**: it compiles the parts it can, runs the rest in normal eager mode, and stitches them together. Your code still produces correct results, but each break is a seam where fusion stops, so too many breaks quietly erase the speedup. The failure mode is silent: no error, just a model that isn't much faster than uncompiled. To find them, set the environment variable `TORCH_LOGS="graph_breaks"` (or call `torch._dynamo.explain(model)(example_input)`) and it will print where and why each break happened. Removing breaks in the hot path — usually by lifting Python control flow out of `forward` — is where most compile tuning happens.

## Combining compile and mixed precision

They are orthogonal and compose cleanly. Compile the model once, and keep the autocast context around the forward pass exactly as before:

```python
model = torch.compile(model.to(device))
scaler = torch.amp.GradScaler("cuda")   # only meaningful for fp16

for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)

    loss.backward()      # bf16: no scaler; for fp16, use the scaler calls above
    optimizer.step()
```

This combination — bf16 autocast plus `torch.compile` — is the default shape of a 2026 GPU training loop.

## Apple Silicon and correctness

The `device_type` argument accepts `"mps"`, and autocast works on Apple Silicon, but support is less mature than CUDA: some ops fall back to `float32`, `float16` is the practical 16-bit type (bf16 support is partial), and there is no tensor-core speedup story like NVIDIA's. `torch.compile` on the MPS backend is likewise newer and more limited than on CUDA — expect more graph breaks and smaller gains. On a Mac, treat both as "try it and measure," not "on by default."

Whichever backend you use, **run a parity check before trusting a speedup.** Both mixed precision and compile change the numerics slightly, and compile can, in rare cases, expose a bug. Take one fixed batch, run it through the plain FP32 eager model and through your compiled/autocast model, and compare:

```python
model.eval()
with torch.no_grad():
    ref = base_model(fixed_batch)                       # fp32 eager
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fast = compiled_model(fixed_batch)
    max_diff = (ref - fast.float()).abs().max().item()
    print(f"max abs diff: {max_diff:.4e}")
```

A small difference (roughly `1e-2` to `1e-3` for bf16, tighter for fp16) is expected and fine. A large difference or a NaN means something is wrong — investigate before you kick off a long run.

## Key takeaways

- Default training is **memory-bound**; mixed precision halves memory traffic and hits the tensor cores, `torch.compile` fuses kernels to cut launch overhead.
- Wrap **only the forward pass and loss** in `torch.autocast(device_type="cuda", dtype=...)`. Import from `torch.amp` — `torch.cuda.amp` is deprecated since PyTorch 2.4.
- **`bfloat16` by default** on Ampere-or-newer GPUs: robust dynamic range, no scaler needed. Use `float16` only under memory pressure or on older hardware, and then add `torch.amp.GradScaler`.
- With fp16, use `scaler.scale(loss).backward()` / `scaler.step(optimizer)` / `scaler.update()` to prevent gradient underflow.
- `torch.compile(model)` is a one-line wrap. The **first iteration is slow** (JIT compile); measure throughput after warmup. Modes: `default`, `reduce-overhead`, `max-autotune`.
- **Graph breaks** silently reduce speedups; find them with `TORCH_LOGS="graph_breaks"`.
- MPS support is thinner than CUDA. Always run a **fixed-batch parity check** against FP32 eager before trusting the fast path.

## Try it

Upgrade the device-agnostic training loop from the previous lesson:

1. Add a `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` context around the forward pass and loss, keeping `backward()` and `step()` outside it.
2. Wrap the model in `torch.compile(model)`. Time the first training step and the tenth — confirm the first is much slower and the rest are faster.
3. Switch the dtype to `torch.float16` and add a `torch.amp.GradScaler("cuda")` with the three scaler calls. Confirm training is still stable.
4. Run `TORCH_LOGS="graph_breaks" python train.py` and check whether your model has any graph breaks in the forward pass.
5. Do a fixed-batch parity check: compare the FP32 eager output to the bf16 compiled output and confirm the max absolute difference is small.
