# 05 — How GPUs Actually Execute Your Code

This chapter is the pivot from "what the model is" to "how to make it fast." You cannot optimize
training or inference without a mental model of the hardware. The single most important idea is
that modern accelerators are almost never limited by how many FLOPs they can do; they are limited
by how fast they can move data. Once you believe that, most performance work makes sense.

We make this concrete by benchmarking real code on real hardware and reading the numbers off
the roofline. The goal of this chapter is to give you the same instinct: to look at an operation,
estimate its arithmetic intensity, and predict whether it will be limited by compute or by memory
before you ever run it.

## The GPU execution model in one page

A GPU is a large number of simple arithmetic units grouped into streaming multiprocessors (SMs
on NVIDIA hardware). An A100 has 108 SMs; an H100 has 132. Work is launched as a kernel: a
function that runs across a grid of threads. Threads are grouped into blocks (up to 1024 threads
per block), blocks are scheduled onto SMs, and within a block threads run in lockstep groups
(warps of 32 on NVIDIA). The programming model is single-instruction, multiple-thread (SIMT): the
same instruction runs on many data elements at once, and if threads in a warp take different
branches they serialize (warp divergence), which is why data-dependent control flow is expensive.

A detail worth emphasizing because it bites in practice is **wave quantization**. Blocks are
scheduled onto SMs in waves. If you launch 109 blocks on a 108-SM A100, the first wave fills all
108 SMs and the last "wave" runs a single block on one SM while the other 107 sit idle. You paid
for two waves to do the work of barely more than one. The lesson: size your grid so the number of
blocks is a multiple of the SM count, and be suspicious of matrix dimensions that produce awkward
tile counts.

The memory hierarchy, fastest and smallest first:

- **Registers:** per-thread, tiny, single-cycle. This is where your working values live.
- **Shared memory / L1:** per-block, on-chip, fast, up to ~192 KB per SM on A100/H100 (split
  between L1 cache and programmer-managed shared memory). Threads in a block cooperate through it.
  This is the "factory floor" where a good kernel does its work.
- **L2 cache:** shared across the whole chip, larger (tens of MB), slower than shared memory.
- **HBM (global memory):** the multi-GB memory you think of as "the GPU's memory" — 40 or 80 GB on
  an A100, 80 GB on an H100. Large but, relative to the compute units, slow to reach. This is
  where your weights and activations live.

A useful analogy: HBM is a warehouse — huge, but far away. Shared memory and registers are
the factory floor — small, but where work actually happens. The whole game of a fast kernel is to
move data from the warehouse to the factory as few times as possible, do as much work as possible
while it is on the floor, and ship back as little as possible. Every trip to HBM is what costs you.

## Tensor cores

Beyond the general (CUDA-core) arithmetic units, modern GPUs have tensor cores: dedicated units
that do small matrix multiply-accumulates in one operation, at much higher throughput than the
general units, in reduced precision (bf16, fp16, fp8). The enormous FLOP numbers on a GPU's spec
sheet are tensor-core numbers at low precision. To get anywhere near them your work has to be
matmul-shaped and in a precision the tensor cores accept. This is a big reason low precision is
not just a memory trick: it is the only way to reach peak compute.

The gap is stark. On an A100:

- FP32 on the general CUDA cores: about 19.5 TFLOP/s.
- BF16/FP16 on tensor cores: about 312 TFLOP/s — roughly 16x higher.

On an H100 the tensor-core numbers are higher again (roughly 990 TFLOP/s bf16 dense, and more with
FP8 and sparsity). If you run your matmuls in fp32, you leave the great majority of the chip
unused. This single fact — that peak compute lives on the tensor cores at low precision — is why
mixed-precision training is universal.

## Arithmetic intensity and the roofline

Here is the model that ties it together. For any operation, define its arithmetic intensity:

```
arithmetic intensity = FLOPs performed / bytes moved to and from HBM
```

Every GPU has two ceilings: a peak compute rate (FLOP/s) and a peak memory bandwidth (bytes/s).
For an A100, peak bf16 tensor compute is ~312 TFLOP/s and peak HBM bandwidth is ~1.5–2.0 TB/s
(1.5 TB/s on the 40 GB card, ~2.0 TB/s on the 80 GB card). For an H100, ~990 TFLOP/s against
~3.35 TB/s of HBM3.

Their ratio is a break-even intensity — the FLOPs-per-byte at which the two ceilings meet:

```
break-even intensity = peak FLOP/s / peak bytes/s
```

For an A100 that is roughly 312e12 / 2.0e12 ≈ 156 FLOPs per byte. For an H100, roughly
990e12 / 3.35e12 ≈ 295 FLOPs per byte. These break-even points are high, and they have been
climbing generation over generation: compute has grown faster than bandwidth, so the bar for being
compute-bound keeps rising.

If your operation's arithmetic intensity is below break-even, you are **memory-bound**: you finish
moving the bytes before you finish the FLOPs, and the compute units sit idle waiting for data. If
it is above, you are **compute-bound**: the arithmetic units are the limit and you are using the
chip well.

Plot achievable performance against arithmetic intensity and you get the roofline: a sloped line
(the bandwidth limit, performance = intensity x bandwidth) that meets a flat line (the compute
limit) at the break-even point. Your kernel's performance sits under whichever part of the roof its
intensity puts it under. The practical use of the roofline is diagnostic: measure an op's achieved
FLOP/s, place it on the plot, and you immediately know whether to chase more compute efficiency
(you are under the flat roof) or fewer HBM bytes (you are under the sloped roof).

## Which operations are memory-bound, and why it matters

Run the intensities:

- **Large matrix multiply** (the MLP, the big attention projections): a matmul of an `M x K` by a
  `K x N` matrix does `2MNK` FLOPs and moves on the order of `MK + KN + MN` elements. As the
  matrices grow, each element loaded participates in many multiply-adds, so intensity scales with
  the shared dimension. Comfortably above break-even. Compute-bound. Good — this is what tensor
  cores are for.
- **Elementwise operations** (activation functions, RMSNorm's scale, adding the residual, dropout):
  you load each element, do one or a few operations, write it back. Intensity near 1. Two to three
  orders of magnitude below break-even. Deeply memory-bound. The GPU's compute is wasted; you are
  just streaming data through.
- **Reductions (softmax, layer statistics):** intermediate intensity, but the naive implementation
  makes several passes over the data (max, subtract, exp, sum, divide), each a separate HBM
  round-trip. Memory-bound, and wasteful in a way fusion fixes directly.
- **Attention softmax and masking:** memory-bound, and worse, the naive implementation
  materializes the full `L x L` score matrix in HBM, which for long sequences is enormous.

This is the key insight that drives the next module. A Transformer forward pass is a few
compute-bound matmuls interleaved with many memory-bound elementwise, reduction, and attention
operations. Each memory-bound op reads its input from HBM and writes its output back to HBM, and
those HBM round-trips, not the FLOPs, are where the time goes. If you can fuse a chain of
memory-bound operations into one kernel so the data makes a single trip from HBM, does all the
work on-chip, and returns once, you win big. That is what kernel fusion and FlashAttention do.

## Occupancy, latency hiding, and why bigger batches help

A GPU hides memory latency by having far more threads ready to run than it can execute at once.
When one warp stalls waiting on HBM, the scheduler instantly swaps in another warp that has its
data ready. This only works if you have given it enough resident parallel work, a quantity called
occupancy (the ratio of active warps to the maximum the SM can hold). Occupancy is limited by
resources: each block consumes registers and shared memory, so a kernel that uses a lot of either
can host fewer blocks per SM and thus expose less parallelism to hide latency behind.

This is the systems-level reason larger batch sizes and longer sequences run more efficiently per
element up to a point: they give the GPU more independent work to hide latency behind, and they
make the matmuls larger and therefore higher-intensity. It is also why tiny inference batches
(one user, one token at a time) run the GPU at a fraction of its capability — the matmuls become
skinny matrix-vector products with intensity near 1, so decoding is memory-bound on the weights.
That inefficiency is the entire motivation for inference batching, which we come to later.

## Benchmarking, correctly

Systems work is empirical, and getting a trustworthy measurement on a GPU takes care
because GPU launches are asynchronous — the CPU queues kernels and returns immediately, so a naive
`time.time()` around a call measures queueing time, not execution time. The methodology:

- **Synchronize before you stop the clock.** Call `torch.cuda.synchronize()` before recording
  start and end times so the CPU actually waits for the GPU to finish.
- **Warm up.** Run several untimed iterations first. The first launches pay for kernel compilation,
  autotuning, cache warming, and cuDNN algorithm selection; timing them pollutes the measurement.
- **Average over many trials** and look at the variance, not just the mean.
- **Profile to attribute time.** The PyTorch profiler (`torch.profiler` with `ProfilerActivity.CPU`
  and `ProfilerActivity.CUDA`, `with_stack=True`) and the NVIDIA tools tell you which kernels
  dominate and whether time is going to compute, memory, or gaps (launch overhead, sync stalls).
- **Watch memory** with `torch.cuda.max_memory_allocated()` to catch the operations whose footprint,
  not whose runtime, is the problem — naive attention being the canonical example.

Benchmarking a simple MLP shows the intensities in action: runtime scales linearly in
depth (`num_layers`) and steps, but quadratically in the hidden `dimension`, because the matmul
FLOPs grow with the square of the dimension while a batch-size increase mostly stresses bandwidth.
Reading those scaling curves is how you confirm which regime an operation is in.

## Practical implications you can act on today

- If a training step is slower than `6ND / peak_FLOPs` predicts, you are memory-bound somewhere,
  not compute-bound. Profile to find the memory-bound ops.
- Use bf16/fp16 so the matmuls hit tensor cores. fp32 matmuls run on the general units at ~1/16 the
  throughput and leave most of the chip unused.
- Fuse elementwise chains. Frameworks do some of this automatically (`torch.compile`), and you can
  do it by hand for the hot paths, as we do when we write kernels.
- Keep tensors in shapes that map cleanly onto tensor-core tile sizes (multiples of 8 or 16 in the
  relevant dimensions), and size grids to a multiple of the SM count to avoid wave quantization.
  Odd shapes leave performance on the floor.
- Larger batches improve GPU utilization (via both higher-intensity matmuls and more occupancy)
  until you hit a memory or diminishing-returns wall.
- Always synchronize and warm up before timing, and profile before optimizing — the bottleneck is
  rarely where intuition first puts it.

## Key takeaways

A GPU is a memory-movement machine with a lot of compute bolted on, not the other way around.
Performance is governed by arithmetic intensity against the roofline: below the break-even point
(~156 FLOP/byte on A100, ~295 on H100) you are memory-bound and the compute idles; above it you are
compute-bound and using the chip well. Matmuls are compute-bound and belong on tensor cores in low
precision, where an A100 offers ~312 bf16 TFLOP/s versus ~19.5 fp32; elementwise ops, reductions,
and naive attention are memory-bound and dominated by HBM round-trips. The way to go fast is to
move data from HBM as rarely as possible, which means low precision for the matmuls and kernel
fusion for everything else. Batch size buys occupancy and matmul size, which is how the GPU hides
memory latency — and every measurement must synchronize, warm up, and profile before you trust it.
