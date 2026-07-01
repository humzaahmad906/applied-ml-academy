# 06 — Kernels, Fusion, and FlashAttention (with Triton)

Earlier we saw that the way to go fast is to stop making round-trips to HBM. This chapter is how
you actually do that: by writing your own GPU kernels that fuse operations, and by understanding
FlashAttention, the single most important applied-kernel result of the last several years. The
tool for writing these without dropping to CUDA is Triton. The systems build has you implement
FlashAttention-2 in Triton yourself, forward and backward, which is the fastest way to internalize
why it works.

## What a kernel is and why you would write one

A kernel is a single GPU function launch — a function that runs in parallel across a grid of
threads. When you write `y = torch.relu(x) + b` in PyTorch, that is (naively) two kernels: one
that loads `x` from HBM, computes the ReLU, and writes the result back to HBM, and a second that
loads that result and `b`, adds them, and writes back. Two loads and two stores of the whole
tensor, for two trivial operations. Both are memory-bound, as we saw when we looked at how GPUs
execute code, so the time is dominated by those four HBM trips.

A fused kernel does the whole chain in one launch: load `x` and `b` once, compute `relu(x) + b`
in registers, write the result once. One load, one store. You have roughly halved the memory
traffic and therefore roughly halved the time, for operations whose FLOPs were never the issue.
The gains compound with the length of the chain: our softmax example fuses the max, subtract,
exponentiate, sum, and divide into a single kernel, cutting the HBM element reads from about
`5MN + M` down to `MN` and yielding a 4–5x speedup — not because it does fewer FLOPs, but because
it stops writing and re-reading the intermediate tensors.

You can write a raw CUDA kernel for this (consider one for GELU: a `__global__` function
where each thread computes one output element `0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 x^3)))`,
indexed by `blockIdx * blockDim + threadIdx`, compiled and bound into Python via PyBind11). It
works, but you manage threads, blocks, indexing, and memory coalescing by hand. Triton exists so
you rarely have to.

## Triton in one paragraph

Triton is OpenAI's Python-embedded language for writing GPU kernels at the block level. Instead of
managing individual threads like CUDA, you write a program that operates on blocks (tiles) of data,
and Triton handles the mapping onto threads, the shared-memory management, memory coalescing, and a
lot of the scheduling. You get most of the performance of hand-written CUDA for far less effort.
The mental model: you launch a grid of program instances; each instance is identified by
`tl.program_id(axis)` and is responsible for one tile of the output; inside an instance you load
tiles of the inputs from HBM into on-chip memory with `tl.load`, compute on them, and write the
result with `tl.store`. Tile sizes are passed as `tl.constexpr` so they are known at compile time
and the kernel can be specialized and autotuned. Masks handle tile edges that run past the tensor
bounds.

A minimal fused elementwise kernel (`y = x * scale + bias`, one HBM round trip) looks like:

```python
import triton
import triton.language as tl

@triton.jit
def fused_kernel(x_ptr, out_ptr, scale, bias, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # which tile am I?
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # my slice of the tensor
    mask = offsets < n                           # guard the ragged last tile
    x = tl.load(x_ptr + offsets, mask=mask)      # one load from HBM
    y = x * scale + bias                         # compute on-chip
    tl.store(out_ptr + offsets, y, mask=mask)    # one store to HBM
```

You launch it with a grid sized to cover `n` (e.g. `grid = (triton.cdiv(n, BLOCK),)`). The whole
point is that `x` makes exactly one trip from HBM and back. In our benchmarks the ordering is
consistent: hand-fused Triton beats `torch.compile`-generated kernels beats unfused eager ops —
though `torch.compile` closes most of the gap for free on simple elementwise chains, which is why
you reach for hand-written Triton only on the hot paths it misses.

## FlashAttention: the important one

Attention is the operation that most needs fusion, for two reasons we established earlier: it is
memory-bound in its softmax and masking, and the naive implementation materializes the full
`L x L` score matrix in HBM. At `L = 8192` that score matrix is 64M entries per head per layer,
which is both a huge memory allocation (`O(L²)`) and a huge amount of HBM traffic to write it out
and read it back for the softmax. This is the operation that OOMs first when you push sequence
length.

FlashAttention removes the materialization entirely. The insight is that you never actually need
the full score matrix in memory at once; you only need the final weighted sum of values. So you
tile the computation: assign each program instance a block of `BLOCK_Q` queries, and stream over
the keys and values in blocks of `BLOCK_K`. For each key block you compute the partial scores for
just those blocks, and accumulate their contribution to the output using an online softmax that
updates a running maximum and running sum as it sees each new block. The `L x L` scores never touch
HBM; only the queries, keys, values, and the final output do.

The consequences:

- **Memory:** attention memory drops from `O(L²)` to `O(L)`. This is what makes long context
  feasible at all.
- **Speed:** far fewer HBM round-trips, so the memory-bound attention runs much faster in practice,
  often several times faster than the naive version, more at longer sequences.
- **Exactness:** FlashAttention is not an approximation. The online-softmax accumulation computes
  the same result as the naive version, up to floating-point ordering. This matters: you are not
  trading quality for speed.

## The online softmax, since it is the crux

Softmax over a vector needs the max (for numerical stability) and the sum of exponentials, both of
which naively require seeing the whole vector. The online version processes the vector in blocks
while maintaining a running max `m` and a running normalizer `l`, and rescales the accumulated
output whenever the max updates:

```
for each key/value block:
    s      = Q @ K_block^T * scale        # this block's scores
    m_new  = max(m, rowmax(s))            # extend the running max
    alpha  = exp(m - m_new)               # correction for the old accumulation
    p      = exp(s - m_new)               # this block's stabilized weights
    l      = l * alpha + rowsum(p)        # rescale old sum, add new
    O      = O * alpha + p @ V_block      # rescale old output, add new
    m      = m_new
O = O / l                                 # final normalization
L = m + log(l)                            # logsumexp, saved for the backward pass
```

That `alpha` term is the whole trick: when a later block reveals a larger max, you retroactively
rescale everything you have accumulated so far — both the normalizer `l` and the output `O` — so
the normalization stays correct no matter what order the blocks arrive in. Get this right and you
have FlashAttention. Get it subtly wrong and your model trains to garbage, which is why
implementing it yourself is so instructive.

The `L = m + log(l)` line is worth noticing: FlashAttention saves the per-row logsumexp (one scalar
per query, `O(L)` memory) instead of the full attention matrix. That single vector is exactly what
the backward pass needs to reconstruct the softmax weights without ever having stored them.

## A real Triton forward kernel, in shape

The forward kernel makes the mapping concrete. Two program IDs index the work — one over
(batch x head), one over query tiles — and the inner loop streams the key/value tiles:

```python
@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
                     stride_qb, stride_qm, stride_qd, ...,   # strides for each tensor
                     N_Q, N_K, D, SCALE, IS_CAUSAL,
                     BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_bh = tl.program_id(0)                 # which (batch, head)
    pid_q  = tl.program_id(1)                 # which query tile
    offs_m = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(Q_ptr + pid_bh*stride_qb + offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd)
    q = q * SCALE

    m_i = tl.full([BLOCK_Q], -float("inf"), tl.float32)   # running max
    l_i = tl.zeros([BLOCK_Q], tl.float32)                 # running sum
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)        # output accumulator

    for start_n in range(0, N_K, BLOCK_K):
        offs_n = start_n + tl.arange(0, BLOCK_K)
        k = tl.load(...); v = tl.load(...)
        scores = tl.dot(q, k)                              # (BLOCK_Q, BLOCK_K), on tensor cores
        if IS_CAUSAL:
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -float("inf"))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p     = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i   = alpha * l_i + tl.sum(p, axis=1)
        acc   = alpha[:, None] * acc + tl.dot(p, v)        # rescale then accumulate
        m_i   = m_new

    o   = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(O_ptr + ..., o)
    tl.store(L_ptr + ..., lse)
```

Note the pieces the roofline predicted: `tl.dot` runs the two matmuls (`QK^T` and `P V`) on the tensor
cores, everything between them stays in registers/SRAM, and the only HBM traffic is loading Q/K/V
tiles and writing O and the logsumexp. The `constexpr` tile sizes (typically 16/32/64/128) are what
Triton autotunes over.

## The backward pass and its recomputation

The backward pass uses the same tiling but adds one idea: rather than store the `L x L` scores from
the forward pass, it **recomputes** them on the fly, reconstructing the softmax weights from the
saved logsumexp `L`. Concretely, for each tile it recomputes `S = QK^T`, gets `P = exp(S - L)` (no
running max needed this time — `L` already encodes it), and then forms the gradients:

```
D  = rowsum(dO * O)                # one scalar per query row
dV = P^T @ dO
dP = dO @ V^T
dS = P * (dP - D)                  # softmax Jacobian, collapsed
dQ = dS @ K
dK = dS^T @ Q
```

Recomputing the scores trades a little extra compute for a large memory saving, which is a good
trade because attention was memory-bound to begin with — you had spare compute. The
`D = rowsum(dO * O)` term is computed in its own small kernel first, then the gradients are computed
in a **two-kernel split** that is the practical crux: one kernel grids over query tiles (each owns a
`dQ` tile and loops over all key tiles), a second grids over key tiles (each owns a `dK`/`dV` tile
and loops over all query tiles). Because each output tile has exactly one owning program, no
program ever writes a tile another might touch — so you need **no `tl.atomic_add` at all**, while
keeping full parallelism. (The alternative single-kernel structure forces atomic accumulation on
the overlapping gradient; the split avoids it, which is why both reference solutions use it.) You
can write the backward either as a pure-PyTorch `torch.autograd.Function` that calls the
recomputation, or fully in Triton for the speed.

FlashAttention has gone through versions (v1, v2, v3) that improve the tiling, the work
partitioning across warps, and the use of newer hardware features. The build targets v2, whose
key change over v1 is parallelizing over the query dimension (each query tile is an independent
program instance) and reducing the non-matmul work, which is exactly the structure above.

## The benchmarking and profiling workflow

The systems build is as much about measuring as implementing. The workflow mirrors the GPU chapter:

- **Time correctly.** Warm up with untimed iterations, call `torch.cuda.synchronize()` around the
  timed region (GPU launches are async), average over many trials, and time forward and backward
  separately.
- **Profile to attribute.** Use the PyTorch profiler and NVTX ranges to see which kernels dominate
  and whether time is in compute, memory movement, or launch gaps.
- **Measure memory, not just time.** `torch.cuda.max_memory_allocated()` is how you demonstrate the
  `O(L²)` -> `O(L)` win: naive attention's memory grows quadratically in sequence length and OOMs
  at long `L`, while FlashAttention stays flat enough to keep going. Several end-to-end Transformer
  configs (large `d_model`, many layers, long `seq_len`) that OOM under naive attention run fine
  under the fused kernel — often the headline result of the build.
- **Check correctness against a reference.** Compare your kernel's output to a naive PyTorch
  attention on a fixed input; the tests pass at `rtol = atol = 1e-2` (generous on purpose — tiled
  computation reorders the floating-point sums), confirming FlashAttention is exact up to ordering,
  not approximate.

To calibrate what "fast" means: on a representative benchmark (1x H100, forward + backward,
`B=1, S=4096, H=16, head_dim=64`, bf16, causal), the naive baseline runs at ~80 ms and a
well-tuned kernel reaches ~5.4 ms — roughly a **15x speedup**, all of it from cutting HBM traffic,
not FLOPs. That gap is the entire lesson of the chapter made numeric.

## When to reach for a custom kernel in real work

Be honest about the cost. A custom kernel is more code, more chances to be wrong, and more
maintenance. Reach for one when profiling shows a specific memory-bound hot spot that
`torch.compile` did not fuse, or when you need an operation the framework does not provide fused
(custom attention masks, fused quantization dequant, a fused RMSNorm plus residual). For most of
your work the right order is: use the framework's fused ops (including FlashAttention, which ships
in PyTorch as scaled-dot-product attention), turn on `torch.compile`, profile, and only then write
a kernel for what remains. For your quantization work specifically, fused dequant-matmul kernels
are exactly the kind of thing that pays off, because the dequantization is a memory-bound
elementwise op you want to fuse into the matmul rather than run separately.

## Key takeaways

A kernel is one GPU launch, and fusing a chain of memory-bound operations into a single kernel
collapses many HBM round-trips into one, which is where the speedup comes from — the fused
softmax shows a 4–5x win purely from cutting intermediate reads/writes. Triton lets you write these
at the block level in Python: a grid of program instances, `tl.program_id` to index tiles,
`tl.load`/`tl.store` for HBM, `tl.dot` for tensor-core matmuls, and `constexpr` tile sizes to
autotune. FlashAttention is the flagship example: by tiling attention and using an online softmax
with a running-max rescale, it never materializes the `L x L` score matrix, cutting attention
memory from quadratic to linear in sequence length and running several times faster, exactly, not
approximately. Its backward pass recomputes the scores from a saved logsumexp rather than storing
them, trading spare compute for memory. Implementing the online softmax and its backward yourself,
then benchmarking it against naive attention and torch SDPA, is the point of the systems build — the
running-max rescale is the subtle part, and the memory profile is the proof.
