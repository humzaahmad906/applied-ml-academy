# 07 — Parallelism: Training Across Many Devices

Once a model or its training state does not fit on one GPU, or one GPU is too slow, you split the
work across devices. There are four ways to split, they compose, and choosing the right
combination for a given model and cluster is one of the harder skills in the field. This chapter is
long because it covers a lot of ground: the collective-communication primitives
and their cost model, then every parallelism strategy built on top of them — data parallelism and
DDP, the ZeRO/FSDP sharding stages, tensor parallelism, pipeline parallelism, sequence/context
parallelism, and how you combine them into 3D parallelism on a real cluster. Throughout, the thing
to hold onto is the communication cost, because it is what decides every design question.

## Why communication is the whole story

A GPU can do on the order of hundreds of teraFLOP/s. Moving a byte between GPUs is comparatively
glacial: NVLink inside a node is fast (hundreds of GB/s), the network between nodes is much slower
(tens of GB/s on good clusters, worse elsewhere). So the moment you split work across devices, the
question is never "can the math be split?" — it almost always can — but "how many bytes must cross
which link, and can I hide that behind computation?" Every parallelism strategy is a different
answer to where the bytes go. Get this framing right and the rest of the module is bookkeeping.

The golden rule, which will recur: **keep the highest-volume communication inside a node, on the
fast interconnect; push only the low-volume communication across nodes.** When you see a strategy
described as "intra-node only" or "fine across nodes," this rule is why.

## The primitives: collective operations

Distributed training is built on collective operations over a *group* of devices (a "process
group" in PyTorch; each device is a "rank," and `world_size` is the number of ranks). The full set
we work through:

- **Broadcast:** one rank sends a tensor to all other ranks.
- **Scatter:** one rank splits a tensor into chunks and sends a different chunk to each rank.
- **Gather:** every rank sends its tensor to one designated rank, which collects them.
- **Reduce:** every rank sends its tensor to one rank, which combines them with an op (sum, min,
  max, avg).
- **All-gather:** every rank has a shard and every rank ends up with the full concatenation of all
  shards.
- **Reduce-scatter:** every rank contributes a full tensor; the tensors are reduced element-wise
  and each rank ends up with one reduced *shard* of the result.
- **All-reduce:** every rank contributes a full tensor and every rank ends up with the full
  reduced result. This is the workhorse of data parallelism.
- **All-to-all:** every rank sends a different piece to every other rank. This is the MoE routing
  pattern and the most expensive general collective.

Two identities are worth memorizing because they explain the cost of everything else:

- **All-reduce = reduce-scatter followed by all-gather.** You first reduce so each rank owns one
  finished shard, then gather so everyone has all shards. This is exactly how the efficient "ring"
  all-reduce is implemented, and it is why all-reduce costs almost precisely twice a
  reduce-scatter.
- **All-gather and reduce-scatter are mirror images** — one distributes, one collects — and each
  moves the same amount of data.

### The cost model

Here is an explicit, simple cost model for the bytes each rank sends. For a tensor of
`size_bytes` on a group of `world_size` ranks using ring algorithms:

- **Reduce-scatter:** $\text{sent\_bytes} = \text{size\_bytes}\cdot\frac{W - 1}{W}$ per rank per pass
  (writing $W$ for `world_size`); in the simplified accounting here, the dominant term is $\text{size\_bytes}\cdot(W - 1)$.
- **All-gather:** the same as reduce-scatter — one pass around the ring.
- **All-reduce:** $\text{sent\_bytes} = \text{size\_bytes}\cdot 2\,(W - 1)$ — the $2\times$ is exactly the
  reduce-scatter + all-gather decomposition.

The crucial feature of the ring formulation is that the bytes each rank sends are *independent of
how many ranks there are* in the large-`world_size` limit: `(world_size - 1) / world_size → 1`.
That is why data-parallel all-reduce scales to hundreds of GPUs without the per-device
communication blowing up — the total volume grows, but the volume *per link* stays roughly one
model's worth of gradients. Contrast all-to-all, whose volume grows with `world_size` and which
therefore does not enjoy this property.

Some benchmarks on NVLink-connected GPUs, worth internalizing as a sanity check:
all-reduce sustained roughly **275 GB/s** on a 100M-element tensor, and reduce-scatter roughly
**75 GB/s**. The rough 2×–3× ratio matches the cost model. Actual wall-clock time is
`latency + sent_bytes / bandwidth`; for small tensors latency dominates (which is the entire
argument for *bucketing*, below), and for large tensors the bandwidth term dominates.

In PyTorch this all lives in `torch.distributed`: you `init_process_group` with a backend (`nccl`
for GPU, `gloo` for CPU), then call `dist.all_reduce(tensor, op=ReduceOp.AVG)`,
`dist.reduce_scatter_tensor(...)`, `dist.all_gather_into_tensor(...)`, and so on. Synchronization
uses `dist.barrier()` and `torch.cuda.synchronize()` — the latter matters when you time things,
because CUDA calls are asynchronous and a naive timer measures only the launch.

## Data parallelism

The simplest thing and the first you reach for. Replicate the full model on every device, give each
device a different slice of the global batch (`local_batch_size = global_batch_size / world_size`),
run forward and backward independently, then **all-reduce the gradients with `ReduceOp.AVG`** so
every replica applies the identical update and stays bit-for-bit in sync. Throughput scales nearly
linearly with device count as long as the all-reduce does not dominate.

The communication volume is one all-reduce of the gradients per step: about `2 * (world_size - 1)`
times the gradient size in bytes, per rank, independent of batch size. Because the gradient is the
same size as the model, this is "one model's worth of bytes" per step regardless of how many GPUs
you add — the scaling property from above. Its limit is *memory*: every device holds the full
model, gradients, and optimizer state. Data parallelism alone does nothing for a model that does
not fit on one GPU. That is what the sharding schemes fix.

### DDP done properly: bucketing and overlap

A naive data-parallel loop runs the entire backward pass, then fires one giant all-reduce. That
wastes the whole backward pass's worth of idle network, and then stalls compute on one huge
transfer. Real DDP (PyTorch's `DistributedDataParallel`, which you also build yourself)
does two things:

- **Overlap.** Gradients become available layer by layer as backprop proceeds from the output back
  to the input. DDP registers autograd hooks so that as soon as a layer's gradient is ready, its
  all-reduce is launched *asynchronously* while backprop continues on earlier layers. By the time
  backward finishes, most of the communication has already happened underneath it. This is the
  single biggest DDP optimization and it is why DDP is nearly free when the network is fast.
- **Bucketing.** Firing one all-reduce per parameter tensor means thousands of tiny transfers,
  each paying the fixed latency cost — and recall that for small tensors latency dominates. So DDP
  coalesces gradients into **buckets** (e.g. 25 MB) and all-reduces a full bucket at once,
  amortizing latency over many parameters. The bucket size is a real tuning knob: too small and you
  pay latency; too large and you lose overlap because you cannot launch the transfer until the last
  gradient in the bucket is ready.

The build makes this concrete — you implement naive DDP, then per-parameter overlap, then
bucketed overlap, and benchmark each. A common and instructive result on a single machine with a
weak interconnect is that distributed training comes out *slower* than one GPU: the communication
overhead swamps the parallelism benefit. That is not a bug; it is the cost model telling you the
truth about your hardware.

## ZeRO and FSDP: sharding the training state

Recall that training state is parameters + gradients + optimizer state, roughly
`18N` bytes for a model of `N` parameters trained in mixed precision with Adam: 2 bytes fp16
params + 2 bytes fp16 grads + (4 fp32 master weights + 4 fp32 momentum + 4 fp32 variance) = 12
bytes of fp32 optimizer state, ~18 bytes total. Pure data parallelism replicates *all* `18N` of it
on *every* device, which is enormously wasteful. ZeRO (Zero Redundancy Optimizer; PyTorch's
incarnation is FSDP, Fully Sharded Data Parallel) removes the redundancy by *sharding* the state
across the data-parallel devices instead of replicating it, while keeping the data-parallel
programming model.

The three stages, with the memory math for `world_size` devices:

- **Stage 1 — shard optimizer state.** Each device stores only its `1/world_size` slice of the
  fp32 master weights and Adam moments. Optimizer state is the biggest chunk (`12N` of the `18N`),
  so this alone cuts per-device memory to roughly `2N + 2N + 12N/world_size`. Communication is
  unchanged from plain DDP in the common formulation (an all-reduce, or equivalently a
  reduce-scatter to place gradients where their optimizer shard lives plus an all-gather of the
  updated params).
- **Stage 2 — also shard gradients.** Each device only ever needs the gradient shard that
  corresponds to the optimizer shard it owns. Per-device memory drops to roughly
  `2N + (2N + 12N)/world_size`. The all-reduce is replaced by a **reduce-scatter** (each rank ends
  up with only its gradient shard) — which, from the cost model, is *half* the bytes of an
  all-reduce.
- **Stage 3 — also shard parameters (this is FSDP).** No device holds the whole model at rest.
  Before a layer runs in the forward pass, the devices **all-gather** that layer's parameters, use
  them, then discard the gathered copy; the backward pass all-gathers them again and
  reduce-scatters the gradients. Per-device memory falls to roughly `18N / world_size`. You have
  traded memory for communication: two extra all-gathers of parameters per layer (forward + back)
  plus the gradient reduce-scatter.

The stage-3 trade is the important one. Total communication volume roughly triples versus plain DDP
(all-gather params in forward, all-gather in backward, reduce-scatter grads), but FSDP overlaps the
next layer's all-gather with the current layer's compute — a "prefetch" — so the wall-clock cost is
much less than 3×. In practice FSDP is the default for models too big for one GPU but where you
still want the simplicity of data parallelism: it scales to large clusters and, unlike tensor
parallelism, its communication is coarse-grained (per layer, not inside every matmul) and hides
well behind compute.

## Tensor parallelism (Megatron-style)

Split *individual layers* across devices. For a matmul `Y = X W`, there are two ways to shard `W`:

- **Column-parallel:** split `W` by columns, `W = [W_1 | W_2 | ...]`. Each device computes
  `Y_i = X W_i`, a slice of the output columns. No communication to produce the partial outputs;
  the results are the concatenation `Y = [Y_1 | Y_2 | ...]`.
- **Row-parallel:** split `W` by rows and correspondingly split the *input* by columns. Each device
  computes a partial sum `Y_i = X_i W_i`, and the final `Y = sum_i Y_i` requires an **all-reduce**.

Megatron's insight is to chain these so the communication cancels where possible. In the **MLP**
block (`Y = GeLU(X A) B`), make the first matmul `A` column-parallel and the second matmul `B`
row-parallel. The column-parallel `A` produces sharded activations with no communication; the
nonlinearity is applied independently on each shard; the row-parallel `B` then needs a single
all-reduce to sum the partials. So the whole MLP costs **one all-reduce in forward** (and one in
backward, denoted the `g` operator; the forward-input split is the `f` operator, which is identity
forward / all-reduce backward). In **attention**, the per-head structure is a natural column split:
put whole heads on different devices (Q, K, V projections column-parallel, so each device does the
full attention math for its heads), then make the output projection row-parallel — again **one
all-reduce** to combine. So a transformer layer costs about **two all-reduces in forward and two in
backward**, on activations.

That is the catch: tensor parallelism communicates *inside every layer*, on *activation-sized*
tensors, and cannot overlap the all-reduce with much because the next matmul depends on its result.
The volume is high and it is on the critical path. This is why tensor parallelism is almost always
confined to a **single node**, where NVLink bandwidth can absorb it. The canonical setup is
tensor-parallel across the 8 GPUs of one node, and something cheaper (data or pipeline) across
nodes. Tensor parallelism is what you use when a single layer is too *wide* to fit or too slow on
one device.

## Pipeline parallelism

Split the model by *depth*: device 0 holds layers 1–8, device 1 holds 9–16, and so on
(`local_num_layers = num_layers / world_size`). A batch flows through the pipeline stage by stage,
with only the activations at each boundary passed forward (point-to-point `send`/`recv`, not a
collective) and the gradients passed back. Communication volume is **low** — you move one boundary
activation per micro-batch, not a whole model's gradients — which is exactly why pipeline
parallelism is the axis you extend *across nodes* on a slow interconnect.

The problem is the **bubble**. Naively, while device 0 works on the batch, devices 1..P-1 sit idle
waiting for its output, then device 0 idles while the rest finish — a triangular waste at the start
and end. The fix is **micro-batches**: split the batch into `m` micro-batches and feed them in like
an assembly line, so all stages stay busy on different micro-batches at once. With `P` pipeline
stages and `m` micro-batches the idle fraction is

$$
\text{bubble\_fraction} = \frac{P - 1}{m + P - 1}
$$

so you shrink the bubble by making `m` large relative to `P`. You never eliminate it. **GPipe**
runs all forwards then all backwards (simple, but holds all micro-batch activations in memory).
**1F1B** (one-forward-one-backward, PipeDream-style) interleaves forward and backward so each stage
starts freeing activation memory sooner — same bubble, much lower peak memory, so it is the
scheme real systems use.

## Sequence and context parallelism

For very long sequences, even a single layer's activations for one sequence do not fit, and the
attention matrix is quadratic in sequence length. **Sequence parallelism** splits the sequence
(token) dimension across devices for the parts of the layer that are elementwise along the sequence
(LayerNorm, dropout, the residual adds) — a natural complement to tensor parallelism that shards
the activations those ops leave replicated, cutting activation memory further. **Context
parallelism** does it for attention specifically: each device holds part of the sequence's queries,
keys, and values, and devices exchange K/V blocks (an all-gather or a ring exchange, as in Ring
Attention) so each can attend over the whole sequence while ever holding only its shard. This is
the newest axis and is what makes million-token-context training feasible.

## Composing them: 3D (and 4D) parallelism

Real large-scale training combines these along orthogonal axes. The canonical recipe maps each
axis to the link whose bandwidth matches its volume:

- **Tensor parallel** *within a node* (e.g. 8-way): highest-volume, per-layer, on-critical-path
  activation all-reduces — placed on NVLink.
- **Pipeline parallel** *across a few nodes*: low-volume boundary activations only — tolerant of
  the slow inter-node network; sized to fit the model's depth.
- **Data parallel with FSDP sharding** *across the remaining replicas*: gradient reduce-scatter /
  parameter all-gather, coarse-grained and overlappable — also fine across nodes.

For MoE models you add a fourth axis, **expert parallelism**, sharding experts across devices,
which introduces the all-to-all routing communication from the mixture-of-experts chapter (and recall all-to-all's
volume grows with `world_size`, so expert-parallel groups are kept modest and, ideally, intra-node).

Choosing how to factor your device count across these axes is a constrained optimization: you must
(1) fit the model and its activations in per-device memory, (2) keep every device busy (small
bubble, good overlap), and (3) keep the highest-volume communication on the fastest links. There
is no universal answer; it shifts every hardware generation as the ratio of compute to memory to
interconnect bandwidth changes.

## The mental model for choosing

Ask, in order:

1. Does the model fit on one GPU with your batch size? If yes, plain DDP with bucketed overlap, and
   stop.
2. Does it fit with FSDP sharding across your data-parallel group? If yes, use FSDP (start at stage
   1, escalate to 2/3 as memory demands).
3. Is a single *layer* too wide or too slow? Add tensor parallelism, kept inside the node on NVLink.
4. Is the model too *deep* to fit even sharded? Add pipeline parallelism across nodes, with enough
   micro-batches to shrink the bubble, using 1F1B for memory.
5. Are sequences too *long*? Add sequence/context parallelism.
6. Is it an MoE? Add expert parallelism for the experts.

Each axis has a communication cost, and the art is placing the high-volume communication on the
fast interconnect and hiding it behind compute wherever the dependency structure allows.

## Relevance to your two-box setup

Your Spark + M2 Max setup is not a training cluster, but the same collective-communication concepts
underlie multi-instance inference sharding and any distributed serving you do across the two boxes.
The interconnect between them (Ethernet/Thunderbolt) is far slower than intra-node NVLink, so the
golden rule bites hard: keep high-volume tensor traffic within a box and send only low-volume data
(activations, routing) across the link. If you ever shard a model across the two, a pipeline-style
split (low, point-to-point communication) will behave far better than a tensor-parallel split (high
volume, on the critical path) over that slow link — this is exactly the cost model predicting your
result before you run it.

## Key takeaways

Everything reduces to communication cost. The collectives have a clean ring cost model:
reduce-scatter and all-gather each move about `size_bytes * (world_size - 1)` per rank, and
all-reduce is exactly their sum (the 2× factor), with per-link volume roughly constant in
`world_size` — which is why data parallelism scales. Real DDP overlaps gradient all-reduce with
backprop and buckets small tensors to beat latency. ZeRO/FSDP shards the `18N` training state
across data-parallel devices: stage 1 (optimizer state), stage 2 (+ gradients, replacing
all-reduce with reduce-scatter), stage 3 (+ parameters, all-gathered per layer for `~18N/world_size`
memory). Tensor parallelism (Megatron) shards layers with column-then-row matmuls, costing ~2
all-reduces per layer per direction on activations — high volume, critical path, so intra-node only.
Pipeline parallelism shards depth with cheap point-to-point activations and a bubble of
`(P-1)/(m+P-1)` shrunk by micro-batches (1F1B for memory) — so it goes across nodes.
Sequence/context parallelism splits long sequences. 3D parallelism assigns each axis to the link
whose bandwidth matches its volume; MoE adds expert parallelism and its all-to-all cost. There is
no fixed recipe — it is a fit-and-utilization optimization that changes with the hardware.

## You can now

- state the ring cost model for the collectives and explain why an all-reduce costs exactly `2×` a reduce-scatter.
- explain why data-parallel per-link communication volume stays roughly constant as you add GPUs, while all-to-all volume grows with `world_size`.
- distinguish the three ZeRO/FSDP stages by what each shards (optimizer state, then gradients, then parameters) and the resulting per-device memory.
- map tensor, pipeline, sequence/context, and expert parallelism onto the interconnect links whose bandwidth matches each one's communication volume.
- work through the decision order — DDP, FSDP, tensor, pipeline, sequence, expert — for choosing a parallelism strategy for a given model and cluster.

## Try it

Bring up `torch.distributed` with the `gloo` backend across a few processes on one machine and time an all-reduce against a reduce-scatter on a large tensor (warm up, then `dist.barrier()` and `torch.cuda.synchronize()` around the timed region). Confirm the roughly `2×` byte ratio the ring cost model predicts. Then implement naive data parallelism (one big all-reduce after backward) and bucketed-overlap DDP on a tiny model, and measure whether overlap actually wins on your interconnect — on a weak link you may find distributed training is *slower* than a single device, exactly as the cost model warns.
