# Language Models From Scratch — A Self-Study Course

A self-contained course that takes you from the raw ideas to a working language model. The goal
is that you understand a language model well enough to build one end to end, not just call one.
Every chapter is written to be read on its own, but the order below is the intended path.

The course is written for someone who already knows PyTorch and wants the mechanics, the numbers,
and the reasons behind the design choices — not just the recipes.

It is also built to be job-relevant, not just academic. The core chapters teach the durable
concepts. A later chapter maps every one of those concepts to the actual 2026 production stack and
recent paper techniques you are expected to name in an interview. The final four chapters are banks
of frontier-lab-level interview questions with worked answers — estimation and systems judgment,
implement-from-scratch coding, whiteboard derivations, and breadth plus rapid-fire — so every kind
of knowledge gets tested the way a real loop tests it. The intent is that after this you can both
build the thing and talk about it the way someone who ships it does.

## Who this is for

You can write non-trivial PyTorch, you know what a gradient is, you have trained something
before, and you are comfortable with matrix notation, basic probability, and the memory
hierarchy of a computer. If quantization, attention, and distributed training are words you
use at work, you are the target reader. If they are not yet, read slower and run the code.

## The five units

The course is organized around five build-it-yourself projects, because each one forces you to
build a thing that actually runs.

1. **Basics** — tokenization, the Transformer, the training loop, and how to account for every
   FLOP and byte of memory.
2. **Systems** — how GPUs actually execute your code, writing kernels, and splitting a model
   across many devices.
3. **Scaling** — scaling laws, compute-optimal training, and how to spend a fixed compute budget
   without wasting it.
4. **Inference and evaluation** — serving the model cheaply and measuring whether it is any good.
5. **Data and alignment** — building a pretraining corpus, then turning a base model into
   something that follows instructions and reasons.

## Chapter list

| # | Topic |
| --- | --- |
| 00 | Overview |
| 01 | Tokenization: bytes, BPE, vocab design |
| 02 | Resource accounting: FLOPs, memory, mixed precision |
| 03 | The modern Transformer, decoder block by block |
| 04 | Mixture of experts: sparse models, routing, load balancing |
| 05 | GPUs: execution model, roofline, arithmetic intensity |
| 06 | Kernels, fusion, FlashAttention, Triton |
| 07 | Parallelism: data, tensor, pipeline, sequence, ZeRO/FSDP |
| 08 | Scaling laws: Chinchilla, IsoFLOP, compute-optimal budgets |
| 09 | Inference: KV cache, prefill/decode, batching, speculative decoding, quantization |
| 10 | Evaluation: perplexity, benchmarks, contamination, judging |
| 11 | Data: web-scale corpora, extraction, filtering, dedup, PII |
| 12 | Alignment: SFT, DPO, RLHF, expert iteration, GRPO/RLVR |
| 13 | The five build-it-yourself projects |
| 14 | The 2026 stack and recent papers mapped to each concept |
| 15 | Interview bank: estimation, systems, inference, quantization, judgment |
| 16 | Interview bank: implementation drills — BPE, attention, KV-cache, sampling, AdamW, DPO/GRPO from scratch |
| 17 | Interview bank: derivations — softmax/attention VJP, RoPE-is-relative, DPO from RLHF, Chinchilla-optimal |
| 18 | Interview bank: breadth and rapid-fire — tokenization, MoE, roofline, scaling-fit, data, eval, sampling, long-context, alignment |

## How to actually use this

Reading is not the course. The course is the five builds. Read a unit, then do its project
before moving on. The projects are ordered so that each one gives you a component you reuse in
the next: the tokenizer feeds the Transformer, the Transformer gets its kernels optimized, the
optimized model gets scaled, the scaled recipe gets fed real data, and the trained base model
gets aligned.

A realistic time budget for someone working full time is one unit per week or two. The systems
unit (kernels and parallelism) is the one that takes longest and pays back the most if your job
touches inference or training throughput.

For interview preparation specifically: read the modern-stack chapter alongside the core chapters
so each concept comes with its current tool name and tradeoff, then drill the four interview banks
last, closing each answer and reconstructing the reasoning out loud. The banks are organized by
question type so you can test every dimension a frontier lab probes: one covers estimation, systems,
inference, and judgment; one covers implement-from-scratch coding (BPE, attention, KV-cache,
sampling, optimizers, DPO/GRPO); one covers whiteboard derivations (VJPs, RoPE, DPO, Chinchilla);
and one covers breadth recall plus rapid-fire across every topic. The builds are what let you answer
the "have you actually done this" questions with specifics instead of theory, so do at least builds
1, 2, and 3 before interviewing for an inference or training-systems role.

## What you can run on

You do not need a cluster. The tokenizer and a small Transformer on TinyStories train on a
single consumer GPU or even a Mac with MPS. The parallelism material you can study on one machine
using process groups over `gloo`/`nccl` with tiny tensors to see the collectives fire, then
scale the ideas up later. The inference material runs fine on a single device. Only the largest
scaling-law sweeps genuinely want multiple GPUs, and those you can shrink.

## A note on honesty

Where I give a concrete number (a FLOP count, a memory figure, a rule of thumb), it is either
derivable from first principles in the module or comes from a specific published result that I
name. If a claim is a heuristic that practitioners use but that has exceptions, I say so. Do not
treat any single number as gospel across hardware and model scales; the point is that you learn
to derive it yourself.
