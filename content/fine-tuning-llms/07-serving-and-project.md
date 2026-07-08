# 07 — Serving, Quantization, and an End-to-End Project

You have a fine-tuned adapter that beats the base model on a clean eval. Now it has to serve real
traffic, cheaply and fast. This last lesson covers the deployment decisions — merge the adapter or
keep it separate, serve one or many adapters with vLLM, and quantize *after* fine-tuning with a
parity check so you don't silently ship a degraded model — and then ties the whole course together
into one end-to-end project you can run.

## Merge or keep the adapter?

A LoRA adapter is a small set of `A`/`B` matrices (tens of MB) that sit on top of a frozen base. You
have two ways to deploy:

**Keep the adapter separate** and load it onto the base at serve time. This is the right default when:

- You serve **many fine-tunes off one base** — one base model in memory, many small adapters swapped
  per request. This is the killer feature (below).
- You want to **hot-swap or A/B** adapters without redeploying the base.
- Storage/versioning matters — a 30 MB adapter is cheap to store and ship per model version.

**Merge the adapter into the base** (`merged = model.merge_and_unload()`) to produce a single
standalone model. This is right when:

- You serve **one** model and want the simplest artifact and zero per-request adapter overhead.
- Your serving stack doesn't support runtime adapters.
- You're about to **quantize** for deployment (merge first, then quantize — see below).

```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("qwen-sft/adapter")
merged = model.merge_and_unload()
merged.save_pretrained("qwen-sft/merged")   # standalone model, base + adapter folded in
```

One sharp edge: **do not merge a bf16 adapter into a 4-bit base.** If you trained QLoRA, the base was
4-bit *for training only*. To merge, reload the base in **fp16/bf16**, attach the adapter, and merge
into full precision — then quantize the merged result with a proper method. Merging into a 4-bit base
compounds quantization error and degrades quality badly.

## Serving LoRA adapters with vLLM

vLLM is the high-throughput serving standard (PagedAttention + continuous batching), and it serves
LoRA adapters natively — including many adapters on one base, selected per request:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules sql-bot=/models/sql-adapter support-bot=/models/support-adapter
```

Then the adapter is chosen by naming it as the `model` in an ordinary OpenAI-compatible request:

```python
import requests
r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "sql-bot",                                  # the adapter name, not the base
    "messages": [{"role": "user", "content": "Count today's signups."}],
})
print(r.json()["choices"][0]["message"]["content"])
```

Why this matters economically: the LoRA math is tiny relative to the base matmuls, so the serving
**overhead is ~3% at rank 16 and ~7% at rank 64.** One 7B base in GPU memory can back dozens of
task-specific adapters, each a few MB, swapped per request. That's how you deploy "a fine-tune per
customer" without a GPU per customer — the single strongest argument for keeping adapters unmerged.
Set `--max-lora-rank` to the largest rank you trained.

## Quantize after fine-tuning — with a parity check

To cut serving memory and boost throughput, quantize the final model to 4-bit for inference. The
critical rule: **quantize *after* fine-tuning, and merge before you quantize.** The order is: train
(QLoRA) → merge adapter into an fp16/bf16 base → quantize the merged model for serving.

Pick the method by target:

- **AWQ** — activation-aware 4-bit, the recommended default for GPU serving via vLLM. Retains ~95% of
  fp16 quality because it identifies and preserves the weights that matter most (including your
  adapter's contribution). Best quality-per-bit for deployment.
- **GPTQ** — another calibrated 4-bit method, ~90% of fp16 quality; widely supported.
- **GGUF** (via `llama.cpp`) — for CPU / Apple Silicon / local serving (Ollama, LM Studio); produces
  multiple bit-depths (Q4, Q5, ...). Use an importance matrix to preserve quality.
- **Avoid bitsandbytes NF4 for the *merged serving* model.** NF4 is excellent for the *frozen base
  during QLoRA training*, but AWQ/GPTQ give materially better results for the post-merge deployment
  artifact.

**The parity check** — never ship a quantized model on faith. Quantization is lossy; verify the loss
is within tolerance on a fixed batch before it goes live:

```python
# Run the SAME fixed prompts through fp16 and the quantized model, compare.
prompts = [...]  # a fixed, representative batch from your eval
fp16_out  = [generate(fp16_model,  p) for p in prompts]
quant_out = [generate(quant_model, p) for p in prompts]

# 1) Behavioral parity: re-run your Lesson 06 task metric on the quantized model.
#    Accept only if the metric drop is within tolerance (e.g. < 1-2 points).
# 2) Numerical spot-check on logits for a fixed input:
import torch
max_abs_diff = (fp16_logits - quant_logits).abs().max().item()
assert max_abs_diff < TOL, f"quantization drifted too far: {max_abs_diff}"
```

Run your frozen eval on the quantized model and confirm the task metric holds. A common, costly
mistake is quantizing, deploying, and discovering weeks later that quality quietly dropped — the
parity check against your eval catches it in minutes.

## The economics of serving your own fine-tune

Fine-tuning only pays off if serving is cheaper than the alternative it replaces, so do the math
before you commit to self-hosting. The levers:

- **Throughput.** vLLM's PagedAttention + continuous batching gives 2–10× the tokens/sec of naive
  `transformers` under concurrent load, by packing many requests' KV-cache into shared GPU memory and
  admitting new requests as others finish. This is what makes a single GPU serve real traffic.
- **Quantization buys throughput and headroom.** A 4-bit AWQ model uses ~1/4 the weight memory of
  fp16, which frees GPU memory for a larger KV cache — more concurrent requests, higher throughput.
  The ~5% quality cost (verified by your parity check) is usually worth it in production.
- **Multi-LoRA amortizes the base.** The break-even story for "a fine-tune per customer": one 7B base
  on one GPU, dozens of adapters swapped per request at ~3% overhead, versus a separate model per
  customer needing a GPU each. The multi-adapter path can be 10–50× cheaper at fleet scale.
- **Compare honestly to the API you're leaving.** Self-hosting has fixed GPU cost whether you serve
  10 or 10,000 requests. Below a volume threshold, a per-token API call to a frontier model is
  cheaper *and* you skip the ops burden. Fine-tuning-plus-self-hosting wins at volume; estimate your
  requests/day against GPU-hour cost before assuming it's cheaper.

The honest summary: fine-tuning's serving story is compelling at scale (high volume, many adapters,
latency-sensitive) and often *not* worth it at low volume, where prompting a hosted model wins on
total cost of ownership.

## End-to-end project: a structured-extraction fine-tune

Put the whole course together. Goal: a small model that extracts structured fields from messy text as
valid JSON, cheaper and more reliable than a prompted frontier model.

1. **Decide it's a fine-tune (Lesson 01).** The gap is *form* — consistent JSON schema and field
   conventions — not *facts*. Prompting is inconsistent on schema; this is a behavior gap. Fine-tune.
2. **Build the data (Lesson 02).** ~500 clean `(text → JSON)` examples in prompt-completion format.
   Validate every completion parses and matches the schema. Split 90/10 on the natural unit, freeze
   and dedup the eval. Generate hard cases synthetically from a stronger model, then *filter to only
   schema-valid ones*.
3. **Configure QLoRA (Lesson 03).** `Qwen2.5-7B-Instruct`, 4-bit NF4 base, `LoraConfig(r=16, α=32,
   target_modules="all-linear")`. Fits on one 24 GB (or even ~8 GB) GPU.
4. **Train (Lesson 04).** `SFTTrainer`, `lr=2e-4`, 2 epochs, effective batch 16, `packing=True`,
   `completion_only_loss=True`, `eval_strategy="steps"`, `load_best_model_at_end=True`. Watch eval
   loss for the overfitting inflection.
5. **Optionally prefer-tune (Lesson 05).** If SFT nails the schema but you want terser, cleaner
   values, build ~200 `(chosen, rejected)` pairs and run DPO at `lr=5e-6`, `beta=0.1`, 1 epoch.
6. **Evaluate (Lesson 06).** Task metric = schema-valid rate + field-level exact match, base vs
   fine-tune on the frozen eval. Add a general-capability probe to rule out forgetting. Confirm no
   train/eval contamination.
7. **Serve and quantize (this lesson).** For one model: merge → AWQ-quantize → parity-check against
   the eval → serve on vLLM. For many customers: keep the adapter, serve multi-LoRA on one base with
   `--enable-lora`.

The deliverable is a model that's smaller, cheaper per token, and more reliable on your narrow task
than the frontier model you started prompting — with an eval number that proves it.

## Local and edge serving

Not every deployment is a GPU server. For local apps, on-device, or Apple Silicon, the path is
**GGUF via `llama.cpp`**, consumed by **Ollama** or **LM Studio**: merge the adapter, convert to
GGUF, pick a bit-depth (Q4_K_M is a good quality/size default), and run with a single command. The
same parity-check discipline applies — run your frozen eval on the GGUF build and confirm the task
metric holds, because aggressive GGUF quantization (Q3 and below) can degrade quality more than AWQ.
On Apple Silicon specifically, the **MLX** stack fine-tunes and serves natively on unified memory and
is often the fastest local option. The decision tree: GPU server at scale → vLLM + AWQ (adapters or
merged); one local machine → GGUF + Ollama; Apple Silicon → MLX. In all three, the artifact you ship
is the merged-and-quantized model or the base-plus-adapter pair you built in this course, gated by a
parity check against the eval you froze in Lesson 06.

## Key takeaways

- **Keep the adapter** to serve many fine-tunes off one base (hot-swap, A/B, per-customer);
  **merge** for a single standalone artifact or before quantizing.
- Never merge a bf16 adapter into a **4-bit** base — reload the base in fp16/bf16, merge, *then*
  quantize.
- vLLM serves LoRA natively: `--enable-lora --max-lora-rank N --lora-modules name=/path`, pick the
  adapter by name per request; overhead is ~3% (r16) / ~7% (r64) — dozens of adapters on one base.
- **Quantize after fine-tuning, merge before quantizing.** Use **AWQ** (~95% fp16) or GPTQ for GPU,
  **GGUF** for CPU/Apple Silicon; avoid NF4 for the merged serving model.
- Always run a **parity check** — re-run your frozen eval and a numerical logit diff on the quantized
  model and accept only within tolerance.
- The end-to-end loop: decide (form vs facts) → curate data → QLoRA → SFT → optional DPO → eval vs
  base → merge/quantize/serve with a parity check.

## Try it

Ship the project end to end at small scale. Take your Lesson 04 fine-tune and: (1) **merge** it —
`AutoPeftModelForCausalLM` → `merge_and_unload()` → `save_pretrained` — and confirm the merged model
loads and generates. (2) **Quantize** the merged model (AWQ if you have a GPU, GGUF via `llama.cpp`
locally otherwise) and run the **parity check**: re-run your Lesson 06 task metric on the quantized
model and record the drop — accept only if it's within a point or two. (3) If you have a GPU, serve
two adapters at once with `vllm serve --enable-lora --lora-modules a=... b=...` and hit each by name
to see per-request adapter selection working. Write a one-paragraph deployment note: merged or
adapter, which quantization, the parity-check numbers, and the base-vs-fine-tune eval delta. That
note is the artifact a team actually reviews before a fine-tune goes to production — and completing
it means you've run the full applied fine-tuning loop, decision to deployment.
