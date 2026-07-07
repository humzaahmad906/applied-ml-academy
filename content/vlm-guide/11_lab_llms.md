# Lab 2 — Decoding and LoRA Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/vlm-guide/notebooks/11_lab_llms.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup. The full write-up and stack alternatives are below.

Load a real instruct model, compare every decoding strategy, inspect how temperature reshapes the logit distribution, apply chat templating correctly, then run a full LoRA fine-tune loop and verify generation before and after. Makes the [LLMs chapter](02_llms.md) concrete.

## Setup

```bash
pip install torch transformers peft trl datasets accelerate
```

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` — ~1 GB fp16. Runs on CPU (slow), MPS (M-series Mac), or CUDA. VRAM requirement: ~1.2 GB fp16 / ~0.4 GB with 4-bit.

```python
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if device in ("cuda", "mps") else torch.float32,
    device_map=device,
)
model.eval()
print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
```

---

## Part A — Decoding Strategies

```python
def gen(prompt: str, **kw) -> str:
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, pad_token_id=tokenizer.eos_token_id, **kw)
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

PROMPT = "The key insight behind the transformer architecture is"

# Greedy — argmax at every step; deterministic, can repeat
print("greedy:", gen(PROMPT, max_new_tokens=40, do_sample=False))

# Temperature — divide logits by temp before softmax
#   temp < 1 → sharper (less random)   temp > 1 → flatter (more random)
print("temp=0.3:", gen(PROMPT, max_new_tokens=40, do_sample=True, temperature=0.3))
print("temp=1.5:", gen(PROMPT, max_new_tokens=40, do_sample=True, temperature=1.5))

# Top-k — zero out all but the k highest-prob tokens, then sample
print("top_k=20:", gen(PROMPT, max_new_tokens=40, do_sample=True, top_k=20, temperature=0.8))

# Top-p (nucleus) — keep the smallest set covering cumulative prob >= p
print("top_p=0.9:", gen(PROMPT, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=0.8))
```

### Temperature reshapes the distribution

```python
ids = tokenizer(PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**ids).logits[0, -1, :]   # last-position logits, [vocab_size]

for temp in [0.1, 0.5, 1.0, 2.0]:
    probs = F.softmax(logits / temp, dim=-1).topk(5)
    toks  = [tokenizer.decode([i]) for i in probs.indices.tolist()]
    print(f"temp={temp}  top-5 probs={probs.values.round(decimals=3).tolist()}  {toks}")
```

At temp → 0 the softmax collapses to a one-hot on the argmax (greedy). At temp → ∞ it flattens to uniform. Rule of thumb: `0.7` for creative tasks, `0.1` for factual/code.

---

## Part B — Chat Templating

Instruct models are trained on a specific conversation format. Feeding raw text instead of a templated prompt puts the model out of distribution — responses degrade immediately.

```python
messages = [
    {"role": "system",  "content": "You are a concise ML tutor. Answer in 2–3 sentences."},
    {"role": "user",    "content": "What problem does the residual connection solve?"},
]

# add_generation_prompt=True appends the <|im_start|>assistant prefix
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(repr(prompt[:180]))   # inspect the ChatML tokens

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    out = model.generate(
        input_ids, max_new_tokens=80,
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True))
```

For multi-turn: append the assistant's prior response to `messages` before the next `apply_chat_template` call. Never construct the special tokens by hand — the template handles all model-specific formatting.

---

## Part C — LoRA Fine-Tuning

```python
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ── tiny instruction dataset ──────────────────────────────────────────────
RAW = [
    ("What is gradient descent?",
     "Gradient descent updates parameters by moving them opposite to the gradient of the loss."),
    ("Explain the residual connection.",
     "A residual connection adds the block input to its output, giving gradients a clean bypass path."),
    ("What does RMSNorm do?",
     "RMSNorm normalizes by root-mean-square instead of mean+std, dropping the bias and being cheaper."),
    ("What is SwiGLU?",
     "SwiGLU multiplies a SiLU-activated linear projection by a second gate projection — empirically better per-parameter than a plain MLP."),
    ("What is the KV cache?",
     "The KV cache stores key/value tensors from prior steps so attention only computes them once per token."),
    ("Explain LoRA.",
     "LoRA freezes the base model and adds low-rank matrices A and B to target layers; only A and B are trained."),
    ("What is top-p sampling?",
     "Top-p sampling keeps the smallest token set whose cumulative probability reaches p, then samples from it."),
    ("What is RoPE?",
     "RoPE rotates Q and K by position-dependent angles so attention scores encode relative offsets."),
]

def _fmt(u, a):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
        tokenize=False, add_generation_prompt=False,
    )

ds = Dataset.from_dict({"text": [_fmt(u, a) for u, a in RAW]})

# ── generate BEFORE ──────────────────────────────────────────────────────
TEST = "Explain LoRA."
before = gen(
    tokenizer.apply_chat_template([{"role":"user","content":TEST}],
                                  tokenize=False, add_generation_prompt=True),
    max_new_tokens=60, do_sample=False,
)
print("BEFORE:", before)

# ── LoRA config ──────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
)

sft_cfg = SFTConfig(
    output_dir="./qwen-lora-lab",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=(device == "cuda"),
    logging_steps=5,
    save_strategy="no",
    max_seq_length=256,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model, args=sft_cfg,
    train_dataset=ds, peft_config=lora_cfg,
)
trainer.train()

# ── generate AFTER ───────────────────────────────────────────────────────
model.eval()
after = gen(
    tokenizer.apply_chat_template([{"role":"user","content":TEST}],
                                  tokenize=False, add_generation_prompt=True),
    max_new_tokens=60, do_sample=False,
)
print("AFTER:", after)
# With 8 examples the change is small — the workflow is what matters here.
# Scale to 1k–10k examples for real adaptation.
```

---

## What you built

- Greedy, temperature, top-k, and top-p decoding with a direct look at how temperature reshapes the logit distribution.
- Correct chat templating — `apply_chat_template` for both inference and training data formatting.
- A full LoRA fine-tune loop: `LoraConfig`, `SFTConfig`, `SFTTrainer`, before/after generation on the same prompt.
- Device-aware dtype selection: bfloat16 on CUDA, float32 fallback on CPU/MPS.

## Build it further

Load the first 500 rows of `HuggingFaceH4/ultrachat_200k`, format them with `apply_chat_template`, and fine-tune for 1 epoch. Then compute token-level perplexity on 10 held-out examples (`model(**inputs).loss`) before and after. Does perplexity drop? Does quality improve on a subjective 5-question evaluation?

---

## Stacks & alternatives

**Ollama — model in one command, zero Python:**

```bash
ollama run qwen2.5:0.5b "What is the residual connection?"
```

```python
import ollama
r = ollama.chat(model="qwen2.5:0.5b",
                messages=[{"role":"user","content":"What is RoPE?"}])
print(r.message.content)
```

Best for quick local experiments and demos. No batching or serving control.

**vLLM — high-throughput serving with an OpenAI-compatible endpoint:**

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000  # CUDA required
```

```python
import requests
r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role":"user","content":"What is RoPE?"}],
    "max_tokens": 80,
})
print(r.json()["choices"][0]["message"]["content"])
```

PagedAttention + continuous batching gives 2–10x higher throughput than HF `transformers` under concurrent load. The right choice when you move from notebook to service.

**MLX / mlx-lm — Apple Silicon native (not PyTorch-MPS):**

```bash
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "What is RoPE?" --max-tokens 100
```

```python
from mlx_lm import load, generate as mlx_gen
m, t = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print(mlx_gen(m, t, prompt="What is RoPE?", max_tokens=100))
```

2–4x faster than PyTorch-MPS on M-series because MLX ops run natively on unified memory with no CPU↔GPU copy overhead.

**Unsloth — faster LoRA with less VRAM:**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", max_seq_length=512, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=8, target_modules=["q_proj","v_proj"],
    lora_alpha=16, use_gradient_checkpointing="unsloth",
)
# plug into the same SFTTrainer call as above
```

2–5x faster training and 60–80% less VRAM than vanilla PEFT via Triton kernels. **Axolotl** is the config-driven alternative (YAML file, no Python) for reproducible team runs — reach for it when the SFT setup is standard and you want version-controlled configs instead of notebooks.
