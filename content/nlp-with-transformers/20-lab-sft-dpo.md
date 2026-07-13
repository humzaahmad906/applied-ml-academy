# 20 — Lab 4: SFT then DPO on a Small Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/nlp-with-transformers/notebooks/20-lab-sft-dpo.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup.

Post-training is where a next-token predictor becomes an assistant. This lab walks the two stages that do almost all of that work in practice: **supervised fine-tuning (SFT)** to teach format and behavior from demonstrations, then **direct preference optimization (DPO)** to nudge the model toward responses people prefer. You will do both on `Qwen2.5-0.5B-Instruct` with LoRA so the whole thing fits on a free T4, and you will measure the change instead of eyeballing it. The one bug that silently wrecks more post-training runs than any other — a chat-template mismatch — gets its own section, because you will hit it on the job.

The mechanism behind SFT and DPO lives in [post-training](07-post-training.md); this lab makes it concrete.

## Setup

```bash
pip install -q "transformers>=4.44" "trl>=0.12" "peft>=0.13" "datasets>=2.20" "accelerate>=0.34"
```

Runtime target: **under 25 min on a Colab T4** (Runtime → Change runtime type → GPU). SFT and DPO are capped with `max_steps` so neither exceeds ~3–4 min. Peak GPU memory stays under ~6 GB — a 0.5B model in bf16 is ~1 GB, and LoRA adds only a few MB of trainable parameters, so **4-bit loading is unnecessary at this size**. You would reach for QLoRA (4-bit base via `bitsandbytes`) only once the base model is 7B+; see [prompting and PEFT](08-prompting-peft.md) for that math.

Seeds are set to 42 on `random`, `numpy`, and `torch` at the top of every section.

---

## Part A — Chat templates and special tokens

An instruct model was trained on turns wrapped in special tokens. Qwen2.5 uses the ChatML format: each turn is `<|im_start|>{role}\n{content}<|im_end|>`. You almost never hand-write these — `apply_chat_template` renders a list of message dicts into exactly the string the model saw in training.

```python
import random, numpy as np, torch
from transformers import AutoTokenizer

random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# add_generation_prompt=True appends the empty assistant-turn opener so the
# model knows it is its turn to speak.
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
print("---")
print("eos token:", tok.eos_token, "| id:", tok.eos_token_id)
print("<|im_start|> id:", tok.convert_tokens_to_ids("<|im_start|>"))
print("<|im_end|>   id:", tok.convert_tokens_to_ids("<|im_end|>"))
```

The rendered string ends with `<|im_start|>assistant\n` and nothing after it — that trailing opener is what `add_generation_prompt=True` adds. `<|im_end|>` is the turn terminator the model learns to emit when it is done; generation stops there.

**The classic template-mismatch bug.** If you feed the model a prompt formatted differently from its training format, it degrades quietly — it rambles past where it should stop, ignores the system prompt, or repeats itself. There is no error; the output just gets worse. Three ways people trigger it:

- Hand-building `"User: ...\nAssistant:"` instead of using the template (wrong delimiters).
- Forgetting `add_generation_prompt=True`, so the model completes the *user's* turn instead of answering.
- Double-templating — calling `apply_chat_template` on text that was already templated, so special tokens get escaped or duplicated.

You will see the first one break generation in Part B.

---

## Part B — Baseline generations and the mismatch bug

Load the model and capture baseline answers on a fixed prompt set *before* any training, so before/after is a fair comparison.

```python
from transformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)

EVAL_PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Write a haiku about gradient descent.",
    "I feel overwhelmed by my workload. Any advice?",
    "What are the tradeoffs between TCP and UDP?",
    "Give me a recipe for a quick breakfast.",
]

@torch.no_grad()
def chat(model, user, system="You are a helpful assistant.", max_new_tokens=128):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

baseline = {p: chat(base, p) for p in EVAL_PROMPTS}
print(baseline[EVAL_PROMPTS[0]])
```

Now watch the template-mismatch bug break decoding. Feed the same question through the correct template versus a plain hand-built string:

```python
@torch.no_grad()
def raw_generate(model, prompt_str, max_new_tokens=128):
    ids = tok(prompt_str, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

q = "What is the capital of France?"
good = tok.apply_chat_template([{"role": "user", "content": q}],
                               tokenize=False, add_generation_prompt=True)
bad = f"User: {q}\nAssistant:"   # wrong delimiters, no special tokens

print("CORRECT TEMPLATE:\n", raw_generate(base, good), "\n")
print("MISMATCHED (hand-built):\n", raw_generate(base, bad))
```

The correct prompt yields a clean, short answer that stops. The mismatched prompt typically runs on — inventing more turns, echoing `User:`/`Assistant:`, or drifting off topic — because the model never sees the `<|im_end|>` it was trained to stop on.

---

## Part C — SFT with LoRA

SFT trains the model to imitate high-quality demonstrations. We use TRL's `SFTTrainer`, which auto-detects the conversational `messages` column, applies the chat template for us, and masks the loss to the assistant turns. LoRA (rank 16) keeps trainable parameters at ~1% of the model.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

sft_ds = load_dataset("trl-lib/Capybara", split="train[:800]")  # conversational: "messages"

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")

sft_args = SFTConfig(
    output_dir="sft-out",
    max_steps=60,                      # cap for ~3 min on T4
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,     # effective batch 8
    learning_rate=2e-4,
    warmup_ratio=0.05,
    logging_steps=10,
    max_length=1024,
    bf16=True,
    seed=42,
    report_to="none",
)

sft_trainer = SFTTrainer(model=base, args=sft_args, train_dataset=sft_ds,
                        peft_config=lora, processing_class=tok)
sft_trainer.train()
sft_trainer.save_model("sft-adapter")
print("SFT done. Trainable params:")
sft_trainer.model.print_trainable_parameters()
```

Watch the loss in the logs drop over 60 steps. That is the model tightening its imitation of the demonstration style — not learning new facts. SFT teaches *format and behavior*; it cannot inject knowledge the base pretraining lacks (the LIMA finding: a few thousand good examples suffice for style, but they don't add capability).

---

## Part D — SFT before/after

Generate on the same fixed prompts with the SFT adapter active and compare side by side.

```python
sft_after = {p: chat(sft_trainer.model, p) for p in EVAL_PROMPTS}

for p in EVAL_PROMPTS[:3]:
    print("PROMPT:", p)
    print("  BASE:", baseline[p][:200])
    print("  SFT :", sft_after[p][:200])
    print()
```

Differences at 0.5B and 60 steps are stylistic, not dramatic — sharper formatting, more consistent tone, fewer trailing digressions. That is what SFT is supposed to do. Do not expect a small-model SFT run to fix reasoning; it changes *how* the model answers, not *what* it knows.

---

## Part E — DPO on preference pairs

DPO skips the separate reward model of classic RLHF. Given a prompt with a **chosen** and a **rejected** response, it directly raises the log-probability of the chosen relative to the rejected, anchored to a reference model by the `beta` KL term. We start from the SFT model (the standard order: DPO refines what SFT produced), merge the SFT adapter into the base, then train a fresh LoRA on preferences.

```python
import gc
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

del sft_trainer, base
gc.collect(); torch.cuda.empty_cache()

# rebuild base, fold in the SFT adapter → this is our starting point for DPO
base2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
sft_model = PeftModel.from_pretrained(base2, "sft-adapter").merge_and_unload()

dpo_ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:600]")  # chosen/rejected

dpo_lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")

dpo_args = DPOConfig(
    output_dir="dpo-out",
    max_steps=60,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,                # DPO wants a much lower LR than SFT
    beta=0.1,                          # KL anchor strength to the reference
    logging_steps=10,
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    seed=42,
    report_to="none",
)

# ref_model=None + peft_config: TRL uses the adapter-disabled model as the reference,
# so we never hold two full copies in memory.
dpo_trainer = DPOTrainer(model=sft_model, ref_model=None, args=dpo_args,
                        train_dataset=dpo_ds, processing_class=tok, peft_config=dpo_lora)
dpo_trainer.train()
print("DPO done.")
```

In the logs, watch `rewards/chosen` rise above `rewards/rejected` and `rewards/margins` grow — that margin is the whole objective. If `rewards/margins` never separates, either the LR is too low or the preference data is too noisy for the signal to show.

---

## Part F — What DPO changed

DPO on `ultrafeedback_binarized` reliably shifts two things: responses get **longer** (the preferred answers in that data tend to be more complete), and the model leans more helpful. Measure the length shift, then read a few generations.

```python
import numpy as np

dpo_after = {p: chat(dpo_trainer.model, p) for p in EVAL_PROMPTS}

def wc(d): return np.array([len(t.split()) for t in d.values()])
print("mean response length (words)")
print(f"  base: {wc(baseline).mean():5.1f}   sft: {wc(sft_after).mean():5.1f}   dpo: {wc(dpo_after).mean():5.1f}")

for p in EVAL_PROMPTS[:2]:
    print("\nPROMPT:", p)
    print("  SFT:", sft_after[p][:220])
    print("  DPO:", dpo_after[p][:220])
```

The length distribution is the fastest tell that DPO did something — and a warning. Length is a well-known confound: preference data often rewards verbosity, so a model can win on the judge by padding without being more correct. Always check length alongside quality, never quality alone.

---

## Part G — A heuristic judged eval

No external API keys here — we score with a transparent, rule-based heuristic. It rewards a sensible length band, penalizes repetition (degenerate loops), and rewards clean sentence endings. It is a *proxy*, not a truth oracle, but it is deterministic and reproducible, which is exactly what a regression gate needs.

```python
def judge(resp: str) -> float:
    words = resp.split()
    n = len(words)
    # length: prefer 15-120 words, decay outside
    length = 1.0 if 15 <= n <= 120 else max(0.0, 1 - abs(n - 60) / 120)
    # repetition: fraction of unique trigrams (1.0 = no repeats)
    tris = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    variety = len(set(tris)) / max(1, len(tris))
    # clean ending
    clean = 1.0 if resp.rstrip().endswith((".", "!", "?", '"', "`")) else 0.5
    return round(0.4 * length + 0.4 * variety + 0.2 * clean, 3)

rows = [("base", baseline), ("sft", sft_after), ("dpo", dpo_after)]
print(f"{'model':>5} | {'mean score':>10} | {'mean words':>10}")
for name, d in rows:
    scores = np.array([judge(t) for t in d.values()])
    print(f"{name:>5} | {scores.mean():>10.3f} | {wc(d).mean():>10.1f}")
```

Read the table honestly. If DPO's score rose only because responses got longer, that is the length confound at work, not real improvement — inspect the generations before you believe the number. This is the core lesson of eval-driven development ([evaluation](10-evaluation.md)): a metric that moves for the wrong reason is worse than no metric.

---

## What you built

- Rendered and dissected a ChatML chat template, and reproduced the template-mismatch bug that silently degrades decoding.
- Ran SFT with `SFTTrainer` + LoRA on an 800-example instruction subset, capped to ~3 min on a T4.
- Compared base vs SFT generations on a fixed prompt set.
- Ran DPO with `DPOTrainer` on preference pairs, starting from the merged SFT model, with an adapter-disabled reference (no second model in memory).
- Measured the DPO response-length shift and scored all three models with a deterministic, keyless heuristic judge — and named the length confound it exposes.

## Exercises

1. **Template ablation.** Regenerate the Part B baseline with `add_generation_prompt=False`. Describe exactly how the outputs change and explain why in terms of what the model thinks its turn is.
2. **Beta sweep.** Rerun DPO with `beta` in {0.01, 0.1, 0.5}. Plot `rewards/margins` and mean response length against beta. Which value best separates chosen from rejected without runaway length growth?
3. **Length-controlled judge.** Add a term to `judge` that neutralizes the length confound (e.g. normalize the score by a length prior, or cap the length reward). Does DPO still beat SFT under the corrected metric?
4. **Data size vs steps.** Hold `max_steps` fixed and vary the SFT subset size (200, 800, 3000 examples). Does more data help at a fixed step budget, or is step count the binding constraint here?
5. **SFT-only vs DPO.** Run DPO directly on the *base* model (skip the SFT stage). Compare against the SFT→DPO pipeline on the heuristic judge and explain why order matters.

## What interviews ask here

- Why do we mask the loss to assistant turns during SFT, and what breaks if we train on the full prompt+response?
- Walk through the DPO loss: what are `beta`, the reference model, and `rewards/margins`, and how does DPO avoid training a separate reward model?
- What is a chat-template mismatch, how does it present at inference (no error, just worse output), and how do you catch it?
- Why does DPO on preference data tend to increase response length, and how would you tell real improvement from the length confound?
- When would you stop at SFT and not run DPO at all? When is the reverse (DPO on a base model) a mistake?
- Why is a deterministic heuristic judge useful even though it's a weak proxy — where does it fit in a production eval loop?
