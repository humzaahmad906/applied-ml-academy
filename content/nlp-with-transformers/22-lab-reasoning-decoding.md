# 22 — Lab 6: Reasoning & Decoding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/nlp-with-transformers/notebooks/22-lab-reasoning-decoding.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup.

Everything in [reasoning](11-reasoning.md) and [inference & decoding](12-inference-decoding.md) is measurable on a 0.5B model. In this lab you watch decoding strategies change the same prompt's output, quantify how chain-of-thought and self-consistency buy accuracy with tokens, try a self-verifier, and time the KV cache doing its job.

**Model choice:** `Qwen/Qwen2.5-0.5B-Instruct`. The 1.5B also fits a T4 in fp16 (~3.1 GB), but it triples generation time and this lab batches up to 200 sampled chains — 0.5B keeps the whole run under the time budget. The 1.5B appears once, as the *target* model in the speculative-decoding bonus, where the 0.5B plays the draft.

## Setup

```bash
pip install -q -U transformers datasets accelerate matplotlib
```

Expected runtime: **15–20 minutes on a free Colab T4** (most of it in Parts B–C). Peak GPU memory ~2 GB; ~5.5 GB if you run the bonus. No API keys, no gated models.

```python
import random, re, time, collections
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"          # decoder-only: pad on the left so generation continues from real tokens
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

# clear Qwen's baked-in sampling defaults (temp 0.7, top_p 0.8, top_k 20) so each experiment is explicit
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None

print(f"{MODEL_ID}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params on {device}")
```

One batched helper does all the generation in this lab:

```python
def chat_generate(prompts, max_new_tokens=256, batch_size=16, **gen_kwargs):
    """Apply the chat template to each user prompt, generate in batches, return completions."""
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id, **gen_kwargs)
        outs.extend(tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    return outs
```

## Part A — Decoding playground

Same prompt, seven decoding configs. Each config sets `temperature`/`top_k`/`top_p` explicitly so nothing leaks between runs.

```python
PROMPT = "Write the opening paragraph of a story about a lighthouse keeper who finds a strange machine."

configs = {
    "greedy":     dict(do_sample=False),
    "temp 0.3":   dict(do_sample=True, temperature=0.3, top_k=0, top_p=1.0),
    "temp 0.7":   dict(do_sample=True, temperature=0.7, top_k=0, top_p=1.0),
    "temp 1.2":   dict(do_sample=True, temperature=1.2, top_k=0, top_p=1.0),
    "top-k 50":   dict(do_sample=True, temperature=1.0, top_k=50, top_p=1.0),
    "top-p 0.9":  dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.9),
    "min-p 0.05": dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=0.05),
}

for name, cfg in configs.items():
    torch.manual_seed(42)
    out = chat_generate([PROMPT], max_new_tokens=80, **cfg)[0]
    print(f"--- {name} ---\n{out.strip()}\n")
```

Read the outputs side by side. Greedy and temp 0.3 are near-identical and safe. Temp 1.2 with no truncation occasionally samples a token from the distribution's junk tail and the text derails from that point — one bad token poisons everything after it. Top-p 0.9 and min-p 0.05 both cut that tail; min-p (Nguyen et al., 2024) scales the cutoff with the top token's probability, so it truncates hard when the model is confident and stays permissive when the distribution is genuinely flat.

**Repetition loops.** Greedy decoding on a raw (non-chat) continuation shows the classic failure — the model falls into an n-gram cycle because the argmax path revisits the same states:

```python
raw = tok("The meaning of life is", return_tensors="pt").to(device)
with torch.no_grad():
    loop = model.generate(**raw, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
text = tok.decode(loop[0], skip_special_tokens=True)
print(text)

grams = [tuple(text.split()[i:i+4]) for i in range(len(text.split()) - 3)]
top_gram, count = collections.Counter(grams).most_common(1)[0]
print(f"\nmost repeated 4-gram: {' '.join(top_gram)!r} x{count}")
```

If your run doesn't loop verbatim, look for phrase-level recycling — the 4-gram counter catches it either way. Re-run with `do_sample=True, temperature=0.7, top_p=0.9` or `repetition_penalty=1.3` and watch the loop break. This is why nobody serves open-ended generation with pure greedy.

## Part B — CoT vs direct answering on GSM8K

Forty test problems, two prompting modes, exact-match on the final number.

```python
from datasets import load_dataset

N = 40
gsm = load_dataset("openai/gsm8k", "main", split=f"test[:{N}]")

def gold_answer(ex):
    return ex["answer"].split("####")[-1].strip().replace(",", "")

def extract_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace("$", ""))
    if not nums:
        return None
    x = nums[-1]
    return x[:-2] if x.endswith(".0") else x     # normalize "42.0" -> "42"

def direct_prompt(q):
    return f"Answer with only the final number. No explanation.\n\nQuestion: {q}\nAnswer:"

def cot_prompt(q):
    return (f"Solve this step by step. End with the final answer on its own line "
            f"as 'Answer: <number>'.\n\nQuestion: {q}")

questions = [ex["question"] for ex in gsm]
golds = [gold_answer(ex) for ex in gsm]

direct_out = chat_generate([direct_prompt(q) for q in questions], max_new_tokens=16, do_sample=False)
cot_out = chat_generate([cot_prompt(q) for q in questions], max_new_tokens=256, do_sample=False)

acc_direct = np.mean([extract_number(o) == g for o, g in zip(direct_out, golds)])
acc_cot = np.mean([extract_number(o) == g for o, g in zip(cot_out, golds)])
print(f"direct: {acc_direct:.1%}   CoT: {acc_cot:.1%}   (n={N})")
```

Expect CoT to land somewhere around 30–50% and direct answering well below it — the gap is the point, not the absolute numbers. CoT works because each generated step is extra serial compute and externalized intermediate state the model conditions on (Wei et al., 2022).

**Honest note on small-model limits.** A 0.5B model is a weak reasoner: it drops carries, misreads quantities, and sometimes writes a correct chain then transcribes the wrong final number. And with n=40, one problem is 2.5 accuracy points — differences smaller than ~10 points here are noise. Treat this as a mechanism demo, not a benchmark. The same harness pointed at a 7B model is a benchmark.

## Part C — Self-consistency: accuracy vs k

Self-consistency (Wang et al., 2023): sample k independent CoT chains at temperature > 0, take the majority-vote answer. Sampling diversifies the *reasoning paths*; correct paths agree on the answer more often than wrong ones do. We sample 8 chains per problem once, then score majority vote on the first k of them for k = 1, 2, 4, 8 — no need to regenerate per k.

```python
K_MAX, N_SC = 8, 25
sc_qs, sc_golds = questions[:N_SC], golds[:N_SC]

torch.manual_seed(42)
prompts = [cot_prompt(q) for q in sc_qs for _ in range(K_MAX)]
chains = chat_generate(prompts, max_new_tokens=256, do_sample=True,
                       temperature=0.7, top_k=0, top_p=0.9)
answers = [[extract_number(chains[i * K_MAX + j]) for j in range(K_MAX)] for i in range(N_SC)]

def majority(ans):
    votes = [a for a in ans if a is not None]
    return collections.Counter(votes).most_common(1)[0][0] if votes else None

avg_tokens = np.mean([len(tok(c).input_ids) for c in chains])
ks = [1, 2, 4, 8]
acc_k = [np.mean([majority(a[:k]) == g for a, g in zip(answers, sc_golds)]) for k in ks]
cost_k = [k * avg_tokens for k in ks]
for k, a, c in zip(ks, acc_k, cost_k):
    print(f"k={k}:  accuracy {a:.1%}   ~{c:.0f} generated tokens/problem")
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(ks, [a * 100 for a in acc_k], color="#4269d0", linewidth=2, marker="o", markersize=7)
ax1.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k (chains sampled)",
        ylabel="accuracy (%)", title="Self-consistency: accuracy vs k")
ax2.plot(ks, cost_k, color="#efb118", linewidth=2, marker="o", markersize=7)
ax2.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k (chains sampled)",
        ylabel="generated tokens / problem", title="Cost vs k")
for ax in (ax1, ax2):
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
```

The two plots together are the test-time-compute tradeoff: accuracy climbs sublinearly (often flattening by k=8 at this scale, and it can be noisy at n=25) while cost climbs exactly linearly. Production systems pick the knee of this curve — and k=2 is pointless, since a 1–1 tie has no majority.

## Part D — Best-of-n with a verifier vs plain majority vote

Majority vote weights every chain equally. A verifier instead *scores* chains and keeps only ones it believes. Here the same model checks its own arithmetic — a self-verifier:

```python
def verify_prompt(q, chain):
    return (f"Question: {q}\n\nProposed solution:\n{chain}\n\n"
            f"Check each arithmetic step of the proposed solution. "
            f"Is the final answer correct? Reply with exactly one word: Yes or No.")

ver_prompts = [verify_prompt(q, chains[i * K_MAX + j])
               for i, q in enumerate(sc_qs) for j in range(K_MAX)]
verdicts = chat_generate(ver_prompts, max_new_tokens=3, do_sample=False)
ok = [v.strip().lower().startswith("yes") for v in verdicts]
print(f"verifier said Yes to {np.mean(ok):.0%} of chains")

bon_correct = 0
for i, g in enumerate(sc_golds):
    kept = [answers[i][j] for j in range(K_MAX) if ok[i * K_MAX + j]]
    pred = majority(kept) if kept else majority(answers[i])   # fall back if nothing verified
    bon_correct += pred == g
print(f"majority vote @8:      {np.mean([majority(a) == g for a, g in zip(answers, sc_golds)]):.1%}")
print(f"verifier-filtered @8:  {bon_correct / N_SC:.1%}")
```

Watch the Yes-rate before you trust the comparison. A 0.5B self-verifier is barely better than a coin flip — it often approves nearly everything, in which case "verifier-filtered" collapses back to plain majority and the two numbers match. That's the finding, not a bug: verification only adds value when the verifier is meaningfully better at *checking* than the generator is at *generating*. This is why production systems train dedicated process reward models rather than asking the policy to grade itself — see [reasoning](11-reasoning.md).

## Part E — KV-cache speed measurement

With `use_cache=True`, each decode step attends over cached K/V and does O(T) work; with `use_cache=False`, every step re-runs the full forward over the whole prefix — O(T²) total.

```python
def timed_generate(use_cache, max_new_tokens=128):
    text = tok.apply_chat_template([{"role": "user", "content": "Explain why the sky is blue, in detail."}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=use_cache, pad_token_id=tok.eos_token_id)
    if device == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = out.shape[1] - enc.input_ids.shape[1]
    return n, dt, n / dt

print(f"{'use_cache':<12}{'tokens':>8}{'seconds':>10}{'tokens/sec':>12}")
for uc in (True, False):
    n, dt, tps = timed_generate(uc)
    print(f"{str(uc):<12}{n:>8}{dt:>10.1f}{tps:>12.1f}")
```

Expect roughly 3–10x at 128 new tokens, and the gap grows with length because the no-cache cost is quadratic. The flip side is memory: the cache for this model is small, but at 70B scale with long contexts and large batches, KV cache — not weights — is what caps serving throughput. That memory math is in [inference & decoding](12-inference-decoding.md).

## Part F (bonus) — Speculative decoding

A small draft model proposes tokens cheaply; the big target verifies them in one batched forward pass and keeps the accepted prefix (Leviathan et al., 2023). Lossless: the output distribution is exactly the target's. Our 0.5B becomes the draft; `Qwen2.5-1.5B-Instruct` (~3.1 GB fp16, same tokenizer family — required) is the target. Skip this cell if you're short on time; it downloads ~3 GB.

```python
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                              torch_dtype=dtype).to(device).eval()
text = tok.apply_chat_template([{"role": "user", "content": "Summarize how photosynthesis works."}],
                               tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt").to(device)

def timed(fn):
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); out = fn()
    if device == "cuda": torch.cuda.synchronize()
    return out, time.perf_counter() - t0

with torch.no_grad():
    plain, t_plain = timed(lambda: target.generate(**enc, max_new_tokens=200,
                                                   do_sample=False, pad_token_id=tok.eos_token_id))
    spec, t_spec = timed(lambda: target.generate(**enc, max_new_tokens=200, do_sample=False,
                                                 assistant_model=model, pad_token_id=tok.eos_token_id))

print(f"vanilla:     {t_plain:.1f}s   speculative: {t_spec:.1f}s   speedup {t_plain / t_spec:.2f}x")
print("outputs identical:", torch.equal(plain[0][:spec.shape[1]], spec[0][:plain.shape[1]]))
```

With greedy decoding the outputs match token for token — that identity check *is* the losslessness claim. The speedup depends on the acceptance rate: on easy, predictable text the draft guesses long runs correctly; on hard text most drafts get rejected and overhead can eat the gain. A 0.5B draft for a 1.5B target is a deliberately modest ratio, so expect ~1.2–1.8x; production pairs (e.g., 1B drafting for 70B) do better.

## What you built

- A decoding playground showing greedy determinism, temperature's risk curve, top-k/top-p/min-p truncation, and a measured greedy repetition loop.
- A GSM8K harness with exact-match scoring: direct vs CoT accuracy on 40 problems.
- Self-consistency with a sample-once-score-many design, plus accuracy-vs-k and cost-vs-k curves.
- Best-of-n with a self-verifier, and evidence for why weak verifiers don't help.
- A KV-cache timing table demonstrating the O(T) vs O(T²) decode gap.
- (Bonus) speculative decoding with a draft/target pair, speedup measured, losslessness verified.

## Exercises

1. Add a `repetition_penalty` sweep (1.0, 1.1, 1.3, 1.5) to the Part A loop prompt. At what value does the loop break, and what does 1.5 do to fluency?
2. In Part C, replace plain majority vote with a vote weighted by each chain's mean token log-probability (`output_scores=True` in `generate`). Does weighting beat counting at k=8?
3. The Part B extractor takes the *last* number in the output. Log the failure cases: how many CoT chains computed the right answer but got scored wrong because of extraction? Write a better extractor and re-measure.
4. Rerun Part C at temperature 0.3 and 1.0 (same k values). Which temperature gives the best k=8 accuracy, and why does too-low temperature defeat self-consistency?
5. In Part F, vary `max_new_tokens` (50, 200, 400) and prompt difficulty. When does speculative decoding's speedup grow, and can you make it *slower* than vanilla?

## What interviews ask here

- Why does greedy decoding loop, and name three mitigations with their tradeoffs. (Argmax revisits high-probability n-gram states; sampling, repetition penalty, no-repeat n-grams — each trades determinism or fluency.)
- Top-p vs min-p: what problem does min-p solve that top-p doesn't? (Fixed cumulative mass keeps too many tokens when the model is confident and too few when it's uncertain; min-p scales the cutoff with peak probability.)
- Why does self-consistency improve accuracy, and what's the cost model? (Independent sampled reasoning paths make errors that are less correlated than the correct answer; cost is exactly linear in k while accuracy is sublinear — pick the knee.)
- When does best-of-n with a verifier beat majority vote? (Only when the verifier's judgment is better than chance in a way uncorrelated with the generator's errors — self-verification with the same small model usually isn't.)
- Walk through why KV caching turns O(T²) decode into O(T) per step, and what it costs. (Cache K/V per layer per head; memory = 2 x layers x heads x head_dim x seq x bytes x batch — the serving bottleneck at scale.)
- Why is speculative decoding lossless, and what determines its speedup? (Rejection sampling against the target's distribution guarantees identical outputs; speedup = f(acceptance rate, draft/target cost ratio).)
