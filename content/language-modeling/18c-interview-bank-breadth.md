# 18 — Interview Bank IV: Breadth & Rapid-Fire — Part 3 of 4: Sampling, long context & alignment mechanics

This is part 3 of 4 of the Interview Bank IV: Breadth & Rapid-Fire lesson. Here we cover sampling and
decoding strategies, the mechanics of serving and extending long context, and the mechanics of
alignment — SFT masking, reward models, PPO's KL leash, DPO, and RLVR/reward hacking.

---

## Part G — Sampling & decoding

**G1. Greedy versus sampling — when does each make sense?**

Greedy decoding takes the argmax token at every step: deterministic, reproducible, and best when there
is one correct continuation — code, math, structured extraction, anything you will verify against
ground truth. Its weakness is that it is repetitive and bland on open-ended text and can get stuck in
loops, because always taking the single most likely token collapses diversity and can walk into
degenerate high-probability cycles. Sampling draws from the model's distribution, which gives variety
and more human-like text for creative and conversational generation, at the cost of determinism and
occasional low-probability mistakes. The rule of thumb: greedy (or low-temperature) for tasks with a
right answer, sampling for tasks where diversity is the point — and for RL rollouts and self-consistency
you *want* sampling, because you need multiple distinct completions per prompt.
*Probes: matching the decode strategy to whether the task has a verifiable answer.*

**G2. Explain temperature, top-k, top-p, and min-p, and how they interact.**

All reshape the next-token distribution before sampling. **Temperature** `T` divides the logits before
softmax: `T < 1` sharpens (more greedy, safer), `T > 1` flattens (more diverse, riskier), `T → 0` is
greedy. **Top-k** keeps only the `k` highest-probability tokens and renormalizes — a fixed-count
truncation. **Top-p (nucleus)** keeps the smallest set of tokens whose cumulative probability exceeds
`p` and renormalizes — an *adaptive* truncation that keeps few tokens when the model is confident and
many when it is unsure, which is why it usually beats top-k. **Min-p** keeps tokens whose probability
is at least `min_p × (max token probability)` — scaling the threshold to the peak, so it is
permissive when the distribution is flat and strict when there is a clear favorite. They *compose* and
order matters: you typically truncate first (top-k/top-p/min-p) then apply temperature to what
remains, and stacking an aggressive temperature on top of a wide nucleus can reintroduce the junk the
nucleus was meant to cut.
*Probes: the mechanics of each truncation and that they combine, with top-p's adaptivity as the key insight.*

**G3. What do repetition and frequency penalties do, and when do they backfire?**

They fight the degenerate looping that greedy and low-temperature decoding fall into. A **repetition
penalty** divides (or subtracts from) the logit of any token that has already appeared, making it less
likely to be repeated. A **frequency penalty** scales the penalty by *how many times* the token
appeared, and a **presence penalty** applies a flat penalty once a token has appeared at all. They
backfire when the task legitimately requires repetition: code has many repeated tokens (`for`, `=`,
indentation, variable names), structured output repeats keys and delimiters, and a language with
limited vocabulary naturally reuses words — an aggressive penalty there degrades correctness or
produces contorted phrasing as the model avoids the natural token. So they are a creative-text tool;
turn them off (or way down) for code and structured extraction.
*Probes: knowing the penalties and that they are harmful on repetitive-by-nature tasks.*

**G4. What is beam search and why do LLMs rarely use it?**

Beam search keeps the `b` highest-probability *partial sequences* at each step, expanding all of them
and pruning back to the top `b`, to approximate the globally most-likely sequence rather than the
greedy locally-most-likely one. It was standard in machine translation, where there is a single best
faithful output and a slightly higher-probability full sequence is genuinely better. LLMs rarely use
it for two reasons. First, on open-ended generation, maximizing sequence probability produces bland,
repetitive, degenerate text — the highest-probability continuation is often the most generic one, and
beam search *amplifies* that pathology. Second, it is expensive (`b` parallel hypotheses) and interacts
badly with sampling. For open-ended text, nucleus sampling gives better output than chasing the mode;
for verifiable tasks, greedy or best-of-n sampling with a verifier beats beam search. Beam survives
mainly in constrained/structured decoding where the objective really is the most probable valid string.
*Probes: that "most probable sequence" is the wrong objective for open-ended LM generation.*

**G5. How do you make sampled generation reproducible, and why might it still drift?**

Fix the random seed for the sampler so the same logits produce the same draws, and pin temperature,
top-k/top-p, and any penalties — with a fixed seed and greedy (`T=0`) decoding you should get identical
output run to run. But it can still drift for reasons *outside* the sampler: floating-point
non-associativity means that changing the batch size, sequence padding, kernel, GPU model, or library
version reorders the reductions inside matmuls and softmax, producing slightly different logits, and
near a tie those tiny differences flip the argmax and the whole continuation diverges. Continuous
batching makes this worse because a request's effective batch composition varies with what else is
in flight. So bit-exact reproducibility across hardware/engine versions is not guaranteed even at
`T=0`; for true determinism you fix the seed *and* pin the batch shape, kernels, and versions, or
accept run-to-run variation as a property of the serving stack.
*Probes: seeding for the sampler plus the deeper floating-point/batching source of non-determinism.*

---

## Part H — Long context

**H1. Why does naively feeding a longer context to a model trained on short context fail?**

Two failures. First, **positional**: RoPE encodes position as a rotation angle `θ_{i,k} = i / Θ^{...}`
that grows with absolute position `i`. A model trained to 4k has only ever seen rotation angles up to
that range, so at position 20k the query/key rotations are in a regime it never learned — the relative-
position signal the attention dot product depends on is out of distribution, and quality collapses.
Second, **the KV cache**: cost grows linearly with sequence length, so at long context one sequence's
cache can exceed the model weights (a 30-layer, 32-KV-head, d_head-128 bf16 model spends ~491 KB per
token — ~4 GB at 8k, ~10 GB at 32k), which caps batch size and can OOM you before quality even matters.
So you cannot just pass more tokens; you have to *extend* the position encoding (interpolation, below)
and *manage* the cache (GQA/MLA, KV quantization, local attention). Long-context
models "bump `Θ` or interpolate positions" as a fine-tuning-time trick on the same RoPE mechanism.
*Probes: naming both the out-of-distribution position problem and the KV-memory wall.*

**H2. What is position interpolation, and how do NTK-aware / YaRN scaling improve on it?**

The RoPE angle grows with position, so to stretch a 4k model to 32k you can **interpolate positions**:
scale every position index down by the extension factor (here 8×) so position 32k maps back into the
0–4k *angle* range the model was trained on — trading angular resolution for reach — then briefly
fine-tune. Plain linear interpolation works but uniformly compresses all frequencies, blurring the
high-frequency (fine-grained, local) position signal the model relies on for nearby tokens.
**NTK-aware scaling** fixes this by changing the RoPE base `Θ` instead of the positions, scaling
frequencies *non-uniformly* — stretch the low-frequency (long-range) components a lot and the
high-frequency (local) components little, so local precision is preserved while range extends.
**YaRN** refines this further with a frequency-dependent interpolation schedule (per-band ramp) plus
an attention-temperature adjustment, reaching longer contexts with less fine-tuning and less quality
loss than either plain interpolation or naive NTK. The through-line: extend where you can afford to
lose resolution (long range), preserve it where you cannot (local).
*Probes: interpolation as trading angular resolution for reach, and why NTK/YaRN scale frequencies non-uniformly.*

**H3. What is the needle-in-a-haystack eval, and what does it and its variants actually test?**

You plant a specific fact (the "needle" — e.g. a random sentence with a magic number) at a controlled
depth inside a long distractor context (the "haystack"), then ask a question only answerable from the
needle, sweeping both context length and needle depth. It tests *retrieval* over long context: can the
model actually attend to and use information anywhere in the window, or does it only see the start and
end (the "lost in the middle" failure, where accuracy sags for needles buried mid-context)? Its limit
is that single-needle retrieval is *easy* — a model can pass it while still failing real long-context
reasoning — so harder variants (multi-needle, needle requiring aggregation across several planted
facts, or reasoning over the retrieved content) are used to distinguish "can find one fact" from "can
reason over the whole context." Passing single-needle is necessary, not sufficient, evidence of usable
long context.
*Probes: knowing the eval construction and that it measures retrieval, not long-context reasoning.*

**H4. What are attention sinks and streaming attention, and what problem do they solve?**

Serving an effectively infinite stream (a long chat) means you cannot keep the whole KV cache — it
grows without bound — so the obvious move is a sliding window that evicts the oldest tokens. Naively
doing that *tanks* quality, and the reason is the **attention sink**: models learn to dump excess
attention probability onto the very first few tokens (softmax must sum to 1, so when no later token
deserves the mass it parks it on the initial tokens as a no-op). Evict those initial tokens and the
softmax distribution is corrupted, so quality collapses. **StreamingLLM's** fix is to *always retain*
the first few "sink" tokens' KV plus a sliding window of recent tokens, discarding the middle. This
keeps the sink the model relies on while bounding the cache, giving stable generation over arbitrarily
long streams without fine-tuning — though note it *forgets* the evicted middle, so it enables endless
*streaming*, not true long-context *recall*.
*Probes: the counterintuitive sink phenomenon and that streaming ≠ long-context recall.*

**H5. At long context, what dominates inference cost — and which levers attack it?**

The **KV cache**, not the weights. It grows linearly with sequence length and batch ($2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot L_{\text{seq}} \cdot \text{batch} \cdot \text{bytes}$), so at long context a single sequence's cache can exceed
the model weights and, across concurrent users, becomes the binding memory constraint that caps batch
size — and since decode is memory-bandwidth-bound, reading that cache also eats bandwidth. The levers
all attack terms in that formula. **GQA/MQA** shrink `n_kv_heads`, cutting the cache by the query-to-KV
ratio. **MLA** compresses the per-token key/value into a small latent, a larger constant-factor
reduction. **Local (sliding-window) attention** interleaved with occasional global layers makes most
layers' cache independent of sequence length. **KV-cache quantization** to int8/int4 attacks the
bytes-per-element term directly. **PagedAttention** does not shrink the cache but eliminates
fragmentation so more of it fits. At long context, KV-cache economics *is* the inference-design problem.
*Probes: that KV cache, not weights, dominates at long context, and mapping each lever to the formula.*

---

## Part I — Alignment mechanics

**I1. Why does SFT mask the prompt, and what exactly is the masked loss?**

You fine-tune on prompt-response pairs with the next-token objective, but you compute loss *only on
the response tokens*. Mechanically you build a `response_mask` that is 0 over prompt tokens and 1 over
response tokens, then average the per-token NLL only over the masked positions:
$\text{loss} = - \sum_t \text{mask}_t \cdot \log p_\theta(y_t \mid y_{<t}) \,/\, \sum_t \text{mask}_t$. The reason: you do not want to spend model
capacity learning to *generate the user's question* — that is input you will always be given, not
behavior you want to produce — only to generate the answer *conditioned on* it. Training on the prompt
tokens would waste gradient on modeling the instruction distribution and can actively hurt. This pairs
with **chat templates**: the pair is wrapped in the model's role markers and turn delimiters, and for
reasoning models structural tags like `<think>...</think>` / `<answer>...</answer>` — which is not
cosmetic, because the reward function later *parses those tags*, so SFT is teaching the exact structure
the reward will grade.
*Probes: the masked-NLL form and *why* the prompt is excluded, plus the template-reward link.*

**I2. How is a reward model trained from preference pairs?**

From comparisons. You collect, for a prompt `x`, a preferred response `y_w` and a dispreferred `y_l`
(judged by humans or an AI), and train a scalar reward model `R(x, y)` under the **Bradley-Terry**
model, which says the probability a human prefers `y_1` over `y_2` is the sigmoid of the reward
difference: $P(y_1 \succ y_2 \mid x) = \sigma(R(x, y_1) - R(x, y_2))$. You fit it by maximum likelihood —
equivalently minimizing $-\log \sigma(R(x, y_w) - R(x, y_l))$ over the pairs. Architecturally it is usually
the same transformer with a scalar head replacing the vocab projection. The crucial property is that
Bradley-Terry only ever sees reward *differences* between two responses to the same prompt, so the
absolute scale of `R` is unidentified — which is exactly the property DPO later exploits to cancel the
intractable partition function.
*Probes: Bradley-Terry as sigmoid-of-difference and the difference-only property.*

**I3. In PPO-style RLHF, what is the KL-to-reference term for, and what breaks without it?**

The objective is $\max_\pi \mathbb{E}_{y \sim \pi}[R(x,y)] - \beta \cdot \mathrm{KL}(\pi(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x))$: maximize reward while staying
close to the frozen SFT reference. The KL term is the leash. Without it, optimization **reward-hacks**
— it finds degenerate outputs the reward model scores highly but humans hate, drifting arbitrarily far
from sensible language because a *learned* reward model is only accurate near the distribution it was
trained on and is exploitable off it. The KL penalty anchors the policy near its SFT starting point
where the reward model is trustworthy. `β` is the main knob in the whole pipeline: too low and you get
reward hacking and mode collapse; too high and the model barely moves off the SFT policy. It also
matters for *safety* — push the policy too far chasing reward and you can knock out lightly-reinforced
refusal behaviors.
*Probes: KL as the anti-reward-hacking anchor and `β` as the central knob.*

**I4. DPO versus PPO — what does DPO trade, and when would you still run PPO?**

DPO exploits that the KL-regularized objective has a *closed-form* optimal policy
$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp(R(x,y)/\beta)$. Invert it to express the implicit reward as
$\beta \log(\pi/\pi_{\text{ref}}) + \beta \log Z(x)$, substitute into the Bradley-Terry loss, and the intractable partition
`Z(x)` cancels (Bradley-Terry sees only differences) — leaving a plain supervised loss on preference
pairs, no reward model and no RL loop:
$L_{\text{DPO}} = - \log \sigma(\beta \cdot (\log \pi(y_w)/\pi_{\text{ref}}(y_w) - \log \pi(y_l)/\pi_{\text{ref}}(y_l)))$. The policy *is* the reward model.
DPO trades away PPO's separate reward model, value network, and finicky on-policy loop for a stable,
simple offline loss that reaches comparable quality on many tasks — the default now. You still reach
for PPO (or online RL) when you need *on-policy* improvement — generating fresh samples, scoring them,
and learning from the model's *own current* outputs — which DPO's fixed offline preference set cannot
give, and which matters when you want the model to explore past the demonstrated/preferred data.
*Probes: the DPO derivation at a high level and the offline-vs-on-policy tradeoff.*

**I5. What is RLVR, and why did verifiable rewards change the field?**

RLVR is reinforcement learning with **verifiable rewards**: where correctness is *checkable* — math
with a known answer, code that passes tests, structured output you can validate — you skip the learned
reward model entirely and reward the objective outcome directly (did the answer match, did the tests
pass). It changed everything because it breaks the dependence on human-labeled or learned rewards,
which are expensive, biased, and *hackable*. A verifiable reward is unlimited and incorruptible, so you
can push *hard* on it without the reward-hacking that plagues learned rewards, and the model can
improve by practicing far past the level of its demonstrations ("SFT memorizes, RL generalizes"). This
is why the reasoning frontier concentrated in checkable domains — math, code, formal tasks — and drove
the 2024–2025 reasoning-model boom via GRPO. The open problem is extending it to domains *without* a
clean verifier, where you are back to learned rewards or clever proxies. For any task with a checkable
answer (does the extracted field match ground truth), this is the move.
*Probes: verifiable = unhackable = can push hard, and the "checkable domains" restriction.*

**I6. What is reward hacking, and how does it differ between learned and verifiable rewards?**

Reward hacking is the policy finding outputs that score high under the *proxy* reward while failing the
*true* objective. On a **learned** reward it is fatal and unbounded: the reward model is only accurate
near its training distribution, so optimization drifts off into degenerate outputs the RM loves and
humans hate — this is precisely what the KL leash exists to prevent. On a **verifiable** reward it is
bounded (the model *does* have to produce a correct answer to get the reward) but it still shows up as
exploiting the *shape* of the reward: the classic case is **length** — models learn longer chains
correlate with correctness and inflate reasoning with degenerate padding — or exploiting a *lenient
answer parser* that accepts near-misses. That is why even with a verifiable reward you keep the KL
leash and the clip, design the reward as format-*plus*-correctness with a *strict* validator (a lenient
validator is a reward-hacking invitation), and watch for length blowup and padding.
*Probes: that verifiable rewards bound but don't eliminate hacking, and the length/lenient-parser failure modes.*

## You can now

- Match a decoding strategy to a task and explain how temperature, top-k/top-p/min-p, and repetition penalties reshape the next-token distribution.
- Explain why beam search is a poor fit for open-ended LM generation, and how to make sampled generation reproducible (and why it can still drift).
- Explain why long context breaks a short-context model (RoPE out-of-distribution, KV-cache growth) and how interpolation/NTK/YaRN extend it.
- Explain what needle-in-a-haystack and attention sinks/StreamingLLM each test, and why the KV cache dominates long-context inference cost.
- Explain SFT's masked loss, reward-model training via Bradley-Terry, PPO's KL leash, the DPO derivation, and RLVR/reward hacking.
