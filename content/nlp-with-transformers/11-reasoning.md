# 11 — Reasoning Models: CoT, Verifiers, and RL with Verifiable Rewards

A base transformer does a fixed amount of computation per token: one forward pass, `O(1)` depth
regardless of how hard the question is. That is the whole problem with reasoning. "What is 17 × 24?"
and "prove there are infinitely many primes" get the same compute budget if you force a single-token
answer. A "reasoning model" is not a new architecture — it is a model trained to *spend more tokens*
when the problem is hard, because tokens are the only way a decoder-only transformer buys extra serial
computation. This module is the mechanism behind that: why chain-of-thought works at all, how you
supervise it, how test-time compute scales, and the RL recipe (GRPO, DeepSeek-R1) that turned "think
step by step" from a prompting trick into a trained capability.

## What "reasoning" means operationally

Drop the philosophy. Operationally, a reasoning model is one that, before emitting a final answer,
generates a variable-length intermediate sequence — a "chain of thought" (CoT) — and is trained so
that this sequence actually improves the answer. Two things are happening, and it helps to separate
them:

- **Serial compute.** Each generated token is a forward pass conditioned on all previous tokens. A
  200-token CoT is 200 sequential transformer evaluations feeding into the final answer, versus 1 for
  a direct answer. The model has literally bought more depth. A single forward pass cannot multiply
  two 3-digit numbers reliably; a model that writes out the partial products can, because it laid the
  computation across the token axis.
- **Externalized state.** The transformer has no scratch memory between the layers of one forward
  pass — nothing persists except what it writes to the context. The CoT *is* the working memory. This
  is why "let the model show its work" beats "just give the answer": the work is the RAM.

Both framings predict the same thing: reasoning helps most on problems that are *compositional and
serial* (multi-step math, code, logic) and helps little on problems that are essentially lookup
(single-fact recall, sentiment). If a task is one hop, CoT is overhead.

## Chain-of-thought and why it works

Chain-of-thought prompting (Wei et al., 2022) showed that simply prepending a few worked examples
with their reasoning steps — or, for instruction-tuned models, just appending "Let's think step by
step" (Kojima et al., 2022) — lifts accuracy on GSM8K-style math from the teens to the double digits
for models that already have the latent capability. It works for the two reasons above: the model
gets serial compute and a place to store intermediate results.

The critical, non-obvious finding: CoT only helps once the model is large/capable enough that the
steps it writes are *mostly correct*. On a weak model, CoT can hurt — it commits to a wrong first
step and the rest of the chain rationalizes it. This is the seed of everything that follows: the raw
prompting trick is unreliable because there is no pressure keeping the chain *faithful* to a correct
computation. The rest of the module is about adding that pressure.

## Self-consistency: sample many, vote

The cheapest reliability upgrade. Instead of one greedy CoT, sample `k` chains at temperature > 0,
extract the final answer from each, and take the majority vote (Wang et al., 2022):

$$
\hat{y} = \arg\max_{y} \sum_{i=1}^{k} \mathbb{1}[\,a(c_i) = y\,]
$$

where $c_i$ is the $i$-th sampled chain and $a(\cdot)$ extracts its answer. The intuition: there are
many correct reasoning paths to a right answer but idiosyncratic, uncorrelated ways to be wrong, so
correct answers pile up while errors scatter. On GSM8K this typically adds 10+ points over a single
chain. It is embarrassingly parallel and needs no extra training — just `k×` the inference cost. Two
caveats: it only works when the answer is *extractable and comparable* (a number, a label — not a
free-form essay), and returns diminish past `k ≈ 8–16` for most tasks. It is your first, always-cheap
test-time compute lever.

## Process vs outcome supervision

If you want to *train* on reasoning, or *verify* it, you face a choice of what signal to use.

- **Outcome supervision (ORM):** score only the final answer. Cheap to collect (you just need the
  ground-truth answer), but it rewards right answers reached by wrong reasoning — the model can learn
  to guess. The reward is sparse and gives no credit assignment across the chain.
- **Process supervision (PRM):** score each *step* of the reasoning. Lightman et al. (2023,
  "Let's Verify Step by Step") showed a process reward model trained on step-level human labels
  substantially outperforms an outcome model as a *verifier* for reranking sampled solutions on MATH.
  A PRM gives dense signal and catches chains that got lucky. The cost: step-level labels are
  expensive, though later work (Math-Shepherd) generates them automatically by rolling out from each
  step and asking "does continuing from here reach the right answer often?"

The practical takeaway for hiring loops: process supervision produces better *verifiers*; outcome
signal (specifically, *verifiable* outcomes — see below) is what scales *RL training*, because you
can check it programmatically for millions of problems with zero human labeling.

## Test-time compute scaling

You now have several ways to convert extra inference FLOPs into accuracy, and they trade off
differently:

- **Longer single chains** — let the model think more per attempt (serial compute).
- **Best-of-n / sampling + verifier** — sample `n` solutions, pick the one a verifier (PRM or a
  learned/rule-based checker) scores highest. Scales accuracy smoothly with `n`.
- **Self-consistency** — the verifier-free special case (majority vote).
- **Search** — tree search over reasoning steps (MCTS-style) guided by a value/process model, useful
  when steps branch and you can prune.

Snell et al. (2024) framed the key result: for a fixed FLOP budget you can often do *better by
spending it at test time on a smaller model* than by using a bigger model with a single shot — up to
a point. The scaling is real but sub-linear; each doubling of samples buys less. Production framing:
test-time compute is a dial you turn per-request based on difficulty and how much a correct answer is
worth. You do not run best-of-64 on "what's the capital of France."

## RL with verifiable rewards: GRPO

Prompting and reranking help, but the leap to R1-class reasoning came from *reinforcement learning
against a reward you can verify programmatically* — for math, "does the final answer match?"; for
code, "do the unit tests pass?". No reward model to hack, no human labels. The RL algorithm that made
this cheap is **GRPO (Group Relative Policy Optimization)**, introduced with DeepSeekMath.

Standard PPO needs a *critic* (value network) — a second model, roughly the size of the policy — to
estimate the baseline $V(s)$ used to compute advantages. That doubles memory and adds a component
that is itself hard to train. GRPO's insight: for these tasks you do not need a learned value
function. For each prompt, sample a **group** of `G` completions from the current policy, score each
with the verifiable reward $r_i$, and use the **group's own statistics as the baseline**. The
advantage for completion `i` is just its reward standardized within the group:

$$
A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}
$$

Then optimize the usual clipped PPO-style objective with these advantages and a KL penalty to a
reference (the pre-RL model) so the policy does not drift into gibberish:

$$
\mathcal{L}_{\text{GRPO}} = -\,\mathbb{E}\!\left[\, \min\!\big(\rho_i A_i,\ \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\,A_i\big) \,\right] + \beta\, D_{\text{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)
$$

where $\rho_i = \pi_\theta(o_i)/\pi_{\theta_{\text{old}}}(o_i)$ is the importance ratio.

**Why no critic works here:** the baseline exists to reduce variance of the policy gradient. A critic
learns $V(s)$; the group mean is an unbiased Monte-Carlo estimate of that same expected reward under
the current policy, computed for free from samples you already drew. When you have a cheap reward and
can afford `G ≈ 8–16` samples per prompt, the empirical mean is a good enough baseline and you delete
an entire model. That is the whole trick — it is a memory and simplicity win, not a magic new
objective. The cost is the extra sampling (`G` completions per prompt) and higher gradient variance
than a well-fit critic would give.

## DAPO-style stability fixes

Naive GRPO is unstable at scale, and the DAPO work (2025) catalogued the fixes that matter, several
of which you should recognize because they show up as knobs:

- **Decoupled / higher clip ("clip-higher").** Using a larger upper clip bound than lower one lets
  low-probability but promising tokens get reinforced, preventing premature entropy collapse where
  the model becomes deterministic too fast and stops exploring.
- **Dynamic sampling.** Groups where *all* completions get the same reward (all right or all wrong)
  produce $A_i = 0$ for everyone — zero gradient, wasted compute. Filter or resample those prompts so
  every batch carries signal.
- **Token-level policy-gradient loss.** Averaging loss per-sequence lets long completions dominate or
  vanish; normalizing at the token level keeps long CoTs contributing proportionally.
- **Overlong-reward shaping.** Penalize or mask completions that hit the length cap so the model does
  not learn to ramble into the truncation.

The theme: reasoning RL runs collapse in specific, diagnosable ways (entropy death, length blowup,
dead batches), and DAPO is the checklist of guards. If your GRPO run's entropy craters in the first
few hundred steps, clip-higher and dynamic sampling are the first two dials.

## The DeepSeek-R1 recipe

DeepSeek-R1 (2025) is the reference open recipe, and its two-stage structure is the thing to be able
to whiteboard.

**Stage 0 — R1-Zero (RL from the base model, no SFT).** Take the base model, apply GRPO with purely
verifiable rewards (answer-match for math, tests for code) plus a simple format reward for putting
reasoning inside `<think>` tags. With *no supervised reasoning data at all*, long chains-of-thought
emerged on their own: the model spontaneously learned to allocate more tokens to harder problems,
backtrack, and self-verify — the widely-quoted "aha moment" where it writes something like "wait, let
me reconsider." This is the load-bearing scientific result: reasoning behavior is *elicited by the
reward*, not copied from demonstrations. R1-Zero's flaw was readability — it mixed languages and
produced messy, human-unfriendly chains.

**Stage 1 — R1 (cold-start SFT → RL → SFT → RL).** To fix readability while keeping the reasoning,
R1 adds a small "cold-start" SFT on a few thousand curated long-CoT examples to seed a clean format,
then runs the verifiable-reward RL again, then does a round of rejection-sampling SFT (keep the good
chains the RL model generates) mixed with general instruction/safety data, then a final RL pass
covering both reasoning and general preferences. The result matches frontier reasoning models on math
and code while staying readable and generally helpful.

The interview-ready compression: **R1-Zero proves RL-from-base elicits reasoning; R1 wraps that in
SFT for readability and generality.**

## Distilling reasoning to small models

You do not need to run RL to *get* a reasoning model. DeepSeek showed that fine-tuning small dense
models (Qwen/Llama, 1.5B–70B) on ~800K reasoning traces *generated by R1* — plain SFT, no RL —
produces small models that dramatically outperform same-size models trained with RL directly. The
lesson: **RL discovers the reasoning behavior; distillation copies it cheaply.** For most teams the
economical path is to buy or generate traces from a strong reasoning model and SFT your small model
on them, reserving RL for when you are pushing a genuinely new frontier. This mirrors the general
distillation story in [transfer learning](06-transfer-learning-tasks.md) and [post-training](07-post-training.md).

## Overthinking and efficiency

More thinking is not free, and past a point it is not even better. Reasoning models "overthink":
they burn hundreds of tokens on trivial questions ("what is 2+2" with a 300-token derivation),
sometimes talking themselves *out* of a correct early answer. This has direct cost consequences —
reasoning tokens are billed like any other, and a request that thinks for 4,000 tokens costs
roughly `4000/50 = 80×` a 50-token direct answer at the same rate, plus the latency hit. Mitigations
in 2026 practice: length penalties during RL, "thinking budget" controls that cap or toggle reasoning
per request (exposed by several 2026 model APIs), and routing — cheap direct model for easy queries,
reasoning model only for hard ones (tie this to difficulty estimation and the eval gates in
[evaluation](10-evaluation.md)).

## What transfers to your job

The single most important production decision is *when to buy reasoning tokens*. Reasoning models win
clearly on math, code, multi-step tool use, and structured planning; they add cost and latency with
little gain on retrieval, classification, extraction, and chat. So: default to a fast non-reasoning
model, gate reasoning behind a difficulty router, cap thinking budgets, and always A/B the accuracy
lift against the token bill on *your* traffic. For decoding-level control of how these chains are
sampled and served, see [inference and decoding](12-inference-decoding.md).

## What interviews ask here

- Why does chain-of-thought improve accuracy? — It buys serial compute (one forward pass per token)
  and externalizes intermediate state into the context, which a single forward pass cannot hold.
- What is self-consistency and when does it fail? — Sample `k` chains, majority-vote the answer;
  fails when answers are not extractable/comparable, and returns diminish past `k≈8–16`.
- Process vs outcome reward models? — PRM scores each step (dense, better verifier, expensive labels);
  ORM scores only the final answer (cheap, sparse, rewards lucky guesses).
- Why does GRPO drop the critic, and what replaces the baseline? — The group mean of `G` sampled
  rewards is a free Monte-Carlo baseline; you delete a policy-sized value net and save memory.
- What is the R1-Zero vs R1 distinction? — R1-Zero is RL-from-base (reasoning *emerges*, but messy);
  R1 adds cold-start SFT plus alternating SFT/RL for readability and generality.
- Name two GRPO instabilities and their fixes. — Entropy collapse → clip-higher; dead all-same-reward
  batches → dynamic sampling.

## Where this shows up on the job

- Deciding per-request whether to route to a reasoning model or a fast one, and setting/capping
  thinking budgets so a math-agent's token bill does not 80× your direct-answer baseline.
- Building a verifier (rule-based checker or PRM) to rerank sampled solutions for a code or math
  feature — best-of-n with a verifier is often cheaper and better than a bigger model.
- Distilling a strong reasoning model's traces into a small deployable model via SFT, rather than
  running your own RL, when you need the behavior on a budget.
- Diagnosing a reasoning RL run that collapsed (entropy death, length blowup) using the DAPO checklist.
