# 12 — Alignment: From Base Model to Assistant

A pretrained base model is a next-token predictor. It will happily continue your prompt in the
style of the internet, which is not the same as answering your question, following instructions,
or reasoning to a correct result. Alignment is the set of post-training methods that turn a base
model into something usable. This is the last unit, it spans three big ideas (RLHF, RLVR, and the
policy-gradient implementation), and it is the one moving fastest, so treat the specifics
as current-as-of-now and the principles as durable.

## The three-stage picture

Post-training is usually staged:

1. **Supervised fine-tuning (SFT):** teach the format and the behavior by imitation.
2. **Preference optimization (DPO / RLHF):** teach the model to prefer better responses over worse
   ones, using human or AI preference data.
3. **Reinforcement learning with verifiable rewards (RLVR, via GRPO and relatives):** teach the
   model to actually get checkable things right (math, code, structured tasks) by rewarding correct
   outcomes.

Not every model uses all three, and the order and emphasis vary, but this is the common spine, and
it is exactly what the alignment build in this course walks through: SFT first, then expert
iteration, then GRPO, all on Qwen2.5-Math-1.5B against grade-school (GSM8K) and competition (MATH)
math. The through-line is that every stage is the *same* objective viewed
differently — maximize expected reward under a KL leash to the model you started from — and the
methods differ only in where the reward comes from and how you estimate its gradient.

## Supervised fine-tuning

The simplest and most important stage. You collect a dataset of prompt-response pairs that
demonstrate the behavior you want (an instruction and a good answer, a chat turn and a good reply)
and fine-tune the base model on them with the same next-token objective, but computing loss only on
the response tokens, not the prompt. The model learns the assistant format, the chat markers, and a
huge amount of behavior just from imitation.

### Loss masking and chat templates

The two mechanical details that matter, and that the build makes you implement by hand:

**Prompt masking.** During collation you build a `response_mask` that is 0 over the prompt tokens
and 1 over the response tokens. The SFT loss is a masked per-token negative log-likelihood — you
compute log-probs over the whole sequence, then average the NLL only over the masked (response)
positions:

$$
\text{loss} = - \frac{\sum_t \text{mask}_t \cdot \log p_\theta(y_t \mid y_{<t})}{\sum_t \text{mask}_t}
$$

Masking the prompt matters: you do not want to spend capacity learning to *generate* the user's
question, only to generate the answer given it. In the microbatch train step this is a
`masked_normalize` over the response mask, giving the mean NLL per response token, then the loss is
scaled by `1 / gradient_accumulation_steps` before `.backward()` so that a micro-batch of 2 with 64
accumulation steps simulates an effective batch of 128.

**Chat templates.** The prompt-response pair is wrapped in the model's chat format (role markers,
turn delimiters, and — for reasoning models — structural tags). The build uses an R1-Zero-style
template that asks the model to emit its reasoning inside `<think>...</think>` and its final answer
inside `<answer>...</answer>`. The template is not cosmetic: the reward function later parses these
tags, so SFT is also teaching the model the exact structure the reward will grade.

### The ceiling of SFT

SFT gets you most of the way to a usable assistant. Its ceiling is that it can only imitate the
demonstrations, and it cannot easily learn to prefer a good answer over a plausible-but-worse one
because it never sees the worse one contrasted with the better. It also tends to make the model
overconfident and can teach it to imitate the surface form of good answers without the substance.
That is what the next stages address. For many practical products, especially narrow ones like
structured document extraction, well-done SFT on high-quality domain demonstrations is enough, and
this is the regime most of your work lives in.

## Preference optimization: RLHF and DPO

To go past imitation you use comparisons. You collect preference data: for a given prompt `x`, two
responses `y_w` (preferred/"winner") and `y_l` (dispreferred/"loser"), judged by humans or an AI
judge. Then you optimize the model to make preferred responses more likely and dispreferred ones
less likely.

### Classic RLHF: reward model then PPO

**Step 1 — reward model.** Train a scalar reward model `R(x, y)` on the preference data under the
**Bradley-Terry** model, which says the probability a human prefers `y_1` over `y_2` is the sigmoid
of the reward difference:

$$
P(y_1 \succ y_2 \mid x) = \sigma\!\left( R(x, y_1) - R(x, y_2) \right)
$$

The reward model is trained by maximum likelihood on this — equivalently, minimizing
`− log σ(R(x, y_w) − R(x, y_l))` over the preference pairs. It is usually the same transformer with
a scalar head replacing the vocab projection.

**Step 2 — RL against the reward, on a KL leash.** Optimize the policy `π` to maximize expected
reward while staying close to the frozen SFT reference `π_ref`. This is the KL-regularized
objective that unifies the whole unit:

$$
\max_\pi \; \mathbb{E}_{x,\, y \sim \pi}\!\left[ R(x, y) \right] - \beta \cdot \mathrm{KL}\!\left( \pi(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x) \right)
$$

Classically this outer optimization is done with **PPO**: sample responses from the current policy,
score them with `R`, estimate a per-token advantage with a learned value network as the baseline,
and take a clipped policy-gradient step (the clip keeps the importance ratio `π/π_old` inside
`1 ± ε` so one batch cannot move the policy too far). It works — it is what made the first
generation of chat assistants good — but it is heavy: you are training and serving a separate
reward model *and* a separate value network *and* running a finicky on-policy RL loop.

### DPO: the same objective without the RL

**DPO (Direct Preference Optimization)** is the simplification that made preference tuning
accessible. The derivation is the payoff here: the KL-regularized objective above has a
*known closed-form optimal policy*,

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp\!\left( R(x, y) / \beta \right)
$$

Invert this to write the reward that any policy implicitly optimizes,
$R(x, y) = \beta \log\!\left( \pi(y \mid x) / \pi_{\text{ref}}(y \mid x) \right) + \beta \log Z(x)$, and substitute it into the Bradley-Terry
loss. The intractable partition function `Z(x)` cancels because Bradley-Terry only ever sees reward
*differences* between two responses to the same prompt. You are left with a plain supervised loss on
preference pairs, no reward model and no RL loop:

$$
L_{\text{DPO}} = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \cdot \left( \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]
$$

Read it as: increase the log-ratio of the winner relative to the reference, decrease it for the
loser, and `β` controls how hard you are allowed to pull away from the reference (it is exactly the
KL coefficient from the original objective). DPO removes the separate reward model precisely because
the policy *is* the reward model — the implicit reward is `β log π/π_ref`. It is far simpler and more
stable to run than PPO and reaches comparable quality on many tasks, which is why DPO (or a variant
like IPO, KTO, ORPO) is the default preference method for most practitioners now.

### The KL term is the whole game

Whether you run PPO or DPO, the KL-to-reference term is what keeps the model honest. Without it,
optimization "reward-hacks": it finds degenerate outputs the reward model scores highly but humans
hate. The KL penalty anchors the model near its SFT starting point. Set `β` too low and you get
reward hacking and mode collapse; too high and the model barely moves off the SFT policy. It is the
main knob in the entire alignment pipeline.

## RL with verifiable rewards: policy gradient, expert iteration, GRPO

The newest and most consequential shift (RLVR). For domains where
correctness is checkable — math with a known answer, code that passes tests, structured extraction
you can validate — you do not need a learned reward model at all. The reward is objective: did the
answer match, did the tests pass. You run reinforcement learning directly against that verifiable
reward, and because the reward is trustworthy you can push hard on it without the reward-hacking
that plagues learned rewards.

### The policy-gradient foundation

Everything here is one estimator. The REINFORCE gradient of expected reward is

$$
\nabla_\theta \mathbb{E}[R] = \mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(y \mid x) \cdot R(x, y) \right]
$$

— reinforce the tokens of high-reward responses, in proportion to the reward. Raw rewards give a
high-variance, always-positive signal, so you subtract a **baseline** `b` and reinforce the
*advantage* `A = R − b` instead. It is worth walking through the variants directly: using
the batch mean as the baseline (`delta = reward − batch_mean`, "centered rewards") or a running
global mean (`delta = reward − baseline_mean`, "normalized rewards"). Centering is the theoretically
grounded choice — it approximates the true advantage `A(s,a) = Q(s,a) − V(s)` — and gives negative
updates to below-average responses, zero update when all responses in a group tie, and empirically
faster convergence than raw-reward weighting.

### Expert iteration (the build's middle stage)

The simplest way to turn a verifiable reward into training signal, no policy gradient required:

1. **Generate.** Sample several responses per prompt from the current model (use
   vLLM for fast batched rollouts, 4 samples per prompt).
2. **Filter.** Score each with the reward function and keep only the trajectories where the reward
   equals 1 — i.e. the ones that reached the correct answer with the right format.
3. **Fine-tune.** Run plain SFT (the same masked-NLL step from above) on those "expert"
   trajectories.
4. **Repeat.** Loop for a few iterations; a good default is 5 EI steps, 100 SFT steps each.

This is rejection-sampling fine-tuning, also called STaR. It is easier than full RL, often captures
much of the benefit, and is the sane thing to try before reaching for GRPO. The model bootstraps: as
it gets better, more of its samples pass the filter, so later iterations train on harder problems it
could not solve before.

### GRPO (the build's final stage)

**GRPO (Group Relative Policy Optimization, from DeepSeek)** is the algorithm that made online RLVR
practical and cheap, and it is the core of the build. PPO needs a
separate value network to estimate the baseline; GRPO removes it. For each prompt it samples a
**group** of `G` responses (use group size 8), scores them all, and uses the
group's own statistics as the baseline. The advantage for response `i` in group with rewards
`{r_1..r_G}` is the group-normalized reward:

$$
A_i = \frac{ r_i - \operatorname{mean}(r_1 \dots r_G) }{ \operatorname{std}(r_1 \dots r_G) + \varepsilon }
$$

Every token in response `i` gets this same scalar advantage. The objective is the PPO-style
per-token clipped surrogate, with the importance ratio between the current policy and the policy
that generated the rollouts:

$$
\begin{aligned}
\text{ratio}_t &= \pi_\theta(y_t \mid \cdot) \,/\, \pi_{\theta_{\text{old}}}(y_t \mid \cdot) \\
L_{\text{GRPO}} &= - \mathbb{E}\!\left[ \min\!\left( \text{ratio}_t \cdot A_i, \; \operatorname{clip}(\text{ratio}_t, 1-\varepsilon, 1+\varepsilon) \cdot A_i \right) \right]
\end{aligned}
$$

with `ε = 0.2` (`cliprange`). The clip is what lets you take several gradient steps on the same
batch of rollouts without the policy running away from the distribution it sampled from. The
build has you implement three modes to see the effect: **naive** (raw policy gradient, no
importance ratio), **unclipped** (importance-weighted but no clip), and **clipped** (the full
GRPO-Clip objective, the default). A subtle but load-bearing implementation detail is computing
`π_θ_old` and the reference log-probs under `torch.no_grad()` so their gradients do not leak into
the policy update. Loss is applied per token and gradient-accumulated (rollout batch 256, 128
accumulation steps) exactly as in SFT.

Compared to PPO, GRPO drops the memory and complexity of the value network and is a natural fit for
the verifiable-reward setting, where a group of samples for the same problem gives you a clean,
cheap baseline for free. This is the "run RL on Qwen Math to improve MATH" move that drove the
2024–2025 reasoning-model boom.

## Reward shaping and reward hacking

The reward function is where the domain knowledge lives, and getting it right is its own topic.
For math reasoning, the `r1_zero_reward_fn` is a two-part reward, exactly the
DeepSeek-R1-Zero recipe:

- **Format reward:** did the output contain the required structure — reasoning inside
  `<think>...</think>` and the final answer inside `<answer>...</answer>`? This is a cheap regex
  check.
- **Answer reward:** parse the content of `<answer>` and check correctness against ground truth with
  a math-aware comparison (normalized string / symbolic equality, not naive string match).

The full reward is 1 only when both format and answer are correct; expert iteration keeps exactly
those. Giving format credit at all is deliberate reward shaping — it hands the model a gradient early,
while it is still learning to structure output, before it can reliably get answers right.

Two failure modes worth stressing:

- **Reward hacking / length bias.** Any proxy reward can be gamed. The classic RLVR failure is
  length: models learn that longer chains correlate with correctness and inflate their reasoning,
  or exploit a lenient answer parser. On a *learned* reward this is fatal (the model finds outputs
  the RM loves and humans hate); on a *verifiable* reward it is bounded but still shows up as
  degenerate padding, which is one reason the KL leash and clip matter even here.
- **Variance.** Policy-gradient signal is noisy, so the group-relative baseline in GRPO and the
  centered/normalized rewards above are not optional niceties — they are what make the training
  converge at all rather than thrash.

## Why verifiable rewards changed everything

The reason RLVR matters so much is that it breaks the dependence on human-labeled or learned
rewards, which are expensive, biased, and hackable. Where you can check the answer, you have an
unlimited, incorruptible reward signal, and the model can improve by practicing against it far
past the level of its demonstrations. This is why the frontier of reasoning is concentrated in
checkable domains: math, code, and formal tasks. The open research problem is extending it to
domains without a clean verifier, where you are back to learned rewards or clever proxies. For your
work, the lesson is direct: any part of your task that has a checkable answer (does the extracted
field match the ground truth, does the structured output parse and validate) is a candidate for
this kind of outcome-driven training, not just imitation.

## Safety alignment

RLHF and DPO are also the mechanism for *safety*, not just helpfulness — the same preference
machinery that teaches "concise and correct over verbose and correct" teaches "refuse to help with
X" and "do not produce Y." The preference dataset carries the values; the optimizer just enforces
them. This is why the two are usually blended (helpful-and-harmless preference data) rather than run
as separate stages, and why the KL leash matters for safety too: push the policy too far off the
reference chasing a reward and you can knock out safety behaviors that were only lightly reinforced.
Instruction tuning, safety alignment, and the classic RLHF pipeline are the "learned-reward"
complement to the verifiable-reward core.

## Practical guidance for a narrow product model

You are usually not building a general assistant. For a narrow, high-accuracy task model:

- Start with strong SFT on high-quality domain demonstrations, with the prompt masked. This alone is
  often 90% of the win for extraction-style tasks and is what your Qwen fine-tunes are doing.
- Add preference tuning (DPO) only if you have a quality axis SFT cannot capture (preferring
  concise correct output over verbose correct output, say). Prefer DPO over PPO unless you have a
  specific reason to run online RL — you avoid a reward model, a value network, and a fragile loop.
- Reach for expert iteration or GRPO when your task has a verifiable reward and SFT has plateaued,
  which is exactly the case for structured extraction where you can score field-level correctness.
  Generate multiple extractions, keep the ones that validate, fine-tune on those — that is expert
  iteration adapted to your domain, and it is strictly simpler than GRPO. Graduate to GRPO only when
  you want the online, on-policy improvement past what filtered SFT gives.
- Design the reward as format-plus-correctness and watch for hacking (a lenient validator is a
  reward-hacking invitation). Keep `β`/the KL leash and the clip in place even with a verifiable
  reward.
- Keep the held-out evaluation clean throughout, because alignment stages are very
  easy to overfit to whatever you are optimizing against.

## Key takeaways

Post-training turns a base next-token predictor into an assistant in stages, and every stage is the
same KL-regularized reward-maximization objective seen from a different angle. **SFT** is masked-NLL
imitation on response tokens (prompt masked, chat template teaching the very structure the reward
later grades); it teaches format and behavior and is often enough for narrow tasks. **Preference
optimization** learns to prefer better over worse: classic RLHF trains a Bradley-Terry reward model
`σ(R(x,y_w) − R(x,y_l))` then runs PPO under a `β·KL` leash, while **DPO** proves the optimal policy
has closed form `π* ∝ π_ref·exp(R/β)`, inverts it, and collapses the whole pipeline into one stable
pair loss `−log σ(β·(log π/π_ref for winner − for loser))` with no separate reward model. **RLVR**
runs policy gradient `∇E[R] = E[∇log π · A]` directly against a *verifiable* reward: **expert
iteration** filters correct rollouts and re-SFTs on them, and **GRPO** samples a group, sets the
advantage to the group-normalized reward `A_i = (r_i − mean)/(std + ε)`, and optimizes the clipped
per-token surrogate `min(ratio·A, clip(ratio, 1±ε)·A)` — dropping PPO's value network and driving
the reasoning-model boom. The reward is format-plus-correctness (R1-Zero style), the KL/clip leash
prevents reward hacking, and verifiable rewards are the key unlock because they give an unlimited,
unhackable signal wherever you can check the answer — exactly the regime much of your
structured-extraction work lives in.

## You can now

You can now:

- implement masked SFT — a `response_mask` that zeros the prompt tokens so the NLL is averaged only over response positions — and explain why the chat template teaches the exact structure the reward later grades.
- derive the DPO loss from the KL-regularized objective: the closed-form optimum $\pi^* \propto \pi_{\text{ref}} \cdot \exp(R/\beta)$, the reward inversion, and why the partition function $Z(x)$ cancels under Bradley-Terry.
- articulate why the $\beta \cdot \mathrm{KL}$ leash is the central knob in the whole pipeline — too low invites reward hacking and mode collapse, too high pins the policy to SFT.
- turn a verifiable reward into training signal three ways — expert iteration (rejection-sampling SFT), and GRPO with the group-normalized advantage $A_i = (r_i - \operatorname{mean})/(\operatorname{std} + \varepsilon)$ and the clipped per-token surrogate — and say why GRPO drops PPO's value network.
- design a format-plus-correctness reward and anticipate its failure modes (length inflation, lenient-parser exploits, advantage collapse when a group ties), and choose the right rung of the ladder for a narrow product model.

## Try it

Implement the GRPO advantage-and-loss step from scratch against a tiny fixture, no RL library. Write `grpo_advantages(rewards)` that group-normalizes a vector of `G` rewards, then `grpo_loss(logp_policy, logp_old, advantages, clip)` that forms the ratio $\pi_\theta / \pi_{\theta_{\text{old}}}$, applies the clipped surrogate $\min(\text{ratio} \cdot A, \operatorname{clip}(\text{ratio}, 1 \pm \varepsilon) \cdot A)$, and averages per token. Verify three things by hand on toy tensors: (1) a group where every reward ties gives ~zero advantage and thus no gradient (advantage collapse); (2) `logp_old` is detached so no gradient leaks into the rollout policy; (3) with the ratio pinned at 1 the clipped and unclipped losses agree. Then flip one reward and confirm the sign of the update moves the right response up. Getting the `no_grad` and the `min` (not `max`) right is exactly what the graded `test_grpo` suite checks.
