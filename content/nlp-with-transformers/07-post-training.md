# 07 — Post-Training: Turning a Base Model into an Assistant

A base model is not a chatbot. Pretraining (see [pretraining](05-pretraining.md)) gives you a
next-token predictor over web text: prompt it with "The capital of France is" and it completes
"Paris," but prompt it with "What is the capital of France?" and it may just as happily continue
with three more quiz questions, because that is what the training distribution rewards. The model
knows an enormous amount; it has no idea it is supposed to *answer you*. Post-training is the
process that installs that idea — first by showing the model what good answers look like (SFT),
then by optimizing it against a signal for what humans actually prefer (RLHF, DPO, and friends).
This is where a raw pretrained model becomes GPT-4o, Claude, or Llama-3-Instruct, and it is the
part of the stack most likely to come up when you interview for an LLM role.

## Base vs assistant: what changes

The weights that make an assistant are a thin layer on top of a base model — post-training touches
a tiny fraction of the compute pretraining used, often under 1%. What it changes is *behavior*, not
knowledge. An assistant model has learned three things a base model lacks: to treat a turn structure
(user says X, assistant responds) as the governing frame; to answer helpfully in a preferred style
and length; and to refuse or hedge on requests it should not satisfy. None of this adds facts. The
recurring interview trap is thinking RLHF "teaches the model things" — it re-weights behaviors the
base model can already produce. If the base model cannot do a task at all, post-training will not
conjure the capability.

## Supervised fine-tuning (SFT)

SFT is ordinary next-token training on curated examples of the behavior you want. The data is a set
of (prompt, response) pairs, and the crucial detail is *format*: they are serialized through a **chat
template** that marks roles with special tokens, e.g.

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

The loss is standard cross-entropy over tokens, but **masked to the assistant turn only** — you do
not want the model spending gradient learning to generate the user's questions. Getting this mask
wrong (training on the whole sequence, or applying a different template at inference than at training)
is the single most common SFT bug, and it shows up as a model that ignores the system prompt or
degenerates. The template is part of the model contract; the tokenizer ships it (`apply_chat_template`)
and you must use the exact one the checkpoint expects.

What SFT can teach: format, tone, instruction-following, tool-call syntax, refusal patterns — any
behavior you can demonstrate. What it cannot teach well: *preferences between two plausible
responses*. SFT only ever sees the "correct" answer; it has no signal that response A is better than
response B when both are fluent. It also suffers from **exposure bias** — trained only on gold
continuations, it never learns to recover from its own mistakes. That gap is exactly what preference
optimization fills.

## The RLHF pipeline

Reinforcement Learning from Human Feedback (Christiano et al. 2017; Ouyang et al. 2022, "InstructGPT")
adds a second stage that optimizes against *comparative* human judgment. It has three parts: collect
preference data, fit a reward model, then optimize the policy against that reward with a KL leash.

### Preference data and the Bradley-Terry reward model

You show annotators a prompt and two candidate responses, and they pick the better one. This gives
you triples $(x, y_w, y_l)$ — prompt, winning ("chosen") response, losing ("rejected") response.
Pairwise comparisons are used because humans are far more reliable at *ranking* two answers than at
assigning an absolute score.

To turn rankings into a scalar reward, you fit a **reward model** $r_\phi(x, y)$ — usually the SFT
model with the LM head replaced by a scalar head — under the **Bradley-Terry** model of pairwise
preference. Bradley-Terry says the probability that $y_w$ beats $y_l$ is a logistic function of the
reward difference:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
= \frac{1}{1 + e^{-(r_\phi(x, y_w) - r_\phi(x, y_l))}}
$$

You train $\phi$ by maximizing the log-likelihood of the observed preferences, i.e. minimizing

$$
\mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)}
\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big].
$$

Only reward *differences* matter, so the scale is arbitrary (a reward model is identified up to an
additive constant per prompt). This is why you never read a raw reward value as "quality" — it is
only meaningful relative to another response on the same prompt.

### PPO with a KL anchor

Now optimize the policy $\pi_\theta$ (the language model) to produce responses the reward model
scores highly. Left alone, this maximization is dangerous: the policy will drift far from fluent
language to chase whatever quirk inflates the reward. The fix is a **KL penalty** anchoring the
policy to the frozen SFT reference $\pi_{\text{ref}}$. The objective is

$$
\max_{\theta}\;\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot\mid x)}
\Big[\, r_\phi(x, y) \;-\; \beta\, \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \Big].
$$

The second term is a per-token KL divergence from the reference; $\beta$ controls how far the policy
may roam. Small $\beta$ → higher reward but more drift and reward hacking; large $\beta$ → stays
close to SFT but barely improves. This objective is optimized with **PPO** (Proximal Policy
Optimization; Schulman et al. 2017), which treats each generated token as an action, uses the
KL-shaped reward as the return, and clips the policy update so no single step moves the policy too
far. PPO needs four models in memory at once — policy, reference, reward model, and a value/critic
head — which is why RLHF is operationally heavy and why the field has been hunting for lighter
alternatives.

### Reward hacking

The reward model is a *proxy* for human preference, not the real thing, and any proxy can be gamed.
Optimize hard enough and the policy finds inputs where $r_\phi$ is high but true quality is not — the
classic symptom is **length bias** (annotators mildly prefer longer, more thorough-looking answers,
so the reward model learns "longer = better," so PPO produces rambling responses), plus sycophancy,
over-formatting, and confident-sounding wrong answers. This is Goodhart's law: when a measure becomes
a target, it stops being a good measure. The KL anchor slows it; it does not stop it. In practice you
detect reward hacking by watching mean response length and KL climb together while human-judged
quality flattens or drops, and you mitigate with length-normalized rewards, better annotation, and
early stopping.

## DPO: preference optimization without the RL

Direct Preference Optimization (Rafailov et al. 2023) removes the reward model and the RL loop
entirely, and it now handles most applied preference tuning. The derivation is worth knowing because
interviewers love it. Start from the same KL-constrained objective above. It has a known
closed-form optimum — the policy that maximizes reward-minus-KL is

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y\mid x)\,
\exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big),
$$

a reference distribution reweighted by exponentiated reward, with partition function $Z(x)$. Solve
this for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x).
$$

The key move: substitute this expression into the Bradley-Terry likelihood. Because Bradley-Terry
depends only on the reward *difference* between two responses to the same prompt, the intractable
$\beta \log Z(x)$ term is identical for both and **cancels**. What is left is a loss you can optimize
directly on preference pairs with no reward model at all:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}
\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
- \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].
$$

Read it plainly: raise the policy's log-prob on the chosen response and lower it on the rejected one,
each measured relative to the frozen reference, with $\beta$ setting how hard. The language model *is*
its own implicit reward model. DPO needs only two models in memory (policy and reference), no
sampling loop, and trains like supervised learning — which is why it democratized RLHF. The mechanics
and code live in the sibling course: [preference tuning](../fine-tuning-llms/05-preference-tuning.md).

**DPO vs RLHF, when to pick which.** DPO is simpler, cheaper, more stable, and the default. Classic
PPO-RLHF can still edge it out when you can *reuse a strong reward model across many training rounds*,
when you want online exploration (the policy generates fresh samples that get scored, versus DPO's
fixed offline pairs), or at the largest frontier scale where labs invest in the reward-model
infrastructure. GRPO and other online RL variants (covered in [reasoning](11-reasoning.md)) matter
specifically when rewards are *verifiable* — math, code — rather than learned from preferences.

## Rejection sampling and best-of-n

There is a middle path that is cheaper than PPO and often surprisingly strong. **Best-of-n** sampling:
at inference, generate $n$ candidates and return the one the reward model scores highest — no training
at all, just spend compute at test time. **Rejection-sampling fine-tuning** (used heavily in the
Llama-2/3 recipes) turns that into a data engine: sample many completions from the current model,
keep only the top-scored ones, and SFT on them. You get much of RLHF's benefit with a plain SFT loop
and no unstable RL. Many production pipelines run several rounds of this before, or instead of, DPO.

## Data quality beats quantity

The most-cited empirical result in post-training is LIMA (Zhou et al. 2023, "Less Is More for
Alignment"): a strong base model fine-tuned on just **1,000 carefully curated** examples produced a
capable assistant, rivalling models tuned on orders of magnitude more data. The interpretation — the
"superficial alignment hypothesis" — is that the base model already learned the knowledge and the
distribution of good responses during pretraining; SFT mostly teaches it *which* sub-distribution to
speak from. A thousand clean, diverse, high-quality examples move that pointer better than a hundred
thousand noisy ones. On the job this flips the instinct: when your fine-tune underperforms, audit and
prune the data before you add more. Duplicates, format inconsistencies, and a handful of wrong labels
do more damage than a smaller-but-clean set. Diversity of instructions matters more than raw count.

## Safety tuning, refusals, and open recipes

Refusal behavior is trained, not intrinsic: safety data (harmful request → appropriate refusal or
safe completion) is mixed into SFT and preference stages. The tension is real — push refusals too hard
and you get **over-refusal** (the model declines benign requests that merely pattern-match to unsafe
ones), too soft and it complies with genuinely harmful ones. This is a preference-optimization problem
like any other, tuned and evaluated against safety benchmarks, and it is why the same base model can
ship with very different safety profiles.

For years these recipes were opaque. That changed with fully open efforts — AllenAI's **Tülu** and
**OLMo** lines publish the data mixtures, code, and preference sets end to end, and are the best way
to see a real modern post-training pipeline (SFT → DPO → verifiable-reward RL) without reverse-engineering
a closed model. If you want to *run* one of these, the [SFT-DPO lab](20-lab-sft-dpo.md) walks a small
model through it on a free GPU.

## What interviews ask here

- Why can't SFT alone produce a good assistant — what does preference optimization add? (Comparative
  signal between plausible responses; SFT only ever sees one gold answer and suffers exposure bias.)
- Write the Bradley-Terry reward-model loss and explain why only reward *differences* are identified.
- Why does RLHF need a KL penalty against the reference model? (Prevent reward hacking / distribution
  drift; $\beta$ trades reward gain against staying fluent and on-distribution.)
- Sketch the DPO derivation — where does the reward model go, and why does $\log Z(x)$ cancel?
- What is reward hacking and how do you detect it? (Proxy Goodharting; watch length + KL rise while
  judged quality flattens; length-normalize, early-stop.)
- DPO vs PPO vs rejection sampling — when is each the right call?

## Where this shows up on the job

- Building an in-house assistant: you almost always run SFT (+ DPO) on an open base rather than PPO;
  getting the chat template and loss masking right is the first thing that breaks.
- Curating and cleaning preference/instruction data is where most of the quality gain actually comes
  from — LIMA's lesson turned into daily practice.
- Diagnosing a "verbose, sycophantic, or over-refusing" fine-tuned model means recognizing reward-model
  or preference-data pathologies, not just twiddling generation settings.
- Choosing DPO over a full RLHF stack is the default cost/stability call for any team without
  reward-model infrastructure.
