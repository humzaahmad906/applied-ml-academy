# 09b — Distillation: Training a Small Model to Imitate a Large One

In the previous chapter we made a fixed model cheaper to serve — quantizing its weights,
paging its KV cache, batching its requests. Distillation attacks the same cost problem from the
other end: instead of squeezing the bytes of a large model, you train a genuinely *smaller* model to
reproduce the large one's behavior. The large model is the **teacher**, the small one is the
**student**, and the whole game is to transfer as much of the teacher's competence as possible into a
network with a fraction of the parameters. Nearly every small open model you can name today — the
DeepSeek-R1 distilled Qwen and Llama checkpoints, Gemma, the compact members of the Qwen and Llama
families — is a distilled artifact, not a from-scratch pretrain. It has become the standard recipe
for producing a small model that punches above its parameter count, so a foundation-model engineer
needs to know it cold.

## Why imitate the distribution instead of the labels

The founding insight (Hinton, Vinyals, Dean, 2015) is that a trained teacher's output distribution
carries far more information than the one-hot label it was trained against. Ask a good model to
predict the next token and it does not merely spike on the correct token; it assigns a graded,
structured distribution over the whole vocabulary — some small mass to plausible synonyms, a little
to grammatically-valid alternatives, almost none to garbage. Hinton called this the **dark
knowledge**: the *relative* probabilities of the wrong answers encode the model's learned similarity
structure. "The capital of France is ___" puts most mass on *Paris* but leaves a telltale trace on
*Lyon* and *France*, and essentially zero on *carburetor*. A one-hot label throws all of that away
and says only "Paris, everything else equally wrong."

For a student, the soft distribution is a much richer training signal than the hard label. Each
token becomes a dense regression target over the whole vocabulary instead of a single class index,
so one distilled token teaches the student roughly as much as many hard-labeled tokens would. That
is the entire reason distillation works: the student is fitting the teacher's *function*, and the
function's shape is visible in the soft outputs. The practical consequence is data efficiency —
students reach teacher-like quality on far fewer tokens than pretraining from scratch would need,
because each token is information-dense and already denoised by a competent teacher.

## The classic recipe: temperature, softmax, and KL

The mechanics turn on the **softmax temperature** `T`. The ordinary softmax over logits $z$ is
$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Dividing the logits by a temperature before the softmax,

$$
p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}},
$$

flattens the distribution as `T` rises. At `T = 1` you get the model's native distribution; at
`T > 1` the probabilities spread out, amplifying exactly the small-but-nonzero masses on the
runner-up tokens that hold the dark knowledge. Without a raised temperature those masses are so
close to zero they contribute almost nothing to the gradient; softening surfaces them. Typical
values are `T` in the range 2–5, with `T = 2` a reasonable default.

The distillation loss then asks the student's softened distribution to match the teacher's softened
distribution, measured by KL divergence, and adds an optional ordinary cross-entropy term against the
ground-truth label when labels exist:

$$
\mathcal{L} = \alpha \, T^2 \cdot \mathrm{KL}\!\left(p^{\text{teacher}}(T) \,\|\, p^{\text{student}}(T)\right) \;+\; (1-\alpha)\,\mathrm{CE}\!\left(y,\, p^{\text{student}}(1)\right).
$$

Two details in that formula earn their keep. The soft term is evaluated at temperature `T`; the hard
term at `T = 1` against the real label. And the soft term carries a factor of `T²`. That factor is
not cosmetic: the gradients of the soft cross-entropy scale as $1/T^2$ (each logit is divided by `T`
inside the softmax, and this shows up squared in the gradient), so multiplying by `T²` keeps the
soft and hard gradient magnitudes comparable as you tune `T`. Forget it and your loss weighting
silently changes every time you change the temperature.

For a language model the same recipe applies **per token position**. The logits have shape
`(batch, seq_len, vocab)`, and you distill the teacher's vocabulary distribution into the student's
at every position, masking out padding. In PyTorch:

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, targets, mask, T=2.0, alpha=0.5):
    # student_logits, teacher_logits: (B, S, V) ; targets: (B, S) ; mask: (B, S) bool
    V = student_logits.size(-1)

    # soft targets: both distributions softened by the same temperature T
    s_logp = F.log_softmax(student_logits / T, dim=-1)     # log-probs (KL input)
    t_prob = F.softmax(teacher_logits / T, dim=-1)         # probs     (KL target)

    # forward KL(teacher || student) per token, summed over the vocab
    kd = (t_prob * (t_prob.clamp_min(1e-9).log() - s_logp)).sum(-1)  # (B, S)
    kd = (kd * mask).sum() / mask.sum() * (T * T)

    # hard cross-entropy against ground truth, at T = 1
    ce = F.cross_entropy(
        student_logits.view(-1, V), targets.view(-1),
        reduction="none",
    ).view_as(targets)
    ce = (ce * mask).sum() / mask.sum()

    return alpha * kd + (1.0 - alpha) * ce
```

(You can equivalently use `F.kl_div(s_logp, t_prob, reduction="none")`, remembering it expects
log-probabilities as the first argument and probabilities as the second; the manual form above makes
the temperature and masking explicit.) This is **white-box, logit-level, off-policy** distillation:
you need the teacher's full logits (white-box), you match the whole distribution (logit-level), and
the training tokens come from a fixed corpus the student did not generate (off-policy). It is the
baseline every other variant is measured against.

### Forward KL vs reverse KL

Which direction of the KL you minimize matters more for language models than the classification
literature suggests. **Forward KL**, $\mathrm{KL}(p^{\text{teacher}} \| p^{\text{student}})$, is
*mass-covering*: it penalizes the student for putting low probability anywhere the teacher puts mass,
so the student tries to cover every mode of the teacher — including modes a small student cannot
represent well, which spreads its probability into low-quality regions and can produce incoherent
generations. **Reverse KL**, $\mathrm{KL}(p^{\text{student}} \| p^{\text{teacher}})$, is
*mode-seeking*: it concentrates the student's mass on the teacher's dominant modes and does not
punish ignoring minor ones. MiniLLM (Gu et al., 2024) showed that reverse KL gives noticeably better
generation quality for LLM distillation precisely because a capacity-limited student is better off
faithfully reproducing the teacher's high-probability behavior than smearing itself thin trying to
cover the tail. Reverse KL has since become the default objective for the generative case, and — as
we'll see — it is what makes on-policy distillation work.

## Sequence-level distillation

The per-token recipe matches distributions at teacher-provided prefixes. **Sequence-level
distillation** (Kim & Rush, 2016) takes a coarser, and often more practical, approach: let the
teacher *generate* complete outputs (greedily or by beam search), then train the student with plain
cross-entropy to reproduce those generated sequences. There is no temperature and no soft
distribution — the teacher's generated text is treated as a hard target, so this is also called
**hard-label distillation**.

Two things make it attractive. First, it targets what the teacher would actually *say*, the mode of
its output distribution, rather than its per-token uncertainty at ground-truth prefixes; the student
learns to produce coherent teacher-like sequences rather than to mimic token-level hedging. Second,
and decisively for practice, it requires nothing but the teacher's *text output*. You do not need
its logits or internals. That makes it the only option when the teacher is a closed API — the
**black-box** setting we return to below — and it is why so much modern small-model training reduces
to "generate a big pile of outputs from a strong teacher, then fine-tune the student on them." The
distilled DeepSeek-R1 checkpoints were built this way: pure supervised fine-tuning on teacher-generated
sequences, no logit matching at all.

## On-policy distillation

Both recipes so far are **off-policy**: the training sequences come from the teacher (or a fixed
corpus), not from the student. This creates the same *exposure bias* / train–test mismatch that has
haunted sequence models since the RNN days. The student is only ever trained on states drawn from
the teacher's distribution — perfect prefixes — but at inference it must condition on *its own*
generations, which drift into regions the teacher never visited and the student was never trained on.
One early mistake compounds: the student lands in an unfamiliar state, has no learned behavior there,
and the errors snowball.

**On-policy distillation** fixes the mismatch by training on the student's own rollouts. The loop is:

1. The **student generates** a trajectory by sampling from its current policy.
2. The **teacher scores** that trajectory — it evaluates the tokens the student actually produced.
3. The student is updated to move toward the teacher on exactly the states it visits.

The elegant formulation, popularized by Google's GKD (Agarwal et al., 2024) and sharpened in
Thinking Machines Lab's October 2025 write-up (Kevin Lu), is to use the teacher's per-token
log-probability as a **dense reward** and optimize a per-token **reverse KL** against it. Concretely,
for each student-sampled token the teacher supplies $\log \pi_{\text{teacher}}(a_t \mid s_t)$, and the
student takes a policy-gradient-style step that rewards tokens the teacher finds probable and
penalizes the "forking tokens" that sent the reasoning astray. This is mathematically an
RL objective — KL-constrained policy optimization with the teacher's log-probs as the reward — but it
is far cheaper and more stable than reward-model RL, because the reward is dense (a signal on *every*
token, not one scalar at the end) and, in the reverse-KL/mode-seeking sense, "unhackable": from the
teacher's view a low KL always corresponds to the desired behavior having high probability, so there
is no degenerate reward to exploit.

A minimal white-box on-policy step, given full teacher access:

```python
# 1. student generates its own trajectory (on-policy)
with torch.no_grad():
    seq = student.generate(prompt, do_sample=True, max_new_tokens=T_gen)

# 2. score every generated token under both models
student_logp = F.log_softmax(student(seq).logits, dim=-1)      # differentiable
with torch.no_grad():
    teacher_logp = F.log_softmax(teacher(seq).logits, dim=-1)

# 3. per-token reverse KL(student || teacher) over the vocab, on visited states
p_s = student_logp.exp()
per_token_rkl = (p_s * (student_logp - teacher_logp)).sum(-1)   # (B, S)
loss = (per_token_rkl * gen_mask).sum() / gen_mask.sum()        # mask to generated span
```

The payoff is large. Because the student practices on its own error states and gets a dense
corrective signal, on-policy distillation is reported to be roughly **9–30× more compute-efficient**
than off-policy distillation for reaching a given quality, and it has moved from research curiosity
to standard post-training ingredient: Qwen3, MiMo, and the GLM series all fold on-policy distillation
into their pipelines. The cost is that you must run the student's generation loop inside training
(slow, sequential decode — see lesson 09) and keep the teacher resident to score, so a step is more
expensive than a static forward pass even though you need far fewer steps overall.

## Reasoning distillation

The highest-profile application in 2025–2026 is distilling a **reasoning** teacher's chain-of-thought
into a smaller student. A large reasoning model like DeepSeek-R1 produces long, deliberate CoT traces
— thousands of tokens of intermediate work before the answer. Distillation transfers that behavior:
generate a large set of R1's reasoning traces on math, code, and logic problems, then fine-tune a
smaller dense model on them (sequence-level, hard-label SFT). DeepSeek open-sourced exactly this — 
1.5B, 7B, 8B, 14B, 32B, and 70B students built on the Qwen2.5 and Llama3 backbones — and reported that
DeepSeek-R1-Distill-Qwen-32B outperformed OpenAI's o1-mini across several reasoning benchmarks.
The striking finding was that distilling R1's traces into a small dense model beat running
large-scale RL directly on that same small model: it is easier to *imitate* a discovered reasoning
pattern than to *rediscover* it from sparse reward. Reasoning ability, it turns out, distills
remarkably well because the CoT trace makes the teacher's problem-solving process explicit in the
token stream, so the student has a fully-worked demonstration to fit rather than just an answer.

The trade-offs are real. Distilled reasoners inherit the teacher's verbosity — they emit long CoT
even on easy questions, inflating inference cost (which lands you right back in the decode-bound
economics of lesson 09), and much recent work targets compressing that. They also inherit the
teacher's mistakes and biases wholesale, and they are ceiling-limited by the teacher: pure imitation
does not exceed the source, so distillation and RL are increasingly *combined* (distill to bootstrap
a competent reasoner, then RL to push past the teacher). Gemma's recent releases likewise use
distillation as a first-class part of both pre- and post-training rather than an afterthought,
signaling that distillation is now baked into how frontier-adjacent small models are made, not just a
compression pass at the end.

## White-box vs black-box distillation

How much of the teacher you can see determines which recipe is even available.

**White-box** distillation assumes full access to the teacher's logits and internal states. You can
match the whole per-token distribution (logit-level KL), run on-policy scoring with exact teacher
log-probs, and even align intermediate representations. This is the setting when you own both models
or use open-weight teachers.

**Black-box** distillation has only the teacher's *text* output through an API — GPT-5, Gemini 2.5,
and other closed models. You cannot see logits (at best a few top-k log-probs), so token-level
divergence matching is off the table, and you are forced to the **sequence level**: sample outputs
from the API and fine-tune the student on them with cross-entropy. Most "distill a small model from a
frontier API" workflows are black-box sequence-level distillation by necessity. (Note the licensing
and terms-of-service dimension here — several providers restrict using their outputs to train
competing models — which is a real constraint on black-box distillation, not just a technical one.)
Recent work is even pushing on-policy ideas into the black-box regime, scoring student rollouts with
only the teacher's text or top-k signal, but the clean per-token reverse-KL story requires white-box
access.

## Where distillation sits among compression methods

Distillation, quantization, and pruning are the three pillars of model compression, and they are
**complementary**, not competing — each attacks a different axis:

- **Quantization** (lesson 09) reduces the *bytes per parameter*. Same architecture, same parameter
  count, fewer bits each. Directly buys decode speed because decode is memory-bound on weight reads.
- **Pruning** removes *parameters* — whole neurons, heads, or layers (structured) or individual
  weights (unstructured, e.g. SparseGPT, Wanda). Fewer parameters to store and, for structured
  pruning, to compute.
- **Distillation** produces a *different, smaller network* trained to behave like the large one. It is
  the only one of the three that lets you change the architecture wholesale — a 70B teacher into a
  7B student with a different depth and width — and the only one that transfers *behavior* rather than
  approximating the existing weights.

Because they operate on different axes they stack. A common production pipeline runs them in the order
**prune → distill → quantize**: prune the large model to trim redundant structure, distill the pruned
(or original) model into a compact student to recover and re-concentrate the capability, then quantize
the student's weights for the final memory and decode-speed win. The distillation step is frequently
what *recovers* the accuracy that aggressive pruning or low-bit quantization sacrificed — you quantize
or prune, then distill the full-precision original into the compressed model to heal the damage. Order
matters empirically (studies find the P→KD→Q sequence robust), and the mental model to carry is:
distill to change *what* the model is, quantize and prune to change *how cheaply* you can run it.

## Practical pitfalls

**The capacity gap.** A student far smaller than its teacher may simply lack the representational room
to fit the teacher's function, and distillation quality *degrades* when the gap is too large — the
student cannot track a distribution it has no capacity to represent. The classic mitigation is a
**teaching assistant**: an intermediate-size model distilled from the teacher, then used as the
teacher for the small student, so no single hop crosses too large a gap. Reverse KL also helps here,
since a mode-seeking student that cannot cover the whole teacher distribution is better off committing
to the dominant modes than smearing itself thin.

**Tokenizer mismatch.** Logit-level distillation assumes the teacher and student share a vocabulary —
you cannot compare a teacher's 50k-dim logit vector to a student's 32k-dim one, and even where the
vocabularies overlap, differing tokenization of the same string misaligns the positions you are trying
to match. If you feed student-tokenized text to a teacher with a different tokenizer, the teacher's
log-probs are computed on a segmentation it finds unnatural, in low-probability regions, and the
distillation signal is garbage. This rules out logit-level and clean on-policy distillation across
tokenizer boundaries; you fall back to sequence-level (which only needs text), or use one of the
recent **cross-tokenizer** methods that distill through a shared byte-level interface or align
distributions by approximate likelihood matching rather than direct logit comparison.

**Distribution coverage.** The teacher's soft labels are only informative on inputs the teacher
actually saw. Off-policy distillation trains on a fixed corpus, so wherever the student later strays —
its own error states — there is no teacher signal, which is the exposure-bias failure on-policy
distillation exists to fix. Even in the on-policy setting, if your prompts do not cover the deployment
distribution the student will be undertrained exactly where it matters. Coverage of the *student's own*
state distribution, not just the teacher's, is what determines whether the distilled model is robust in
production.

## Key takeaways

Distillation trains a small student to imitate a large teacher, and it works because the teacher's soft
output distribution carries "dark knowledge" — the relative probabilities of wrong answers — that a
one-hot label discards, making each distilled token an information-dense regression target. The classic
white-box recipe softens both distributions with a temperature `T`, minimizes their KL (remembering the
`T²` factor that rebalances the gradient), and optionally adds hard-label cross-entropy; for generative
LLMs, reverse KL (mode-seeking) beats forward KL (mass-covering) because a capacity-limited student
should commit to the teacher's dominant modes. Sequence-level distillation trains on the teacher's
generated *text* with cross-entropy, needs no logits, and is the only option for black-box API teachers.
On-policy distillation trains on the *student's own* rollouts scored by the teacher — dense per-token
reverse-KL reward — which cures the exposure bias of off-policy training and is 9–30× more
compute-efficient, now standard in Qwen3, MiMo, and GLM. Reasoning distillation transfers long CoT
traces from a reasoning teacher (DeepSeek-R1 into Qwen and Llama students) and can beat running RL on the
small model directly, at the cost of inherited verbosity and a teacher-imposed ceiling. Distillation is
complementary to quantization and pruning — it changes *what* the model is while they change how cheaply
it runs — and the three stack, often prune → distill → quantize. Watch the capacity gap (use a teaching
assistant), tokenizer mismatch (fall back to sequence-level or a byte-level interface), and coverage of
the student's own state distribution.

## You can now

- explain why a teacher's soft distribution is a richer training signal than hard labels, and what "dark knowledge" means.
- write the temperature-scaled KL distillation loss, justify the `T²` gradient-rebalancing factor, and implement it per-token over a vocabulary in PyTorch.
- choose between forward KL (mass-covering) and reverse KL (mode-seeking) and say why reverse KL suits capacity-limited generative students.
- distinguish logit-level, sequence-level, off-policy, and on-policy distillation, and explain why on-policy cures exposure bias and is dramatically more compute-efficient.
- describe reasoning distillation (DeepSeek-R1 into smaller Qwen/Llama students), why it can beat direct RL on the small model, and its verbosity and ceiling trade-offs.
- place distillation alongside quantization and pruning as complementary compression, and reason about a prune → distill → quantize pipeline.
- diagnose the three main failure modes: capacity gap, tokenizer mismatch, and distribution coverage.

## Try it

Take an open-weight teacher (say a 7–8B instruct model) and a smaller student sharing its tokenizer. First run **off-policy** logit distillation: collect a corpus, compute the teacher's logits, and train the student with the temperature-KL loss above; evaluate on a held-out reasoning or QA set. Then run **sequence-level** distillation: have the teacher generate completions, fine-tune the student on them with plain cross-entropy, and compare. Finally, if you have the compute, wire up the **on-policy** loop — student generates, teacher scores per-token reverse KL — and confirm it reaches the off-policy quality in far fewer gradient steps. Watching the same student improve fastest under on-policy training makes the exposure-bias argument concrete, and comparing sequence-level against logit-level shows you exactly what the soft distribution was buying you.
