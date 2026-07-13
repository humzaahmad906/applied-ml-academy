# 13 — Interpretability: Reading What the Model Is Doing

A transformer is a program you never wrote. It has billions of parameters, no comments, and behavior
that emerges from data rather than from anything you specified. When it misclassifies a support
ticket, leaks a fact it shouldn't know, or refuses a benign request, you can't set a breakpoint and
step through — the "logic" is smeared across matrices. Interpretability is the toolkit for reading
that program anyway: figuring out *what* a model computes and *why*, at a level you can act on. This
is not an academic luxury. It's how you debug an eval regression that has no code cause, how you
justify a model to a risk committee, and increasingly how you *steer* behavior without retraining.

This lesson gives you the ladder practitioners actually climb — behavioral, then attributional, then
mechanistic — the honest limits of each rung, and the handful of findings the field has nailed down
well enough to build on.

## The three levels, from cheap to deep

Think of interpretability as a ladder. You climb only as high as your problem forces you to.

- **Behavioral** — treat the model as a black box and probe its input/output map. Change one thing,
  watch what moves. Cheap, always available, and the level most production debugging lives at.
- **Attributional** — ask *which parts of the input or the network* were responsible for an output.
  Attention maps, saliency, gradient attribution. Seductive and often misleading.
- **Mechanistic** — reverse-engineer the actual computation into human-legible algorithms: circuits,
  features, the flow of information through the residual stream. Expensive, still research-grade, but
  the only level that tells you *how* rather than just *what*.

Most of your job lives on rung one. But knowing rungs two and three is what separates "the model is
weird" from "the model copies the previous occurrence of this token via an induction head, and my
prompt breaks that pattern."

## Behavioral: probing and ablation

The workhorse is the **linear probe**. Freeze the model, run your inputs, cache the hidden state at
some layer, and train a cheap linear classifier to predict a property (part of speech, sentiment,
whether the subject is singular) from that hidden state. If a linear probe recovers the property with
high accuracy, the information is *linearly available* at that layer. Probes across depth give you a
picture of where information appears — early layers encode surface/syntactic features, middle layers
carry the most transferable semantics, late layers specialize toward the output distribution.

The classic trap: a probe measures *decodability*, not *use*. A property being linearly readable at
layer 12 does not mean the model uses it downstream. To test use, you **ablate** — zero out or patch
a component and measure the behavioral delta. Ablation is causal in a way probing is not: if removing
a head collapses subject-verb agreement, that head is doing agreement work. Probing tells you what's
*present*; ablation tells you what *matters*.

For LLMs the behavioral toolkit also includes **counterfactual prompting** (swap one entity, see if
the answer swaps) and **input attribution by occlusion** (mask spans, watch the logit move). These
are model-agnostic and API-friendly, which is why they dominate production debugging.

## Attributional: attention is not explanation

The most abused chart in NLP is the attention heatmap. It's tempting: attention weights literally say
"token *i* attended to token *j* with weight 0.8," so surely the high-weight tokens are the
explanation. They usually aren't.

Two results settled this. Jain and Wallace (2019), "Attention is not Explanation," showed you can
often find *entirely different* attention distributions that produce the *same* prediction — so the
attention you observed was not a necessary cause. Wiegreffe and Pinch (2019) pushed back with "Attention
is not *not* Explanation," arguing attention is *a* plausible explanation under constraints, not *the*
explanation. The practitioner takeaway is firm: **attention weights are a routing statistic, not an
attribution.** A head can place high weight on a token whose value vector it then multiplies by
near-zero; the weight is loud, the contribution silent. Attention also gets diluted by **attention
sinks** (see [transformer architecture](04-transformer-architecture.md)) — heads dumping probability
mass on the first token as a no-op. Never ship "the model focused on X" as an explanation backed only
by an attention map.

Gradient-based **saliency** (gradient × input, integrated gradients) is more principled — it measures
sensitivity of the output to each input dimension — but it's noisy, sensitive to the baseline you
integrate from, and can be gamed. Treat saliency as a hypothesis generator, never as proof.

## Mechanistic: the residual stream view

Mechanistic interpretability starts from a reframing of the architecture. Stop seeing a stack of
layers and start seeing a **residual stream**: a running vector, one per token position, that every
component *reads from* and *writes to* additively. Attention heads and MLPs don't transform the stream
in place — they compute an output and *add* it back. Because everything is additive, the final logits
are a **sum of contributions** from every component across every layer:

$$
\text{logits} = W_U \Big( x_0 + \sum_{\ell} \text{attn}_\ell + \sum_{\ell} \text{mlp}_\ell \Big)
$$

where $W_U$ is the unembedding and $x_0$ the token+position embedding. This linearity is the whole
game: it lets you decompose an output into "who contributed what," and it lets you read the stream at
any depth by projecting through $W_U$.

### The logit lens, with a worked example

The **logit lens** (nostalgebraist, 2020) is the simplest mechanistic tool and the one you'll use
first. The trick: apply the model's *final* unembedding $W_U$ (and final layernorm) to the residual
stream at an *intermediate* layer, as if the model stopped there. This gives you the model's "current
best guess" for the next token at every depth.

Worked example. Prompt: `"The Eiffel Tower is located in the city of"`. Cache the residual at the last
position after each layer and project to vocabulary:

```python
import torch
# resid[l]: [d_model] residual at final position after layer l
# ln_f, W_U: final layernorm and unembedding of the model
for l in range(model.n_layers):
    logits_l = W_U @ ln_f(resid[l])          # [vocab]
    top = logits_l.topk(3).indices
    print(l, [tokenizer.decode(t) for t in top])
```

You'll see something like: early layers predict frequent generic tokens (`the`, `a`, punctuation);
around the middle the geography features resolve and `France` climbs; by the last few layers `Paris`
dominates. The lens makes the *trajectory* of the prediction visible — you watch the answer form. It
also exposes failures: if `Paris` never rises, the fact isn't retrieved; if it rises then gets
*suppressed* in the last layer, a late component is overriding it (a real pattern behind some
refusals and some hallucinations). Caveats: the lens assumes intermediate states live in the same
basis as the output, which is only approximately true; **tuned-lens** (Belrose et al., 2023) fits a
small per-layer transform to fix the bias and gives cleaner trajectories.

### Induction heads and the IOI circuit

Two findings are solid enough to teach as fact.

**Induction heads** (Olsson et al., 2022, "In-context Learning and Induction Heads") are a two-head
circuit that implements the rule "if the pattern `[A][B]` appeared earlier, then after a later `[A]`,
predict `[B]`." Head one is a *previous-token head* that copies each token's identity into the next
position's stream; head two is the induction head that, at the current `[A]`, attends to the position
*after* the earlier `[A]` and copies its token to the output. This is literally prefix matching plus
copy. It matters because induction heads form during a sharp phase change in training that coincides
with the model acquiring in-context learning — they are a large part of the mechanism behind few-shot
prompting working at all.

**The IOI circuit** (Wang et al., 2022, "Interpretability in the Wild") is the field's cleanest
end-to-end case study. Task: "Indirect Object Identification" — given `"When Mary and John went to the
store, John gave a drink to"`, the model correctly predicts `Mary`. Wang et al. reverse-engineered the
full circuit in GPT-2 small: **duplicate-token heads** notice `John` appears twice; **S-inhibition
heads** write a signal that suppresses the duplicated name; **name-mover heads** then attend to and
copy the *remaining* name (`Mary`) to the output. They validated every claim causally with **activation
patching** — replacing a component's activation with the one from a counterfactual prompt and measuring
the logit shift. IOI is the proof of concept that a real behavior in a real model decomposes into a
small, named, causally-verified algorithm.

### Superposition and SAEs, conceptually

Here's the wall. If you inspect a single neuron in an MLP, it usually fires for a grab-bag of unrelated
concepts — it's **polysemantic**. The leading explanation is **superposition** (Elhage et al., 2022):
models represent far more features than they have neurons by encoding features as *directions* in
activation space that overlap, tolerating a little interference because most features are rare and
don't co-occur. So the natural unit of computation isn't the neuron; it's a direction.

**Sparse autoencoders (SAEs)** are the tool built to recover those directions. Train a wide
autoencoder on a layer's activations with a sparsity penalty, forcing it to reconstruct each activation
as a sparse sum of learned dictionary directions. Those directions turn out to be far more
**monosemantic** — one fires for the Golden Gate Bridge, another for DNA sequences, another for
"code that reads a file" (Bricken et al. and Templeton et al., 2023–2024, "Towards Monosemanticity" /
"Scaling Monosemanticity" on Claude). SAEs are how you go from "layer 20 has some structure" to "here
are 30,000 named features and I can watch which fire." They're expensive to train, incomplete (dead
features, reconstruction error), and there's active debate about whether the recovered features are
the model's "real" units — but they're the current best bridge from activations to concepts.

## What interpretability buys you today

Be honest about maturity: mechanistic interp does not yet let you fully explain a frontier model's
answer on demand. What it *does* deliver in production:

- **Eval debugging.** When a metric drops with no code change (see [evaluation](10-evaluation.md)),
  logit-lens trajectories and probes localize whether the failure is retrieval (fact never surfaces),
  representation (probe accuracy collapsed), or a late-layer override.
- **Steering vectors.** Take activations on "positive" vs "negative" prompts, subtract to get a
  direction, and *add* it to the residual stream at inference to push behavior (more formal tone, more
  refusal, a chosen topic). Contrastive Activation Addition (Rimsky et al., 2023) and the "Golden Gate
  Claude" demo are the canonical examples. It's a training-free knob — cheaper than a fine-tune, and
  reversible.
- **Guardrail features.** SAE features for "deception," "PII," or a policy-violating concept can be
  monitored or clamped as a detection signal, complementing behavioral filters
  (see [risks and safety](15-risks-and-safety.md)).

## Honest limits

Keep the skepticism calibrated. Probes conflate decodability with use. Attention is not attribution.
Saliency is a hypothesis, not a proof. Circuit analysis has been validated mainly on small models and
narrow tasks; scaling it is unsolved. SAEs leave reconstruction error and dead features on the table,
and the features they find may be an artifact of the dictionary as much as the model. Nothing here is
a lie detector. The right stance: interpretability sharpens hypotheses and occasionally gives you a
causal handle — use it to debug and steer, not to *certify*.

## What interviews ask here

- Why is an attention heatmap not an explanation? Name the failure mode (high weight, near-zero value
  contribution; sinks) and cite that you can find different attention with the same prediction.
- What's the difference between a probe and an ablation? Decodability (present) vs causal use (matters).
- Explain the logit lens in one breath. Project an intermediate residual through the final unembedding
  to read the model's running next-token guess; caveat: assumes shared basis, tuned-lens fixes it.
- What are induction heads and why do they matter? Prev-token head + match-and-copy head implementing
  in-context pattern completion; they underlie few-shot ICL and appear at a training phase change.
- What problem do SAEs solve? Superposition/polysemanticity — recover monosemantic feature directions
  from activations via a sparse, over-complete autoencoder.
- What is a steering vector? A contrastive activation direction added to the residual stream to shift
  behavior at inference, no retraining.

## Where this shows up on the job

- Debugging an eval regression whose cause isn't in the diff — localize it to retrieval vs
  representation vs a late-layer override before you touch training.
- Shipping a training-free behavior tweak (tone, refusal, topic emphasis) via steering vectors when a
  fine-tune is too slow or too heavy.
- Building or consuming feature-based guardrails and monitoring for a safety/trust team.
- Explaining model behavior to a risk or product stakeholder honestly — including saying "we can
  hypothesize but not certify," which is the answer that keeps you credible.
