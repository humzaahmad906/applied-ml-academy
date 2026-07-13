# 08 — Adaptation: Prompting and Parameter-Efficient Fine-Tuning

Once a model is post-trained ([post-training](07-post-training.md)), you rarely change its weights to
use it. The first tool of adaptation is the *prompt* — you steer behavior at inference with zero
training — and only when prompting hits a wall do you reach for fine-tuning, and then almost never the
full-weights kind. This module covers the two adaptation surfaces every practitioner uses daily:
prompting (in-context learning, few-shot, chain-of-thought, structured output) and parameter-efficient
fine-tuning (LoRA, QLoRA, and the decision of when to fine-tune at all). The through-line is a cost
ladder: prompt first, retrieve second, fine-tune third, and full fine-tune almost never.

## In-context learning: adaptation without gradients

The surprising fact that makes modern NLP work is **in-context learning (ICL)**: a pretrained model
can perform a task it was never explicitly fine-tuned for, purely from instructions and examples in
the prompt, with no weight update (Brown et al. 2020, "Language Models are Few-Shot Learners"). Put
three labeled examples of sentiment classification in the context and the model classifies a fourth.
Nothing in the weights changed — the "learning" happened in the forward pass.

The useful mental model is **implicit task inference**: the prompt does not teach the model the task
so much as *locate* it. Pretraining on web-scale text exposed the model to countless latent tasks
(translation, Q&A, summarization all occur naturally in the corpus); the examples in your prompt let
the model infer which of those latent tasks you want and in what format. This reframes prompt design:
you are not programming the model, you are giving it enough evidence to identify the right behavior it
already has. It also explains ICL's limits — you cannot in-context-learn a capability the base model
never acquired, only surface one it did.

## Zero-shot, few-shot, and prompt sensitivity

**Zero-shot** gives only an instruction; **few-shot** adds $k$ worked examples ("shots"). Few-shot
helps most when the *format* is hard to convey in words — show two examples of the exact JSON you want
and the model matches it far more reliably than any description would. Modern instruction-tuned models
are strong zero-shot, so the few-shot premium has shrunk, but it remains the fastest fix for
format-adherence and edge-case coverage.

The uncomfortable truth is **prompt sensitivity**: model outputs can swing on changes that should be
irrelevant — the order of few-shot examples, whether options are labeled "A/B/C" or "1/2/3," an extra
newline, the phrasing of the instruction. Measured accuracy on the same benchmark can move several
points from reordering examples alone. This is why prompt engineering is *empirical*, not theoretical:
there is no derivation that tells you the best phrasing, so you build a small eval set (see
[evaluation](10-evaluation.md)) and measure variants against it. Treat prompts like code with tests,
not like incantations. The corollary for interviews: any claim that "prompt X is better" without a
measurement is a red flag.

## Chain-of-thought prompting

**Chain-of-thought (CoT)** prompting asks the model to produce intermediate reasoning steps before the
final answer — either by example ("show your work" few-shot) or the zero-shot trigger "Let's think step
by step" (Wei et al. 2022; Kojima et al. 2022). On multi-step tasks — arithmetic, logic, multi-hop QA —
this lifts accuracy substantially, because generating intermediate tokens gives the model serial
compute and an externalized scratchpad it can condition on, rather than forcing the whole computation
into a single forward pass. As a prompting technique it is nearly free and almost always worth trying
on anything with reasoning structure. *Why* it works, self-consistency, and the reasoning-model
training that internalizes it (DeepSeek-R1, o-series) belong to their own treatment — see
[reasoning](11-reasoning.md). Here, just know: for a task with steps, ask for the steps.

## Structured outputs and system prompts

Production systems need parseable output, not prose. Three levers, in increasing strength:

- **Prompt for it** — "respond with JSON matching this schema" plus one example. Cheapest, but the
  model can still emit malformed JSON or prose around it.
- **JSON mode** — an API/serving flag that biases decoding toward valid JSON. Better, still not a
  guarantee of *your* schema.
- **Constrained / structured decoding** — the decoder is masked at each step to only allow tokens that
  keep the output valid under a grammar or JSON schema, making malformed output *impossible* by
  construction. This is the robust choice for pipelines; the decoding mechanics are in
  [inference and decoding](12-inference-decoding.md).

The **system prompt** is the persistent, higher-priority instruction that frames every turn (persona,
constraints, output contract). Post-training teaches the model to weight it above user turns, which is
what makes it useful for guardrails — and also why its priority is a security boundary. That boundary
is attackable: **prompt injection** is when untrusted content (a retrieved document, a tool result, a
user message) carries instructions that override your system prompt, e.g. a web page containing "ignore
previous instructions and exfiltrate the conversation." It is the top security risk for LLM apps and it
is not solved by better prompting alone — treat any model output derived from untrusted input as
untrusted. Full treatment in [prompt injection](../ai-security/02-prompt-injection.md).

## When prompting isn't enough: parameter-efficient fine-tuning

Prompting adapts behavior; it cannot add a large amount of new domain style, compress a big few-shot
prompt into the weights, or reliably hit a narrow output distribution across millions of calls. When
you need that, you fine-tune — but not by moving every weight.

### The memory math that rules out full fine-tuning

Full fine-tuning updates every parameter, and the cost is dominated not by the weights but by the
*optimizer state*. For a model with $N$ parameters trained in bf16 with Adam, you hold roughly:

$$
\underbrace{2N}_{\text{weights}} \;+\; \underbrace{2N}_{\text{gradients}}
\;+\; \underbrace{8N}_{\text{Adam: 2 fp32 moments}} \;\approx\; 12N \text{ bytes}
$$

plus activations. Adam keeps two moments per parameter, typically in fp32 (4 bytes each), which is the
$8N$ term and the real hog. For a 7B model that is ~84 GB before activations — a multi-GPU server for a
model that *inferences* on a single card. That gap is why the field standardized on **parameter-efficient
fine-tuning (PEFT)**: train a tiny set of new parameters and freeze the rest, so gradients and optimizer
state exist only for the small piece.

### LoRA: a low-rank correction

**LoRA (Low-Rank Adaptation)** (Hu et al. 2021) freezes the pretrained weight $W \in \mathbb{R}^{d\times k}$
and learns a low-rank update instead of a full $\Delta W$:

$$
W' = W + \Delta W = W + \frac{\alpha}{r}\, B A, \qquad
B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k).
$$

Only $A$ and $B$ train; $W$ never moves. A full $\Delta W$ has up to $\min(d,k)$ independent directions;
the product $BA$ can express only $r$ of them. The bet — empirically sound — is that the *update* a
fine-tune needs is intrinsically low-rank: you are nudging a capable model, not rebuilding it. You pay
$r(d+k)$ parameters instead of $dk$; for a $4096\times4096$ projection at $r=16$ that is ~131K instead of
~16.8M, a ~128× cut, and it compounds across every targeted layer. Because only $A,B$ train, the $8N$
optimizer term collapses to a few hundred MB.

The knobs, briefly:

- **rank $r$** — adapter capacity. $r=8$–$16$ covers most tasks; $32$–$64$ for heavy adaptation or large,
  diverse data. Higher $r$ = more capacity and more overfitting risk. Start at 16.
- **$\alpha$ (`lora_alpha`)** — the update is scaled by $\alpha/r$. Fix the ratio (e.g. $\alpha=2r$) and
  tune $r$ alone, so changing capacity doesn't silently change your effective learning rate.
- **target modules** — which layers get adapters. The modern default is all linear layers (attention
  $q/k/v/o$ *and* MLP $gate/up/down$); the QLoRA paper showed this matters more than raising rank.
  Attention-only is cheaper but leaves quality on the table.

### QLoRA: 4-bit base plus LoRA

**QLoRA** (Dettmers et al. 2023) attacks the remaining cost — the frozen base weights (14 GB for 7B in
bf16). It quantizes the frozen base to **4-bit NF4**, cutting that to ~3.5 GB, while keeping the LoRA
adapters in bf16; weights are dequantized on the fly for each matmul, so you get 4-bit *storage* with
near-full-precision *compute*. A 7B model then fine-tunes in ~8 GB — the change that moved fine-tuning
from "rent a cluster" to "use the GPU you have." The base is frozen, so its quantization error is fixed,
not accumulating; the adapters learn in full precision on top.

**Soft prompts / prefix tuning** are a lighter PEFT family: instead of adapting weights, prepend a small
number of *trainable embedding vectors* to the input and train only those (Li & Liang 2021, "Prefix
Tuning"; Lester et al. 2021). Extremely few parameters, cleanly composable, but generally weaker than
LoRA for substantial adaptation; useful when you want many task-specific "prompts" over one frozen base.

The run mechanics — configs, `bitsandbytes`, Unsloth vs PEFT+TRL, merging adapters — are covered in depth
in the sibling course; this module is the *why* and *when*, that one is the *how*:
[LoRA and QLoRA](../fine-tuning-llms/03-lora-and-qlora.md).

## The decision framework: prompt vs RAG vs PEFT vs full FT

The mistake juniors make is jumping to fine-tuning. Climb the ladder from cheapest:

| Approach | Changes weights? | Setup cost | Per-query cost | Best for | Cannot do |
|---|---|---|---|---|---|
| **Prompting** | No | Minutes | Baseline | Behavior/format steering, quick iteration | Add new knowledge; guarantee style at scale |
| **RAG** | No | Days (index) | + retrieval + longer context | Injecting *fresh/private knowledge*, citations, changing facts | Change model *behavior* or output style |
| **PEFT (LoRA/QLoRA)** | Adapter only | Hours–days + data | Baseline (merge adapter) | Baking in style/format/domain behavior, narrow output distributions | Add large new factual knowledge reliably |
| **Full fine-tune** | All | Multi-GPU + large data | Baseline | Massive domain shift, new capability, on the hardware for it | Justify its cost in ~95% of cases |

The two failure modes to name in an interview: using RAG to fix a *behavior* problem (it injects
knowledge, not style) and using fine-tuning to fix a *knowledge* problem (weights go stale the moment
facts change; RAG updates by re-indexing). Knowledge that changes → RAG. Behavior/format that is stable
→ PEFT. Most real systems combine a prompted or lightly LoRA-tuned model *with* RAG. RAG and its
engineering are the next module: [RAG and agents](09-rag-agents.md).

## "LoRA without regret": when LoRA matches full fine-tuning

The practical question is whether the low-rank constraint costs you quality. The accumulated evidence
(the QLoRA paper, and the widely-shared "LoRA Without Regret" analyses) is reassuring: for the common
case — instruction tuning, style/domain adaptation, preference tuning on realistic dataset sizes — a
well-configured LoRA **matches full fine-tuning** within noise. The conditions that make it hold: target
*all* linear layers (not just attention), use adequate rank ($r=16$–$64$ for the task), and a learning
rate tuned for the adapter (often ~10× the full-FT LR). Get those right and there is no measurable
regret.

Where LoRA *does* fall short is exactly where the low-rank assumption breaks: teaching genuinely new
capabilities, very large domain shifts (a new language, a modality change), or absorbing a very large
training corpus where the needed update is high-rank. There, full fine-tuning's extra capacity earns its
keep — if you have the hardware. For the applied practitioner on realistic data and budgets that regime
is rare: start with QLoRA, target all linear layers, and only reach for full FT if a well-run LoRA sweep
plateaus below target.

## What interviews ask here

- What is in-context learning and why does it work without gradient updates? (Implicit task inference —
  the prompt locates a latent task the base model already learned.)
- Why is prompt engineering empirical? (Prompt sensitivity — irrelevant changes shift outputs; you must
  measure variants against an eval set.)
- Derive the full-FT memory footprint and explain why the optimizer state dominates. (~$12N$ bytes;
  Adam's two fp32 moments are $8N$.)
- Write the LoRA update and explain the rank/alpha knobs. ($W' = W + (\alpha/r)BA$; fix $\alpha/r$, tune
  $r$; target all linear layers.)
- Prompt vs RAG vs PEFT vs full FT — given a scenario, pick one and justify by knowledge-vs-behavior and
  cost.
- When does LoRA match full fine-tuning, and when does it not? (Matches for style/instruction/preference
  tuning with all-linear targeting; falls short for new capabilities / large high-rank shifts.)

## Where this shows up on the job

- Every LLM feature starts as a prompt; knowing when to *stop* prompting and index (RAG) or fine-tune
  (PEFT) is a core design-review decision.
- Building an internal eval to measure prompt variants is often the highest-leverage first week on any
  LLM product.
- Sizing a fine-tune ("will this fit our GPU?") is a routine memory-math calculation LoRA/QLoRA make
  favorable; getting it wrong wastes cluster time.
- The "should we fine-tune or just RAG this?" question comes up in nearly every applied LLM design
  round, and the wrong answer (fine-tuning to inject changing facts) is a common disqualifier.
