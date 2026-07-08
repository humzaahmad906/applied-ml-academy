# 06f — Prompt Management and LLM Evaluation Pipelines

This lesson teaches the two disciplines that the rest of the catalog name-checks but never unpacks: managing prompts as versioned artifacts, and building evaluation pipelines that gate prompt and model changes the way tests gate application code. By 2026 this is table-stakes LLMOps — a company shipping LLM features without a prompt registry and an eval gate is running the equivalent of pushing to production with no tests and no version control. Everything here is the LLM-layer analog of the CI/CD and continuous-training machinery you already know for classical ML.

## Why a Prompt Is a Production Change with No Type Checker

A prompt is source code. It has inputs (variables), control flow (conditionals, few-shot examples), and outputs (structured or free text). It is the highest-leverage line of code in an LLM system: a single word change can swing accuracy, cost, latency, and safety. And unlike real code, nothing stops you from breaking it. There is no compiler, no type checker, no linter that catches "I removed the JSON schema instruction and now 4% of responses fail to parse." The failure surfaces in production, silently, as a quality regression that no stack trace points to.

The failure modes are specific and they compound:

- **Silent regressions.** Rewording the system prompt to fix one edge case degrades ten others you weren't looking at.
- **Untracked changes.** A prompt edited directly in a vendor playground, copy-pasted into code, and shipped — with no diff, no author, no rollback path.
- **Model drift under a frozen prompt.** The provider ships a new snapshot of the model behind the same API alias; your prompt is unchanged but its behavior isn't.
- **Prompt-model coupling.** A prompt tuned for one model breaks on another. When you route across providers for cost or availability (see 06d), the prompt that worked on one is not guaranteed on the next.

The discipline that fixes this is the same discipline you apply to models: **version it, test it against a held-out set, gate the change on metrics, canary the rollout, and observe it in production.** The rest of this lesson is that loop.

---

## Prompt Versioning and Registries

The first move is to **decouple prompts from application code.** A prompt hardcoded in a Python string ships on the application's release cadence, is invisible to non-engineers, and can't be rolled back without a redeploy. A prompt in a registry is an independent artifact with its own version history, its own deploy labels, and its own rollback — changeable without shipping code.

A prompt registry gives you:

- **Immutable, auto-incrementing versions.** Each save is a new numbered snapshot. Nothing is edited in place.
- **Deploy labels / aliases.** A moving pointer like `production` or `staging` resolves to a specific version. You promote by moving the label, and roll back by moving it back — no redeploy.
- **Diffs and authorship.** Who changed what, when, and why — a Git-like history for prose.
- **Runtime fetch with caching.** Application code fetches the prompt by name and label at runtime, with a local cache and a bundled fallback so a registry outage can't take down serving.

### The 2026 tool landscape

| Tool | Role | Notes |
|---|---|---|
| **Langfuse** | Open-source LLM engineering platform | Linear versioning with `production`/`staging` labels, side-by-side diff UI, SDK fetch with client-side caching. Self-hostable. |
| **LangSmith** | LangChain's observability + eval platform | Prompt hub with version history; tight coupling to LangChain/LangGraph and its Playground. |
| **MLflow 3 Prompt Registry** | GenAI extension of MLflow | Git-like immutable versions with auto-incrementing numbers and **aliases**; natural fit if your classical-ML stack is already on MLflow. |
| **PromptLayer / Humanloop** | Product-oriented prompt CMS | Visual registries aimed at letting PMs and domain experts edit prompts without touching code. |

This course already standardizes on Langfuse and LangSmith for observability and MLflow 3 for the model registry, so the pragmatic default is: **prompts live next to whichever registry already holds your traces or your models.** Don't add a fourth vendor for prompts alone.

A runtime fetch looks like this — note the label indirection and the fallback:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Resolve the "production"-labelled version at request time; cache locally.
prompt = langfuse.get_prompt("support-classifier", label="production", cache_ttl_seconds=60)
compiled = prompt.compile(ticket_text=incoming_ticket)  # fills {{ticket_text}}
# prompt.version is captured on the trace, so every response is tied to an exact prompt version.
```

The label indirection is what makes A/B and rollback cheap: point `production` at version 8, watch the metrics, and if it regresses, point it back at 7. No code ships either way.

---

## Prompt CI: Regression-Testing Every Change

Versioning tells you *what* changed. It doesn't tell you whether the change is *good*. For that you need the LLM-layer equivalent of a test suite: a **held-out eval set** and a job that runs every prompt (or model) change against it and **gates the merge on metrics.**

The dominant open tool here is **promptfoo** — a declarative, YAML-configured eval runner with first-class CI/CD integration (it's now part of OpenAI, and is used internally at both OpenAI and Anthropic). You define prompts, providers, test cases, and per-case assertions, then run `promptfoo eval` in a GitHub Action on every PR that touches a prompt.

A minimal `promptfoo.config.yaml`:

```yaml
description: Support-ticket classifier regression suite

prompts:
  - file://prompts/support-classifier.txt   # the candidate prompt under test

providers:
  - anthropic:messages:claude-sonnet-4-5
  - openai:gpt-5.1                            # test across the models you route to

defaultTest:
  assert:
    - type: is-json                           # deterministic: must parse
    - type: latency
      threshold: 3000                         # ms, per response
    - type: cost
      threshold: 0.002                        # USD, per response

tests:
  - vars:
      ticket_text: "My invoice charged me twice this month."
    assert:
      - type: javascript
        value: JSON.parse(output).category === "billing"   # exact-match ground truth
  - vars:
      ticket_text: "The app crashes when I upload a PDF."
    assert:
      - type: contains-json
      - type: llm-rubric                       # graded, not exact-match
        value: >
          The category is a reasonable classification of a technical bug report,
          and the response includes a non-empty summary field.
```

The mix of assertion types is the point. **Deterministic checks** (`is-json`, `contains`, regex, latency, cost) are cheap, fast, and flake-free — use them for everything you can express as a hard rule. **Graded checks** (`llm-rubric`) fall back to an LLM judge only for the subjective slice that no rule captures. In CI, wire `promptfoo eval` to fail the build when the pass rate drops below a threshold or regresses against the base branch; the diff view surfaces exactly which cases flipped.

The eval set itself is the asset, and it obeys the same rules as any ML eval set (see the eval chapters): **frozen, versioned, and grown by addition, never by rebalancing.** Seed it from real production traces — the failures and edge cases your users actually hit — not from cases you imagined. Adding a case that a prompt got wrong is how the suite gets sharper over time; quietly deleting a case you can't pass is how it rots.

---

## LLM-as-Judge: Building One That You Can Trust

Most interesting LLM outputs — a summary, an answer, a rewrite — have no single correct string, so exact match is useless. The 2026 workhorse for grading them at scale is **LLM-as-judge**: a strong model scores outputs against a rubric. It's cheaper than human review and far faster, but a naive judge is a random-number generator with good grammar. Building one you can trust takes care.

### Pointwise vs pairwise

- **Pointwise (direct scoring):** the judge scores one output against a rubric — a Likert 1–5, or 0/1 pass-fail on specific criteria. Simple, absolute, easy to threshold in CI. This is what promptfoo's `llm-rubric` and MLflow's scorers do.
- **Pairwise (comparison):** the judge picks the better of two outputs (A vs B). More reliable for ranking two prompt versions against each other — humans and models both compare more consistently than they score in the absolute — but it doesn't give you an absolute number to gate on.

Rule of thumb: **pointwise for CI gates and monitoring, pairwise for choosing between two candidate prompts.**

### Rubric design

A good rubric is specific, decomposed, and grounds every judgment. Ask for one criterion at a time, force a reasoning step before the score, and demand structured output:

```text
You are grading a customer-support reply. Judge ONLY the criteria below.
Do not reward length, fluency, or politeness beyond what the criteria state.

Criteria:
1. Factual grounding — every claim is supported by the provided ticket and KB context.
2. Resolution — the reply addresses the customer's actual problem, not an adjacent one.
3. Safety — no promises about refunds/policy the KB does not authorize.

Ticket: {{ticket_text}}
KB context: {{context}}
Reply to grade: {{output}}

For each criterion, give a one-sentence justification, then a score of 0 or 1.
Return JSON: {"grounding": {...}, "resolution": {...}, "safety": {...}}.
```

### The biases you must control

LLM judges have systematic, measurable biases. Ignoring them means your eval numbers are noise:

- **Position bias.** In pairwise, judges favor whichever answer is in slot A (or slot B). This is large — studies put GPT-4-class inconsistency around 40%. **Mitigate:** run each pair in both orders and only count a win if the judge is consistent across the swap; or score pointwise and derive the comparison post-hoc.
- **Verbosity bias.** Longer answers get rated higher regardless of quality (~15% inflation observed). **Mitigate:** explicitly instruct the rubric to ignore length, and add a deterministic length cap as a separate assertion.
- **Self-preference bias.** A judge over-rewards outputs from its own model family (a 5–7% boost). **Mitigate:** use a judge from a *different* family than the model you're evaluating — never let a model grade its own homework — or ensemble multiple judges.

The non-negotiable step: **calibrate the judge against human labels.** Hand-label a representative sample, compute agreement (Cohen's kappa) between judge and humans, and only trust the judge on the slices where it agrees with people. A judge validated against 200 human labels can be more reliable and vastly cheaper than continuous human review; an uncalibrated judge is folklore with a temperature setting.

MLflow 3's `mlflow.genai.evaluate()` ships research-validated built-in judges (Correctness, RelevanceToQuery, Safety, Groundedness, Guidelines) plus custom `@scorer` functions, and — critically — the same scorers run in offline eval and in production monitoring, so your gate metric and your live metric are the same metric.

---

## Offline vs Online Eval, and Canarying a Prompt Change

There are two evaluation regimes and you need both.

**Offline eval** is the pre-ship gate: run the candidate prompt/model against the frozen dataset, in a controlled setting, and block the release if metrics regress. This is the promptfoo/CI loop above — the unit and integration tests of the LLM layer.

**Online eval** runs in production against live traffic: sample real interactions and score them continuously (deterministic checks plus a monitoring judge on a sample) to catch what offline missed. Offline eval can only measure the failures you anticipated; online eval catches the distribution shift you didn't.

Offline green is necessary but not sufficient, because your eval set is never the full production distribution. So you **canary**, exactly as you would a model deploy:

1. Move `production` to point at the new prompt version for a small traffic slice (say 5%), leaving the rest on the old version.
2. Run online eval on both slices in parallel and compare — judge scores, parse-failure rate, token cost, latency, refusal rate, user thumbs.
3. Ramp the slice only if the new version holds or improves. If any guardrail metric regresses, move the label back — an instant rollback with no redeploy, which is the entire payoff of label indirection.

This is A/B testing for prompts, and the registry's label mechanism is what makes it a config change rather than a deploy.

---

## Observability: Tracing Prompts, Responses, and Cost

You cannot evaluate — offline or online — what you don't capture. LLM observability means **tracing every call**: the resolved prompt (and its version), the rendered inputs, the full response, the model and parameters, token counts in and out, latency, and computed cost.

The emerging standard is the **OpenTelemetry GenAI semantic conventions**, defined by the GenAI SIG. They give LLM calls standardized `gen_ai.*` span attributes so your traces are portable across vendors instead of locked to one SDK's schema:

```text
gen_ai.system            = "anthropic"
gen_ai.request.model     = "claude-sonnet-4-5"
gen_ai.usage.input_tokens  = 1830
gen_ai.usage.output_tokens = 240
gen_ai.response.finish_reason = "stop"
```

Two caveats for 2026: most of these conventions are still **experimental** (not yet API-stable), and instrumentation maturity varies — the OpenAI SDK auto-instrumentation is furthest along, with Anthropic, Bedrock, and others covered via community libraries. Emit token counts on every span, because with per-token billing that's what lets you compute near-real-time spend and attribute cost per prompt version, per feature, per customer.

In practice you get these conventions "for free" from the platforms already in this course: **Langfuse** and **LangSmith** ingest traces, tie each response back to the exact prompt version that produced it, and are where your online-eval scores and dashboards live. For an agent, each tool call, retrieval, and LLM invocation becomes a child span, so you get the full reasoning chain, not just the final answer. A trace that carries the prompt version is what closes the loop: a production regression flagged by online eval points you straight at the version that caused it and the rollback target.

---

## Tying It Back: This Is CI/CT for the LLM Layer

None of this is a new paradigm. It is the MLOps loop you already know, re-expressed for prompts and generative outputs:

| Classical ML | LLM layer |
|---|---|
| Model registry (versions, stages) | Prompt registry (versions, labels) |
| CI on code + data | Prompt CI (promptfoo) on prompt changes |
| Frozen held-out eval set + metric gate | Frozen eval set + judge/deterministic gate |
| Offline eval before promotion | Offline eval before label move |
| Canary + automated rollback on a model deploy | Canary + label rollback on a prompt change |
| Monitoring + drift detection | Online eval + trace observability |
| Continuous training triggered by drift | Prompt/model re-eval triggered by regression |

The prompt is a versioned artifact; the eval set is your frozen test bench; the judge is your automated grader; the registry label is your deploy switch and your rollback; the trace is your monitoring. A team that has internalized continuous integration and continuous training for models already owns the mental model — the only new pieces are the non-determinism of the output (which is why the judge and its bias controls matter) and the fact that the prompt is prose a non-engineer can and will edit (which is why the registry and the CI gate matter). Treat the LLM layer with the same discipline as the model layer, and it stops being the flaky, un-versioned part of your stack.

---

## Exercises

1. Move a hardcoded prompt from one of your projects into Langfuse (or MLflow 3's Prompt Registry). Fetch it at runtime by `production` label with a bundled fallback. Change it, promote the new version by moving the label, then roll back.
2. Build a 30-case promptfoo suite seeded from real traces for that prompt. Mix deterministic assertions (`is-json`, `cost`, `latency`) with `llm-rubric`. Wire `promptfoo eval` into a GitHub Action that fails the PR on a pass-rate regression.
3. Write a pointwise judge prompt with a decomposed rubric. Hand-label 50 outputs, compute Cohen's kappa between your judge and your labels, and iterate the rubric until agreement is acceptable. Then flip it to pairwise and measure position bias by running each pair in both orders.
4. Instrument your LLM calls with OpenTelemetry `gen_ai.*` attributes (or via Langfuse/LangSmith). Build a dashboard of token cost per prompt version, and confirm every trace carries the version that produced it.

---
## You can now

- Explain why a prompt is a production change with no type checker, and enumerate the failure modes (silent regressions, untracked edits, model drift under a frozen prompt, prompt-model coupling) that versioning and evals exist to catch.
- Stand up a prompt registry (Langfuse, LangSmith, MLflow 3, or PromptLayer/Humanloop), decouple prompts from code, and use label indirection to A/B and roll back a prompt without a redeploy.
- Build a promptfoo-style regression suite over a frozen, trace-seeded eval set that mixes deterministic and graded assertions, and gate every prompt change on it in CI.
- Design an LLM-as-judge — pointwise vs pairwise, a decomposed rubric, and calibration against human labels — while controlling position, verbosity, and self-preference bias.
- Distinguish offline from online eval and canary a prompt change against live traffic with an instant label rollback on regression.
- Trace prompts, responses, tokens, and cost with the OpenTelemetry `gen_ai` semantic conventions and Langfuse/LangSmith, tying each production response to the exact prompt version that produced it.
- Map the whole thing onto the MLOps loop you already know — this is CI/CT for the LLM layer, not a new paradigm.
</content>
</invoke>
