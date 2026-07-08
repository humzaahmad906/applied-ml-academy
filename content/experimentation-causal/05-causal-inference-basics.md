# 05 — Causal Inference Basics

The A/B test is the gold standard, and if you can run one, you should. But a huge fraction of the causal questions that matter cannot be answered with a coin flip. Does smoking cause cancer? You can't randomly assign people to smoke for thirty years. Did the new pricing policy grow revenue? It launched everywhere at once — there's no control group. Does a minimum-wage law reduce employment? You can't A/B test legislation. For all of these, you're stuck with **observational data**: outcomes from a world that chose its own treatments. This lesson builds the conceptual toolkit — confounding, DAGs, the backdoor idea — for reasoning about cause from data you didn't get to randomize, and it's honest about when you simply can't.

## Confounding, the core disease

Recall the ice-cream-and-drowning correlation: both are driven by hot weather. That's a **confounder** — a variable that causes both the treatment and the outcome, manufacturing a correlation between them that has nothing to do with one causing the other. Confounding is *the* central problem of observational causal inference. In lesson 01 we saw it inflate a true +5% feature effect into a +28% mirage because engaged users both adopted the feature and retained on their own.

The reason randomization is so powerful is that it *severs* the arrow from confounders to treatment. When a coin decides who gets treated, nothing about the user — engagement, wealth, weather, mood — can influence assignment, so no confounder can bias the comparison. Observational data gives you no such guarantee: treatment was assigned by the messy, self-selecting real world, and every difference between the treated and untreated groups is a candidate confounder. Your entire job becomes *ruling confounders out* — which requires first being able to reason about them clearly.

## DAGs: drawing your causal assumptions

The tool for that reasoning is the **Directed Acyclic Graph (DAG)**. It's exactly what it sounds like: variables are nodes, and an arrow X → Y means "X directly causes Y." *Directed* (arrows have direction), *acyclic* (no loops — nothing causes itself through a cycle). A DAG is not learned from data; it's a picture of *your assumptions* about how the world works, made explicit so others can challenge them.

Consider the feature-retention question, with engagement as a confounder:

```
         engagement
          /        \
         v          v
   feature  ----->  retention
```

Engagement points to both feature adoption *and* retention; feature also points to retention (the effect we want). This little picture encodes the whole problem: there are two reasons feature and retention correlate — the direct causal arrow (what we want) and the **backdoor path** feature ← engagement → retention (the confounding we must remove).

Three canonical structures cover most of what you'll meet:

- **Chain (mediator):** X → M → Y. M is a mediator; X affects Y *through* M. If you're after the total effect of X, do **not** control for M — you'd block the very pathway you're measuring.
- **Fork (confounder):** X ← Z → Y. Z is a confounder creating a spurious X–Y association. You **must** control for Z to get the causal effect.
- **Collider:** X → C ← Y. C is a common *effect* of X and Y. Here the trap inverts: controlling for a collider *creates* a spurious association that wasn't there. This is subtle and catches experts — conditioning on a collider (or anything downstream of it) opens a path rather than closing one.

The collider point deserves a beat because it's counterintuitive: adjusting for *more* variables is not always safer. Control for the wrong one — a mediator or a collider — and you *introduce* bias. Which variables to control for is a question the DAG answers; blindly throwing every column into a regression does not.

## The backdoor criterion, intuitively

So which variables *do* you control for? The **backdoor criterion** gives the rule. A "backdoor path" is any non-causal path from treatment to outcome that starts with an arrow pointing *into* the treatment (like feature ← engagement → retention). These paths are the conduits of confounding. The criterion says: **find a set of variables that blocks every backdoor path, while not opening any new ones (don't include colliders or mediators).** If such a set exists and you can measure it, adjusting for it lets you recover the causal effect from observational data.

In the feature example, controlling for `engagement` blocks the single backdoor path, and engagement isn't a collider or mediator here — so `{engagement}` is a valid **adjustment set**. Measure engagement, compare feature-vs-no-feature *within* users of similar engagement, and the confounding dissolves. This is the logic under every method in the next lesson: they are all machinery for adjusting on a backdoor set.

Here's the humbling part, stated plainly. The backdoor criterion only works for confounders you can **name and measure**. Randomization balances confounders you never even thought of; observational adjustment can only handle the ones on your DAG. If engagement drives everything but you never logged it, no statistical method recovers the truth — you'll adjust for what you have and quietly inherit the bias from what you don't. This is **unmeasured confounding**, and it is the permanent asterisk on every observational causal claim. Honest practice names the confounders you *couldn't* measure and reasons about which way they'd bias the answer.

## A whiff of do-calculus

You'll hear the phrase **do-calculus** (Judea Pearl's framework). The one idea worth carrying from it is the distinction between two superficially similar quantities:

- **P(Y | X = x)** — the distribution of Y among units we *observed* to have X = x. This is prediction; it's soaked in confounding.
- **P(Y | do(X = x))** — the distribution of Y if we *intervened* to set X = x for everyone, reaching in and overriding whatever would naturally have determined X. This is causation.

The `do` operator is the mathematical version of the coin flip: it wipes out all the arrows *into* X (nobody's engagement decides their treatment anymore — you decided it), leaving only X's downstream effects. Causal inference is the discipline of computing a `do` quantity from `observed` (non-do) data. Randomization does it physically; the backdoor criterion tells you when adjustment can do it on paper. Do-calculus is the full set of rules for when such a translation is possible at all — you don't need the machinery now, only the mental distinction: **seeing X = x is not the same as setting X = x**, and confounding is exactly the gap between them.

```python
# The gap between "seeing" and "doing", concretely — the lesson-01 setup again.
import numpy as np
rng = np.random.default_rng(0)
n = 20_000
engagement = rng.random(n)
feature = rng.random(n) < engagement                 # self-selected
retention = rng.random(n) < (0.2 + 0.6 * engagement + 0.05 * feature)

# P(retention | feature) — observational, confounded:
seeing = retention[feature].mean() - retention[~feature].mean()

# Approx P(retention | do(feature)) — adjust for engagement by binning (backdoor):
bins = np.digitize(engagement, np.linspace(0, 1, 11))
doing = np.mean([
    retention[(feature) & (bins == b)].mean() - retention[(~feature) & (bins == b)].mean()
    for b in range(1, 11) if ((feature) & (bins == b)).any() and ((~feature) & (bins == b)).any()
])
print("seeing (P|X):", round(seeing, 3), " doing (adjusted):", round(doing, 3))
# output: seeing (P|X): 0.246  doing (adjusted): 0.058
```

Adjusting within engagement bins pulls the confounded 0.246 back to 0.058 — essentially the true 0.05, the backdoor criterion in action, and a concrete instance of turning a "seeing" number into a "doing" number.

## When you genuinely can't A/B test

Sometimes the right move is not a clever observational method but the honest admission that a controlled experiment is impossible or wrong:

- **Ethics.** You cannot randomize people to smoke, to skip a proven treatment, or to receive a harmful experience. Deliberately degrading some users' safety or health to measure an effect is off the table.
- **Feasibility.** Some treatments have no unit-level control group: a nationwide law, a company-wide reorg, a one-time pricing change that shipped to everyone. There's no one left to be the control.
- **Scale and spillover.** Network effects (marketplaces, social graphs) can make a clean control impossible because treating some units contaminates the others.
- **Cost and speed.** A proper RCT for a rare, slow outcome (five-year churn) may take longer than the decision can wait.

In every one of these, observational methods are not a lazy substitute for an experiment you were too impatient to run — they're the *only* tool available, and their conclusions come with the unmeasured-confounding asterisk attached. The next lesson is the toolbox; this lesson is the judgment that tells you when to open it, and the DAG discipline that tells you *what to adjust for* once you do.

## Key takeaways

- **Confounding** — a variable causing both treatment and outcome — is the central problem of observational causal inference; randomization defeats it, observational data does not.
- A **DAG** draws your causal assumptions explicitly: arrows are direct causes; the three structures are chains (mediators), forks (confounders), and colliders (common effects).
- Controlling for the right variable (a confounder) removes bias; controlling for the *wrong* one (a mediator or collider) *adds* bias — more adjustment is not always safer.
- The **backdoor criterion** says: block every backdoor path with a measurable adjustment set, without opening new ones. Every observational method is machinery for doing this.
- Observational adjustment only handles confounders you can name and measure; **unmeasured confounding** is the permanent asterisk. Randomization handles the ones you never imagined.
- **P(Y | X)** (seeing) ≠ **P(Y | do(X))** (setting). Causal inference computes a "doing" quantity from "seeing" data — the whole game in one line.
- Ethics, feasibility, spillover, and cost make some questions un-randomizable; observational methods are then the only option, not a shortcut.

## Try it

Draw the DAG for this scenario and identify what to control for. A hospital finds that patients who receive a new drug die *more* often than those who don't. You suspect the drug is given preferentially to the sickest patients. Sketch nodes for `severity`, `drug`, and `death`, add the arrows your story implies, and name the backdoor path. Is `severity` a confounder, mediator, or collider? Then argue in two sentences why comparing raw death rates is misleading and what adjustment set the backdoor criterion prescribes. Bonus: modify the lesson's binning code so `severity` (not engagement) is the confounder and confirm adjustment reverses the naive conclusion.
