# 01 — Why Experiment?

Ice cream sales and drowning deaths rise together every summer. Nobody sane concludes that eating ice cream drowns people — a third thing, hot weather, drives both. Yet the moment the variables are less familiar, this exact mistake sneaks into product decisions, medical claims, and ML deployments every day. "Users who adopt our new feature retain 40% better" sounds like a reason to push the feature to everyone. It might be. Or the kind of user who adopts new features was already the kind who sticks around. This course is about telling those two worlds apart, and the first lesson is about *why* that is genuinely hard — and what the one reliable escape hatch is.

## Correlation is not causation, stated precisely

You have heard the slogan. Here is what it actually means. Two variables are **correlated** when knowing one tells you something about the other: they move together. Correlation is a property of observed data, and it is symmetric — if X predicts Y, then Y predicts X equally well.

**Causation** is different and asymmetric: X causes Y if *intervening* to change X would change Y. Flipping the switch changes the light; the light does not change the switch.

Correlation between X and Y can arise in several ways, and only one of them is "X causes Y":

- X really does cause Y (feature adoption causes retention).
- Y causes X (retained users stick around long enough to discover the feature — reverse causation).
- A third variable Z causes both (engaged users adopt features *and* retain — Z is a **confounder**).
- Pure chance in a finite sample (you tested twenty things; one lined up).

From the correlation alone, you cannot tell which story you are in. That is the whole problem. A model trained on observational data learns the correlation faithfully and will happily predict retention from feature adoption — but a *prediction* is not a *policy*. The question "who will retain?" is answered by correlation. The question "if I make everyone adopt this, will retention rise?" is a causal question, and correlation is silent on it.

## The counterfactual: the thing you can never see

To make "causation" rigorous, we need the idea of a **counterfactual**. Take a single user, Maya, and a single decision: show her the new onboarding flow, or the old one. There are two possible futures:

- Y₁ = Maya's retention **if she sees the new flow**
- Y₀ = Maya's retention **if she sees the old flow**

The **causal effect of the new flow on Maya** is the difference between these two futures: Y₁ − Y₀. If showing her the new flow makes her retain when she otherwise wouldn't have, that difference is real and it is caused by the flow.

Here is the catch, sometimes called the **fundamental problem of causal inference**: for any single person, you only ever observe *one* of these. Maya either saw the new flow or she didn't. The other future — the counterfactual — is forever unobserved. You cannot rewind Maya, wipe her memory, and run the other version. The individual causal effect is, for a single unit, unknowable.

This framing — every unit has a pair of *potential outcomes* (Y₀, Y₁), and we see only one — is the **potential outcomes framework** (also called the Rubin causal model). It is the backbone of everything in this course, so sit with it. The effect is defined in terms of a comparison we can never fully make for one person.

## From impossible individuals to achievable averages

If individual effects are hopeless, what can we recover? **Averages.** We give up on "what would this exact person have done otherwise" and aim for the **Average Treatment Effect (ATE)** across a population:

> ATE = E[Y₁] − E[Y₀]

the average outcome *if everyone got the treatment* minus the average outcome *if everyone got the control*. Both terms are still counterfactual — we never see a world where literally everyone is treated — but averages are estimable in a way individual effects are not, *if we set things up correctly.*

The naive approach is to compare the people who happened to get the treatment against the people who happened not to. Let's see why that breaks.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 10_000

# "engagement" is a hidden trait, 0..1. Engaged users self-select into the feature.
engagement = rng.random(n)
adopts = rng.random(n) < engagement          # engaged users adopt more
# True retention: baseline + a real +0.05 causal lift from adopting.
base = 0.2 + 0.6 * engagement
retention = rng.random(n) < (base + 0.05 * adopts)

naive = retention[adopts].mean() - retention[~adopts].mean()
print("naive adopter vs non-adopter gap:", round(naive, 3))
# output: naive adopter vs non-adopter gap: 0.247
```

The true causal lift we baked in is 0.05, but the naive comparison reports 0.247 — about five times too large. The gap is inflated by engagement: adopters were more retentive *before* the feature ever touched them. This is **confounding**, and it is not a small correction. It flipped a modest real effect into a headline number that would get the feature shipped for the wrong reason.

## Randomization: the one clean escape hatch

Now change one line. Instead of letting users self-select, we *assign* the feature by coin flip.

```python
# Randomized assignment: a coin flip, independent of engagement.
treat = rng.random(n) < 0.5
retention_rct = rng.random(n) < (base + 0.05 * treat)

ate_hat = retention_rct[treat].mean() - retention_rct[~treat].mean()
print("randomized estimate:", round(ate_hat, 3))
# output: randomized estimate: 0.053
```

Now the estimate is 0.053 — essentially the true 0.05. Nothing about the causal effect changed; the only thing that changed is *how people ended up in each group.*

Why does this work? Because the coin flip is independent of engagement, of the weather, of everything. On average the treatment and control groups have the *same* distribution of engagement, the same everything — they are two random halves of the same population. So any hidden confounder is, in expectation, balanced across the two arms. The difference in outcomes can then only be caused by the one thing we deliberately made different: the treatment. In potential-outcomes language, randomization makes treatment assignment **independent of the potential outcomes**, which is exactly the condition that turns the observed difference-in-means into an unbiased estimate of the ATE.

This is why the **randomized controlled trial (RCT)** — in industry, the **A/B test** — is called the gold standard. It does not require you to know or measure the confounders. You do not need to think of engagement, or the weather, or the day of the week. Randomization neutralizes all of them at once, including the ones you never thought of. That is a genuinely remarkable property, and no amount of clever statistics on observational data can fully replicate it.

## What randomization does not do

It is not magic, and honest practice means naming the limits:

- **Randomization balances groups only *on average* and only *at scale*.** With 10 users per arm, one arm can easily draw all the whales. Small experiments can still be badly imbalanced by chance — which is why sample size (lesson 02) matters.
- **It estimates an average, not your effect.** The ATE can be +5% overall while the feature *hurts* a segment. Averages hide heterogeneity (lesson 07).
- **You have to actually run it cleanly.** Broken randomization, users leaking between arms, or peeking at results early (lesson 04) all reintroduce the bias you paid randomization to remove.
- **Sometimes you simply can't randomize** — for ethics (you can't randomly assign smoking), feasibility (you can't randomize which country gets a law), or cost. That entire situation is what the second half of this course, causal inference from observational data, is about.

For now, hold onto the core mental model: every unit has two potential outcomes, you see one, the effect lives in the gap, and randomization is what lets you estimate the *average* gap without knowing what's confounding you. Everything else is refinement on this one idea.

## Key takeaways

- Correlation is symmetric and observational; causation is asymmetric and about *intervening*. A correlation can come from X→Y, Y→X, a common cause Z, or chance — the data alone can't tell you which.
- The potential outcomes framework says every unit has two outcomes, Y₀ and Y₁; the causal effect is Y₁ − Y₀, and the *fundamental problem* is that you only ever observe one of them per unit.
- Individual effects are unknowable, but the **Average Treatment Effect** E[Y₁] − E[Y₀] is estimable — if assignment is set up right.
- Naively comparing self-selected groups is confounded and can be wildly wrong (we saw a true +0.05 read as +0.28).
- **Randomization** makes assignment independent of potential outcomes, balancing every confounder — known and unknown — on average. That is why the A/B test is the gold standard.
- Randomization still needs adequate sample size, only gives an average, must be run cleanly, and isn't always possible.

## Try it

Reproduce the confounding demo above, then vary the strength of self-selection. Replace `adopts = rng.random(n) < engagement` with `adopts = rng.random(n) < engagement**k` and sweep `k` from 0 (adoption independent of engagement) up to 4 (strong self-selection). For each `k`, print both the naive gap and the randomized estimate. Watch the randomized estimate stay glued near 0.05 while the naive gap swings around. Write one sentence explaining why only one of the two numbers depends on `k`.
