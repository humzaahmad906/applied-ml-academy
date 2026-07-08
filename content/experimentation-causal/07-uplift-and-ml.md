# 07 — Uplift and ML

Every method so far has chased a single number: the *average* treatment effect. But averages hide people. A discount that lifts purchases by 3% on average might do +15% for bargain-hunters, nothing for loyalists who'd buy anyway, and *−5%* for a segment that finds the coupon spammy and unsubscribes. Ship it to everyone and you leave money on the table and actively harm some users. The question that actually drives decisions is not "does the treatment work on average?" but "**for whom** does it work, and who should I treat?" This lesson is about estimating effects that vary across people — heterogeneous treatment effects — and using them to target. It's where causal inference meets machine learning, and it's the beating heart of modern growth, marketing, and personalization systems.

## From ATE to CATE

The average treatment effect, ATE = E[Y₁ − Y₀], collapses everyone into one number. The **Conditional Average Treatment Effect (CATE)** keeps the covariates:

> τ(x) = E[Y₁ − Y₀ | X = x]

the treatment effect *for units with characteristics x*. If x describes a bargain-hunter, τ(x) is large; for a loyalist, τ(x) ≈ 0; for the coupon-averse, τ(x) < 0. Estimate τ(x) well and you can rank every user by how much the treatment would help *them* — the foundation of targeting.

But recall the fundamental problem from lesson 01: for any single user you see *one* potential outcome, never both. You can never observe an individual's true Y₁ − Y₀, so you can't just train a model to predict it directly — there's no label. **Uplift modeling** is the collection of tricks for estimating τ(x) anyway, and the good news is that they mostly reduce the causal problem to ordinary supervised learning you already know. The essential prerequisite: the data must come from a **randomized experiment** (or a setting where you've credibly removed confounding via lesson 06's methods), so that within any group defined by x, treatment is as-good-as-random and the group means are unbiased.

## Uplift is not response

The most important and most-violated distinction: **uplift (who the treatment changes) is not the same as response (who has a good outcome).** Marketers instinctively target people *likely to buy*. But four types of people exist under a treatment like a discount:

- **Persuadables** — buy *only if* treated. τ(x) large and positive. **These are the only people worth treating.**
- **Sure things** — buy either way. τ(x) ≈ 0. Treating them wastes the discount on a sale you'd have gotten free.
- **Lost causes** — buy under neither. τ(x) ≈ 0. Treatment is wasted; they won't convert regardless.
- **Sleeping dogs** — buy *unless* treated; the treatment annoys them into leaving. τ(x) **negative**. Treating them actively backfires.

A response model targets "likely buyers," which lumps persuadables together with sure things — so you spend heavily on people who'd have bought anyway, and worse, a response model is blind to sleeping dogs. Uplift modeling targets τ(x) directly, isolating the persuadables and *avoiding* the sleeping dogs. This is the entire commercial reason the field exists: the same budget spent on high-uplift users instead of high-response users can multiply incremental revenue.

## The meta-learners

**Meta-learners** estimate CATE by wiring together standard regressors/classifiers (call the base learner a "learner" — any of random forest, gradient boosting, a neural net). Three dominate.

**S-learner (Single).** Train *one* model on all the data with the treatment indicator `T` as just another feature: `μ(x, t)`. Then τ̂(x) = μ(x, 1) − μ(x, 0) — predict each unit twice, once with the treatment flag on, once off, and difference. Simple and data-efficient, but its weakness is real: if the model regularizes the treatment feature away (easy when the effect is small relative to the outcome), it predicts zero uplift for everyone.

**T-learner (Two).** Train *two separate* models — one on the treated units (`μ₁`), one on the control units (`μ₀`) — and set τ̂(x) = μ₁(x) − μ₀(x). It can't ignore treatment (it's baked into the split) and handles arms with different structure well. Its weakness: it splits your data in two, and errors in the two independently-fit models compound, which hurts when one arm is small.

**X-learner (Cross).** A cleverer, multi-step scheme that shines with **imbalanced arms** (e.g., 5% treated, 95% control — common when treatment is expensive). It (1) fits T-learner outcome models, (2) *imputes* each unit's individual treatment effect using the other arm's model, (3) fits regressors to those imputed effects, and (4) combines them with a propensity-based weight that leans on the larger arm where each model is more trustworthy. More moving parts, but strong when treatment is rare.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
n = 8000
X = rng.normal(0, 1, (n, 3))
T = rng.binomial(1, 0.5, n)                       # randomized treatment
# True heterogeneous effect: strong when X[:,0] > 0, negative when X[:,0] < -1.
tau = 2.0 * (X[:, 0] > 0) - 1.0 * (X[:, 0] < -1)
Y = X[:, 1] + 0.5 * X[:, 2] + tau * T + rng.normal(0, 1, n)

# T-learner: separate model per arm.
m1 = GradientBoostingRegressor().fit(X[T == 1], Y[T == 1])
m0 = GradientBoostingRegressor().fit(X[T == 0], Y[T == 0])
tau_hat = m1.predict(X) - m0.predict(X)

# Does predicted uplift track the truth? Check top vs bottom predicted-uplift groups.
top = tau_hat > np.quantile(tau_hat, 0.8)
bot = tau_hat < np.quantile(tau_hat, 0.2)
print("true effect, top-uplift group:", round(tau[top].mean(), 2))
print("true effect, bottom-uplift group:", round(tau[bot].mean(), 2))
# output: true effect, top-uplift group: 2.0
# output: true effect, bottom-uplift group: -0.78
```

The T-learner ranked users well: the group it flagged as high-uplift genuinely has a ~+2.0 effect, while the group it flagged low sits *negative* — those are the sleeping dogs you'd want to *exclude*. Note we never needed individual ground-truth effects to validate the ranking; we leaned on the randomized arms. In production, reach for the well-tested implementations — Microsoft's **EconML** (`SLearner`, `TLearner`, `XLearner`, `CausalForestDML`) or Uber's **CausalML** — rather than hand-rolling, especially for the X-learner's bookkeeping. **Causal forests** are a popular alternative that estimate τ(x) directly with honest, split-sample trees and give confidence intervals for free.

## Evaluating uplift: the Qini and uplift curves

You can't score an uplift model with accuracy or RMSE — there's no per-user label to compare against. Instead you evaluate the **ranking**: if the model is good, treating users in its predicted-uplift order should capture incremental conversions faster than random targeting. The **uplift curve** (and its close relative the **Qini curve**) plots cumulative incremental gain as you treat more of the population in model-ranked order; the area between that curve and the random-targeting diagonal — the **Qini coefficient** — is the standard summary, analogous to AUC. A large-scale 2024–2025 benchmark on the 14-million-row Criteo uplift dataset found that the best meta-learner's top-20% ranked users captured roughly 78% of *all* incremental conversions — a vivid illustration of why ranking by uplift beats treating everyone.

The evaluation logic: bucket users by predicted uplift, then *within each bucket* compute the actual treated-minus-control outcome difference (valid because treatment was randomized). A good model shows a monotone gradient — high-predicted-uplift buckets have large real effects, low buckets near zero or negative. That gradient, not any single accuracy number, is what tells you the model found the persuadables.

## Targeting: turning τ̂(x) into a decision

A CATE estimate is not yet a policy. The final step is turning τ̂(x) into *who gets treated*, and that's a business optimization, not a statistics problem:

- **Threshold rule:** treat everyone with τ̂(x) > c. Setting c = 0 already beats treating-everyone by dropping sure things and sleeping dogs. Raise c to be more selective.
- **Cost-aware rule:** a discount costs money and each conversion has value, so treat when `τ̂(x) × value > cost`. This naturally excludes low-uplift users whose expected incremental value doesn't cover the treatment's cost.
- **Budget-constrained rule:** if you can only treat k% of users, rank by τ̂(x) and treat the top k% — exactly what the Qini curve says maximizes incremental return.

Two honest cautions. First, uplift estimates are *noisier* than average effects — you're slicing the data finely, so each τ̂(x) rests on less signal; validate that the ranking holds on a fresh holdout before trusting it with real budget. Second, the whole edifice assumes the training data came from a randomized (or de-confounded) experiment — feed an uplift model naive observational data and it will confidently learn confounding as if it were treatment effect, reproducing exactly the lesson-01 mirage at the individual level. Uplift modeling is the powerful synthesis of this course — potential outcomes from lesson 01, randomization from lessons 02–04, and supervised ML — but it inherits every one of their assumptions.

## Key takeaways

- The **CATE** τ(x) = E[Y₁ − Y₀ | X=x] captures how the treatment effect *varies* across people; averages hide winners, non-responders, and those the treatment harms.
- **Uplift ≠ response:** target **persuadables** (buy only if treated), skip **sure things** and **lost causes**, and avoid **sleeping dogs** (treatment backfires). Response models can't see sleeping dogs.
- **Meta-learners** turn CATE into supervised learning: **S-learner** (one model, treatment as a feature), **T-learner** (separate model per arm), **X-learner** (imputation + weighting, best for imbalanced arms).
- You can't observe individual effects, so evaluate the **ranking** with uplift/Qini curves, not accuracy; a good model shows a monotone gradient of real effects across predicted-uplift buckets.
- Turn τ̂(x) into a **targeting policy** with a threshold, a cost-aware rule (`τ̂·value > cost`), or a budget-constrained top-k selection.
- Uplift inherits every earlier assumption — above all it needs randomized or de-confounded data, or it learns confounding as if it were effect. Use EconML/CausalML in production and validate on a holdout.

## Try it

Using the meta-learner dataset, build a simple uplift curve. Sort users by predicted τ̂(x) descending. Then, sweeping a treatment fraction from 0% to 100%, compute the cumulative incremental gain if you treated the top-ranked fraction — approximate it by, within the treated-fraction cohort, taking (mean Y of actually-treated) − (mean Y of actually-control) scaled by the cohort size. Plot this against the straight diagonal you'd get from random targeting. Confirm the model's curve bows above the diagonal, and read off how much of the total incremental gain the top 20% of users captures. Then flip to targeting the *bottom* 20% and note the negative gain — the sleeping dogs — and write one sentence on what that implies for a naive "treat the most likely buyers" campaign.
