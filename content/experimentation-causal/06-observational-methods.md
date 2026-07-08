# 06 — Observational Methods

You've accepted you can't run an experiment, drawn the DAG, and identified the confounders you need to neutralize. Now what do you actually *compute*? This lesson is the practical toolbox: four workhorse methods for estimating causal effects from observational data, each built on a different assumption about *why* treatment and control differ, and each with a clear "use this when" signature. None of them is magic — every one trades the ironclad guarantee of randomization for an assumption you must argue is plausible. The skill is matching the method to the structure of your problem and being honest about the assumption you're leaning on.

## Matching and propensity scores

**The idea.** If confounding comes from treated and untreated units having different characteristics, then *manufacture* comparability: for each treated unit, find untreated units that look just like it on the confounders, and compare within those pairs. It's the backdoor criterion made operational — you're conditioning on the adjustment set by literally matching on it.

Matching directly on many covariates fails (the "curse of dimensionality" — nobody matches exactly on 20 variables). The fix is the **propensity score**: the probability a unit gets treated given its covariates, `e(x) = P(treatment = 1 | X = x)`, estimated with a plain logistic regression. Rosenbaum and Rubin's key result is that matching on this single number is enough — units with the same propensity score have, on average, the same covariate distribution. So you collapse 20 confounders into one score and match (or weight, or stratify) on it.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 5000
age = rng.normal(40, 10, n)
income = rng.normal(50, 15, n)
# Older, higher-income users self-select into treatment (confounding):
logit = -6 + 0.08 * age + 0.03 * income
treat = rng.random(n) < 1 / (1 + np.exp(-logit))
# Outcome depends on confounders + a true +2.0 treatment effect:
y = 10 + 0.2 * age + 0.1 * income + 2.0 * treat + rng.normal(0, 3, n)

naive = y[treat].mean() - y[~treat].mean()

# Propensity score, then inverse-probability weighting (IPW):
ps = LogisticRegression().fit(np.c_[age, income], treat).predict_proba(np.c_[age, income])[:, 1]
w = np.where(treat, 1 / ps, 1 / (1 - ps))          # weight to balance the arms
ate_ipw = (np.sum(w * treat * y) / np.sum(w * treat)
           - np.sum(w * ~treat * y) / np.sum(w * ~treat))
print("naive:", round(naive, 2), " IPW:", round(ate_ipw, 2), " (truth 2.0)")
# output: naive: 4.03  IPW: 1.99  (truth 2.0)
```

The naive comparison reads 4.03 because treated users are older and richer (which independently raises `y`); reweighting by the propensity score rebalances the arms and recovers the true 2.0. **Use it when** you believe you've measured all the confounders (selection on observables) and you have cross-sectional data. **The assumption you're betting on:** no unmeasured confounding — the propensity model saw everything that drives both treatment and outcome. Also check **overlap**: if some units have propensity ≈ 0 or ≈ 1, there are no comparable counterparts and the weights explode. Matching cannot fix a confounder you didn't measure — that's the one thing it shares with every method here.

## Difference-in-differences

**The idea.** A treatment turns on for one group at one time (a policy rolls out in one state, a feature launches in one region). You have a treated group and an untreated group, each measured *before and after*. Comparing treated after-vs-before is confounded by anything that changed over time for everyone; comparing treated-vs-control after is confounded by pre-existing differences between the groups. **Difference-in-differences (DiD)** cancels both by taking a difference *of* differences: (treated after − treated before) − (control after − control before). The control group's before/after change estimates what *would* have happened to the treated group absent treatment — the counterfactual trend — and subtracting it isolates the treatment effect.

```python
# minimum wage rises in NJ, not in PA (the classic Card-Krueger shape)
nj_before, nj_after = 20.4, 21.0     # avg employment, treated state
pa_before, pa_after = 23.3, 21.2     # control state (declined anyway)
did = (nj_after - nj_before) - (pa_after - pa_before)
print("DiD estimate:", round(did, 2))
# output: DiD estimate: 2.7
```

Employment in NJ rose 0.6 while PA fell 2.1; DiD attributes +2.7 to the policy relative to the common trend. **Use it when** you have panel/repeated data spanning a discrete treatment event with a plausible comparison group. **The assumption you're betting on — parallel trends:** absent treatment, the two groups would have moved in *parallel*. You can't prove it, but you can support it by plotting pre-treatment periods and checking the lines tracked each other before the intervention. If they were already diverging, DiD is not for you.

## Instrumental variables

**The idea.** What if there's unmeasured confounding you *can't* adjust away — matching and DiD are both dead? Sometimes you can find an **instrument**: a variable that nudges the treatment but affects the outcome *only through* treatment, with no backdoor of its own. The instrument acts like a natural randomizer. The canonical example: to estimate the effect of military service on earnings, use the **draft lottery number** — it randomly pushed people into service (relevance) but has no other path to earnings (exclusion). You then use only the variation in treatment that the instrument *explains* — the "as good as random" slice — and discard the confounded rest.

Mechanically this is **two-stage least squares**: regress treatment on the instrument (stage 1) to get the instrument-driven part of treatment, then regress the outcome on *that* (stage 2).

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(1)
n = 5000
Z = rng.binomial(1, 0.5, n)                  # instrument (e.g., lottery)
U = rng.normal(0, 1, n)                       # UNMEASURED confounder
D = (0.3 * Z + 0.5 * U + rng.normal(0, 0.3, n) > 0.4).astype(float)  # treatment
Y = 1.5 * D + 2.0 * U + rng.normal(0, 0.5, n)  # true effect of D is 1.5; U confounds

# Stage 1: predict D from Z. Stage 2: regress Y on predicted D.
d_hat = sm.OLS(D, sm.add_constant(Z)).fit().predict()
iv = sm.OLS(Y, sm.add_constant(d_hat)).fit().params[1]
ols = sm.OLS(Y, sm.add_constant(D)).fit().params[1]
print("naive OLS:", round(ols, 2), " IV (2SLS):", round(iv, 2), " (truth 1.5)")
# output: naive OLS: 4.23  IV (2SLS): 1.4  (truth 1.5)
```

Plain OLS reads 4.23, badly biased by the unmeasured `U`; 2SLS recovers 1.40 — close to the true 1.5 — using only the lottery-driven variation. **Use it when** unmeasured confounding defeats adjustment but you have a credible natural experiment. **The assumptions you're betting on:** the instrument is *relevant* (genuinely moves treatment — a *weak* instrument gives wildly unstable estimates) and satisfies *exclusion* (affects the outcome through no other path — untestable and where most IV analyses live or die). Good instruments are rare and precious; a bad one is worse than no analysis.

## Synthetic control

**The idea.** DiD needs a comparison group, but what if only *one* unit is treated and no single other unit is a good match? California passes a tobacco law — no other state is "like California." **Synthetic control** builds a fake California: a *weighted combination* of the untreated states (a bit of Nevada, some Utah, etc.) chosen so the weighted blend tracks real California's outcome closely *in the pre-treatment period*. After the law, the gap between real California and this "synthetic California" is the estimated effect. It's DiD's sophisticated cousin — instead of one control with a parallel-trends assumption, you construct the best possible synthetic control from a *donor pool* and let the pre-period fit vouch for it.

```python
# Sketch of the logic (real use: the `pysyncon` or `SparseSC` libraries).
# Find donor weights w >= 0 summing to 1 that best reproduce the treated
# unit's PRE-treatment outcomes, then apply those weights POST-treatment:
#   synthetic_post = sum_j  w_j * donor_j_post
#   effect         = treated_post - synthetic_post
# The pre-treatment fit is the credibility check: if the synthetic can't
# track the treated unit BEFORE treatment, don't trust it after.
```

**Use it when** you have one (or few) treated units, many untreated donors, and a long pre-treatment history — comparative case studies of policies, big one-off launches. **The assumption you're betting on:** a convex combination of donors that matched the treated unit's *past* will keep matching its (untreated) *future*, and no donor was itself hit by the treatment's spillover. The pre-treatment tracking is your visible, checkable evidence — a synthetic control that fits the pre-period tightly is far more persuasive than one that doesn't.

## Choosing among them

A quick decision guide, since the methods aren't interchangeable — each answers a different data shape:

- **Cross-sectional data, all confounders measured** → propensity scores / matching / IPW.
- **A treatment event with before/after data and a comparison group** → difference-in-differences.
- **Unmeasured confounding, but a credible natural randomizer exists** → instrumental variables.
- **One or few treated units, many donor controls, long history** → synthetic control.

And the assumption to interrogate first, respectively: *did I measure every confounder* (matching), *parallel trends* (DiD), *exclusion restriction* (IV), *pre-period fit and no donor contamination* (synthetic control). Whichever you reach for, state the assumption out loud, show whatever evidence you can (overlap plots, pre-trends, first-stage strength, pre-period tracking), and remember the asterisk from lesson 05: none of these recovers the effect of a confounder you never saw. They buy you a defensible causal estimate, not the certainty a clean A/B test would have given you.

## Key takeaways

- **Propensity score** methods collapse many measured confounders into one score and match/weight on it (IPW recovered a true effect from a badly biased naive gap); they assume *no unmeasured confounding* and adequate overlap.
- **Difference-in-differences** cancels both time trends and fixed group differences via a difference of differences; it assumes **parallel trends**, checkable against pre-treatment periods.
- **Instrumental variables** (2SLS) exploit a natural randomizer to beat *unmeasured* confounding, using only the instrument-driven variation; they hinge on relevance and the untestable **exclusion restriction**.
- **Synthetic control** builds a weighted blend of donor units to serve as the counterfactual for one treated unit; pre-treatment fit is its credibility test.
- Match the method to the data shape: cross-sectional → matching; event + panel → DiD; unmeasured confounding + instrument → IV; single treated unit + donors → synthetic control.
- Every method trades randomization's guarantee for an assumption — name it, show evidence for it, and keep the unmeasured-confounding asterisk in view.

## Try it

Take the propensity-score dataset and stress-test the "no unmeasured confounding" assumption. Add a third confounder that drives both treatment and outcome — say `risk_tolerance` — but then fit the propensity model on *only* `age` and `income`, omitting it. Compare the IPW estimate to the true effect and watch the bias reappear. Then include `risk_tolerance` in the propensity model and confirm the estimate snaps back near the truth. Write two sentences explaining why the omitted confounder broke IPW, and connect it to the backdoor criterion from lesson 05.
