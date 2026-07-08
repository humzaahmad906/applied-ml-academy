# 02 — Designing an A/B Test

Most failed experiments were doomed before a single user was bucketed. The team picked a metric that couldn't move, ran on a sample too small to see the effect they cared about, or never wrote down what "success" meant, so every result became a debate. Analysis gets all the glory, but design is where experiments are won or lost. This lesson walks the design checklist a good experimenter fills in *before* launch: the hypothesis, the metrics, the sample size, and the randomization unit — ending with a power calculation you can run yourself.

## Start with a sharp hypothesis

"Let's see if the new checkout does better" is not a hypothesis; it's a vibe. A usable hypothesis names a change, a metric, a direction, and ideally a magnitude:

> *Replacing the three-step checkout with a one-page checkout will increase completed purchases per session, and we care about a lift of at least 1 percentage point.*

That sentence already forces the three decisions the rest of design depends on: what you're changing (the treatment), what you're measuring (completed purchases per session), and how big an effect matters to you (1 pp). Write it down. It is the contract you'll hold your future, result-motivated self to when the data comes back messy.

Formally this maps onto the hypotheses from your stats course: the **null** H₀ says the new checkout changes nothing, and the **alternative** H₁ says it changes the metric. But the business framing above is what keeps the test honest.

## Choosing metrics: the OEC and its guardrails

You cannot optimize everything at once, so you pick one metric to be the arbiter. This is the **Overall Evaluation Criterion (OEC)** — the single number that, if it moves the right way, means the change was good. A strong OEC is:

- **Sensitive** — it can actually move within the experiment's window. Annual revenue per user is too slow; conversion rate in a session is measurable now.
- **Aligned with long-term value** — clicks are easy to move and easy to game (a confusing UI generates lots of frustrated clicks). Prefer metrics closer to real value: completed purchases, not add-to-cart taps.
- **Attributable to the change** — it responds to what you touched.

Then you add **guardrail metrics**: things you are *not* trying to improve but refuse to harm. A checkout change might lift conversion while quietly increasing refund rate, page-load time, or customer-service tickets. Guardrails catch the change that "wins" on the OEC by cannibalizing something you care about more. A common industry setup is one OEC plus three to five guardrails (latency, crash rate, revenue, an engagement counter). If the OEC goes up but a guardrail breaks, you have not won — you have found a tradeoff to escalate.

## Effect size and the MDE

The single most important design number is the **Minimum Detectable Effect (MDE)**: the smallest true effect you want the experiment to be able to catch reliably. It is a *business* decision, not a statistical one. If a 0.1 pp lift in conversion wouldn't change what you ship, don't build an experiment that strains to detect it — you'll just burn traffic. If 1 pp would be a clear win, set the MDE around there.

The MDE, the baseline rate, and your tolerances for the two errors from the last course together *pin down the sample size.* You cannot choose all four freely — they trade off:

- **α (significance level):** your tolerated false-positive rate, usually 0.05. Lower α → need more data.
- **power (1 − β):** the probability of detecting a true effect of MDE size, usually 0.80. Higher power → need more data.
- **MDE:** smaller effects need more data (quadratically — halving the MDE roughly *quadruples* the sample).
- **baseline variance:** noisier metrics need more data.

## The sample-size formula, and why it looks the way it does

For comparing two proportions (like conversion rates), the required sample size *per arm* is approximately:

```
n ≈ (z_{α/2} + z_β)² · [ p_A(1−p_A) + p_B(1−p_B) ] / (p_B − p_A)²
```

Every piece earns its place. The numerator's z-terms encode your α and power (stricter α or higher power → bigger z's → bigger n). The bracket is the variance of the two proportions — noisier metrics cost data. The denominator is the *squared* effect size, which is why small MDEs are so expensive: the effect enters squared, so detecting half the effect costs four times the users.

Let's make it concrete. Baseline conversion is 10%, we want to detect a lift to 11% (a 1 pp MDE), at α = 0.05 and 80% power.

```python
import numpy as np
from scipy import stats

p_a, p_b = 0.10, 0.11          # baseline and target
alpha, power = 0.05, 0.80

z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-sided: 1.959...
z_beta  = stats.norm.ppf(power)           # 0.8416...

num = (z_alpha + z_beta) ** 2 * (p_a * (1 - p_a) + p_b * (1 - p_b))
n_per_arm = num / (p_b - p_a) ** 2
print("n per arm:", int(np.ceil(n_per_arm)))
# output: n per arm: 14749
```

About 14,700 users *per arm* — roughly 30,000 total — to reliably catch a 1 pp lift on a 10% baseline. If you only have 5,000 users a week, that is about a six-week experiment, and now you know that *before* you start, not three weeks in when someone asks "how much longer?"

## The same thing with statsmodels

Hand-rolling the formula is great for intuition, but for production use `statsmodels`, which also handles the arcsine effect-size transform for proportions cleanly via `proportion_effectsize`.

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

effect = proportion_effectsize(0.11, 0.10)     # standardized (Cohen's h)
analysis = NormalIndPower()
n = analysis.solve_power(effect_size=effect, alpha=0.05, power=0.80,
                         ratio=1.0, alternative="two-sided")
print("n per arm (statsmodels):", int(np.ceil(n)))
# output: n per arm (statsmodels): 14745
```

The two numbers agree to within a handful of users (14,749 vs 14,745): the closed-form uses the raw proportion variance while `proportion_effectsize` uses the variance-stabilizing arcsine transform (Cohen's h), and for a small effect on a moderate baseline the two land in almost the same place. Both are standard; where sample-size tools disagree it's usually the effect-size convention, and the differences are well within the fuzz of your baseline estimate. For continuous metrics (revenue per user, session length), swap in `TTestIndPower` and express the effect size as (mean difference) / (standard deviation).

A useful sanity habit: run the calc, then flip it. Given the sample you can *actually* get in a reasonable window, `solve_power` for `power` — if it comes back at 0.4, your experiment is underpowered and likely to waste everyone's time.

## The randomization unit: the quiet killer

You must decide *what* gets randomized, and getting this wrong silently corrupts everything downstream. Common units:

- **User** (most common): each user is consistently in one arm across sessions and devices. Correct when the change could be noticed across visits — you don't want a user seeing the new checkout on Monday and the old one on Tuesday.
- **Session / request:** finer-grained, more statistical power, but wrong when the treatment has memory. Randomizing per session leaks the experience across arms for the same person.
- **Cluster** (account, classroom, city): required when users *interact*, or the effect **spills over** between them. If treating one user changes an untreated friend's behavior (social features, marketplaces, two-sided networks), user-level randomization violates the assumption that units don't affect each other (SUTVA), and you must randomize at the level of the interacting group even though it costs power.

Two more design musts. First, the **randomization unit should match the analysis unit** — if you randomize by user, analyze per user, or your variance math is wrong. Second, decide the **split** (usually 50/50, which maximizes power for a fixed total) and *commit to the duration up front.* You compute a sample size, you run until you hit it, you look once. The temptation to peek early and stop when it looks good is the single most common way to turn a well-designed experiment into a false positive — and it is the first pitfall we tackle in lesson 04.

## Key takeaways

- Design before you launch: a sharp hypothesis, the right metrics, and a sample size computed in advance prevent most experiment failures.
- The **OEC** is your single arbiter metric — sensitive, aligned with long-term value, and hard to game. **Guardrails** are metrics you refuse to harm while chasing it.
- The **MDE** is a business decision: the smallest effect worth detecting. It, α, power, and baseline variance jointly determine sample size — you can't pick all of them freely.
- Sample size scales with the *square* of the inverse effect size: halving the MDE quadruples the users needed.
- Compute sample size with the closed-form for intuition and `statsmodels` (`NormalIndPower`/`TTestIndPower`) in practice; flip the calc to check the power your realistic sample actually buys.
- Choose the randomization unit deliberately — user for cross-session changes, cluster when units interact or spill over — and match analysis unit to randomization unit.

## Try it

Build an MDE-vs-sample-size curve. For a 10% baseline, loop the target conversion from 10.2% up to 13% in small steps and, for each, compute the required n per arm with `NormalIndPower().solve_power`. Plot n against the MDE. You should see the cost explode as the MDE shrinks toward zero. Then annotate the point matching a sample size you could realistically collect in two weeks, and read off the smallest effect that experiment could actually detect. Write one sentence on what that means for a team that wants to "detect any improvement."
