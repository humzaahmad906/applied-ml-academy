# 05 — Actor-Critic and PPO

The last lesson ended on a cliffhanger: policy gradients need a baseline to tame their variance, the best baseline is `V(s)`, and estimating `V(s)` means training a second network. This lesson picks up exactly there. We'll combine a policy (the **actor**) with a value estimator (the **critic**) into **actor-critic** methods, sharpen the advantage estimate with **GAE**, and then build up to **PPO** — the algorithm that is, quite literally, the workhorse of both game-playing RL and modern LLM alignment. If you understand PPO, lesson 06 on LLMs will feel like a small step rather than a leap.

## Actor and critic

An **actor-critic** agent holds two learned functions:

- The **actor** `π_θ(a|s)` — the policy that chooses actions. Trained by the policy gradient, exactly as in REINFORCE.
- The **critic** `V_φ(s)` — a value network that estimates the expected return from a state. Trained by regression to observed returns (or bootstrapped targets, TD-style).

The critic's whole job is to serve as the baseline the actor needs. The actor's update becomes advantage-weighted:

```
∇_θ J(θ) = E [ ∇_θ log π_θ(a_t|s_t) · A_t ]
```

where the advantage `A_t` is estimated using the critic. The two learn together in a tight loop: the critic watches the actor's experience and learns to predict returns; the actor uses the critic's predictions to figure out which actions beat expectations and shifts probability toward them.

Why is this better than plain REINFORCE? Because the critic lets us **bootstrap**. REINFORCE had to wait for a full episode to compute the return `G_t`. With a critic, we can estimate the value of the rest of the episode instead of living it. The simplest advantage estimate uses a single step:

```
A_t ≈ r_t + γ·V_φ(s_{t+1})  −  V_φ(s_t)
```

That's just the TD error again — the same quantity from Q-learning, now serving as a low-variance (but somewhat biased, because the critic is imperfect) advantage. We've traded REINFORCE's unbiased-but-noisy Monte Carlo returns for the critic's biased-but-stable estimates. That bias–variance dial is the next idea.

## GAE: tuning the bias–variance dial

There's a spectrum of ways to estimate the advantage, and they trade bias against variance:

- **One-step TD** (`r_t + γV(s_{t+1}) − V(s_t)`): low variance (only one real reward involved), high bias (leans hard on the imperfect critic).
- **Full Monte Carlo** (`G_t − V(s_t)`, the REINFORCE-with-baseline estimate): unbiased (uses real returns), high variance.
- Everything in between: 2-step, 3-step, ... n-step returns.

**Generalized Advantage Estimation (GAE)** elegantly blends *all* of these into one estimator with a single knob, `λ` (lambda) in `[0, 1]`. It's an exponentially-weighted average of the n-step TD errors:

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)              # the one-step TD error at time t
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

The `λ` knob interpolates the whole spectrum: `λ = 0` collapses to one-step TD (low variance, high bias); `λ = 1` recovers the full Monte Carlo advantage (unbiased, high variance). In practice `λ ≈ 0.95` and `γ ≈ 0.99` are near-universal defaults — mostly low-variance, with a little Monte Carlo honesty mixed in. GAE is computed with the same backward accumulation trick as the return, over a batch of collected steps:

```python
# NN — Generalized Advantage Estimation
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    # values has one extra entry: the bootstrapped value of the final next-state
    advantages, gae = [], 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]   # one-step TD error
        gae = delta + gamma * lam * gae                          # exponential blend
        advantages.insert(0, gae)
    return advantages
```

GAE is the standard advantage estimator inside PPO — including the PPO that trains LLMs — so it's worth recognizing on sight.

## The problem PPO solves

We have an advantage-weighted policy gradient. Why not just take big steps up it? Because policy gradients are **on-policy**, and that makes large steps dangerous. The gradient is only valid *at the current policy*; take one big step and the policy you land on may be so different that the data you collected no longer describes it. A single oversized update can collapse the policy into garbage, and — unlike supervised learning — there's no fixed dataset to recover from, because the ruined policy now collects ruined data. RL training is a feedback loop, and instability compounds.

So we want to squeeze *multiple* gradient steps out of each batch of collected experience (for sample efficiency) without letting the policy wander too far from the one that gathered that data. That's precisely the tension PPO manages.

## PPO's clipped objective

**Proximal Policy Optimization (PPO)** works with the probability *ratio* between the new policy and the old one that collected the data:

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

If the new policy is more likely to take `a_t` than the old was, `r_t > 1`; less likely, `r_t < 1`; identical, `r_t = 1`. The vanilla surrogate objective is `r_t(θ)·A_t` — increase the ratio for good actions (positive advantage), decrease it for bad ones. But maximized freely, that objective would drive the ratio to extremes, exactly the runaway step we're trying to prevent.

PPO's contribution is the **clipped surrogate objective**, which caps how much the ratio is allowed to help:

```
L_CLIP(θ) = E_t [ min( r_t(θ)·A_t ,  clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]
```

with `ε` typically `0.2`. Unpack the `min` and the `clip` by cases — it's more intuitive than it looks:

- **Good action (`A_t > 0`):** we want to raise `r_t`. But once `r_t` exceeds `1+ε`, the clipped term flattens the objective — there's no further reward for pushing that action's probability even higher. The update is capped, so one lucky action can't dominate.
- **Bad action (`A_t < 0`):** we want to lower `r_t`. Once `r_t` drops below `1−ε`, the objective again flattens — no extra credit for slamming the probability to zero.
- The `min` ensures the clipping only ever *removes* incentive to move too far; it never rewards moving further than the trust region allows.

The effect is a **pessimistic, self-limiting objective**: the policy is free to improve within a trust region of width `ε` around the old policy, but gets no benefit from leaving it. This is what lets PPO safely take *several* epochs of minibatch gradient steps on the same batch of data before collecting fresh experience — a big win in sample efficiency over single-use REINFORCE.

```python
# NN — PPO clipped policy loss
import torch

def ppo_clip_loss(logp_new, logp_old, advantages, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)           # r_t = pi_new / pi_old (in log space)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return -torch.min(unclipped, clipped).mean()      # negative: maximize the surrogate
```

## Putting the full PPO loss together

The clipped policy loss is one of three terms. The complete PPO objective, optimized jointly, is:

```
L = L_CLIP(θ)  −  c1 · L_value(φ)  +  c2 · H[π_θ]
```

- **`L_CLIP`** — the clipped policy (actor) loss above.
- **`L_value`** — the critic's loss, plain MSE between `V_φ(s)` and the observed returns. This trains the value network whose estimates GAE relies on. (`c1 ≈ 0.5`.)
- **`H[π_θ]`** — an **entropy bonus**, added to keep the policy from collapsing to determinism too early. Higher entropy = more exploration; the bonus gently discourages premature confidence. (`c2 ≈ 0.01`.)

The full loop: collect a batch of experience with the current policy, compute advantages with GAE and returns for the critic, then run several epochs of minibatch SGD on `L` — reusing the batch under the protection of the clip — then discard the batch and repeat. Actor and critic often share a network trunk to save compute.

That's the entire algorithm. PPO is popular for one reason: it is *robust*. It's not the most sample-efficient method on paper, but it works across an enormous range of problems with barely any tuning, rarely blows up, and is straightforward to implement. That reliability is why, when researchers needed an RL algorithm to fine-tune language models on human feedback, they reached for PPO — and why the next lesson can treat PPO as a known quantity and focus on what's genuinely new about the LLM setting.

## Key takeaways

- **Actor-critic** pairs a policy (**actor**, trained by policy gradient) with a value network (**critic**, trained by regression) that supplies the advantage baseline and enables **bootstrapping**.
- The **advantage** `A_t` measures how much an action beat the expected value; estimating it well is a **bias–variance tradeoff**.
- **GAE** blends n-step advantage estimates with one knob `λ`: `λ=0` is low-variance/high-bias one-step TD, `λ=1` is unbiased/high-variance Monte Carlo. Defaults `γ=0.99, λ=0.95`.
- On-policy gradients are unsafe at large step sizes; **PPO** constrains updates with a **clipped surrogate objective** on the probability ratio `r_t = π_new/π_old`, keeping the policy in a trust region of width `ε` (~0.2).
- Clipping makes the objective **pessimistic and self-limiting**, enabling multiple epochs of reuse per batch — far more sample-efficient than REINFORCE.
- Full PPO loss = **clipped policy loss + value loss + entropy bonus**. PPO's defining virtue is **robustness**, which is why it became the default for RLHF.

## Try it

1. Extend your REINFORCE CartPole agent into A2C: add a critic head, replace `G_t` with the TD-error advantage, and train both. Compare stability against plain REINFORCE.
2. Swap the one-step advantage for `compute_gae`. Sweep `λ ∈ {0, 0.5, 0.95, 1.0}` and describe the effect on the reward curve's smoothness and final performance.
3. Implement `ppo_clip_loss` and run 4 epochs of minibatch updates per collected batch. Then set `eps` very large (e.g. `100`) to effectively disable clipping — what happens to training stability, and why?
4. Remove the entropy bonus. Does the policy converge faster or collapse to a suboptimal deterministic action? Relate this to exploration.
