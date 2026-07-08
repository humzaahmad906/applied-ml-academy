# 04 — Policy Gradients

Value-based methods learn a `Q` function and read off a policy by taking `argmax`. That works, but it has two nagging limits: the `argmax` makes the policy deterministic and awkward for continuous or very large action spaces, and it never optimizes what we actually care about — the policy — directly. **Policy gradient methods** flip the approach on its head. They *parameterize the policy itself* as a neural network and improve it with gradient ascent on expected return. This is the branch of RL that leads straight to how modern LLMs are trained, so it's worth building carefully.

## Parameterizing the policy directly

Instead of learning values, we make the policy a function with weights `θ`: `π_θ(a | s)`, a network that takes a state and outputs a probability distribution over actions. For a small discrete action space, that's a softmax over action logits; for continuous actions, it's often the mean and standard deviation of a Gaussian.

```python
# NN — A policy network for discrete actions
import torch, torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, s):
        logits = self.net(s)
        return torch.distributions.Categorical(logits=logits)  # a sampleable distribution
```

The agent acts by *sampling* from this distribution: `a ~ π_θ(·|s)`. That sampling gives us exploration for free — a stochastic policy naturally tries different actions — and it means the policy can express genuinely random optimal behavior (think rock-paper-scissors, where any deterministic policy is exploitable). Training now means one thing: **adjust `θ` so the policy assigns more probability to actions that lead to high return.**

Why bother, when Q-learning already worked? Three reasons make direct policy optimization worth it. **Continuous actions:** a robot arm's torque is a real number, and you can't take an `argmax` over a continuum — but a policy can just output the mean of a Gaussian. **Stochastic optimality:** some problems *require* randomness, and a value-greedy policy can't provide it. **Directness:** we optimize the thing we deploy, with no `max` operator to overestimate values (the DQN trouble from last lesson) and no requirement that the policy be a deterministic function of learned values. The cost, as we'll see, is variance — and that's exactly what LLM training also grapples with, which is why this branch is the one that leads to RLHF.

## The objective and its gradient

Define the objective as the expected return under the policy:

```
J(θ) = E_{τ ~ π_θ} [ R(τ) ]
```

where `τ` is a trajectory (a full episode) the policy generates, and `R(τ)` is its total return. We want to climb this — do gradient *ascent*: `θ ← θ + α·∇_θ J(θ)`. The difficulty is that `θ` affects `J` through *which trajectories get sampled*, and you can't obviously differentiate through a sampling process. The environment's dynamics `P(s'|s,a)` are also unknown and not differentiable.

The **policy gradient theorem** resolves this, and the result is beautifully clean:

```
∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t  ∇_θ log π_θ(a_t | s_t) · R(τ) ]
```

Do not let the derivation intimidate — the *meaning* is what matters. The key trick (the "log-derivative trick") converts a gradient of an expectation into an expectation of a gradient, which we can estimate just by sampling trajectories and averaging. And notice what's absent: no gradient of the environment appears. We only ever differentiate `log π_θ`, which is *our own network* and fully differentiable. The unknown, non-differentiable world shows up only as the scalar `R(τ)` — a number we observe, not something we backprop through.

Read the formula as an instruction: **`∇_θ log π_θ(a_t|s_t)` is the direction in weight space that makes action `a_t` more likely. Multiply it by the return `R(τ)`, and you push probability up on actions from good trajectories and down on actions from bad ones**, in proportion to how good or bad. Reward-weighted maximum likelihood. That's the entire idea.

## REINFORCE: the algorithm

Turn the theorem into code and you get **REINFORCE** (Williams, 1992), the original policy-gradient algorithm. The recipe:

1. Run the current policy to collect one or more complete episodes.
2. Compute the return `G_t` from each time step to the episode's end.
3. Form the loss `−Σ_t log π_θ(a_t|s_t) · G_t` (negative because optimizers minimize).
4. Backprop and step. Repeat.

```python
# NN — REINFORCE update from one episode
def reinforce_update(policy, optimizer, states, actions, rewards, gamma=0.99):
    # 1) discounted return from each step onward (reward-to-go)
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # 2) reward-weighted log-likelihood loss
    loss = 0.0
    for s, a, Gt in zip(states, actions, returns):
        dist = policy(s)
        loss = loss - dist.log_prob(a) * Gt      # push up log-prob of actions, scaled by return
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

Two refinements are already baked in above and both are standard. First, we use the **reward-to-go** `G_t` (return from step `t` onward) rather than the whole-episode `R(τ)` for every step. This is correct because an action can only influence rewards that come *after* it — crediting it with rewards from *before* it is pure noise. Same expected gradient, less variance. Second, REINFORCE is fundamentally **on-policy** and **Monte Carlo**: it needs complete episodes (to compute real returns) and the data must come from the *current* policy. The moment you update `θ`, the old episodes are stale and must be thrown away. That's a real inefficiency, and it's exactly what PPO in the next lesson works around.

Contrast with Q-learning: no bootstrapping (we use actual returns, not estimated ones), no `max` operator, no target network, and it optimizes the policy we deploy. The tradeoff is variance, which we turn to now.

## The variance problem, and baselines

REINFORCE works, but it is *noisy*. The return `R(τ)` of a single episode is a high-variance random variable — the same policy can produce a triumphant episode and a disastrous one back to back, mostly due to environment randomness the agent didn't control. Multiply a wildly varying `R(τ)` into every gradient and the updates jitter, so you need tiny learning rates and enormous numbers of episodes to average the noise out. High variance is *the* practical weakness of vanilla policy gradients.

The cure is the single most important trick in the whole family: **subtract a baseline.** Replace the raw return with the return *minus a reference value* `b(s)`:

```
∇_θ J(θ) = E [ Σ_t ∇_θ log π_θ(a_t|s_t) · ( G_t − b(s_t) ) ]
```

Here's the crucial fact: as long as the baseline `b(s)` doesn't depend on the action, **subtracting it leaves the gradient's expected value unchanged** — it introduces zero bias — while it can dramatically *reduce variance*. Intuitively, what should drive an update is not "was the return large in absolute terms?" but "was this action *better than typical* from this state?" Subtracting a baseline recenters the signal around zero: actions above the baseline get pushed up, actions below get pushed down, and an action that was merely average produces almost no update instead of a big one.

The natural, near-optimal baseline is the state-value function `V(s)` — the expected return from that state. Then the weighting becomes `G_t − V(s_t)`, which is an estimate of the **advantage**:

```
A(s, a) = Q(s, a) − V(s)
```

The advantage answers precisely the right question: *how much better than average is taking action `a` in state `s`?* A positive advantage means the action beat expectations (increase its probability); negative means it underperformed (decrease it). Advantage-weighted policy gradients are the workhorse form, and reframing the update around advantage — rather than raw return — is the conceptual hinge between this lesson and the next.

But wait: to use `V(s)` as a baseline, we need to *estimate* `V(s)`. That means learning a value function alongside the policy — one network (the **actor**) that acts, and another (the **critic**) that estimates value to reduce the actor's variance. That combination is **actor-critic**, and it's the subject of the next lesson and the foundation of PPO.

## Key takeaways

- **Policy gradient methods** parameterize the policy `π_θ(a|s)` as a network and do gradient *ascent* on expected return, optimizing the policy directly instead of deriving it from `Q`.
- The **policy gradient theorem** gives `∇_θ J = E[Σ_t ∇_θ log π_θ(a_t|s_t)·R(τ)]` — reward-weighted maximum likelihood. Only the policy is differentiated; the unknown environment enters only as observed scalar rewards.
- **REINFORCE** implements it: collect full episodes, weight each action's log-prob by its **reward-to-go** `G_t`, and step. It is **on-policy** and **Monte Carlo** — episodes are single-use.
- Vanilla policy gradients suffer from **high variance**. Subtracting a **baseline** `b(s)` reduces variance with **zero added bias**.
- The best baseline is `V(s)`, giving the **advantage** `A(s,a) = Q(s,a) − V(s)` — "how much better than average was this action." Advantage weighting is the standard form.
- Estimating `V(s)` requires a second network, leading to **actor-critic** methods (next lesson).

## Try it

1. Implement `reinforce_update` and train `PolicyNet` on CartPole (`gym`/`gymnasium`). Plot episode reward over training — expect noisy but rising curves.
2. Replace reward-to-go `G_t` with the whole-episode return `R(τ)` for every step. Does learning get noisier or slower? Explain using the causality argument.
3. Add a constant baseline equal to the running mean of episode returns (`G_t − mean_return`). Compare the variance of the reward curve with and without it.
4. Explain in one sentence why subtracting a baseline that depends only on the *state* leaves the gradient unbiased, but a baseline that depended on the *action* would not.
