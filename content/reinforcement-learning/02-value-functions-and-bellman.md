# 02 — Value Functions and the Bellman Equation

The last lesson said the agent's job is to maximize expected return. But return is a property of an entire future trajectory — you only know it after the fact. To *act* well in the moment, the agent needs a way to judge a situation before the future has played out. That judgment is the **value function**, and the equation that ties present value to future value — the **Bellman equation** — is the single most important idea in all of RL. Almost every algorithm in this course is a way of solving, approximating, or sidestepping Bellman.

## What a value function measures

A **value function** answers the question: "Starting from here and acting according to my policy, how much return should I expect?" It converts the messy, delayed, stochastic future into one number attached to a state. That number is what lets an agent prefer one situation over another without simulating the rest of time.

There are two flavors, and the distinction matters throughout the course.

The **state-value function** `V^π(s)` is the expected return starting from state `s` and following policy `π` thereafter:

```
V^π(s) = E_π[ G_t | s_t = s ]
```

The **action-value function** `Q^π(s, a)` is the expected return starting from state `s`, taking action `a` *first*, and following `π` afterward:

```
Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]
```

The `E_π` (expectation under `π`) is doing heavy lifting: because the world is stochastic and the policy may be too, the return is a random variable, and we care about its average. Both functions are always tied to a specific policy — value is not an absolute property of a state, it's the value *of that state to that policy*. Change the policy and the numbers change.

Why keep both? `V` tells you how good it is to *be* somewhere. `Q` tells you how good it is to *do* something. And `Q` has a decisive practical advantage: if you know `Q`, choosing the best action is trivial — just pick the `a` with the largest `Q(s, a)`, no model of the environment required. That property is why the whole family of Q-learning methods (next lesson) exists. The two are related by averaging `Q` over the policy's action choices:

```
V^π(s) = Σ_a  π(a|s) · Q^π(s, a)
```

## The Bellman equation: value defined by future value

Here's the pivotal observation. The return has a recursive structure — the return from now equals the immediate reward plus the (discounted) return from the next step:

```
G_t = r_{t+1} + γ·G_{t+1}
```

Take the expectation of both sides and the same recursion appears in the value function. This is the **Bellman expectation equation** for `V^π`:

```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a) + γ·V^π(s') ]
```

Strip away the sums and it reads in plain English: **the value of a state is the immediate reward you expect plus the discounted value of wherever you land next.** The two summations are just averaging — over the actions your policy might take (`π(a|s)`) and over the states the environment might send you to (`P(s'|s,a)`).

This is a profound move. It turns a statement about an infinite future (`V` is defined via `G_t`, an infinite discounted sum) into a **local** relationship between a state and its immediate neighbors. You no longer need to simulate to the end of time; you only need to relate each state to the ones one step away. That locality is what makes value functions computable.

The same recursion holds for `Q`:

```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a) + γ Σ_{a'} π(a'|s') Q^π(s',a') ]
```

The Bellman equation is a **consistency condition**. A correct value function is one that is self-consistent under this equation everywhere. If your estimated values don't satisfy Bellman, the mismatch — the **Bellman error** — is precisely the signal that tells you how to fix them. Every value-based learning algorithm is, at heart, a procedure for driving the Bellman error to zero.

## From "evaluate a policy" to "find the best one"

So far we've been *evaluating* a fixed policy. But we want the *best* policy. Define the **optimal value functions** as the best achievable over all policies:

```
V*(s) = max_π V^π(s)        Q*(s,a) = max_π Q^π(s,a)
```

These satisfy a sharper relation, the **Bellman optimality equation**, where the average-over-actions is replaced by a `max` — instead of averaging over what some policy does, we assume we act optimally, taking the best action every time:

```
V*(s)   = max_a  Σ_{s'} P(s'|s,a) [ R(s,a) + γ·V*(s') ]
Q*(s,a) =        Σ_{s'} P(s'|s,a) [ R(s,a) + γ·max_{a'} Q*(s',a') ]
```

The `max` is what turns evaluation into optimization. And it hands us the payoff for `Q`: once you have `Q*`, the optimal policy is simply

```
π*(s) = argmax_a Q*(s,a)
```

Greedy action selection on `Q*` is optimal behavior. No environment model, no planning — just look up the row for the current state and take the biggest entry. This is the target that Q-learning chases.

## Policy versus value: two ways to hold a solution

Step back and notice there are two distinct objects in play, and this split organizes the entire field:

- **Value-based methods** learn a value function (`V` or `Q`) and derive the policy from it implicitly, by acting greedily. The policy is a *consequence* of the values. Q-learning and DQN (next lesson) live here.
- **Policy-based methods** parameterize and learn the policy `π` *directly*, without necessarily learning values at all. REINFORCE and policy gradients (lesson 04) live here.
- **Actor-critic methods** do both — learn a policy (actor) *and* a value function (critic) that helps train it. PPO (lesson 05) lives here, and PPO is the bridge to LLMs.

Keep this three-way map in your head; it's the skeleton the rest of the course hangs on.

## Dynamic programming: solving Bellman when you know the world

If you actually *know* the MDP — the transitions `P` and rewards `R` — you can solve the Bellman equations directly with **dynamic programming (DP)**. DP won't scale to real problems (you rarely know `P`, and you can't enumerate huge state spaces), but it makes the machinery concrete and every later algorithm is a stochastic, sampled approximation of it.

The core routine is **value iteration**: start with arbitrary values and repeatedly apply the Bellman optimality equation as an *update rule*, sweeping over all states until the numbers stop changing.

```python
# NN — Value iteration on a known MDP
import numpy as np

def value_iteration(states, actions, P, R, gamma=0.9, tol=1e-6):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            v_old = V[s]
            # Bellman optimality: value of s = best action's (reward + discounted next value)
            V[s] = max(
                sum(P[(s, a)][s2] * (R[(s, a)] + gamma * V[s2]) for s2 in states)
                for a in actions
            )
            delta = max(delta, abs(v_old - V[s]))
        if delta < tol:          # values have converged — Bellman error is ~0 everywhere
            return V
```

Each sweep pushes the estimates one step closer to self-consistency; the `max` change per sweep (`delta`) shrinks toward zero, and the Bellman contraction guarantees convergence to `V*`. Once you have `V*` (or `Q*`), reading off the greedy policy gives you the optimal controller.

The essential thing to carry forward: **value iteration bootstraps** — it updates each estimate using other current estimates (`V[s2]`), not ground-truth returns. This "learn a guess from a guess" idea, called **bootstrapping**, is what lets learning happen online from single transitions rather than complete trajectories. It's also what makes value learning fast, and occasionally unstable. Q-learning in the next lesson is exactly value iteration with the known `P` and `R` replaced by samples the agent collects itself.

There's a close cousin worth naming: **policy iteration**, which alternates two steps — *policy evaluation* (solve the Bellman expectation equation for the current policy's `V^π`) and *policy improvement* (make the policy greedy with respect to those values). Each round produces a strictly better policy until no improvement is possible, at which point you've reached `π*`. Value iteration is essentially policy iteration with the evaluation step truncated to a single sweep. The reason to know this pair is that the *generalized* pattern — evaluate the current behavior, then improve it, repeat — is the skeleton of nearly every RL algorithm in the course, including the actor-critic and PPO methods that train language models. The critic evaluates; the actor improves. Same loop, sampled and scaled.

## Key takeaways

- A **value function** summarizes the expected future return of a situation: `V^π(s)` for being in a state, `Q^π(s,a)` for taking an action then following `π`. Value is always relative to a policy.
- `Q` is prized because acting greedily on it — `argmax_a Q(s,a)` — needs no model of the world.
- The **Bellman equation** makes value *local*: value now = expected immediate reward + discounted value of the next state. It's a consistency condition; violating it produces the **Bellman error** that drives learning.
- The **Bellman optimality equation** swaps the average-over-actions for a `max`, turning evaluation into optimization; `π*(s) = argmax_a Q*(s,a)`.
- The field splits into **value-based**, **policy-based**, and **actor-critic** methods — a map for the whole course.
- **Dynamic programming** (value iteration) solves Bellman when the MDP is known, by **bootstrapping** — updating estimates from other estimates. Real algorithms replace the known model with samples.

## Try it

1. Build a tiny 3-state MDP by hand (states A, B, C; C is terminal with reward 1). Define transitions and run the `value_iteration` function. Print `V` each sweep and watch it converge.
2. Set `γ = 0` and rerun. What do the values become, and why? Then set `γ = 0.99` and describe how the values far from the reward change.
3. Given a converged `V*`, write a few lines that extract the greedy policy (the best action in each state). Confirm it points "toward" the reward.
4. Explain in one sentence why knowing `Q*` frees you from needing `P`, but knowing `V*` does not.
