# 03 — Q-Learning and DQN

Value iteration solved the MDP, but it cheated: it needed the transition probabilities `P` and reward function `R` handed to it in advance. Real agents don't get that. They're dropped into a world they don't understand and have to learn purely from experience — from tuples of `(state, action, reward, next state)` that they collect by acting. **Q-learning** is the algorithm that does this. It learns `Q*` directly from samples, without ever building a model of the environment. Then we'll scale it from a lookup table to a neural network and arrive at **DQN**, the method that first made deep RL work on raw pixels.

## Tabular Q-learning

Start in the simplest possible setting: few enough states and actions that we can store one number for every `(state, action)` pair in a table `Q[s][a]`. The agent's entire knowledge is that table.

Recall the Bellman optimality equation for `Q`. Value iteration turned it into an update by averaging over the known transition model. Q-learning does the same thing but replaces the expectation with a *single observed sample*. After taking action `a` in state `s`, seeing reward `r`, and landing in `s'`, it nudges `Q[s][a]` toward the value that sample implies:

```
Q[s][a]  ←  Q[s][a]  +  α · ( r + γ · max_{a'} Q[s'][a']  −  Q[s][a] )
```

This one line is the heart of the lesson, so read it slowly.

- `r + γ · max_{a'} Q[s'][a']` is the **TD target** — our best current estimate of what `Q[s][a]` *should* be, given this transition. It's the immediate reward plus the discounted value of acting greedily from the next state. It is exactly the right-hand side of the Bellman optimality equation, but for this one sampled `s'` instead of an average over all possible `s'`.
- `r + γ·max Q[s'][a'] − Q[s][a]` is the **TD error** (temporal-difference error) — the gap between the target and what we currently believe. This is the sampled Bellman error from the last lesson, made concrete.
- `α` (alpha) is the **learning rate**: how far to move toward the target on each step. Small `α` means slow, stable learning; large `α` means fast but jumpy.

So Q-learning says: *whenever reality disagrees with your estimate, move your estimate a fraction `α` of the way toward what reality just showed you.* Do this enough times, visiting every state-action pair often enough, and the table provably converges to `Q*`.

Two properties make this remarkable. First, it **bootstraps**: the target uses `Q[s']`, our own current estimate, not a real observed return. We learn a guess from a guess — and it still works, because the reward `r` injects a grain of ground truth at every step. Second, it's **off-policy**: the update uses `max_{a'} Q[s'][a']` regardless of what action the agent *actually* took next. This means the agent can explore with a random, exploratory policy while still learning about the *optimal* greedy policy. Off-policy learning is what lets us reuse old experience — a fact DQN exploits heavily.

```python
# NN — Tabular Q-learning update
def q_update(Q, s, a, r, s_next, done, alpha=0.1, gamma=0.99):
    best_next = 0.0 if done else max(Q[s_next])      # no future value past a terminal state
    td_target = r + gamma * best_next
    td_error = td_target - Q[s][a]
    Q[s][a] += alpha * td_error                       # step toward the target
    return td_error
```

## A gridworld to make it real

Picture a 4×4 grid. The agent starts top-left, the goal is bottom-right (reward +1, ends the episode), every other step gives reward 0, and actions are up/down/left/right. There's no model given — the agent learns which moves pay off purely by wandering and updating.

```python
# NN — Q-learning on a 4x4 gridworld
import numpy as np, random

SIZE = 4
GOAL = (SIZE - 1, SIZE - 1)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

def step(pos, a):
    dr, dc = ACTIONS[a]
    r, c = pos
    nr, nc = min(max(r + dr, 0), SIZE - 1), min(max(c + dc, 0), SIZE - 1)  # walls clamp
    npos = (nr, nc)
    reward = 1.0 if npos == GOAL else 0.0
    return npos, reward, npos == GOAL

Q = np.zeros((SIZE, SIZE, 4))
alpha, gamma, epsilon = 0.1, 0.95, 0.2

for episode in range(2000):
    pos = (0, 0)
    for _ in range(100):                          # cap steps so a lost agent still ends
        if random.random() < epsilon:
            a = random.randrange(4)               # explore
        else:
            a = int(np.argmax(Q[pos[0], pos[1]]))  # exploit
        npos, reward, done = step(pos, a)
        best_next = 0.0 if done else np.max(Q[npos[0], npos[1]])
        Q[pos[0], pos[1], a] += alpha * (reward + gamma * best_next - Q[pos[0], pos[1], a])
        pos = npos
        if done:
            break

# Greedy path length from the learned Q
greedy = [int(np.argmax(Q[r, c])) for r in range(SIZE) for c in range(SIZE)]
print("Value of start state:", round(float(np.max(Q[0, 0])), 3))
# output: Value of start state: 0.735   (~ gamma^6, the 6 optimal steps to the goal)
```

Watch what happened. Early episodes are almost pure luck — the agent stumbles to the goal by chance, and a single +1 reward updates only the last cell. But bootstrapping propagates that value *backward* over subsequent episodes: the cell next to the goal learns it's valuable, then the cell before that, and so on, until value has flowed all the way back to the start. The final `Q[0,0]` sits near `γ⁶ ≈ 0.735`, exactly the discounted value of a 6-step optimal path. The agent discovered the shortest route without ever being told the layout.

## When the table won't fit

Tabular Q-learning is beautiful and useless for real problems. A gridworld has 16 states; a chess position has ~10⁴⁵; an Atari screen is 210×160 pixels — more distinct states than atoms in the observable universe. You cannot store a row per state, and even if you could, you'd never visit each one to fill it in.

The fix is **function approximation**: instead of a table, use a parameterized function `Q(s, a; θ)` that *generalizes* across states. Feed it a state, get out `Q` values for each action. Similar states produce similar values, so learning about one state teaches you about its neighbors — the agent no longer needs to visit every state, only enough to fit the function. When that function is a neural network, we call it a **Deep Q-Network (DQN)**.

The idea sounds like a trivial swap — replace the table update with a gradient step — but doing it naively *diverges*. Two problems, two fixes, and those fixes are what DQN actually contributed.

## DQN: making function approximation stable

**Problem 1 — correlated data.** An agent's consecutive experiences are highly correlated (frame `t` looks like frame `t+1`). Training a network on a stream of correlated samples is like training a classifier on a dataset sorted by label — it overfits to the recent past and forgets the rest. The fix is a **replay buffer**: store every transition `(s, a, r, s')` in a large memory, and train on *random mini-batches* sampled from it. This breaks the correlations (samples in a batch come from all over the agent's history) and reuses each experience many times — which is only sound because Q-learning is off-policy, so old data collected under an older policy is still valid to learn from.

**Problem 2 — a moving target.** In the update `r + γ·max Q(s'; θ) − Q(s; θ)`, the same network `θ` appears in both the prediction and the target. So the target shifts every time you update the weights — you're chasing a goalpost you move with your own feet, and the feedback loop can blow up. The fix is a **target network**: a second copy of the network, `θ⁻`, used only to compute the target, and frozen for many steps (then periodically copied from the live network). Now the target holds still long enough for the live network to converge toward it.

Put together, the DQN loss for a mini-batch is a supervised-looking regression toward the (frozen-target) Bellman target:

```
L(θ) = E_{(s,a,r,s') ~ buffer} [ ( r + γ·max_{a'} Q(s', a'; θ⁻)  −  Q(s, a; θ) )² ]
```

```python
# NN — One DQN gradient step (PyTorch sketch)
import torch, torch.nn.functional as F

def dqn_step(q_net, target_net, batch, optimizer, gamma=0.99):
    s, a, r, s_next, done = batch                       # tensors from the replay buffer
    q_pred = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)   # Q(s,a) for actions taken
    with torch.no_grad():                                # target network is not trained
        q_next = target_net(s_next).max(dim=1).values    # max_a' Q(s', a'; theta-minus)
        td_target = r + gamma * q_next * (1 - done)       # zero out future past terminal
    loss = F.mse_loss(q_pred, td_target)                  # regress prediction -> target
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

Everything else is Q-learning you already know: act ε-greedily to explore, store transitions, sample batches, take gradient steps, and every `C` steps copy `θ` into `θ⁻`. With these two stabilizers, the 2015 DQN learned to play dozens of Atari games from raw pixels at human level using one architecture and one set of hyperparameters — the result that kicked off the deep-RL era.

DQN also has a well-known bias worth flagging: the `max` in the target systematically **overestimates** action values (you're taking the max of noisy estimates, and max-of-noise skews high). **Double DQN** fixes it by using the live network to *choose* the next action and the target network to *evaluate* it. Keep the phenomenon in mind; the `max` operator is a recurring troublemaker in value-based RL, and it's part of why the policy-gradient methods in the next lesson avoid it entirely.

## Key takeaways

- **Q-learning** learns `Q*` from sampled transitions with the update `Q[s][a] ← Q[s][a] + α·(r + γ·max Q[s'][a'] − Q[s][a])` — no model of the environment required.
- The **TD error** `r + γ·max Q[s'] − Q[s][a]` is the sampled Bellman error; **bootstrapping** propagates reward backward over episodes.
- Q-learning is **off-policy** (learns the greedy policy while exploring), which is what makes experience reuse valid.
- Tables don't scale; **function approximation** with a neural net (**DQN**) generalizes across states, but naive training diverges.
- **DQN's two fixes:** a **replay buffer** (decorrelate and reuse data) and a **target network** (hold the Bellman target still). The loss is MSE regression toward the frozen target.
- The `max` overestimates values; **Double DQN** decouples action selection from evaluation to reduce the bias.

## Try it

1. Run the gridworld code. Print `Q[0,0]` every 200 episodes and watch value flow back from the goal. Then set `epsilon = 0` (no exploration) and explain why learning stalls or gets stuck.
2. Raise `gamma` to `0.999` and drop it to `0.5`. How does the learned start-state value change, and how does that match `γ^(steps to goal)`?
3. In the DQN sketch, delete the target network (use `q_net` for both prediction and target). Describe what could go wrong and why the frozen copy helps.
4. Explain in your own words why a replay buffer is only valid *because* Q-learning is off-policy.
