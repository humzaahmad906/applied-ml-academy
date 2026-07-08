# 07 — Project: Train an Agent from Scratch

You've seen the ideas; now build one end to end. This capstone walks you through training a REINFORCE-with-baseline agent (a minimal actor-critic) on **CartPole**, the "hello world" of RL: balance a pole on a moving cart by pushing it left or right. It's small enough to train on a laptop CPU in a couple of minutes, but it exercises every concept from the course — policy network, sampling, returns, advantage, policy-gradient loss — and its failure modes are the same ones you'll hit on any RL problem, including LLM fine-tuning. The point isn't CartPole; it's to feel the full loop working and learn to read its pathologies.

## The environment

CartPole (from Gymnasium, the maintained successor to OpenAI Gym) gives you:

- **State:** 4 numbers — cart position, cart velocity, pole angle, pole angular velocity.
- **Actions:** 2 — push left (`0`) or right (`1`).
- **Reward:** `+1` for every timestep the pole stays up.
- **Episode ends** when the pole falls past a threshold, the cart runs off-screen, or you reach 500 steps. So the return equals the number of steps survived — the agent's entire goal is *survive longer*, and it must discover that by trial and error, never being told which push was correct.

```python
# NN — Setup
import gymnasium as gym
import torch, torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")
STATE_DIM = env.observation_space.shape[0]   # 4
N_ACTIONS = env.action_space.n               # 2
torch.manual_seed(0); np.random.seed(0)      # reproducibility — set seeds in every RL run
```

## The agent: shared actor-critic network

One network with two heads — a policy head (actor) and a value head (critic). Sharing the trunk is standard and saves compute; the actor picks actions, the critic estimates `V(s)` to serve as the advantage baseline (lessons 04–05).

```python
# NN — Actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Linear(128, n_actions)   # -> action logits
        self.critic = nn.Linear(128, 1)          # -> V(s) estimate

    def forward(self, s):
        h = self.shared(s)
        dist = torch.distributions.Categorical(logits=self.actor(h))  # the policy
        value = self.critic(h).squeeze(-1)                            # the baseline
        return dist, value

policy = ActorCritic(STATE_DIM, N_ACTIONS)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
```

## The training loop

The full loop: collect a complete episode by sampling actions, compute discounted returns and advantages, then take one policy-gradient step. The advantage is `return − critic estimate`, and we train the critic to predict the returns.

```python
# NN — REINFORCE with a learned baseline (minimal actor-critic)
def run_episode(env, policy, gamma=0.99):
    log_probs, values, rewards = [], [], []
    state, _ = env.reset()
    done = False
    while not done:
        s = torch.as_tensor(state, dtype=torch.float32)
        dist, value = policy(s)
        action = dist.sample()                       # sample from the policy (exploration)
        log_probs.append(dist.log_prob(action))
        values.append(value)
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # discounted return (reward-to-go) from each step
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # normalize: variance control
    return torch.stack(log_probs), torch.stack(values), returns, sum(rewards)

def update(log_probs, values, returns, optimizer):
    advantages = returns - values.detach()           # detach: don't send actor grads into critic
    actor_loss = -(log_probs * advantages).mean()    # reward-weighted log-likelihood
    critic_loss = (returns - values).pow(2).mean()   # regress critic -> returns
    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()

# Train
recent = []
for episode in range(600):
    log_probs, values, returns, ep_reward = run_episode(env, policy)
    update(log_probs, values, returns, optimizer)
    recent.append(ep_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(recent[-50:])
        print(f"episode {episode+1:4d}   avg reward (last 50): {avg:.1f}")
# output (approximate — RL is stochastic, exact numbers vary by seed):
# episode   50   avg reward (last 50): 28.4
# episode  200   avg reward (last 50): 96.7
# episode  400   avg reward (last 50): 210.3
# episode  600   avg reward (last 50): 475.9   <- near the 500 cap: solved
```

If it works, you'll watch the average reward climb from ~20 (random flailing) toward the 500 cap. The agent taught itself to balance the pole from nothing but a `+1`-per-step signal — no demonstrations, no labels. That is the whole thesis of the course made concrete.

Trace the concepts as they appear in the code, because every one of them is something you'll meet again in a production LLM training loop:

- **The policy** is `dist` — a `Categorical` over actions, the direct analog of an LLM's softmax over the vocabulary.
- **Sampling** (`dist.sample()`) is the exploration mechanism; the same call, at LLM scale, is generating candidate responses to a prompt.
- **The return** is computed by the identical backward recursion (`G = r + gamma * G`) you first saw in lesson 01.
- **The advantage** (`returns - values.detach()`) is the "better than expected?" signal; GRPO computes the same quantity as "better than the group's average?"
- **The policy-gradient loss** (`-(log_probs * advantages)`) is reward-weighted maximum likelihood — push up the log-probability of actions that beat the baseline.

The `detach()` on the critic's values deserves a note: it stops the actor's loss from flowing gradients into the critic, so the two heads train on their own objectives (the actor on advantage, the critic on regressing to returns). Forgetting it is a subtle, common bug that quietly entangles the two losses.

## Pitfalls: how to read what goes wrong

RL fails differently from supervised learning, and the failures are informative. When (not if) your run misbehaves, match the symptom:

**Reward climbs, then collapses.** The single most common RL pathology. The policy improves, then suddenly craters back to random. Usually the learning rate is too high — one oversized update pushed the policy off a cliff, and because RL data is self-generated, the wrecked policy collects wrecked data and can't recover. Lower `lr`, or reach for PPO's clipped objective, which exists precisely to prevent this. This is the on-policy instability from lesson 05, live.

**No learning at all (flat at ~20).** Check the sign of your loss first — a flipped sign (gradient *descent* on return instead of ascent) is the classic bug, and it silently does the opposite of what you want. Then check that advantages aren't all near zero (a broken baseline) and that gradients are actually flowing to the actor.

**Wild variance, jagged curve.** Inherent to policy gradients (lesson 04). Mitigate with return normalization (already in the code above), the value baseline, or averaging gradients over several episodes per update instead of one.

**Premature convergence to a mediocre policy.** The policy went deterministic too early and stopped exploring — it exploits a so-so strategy forever. Add an **entropy bonus** to the loss (`− c · dist.entropy().mean()`) to keep the policy stochastic longer. This is the exploration–exploitation tradeoff from lesson 01.

**Seeds matter, and results are noisy.** The same code with two different seeds can look like success and failure. Always seed torch/numpy/env, and judge performance over *many* episodes and *several* seeds, never a single lucky run. This is a discipline that carries straight into LLM RL, where a single run tells you almost nothing.

**Reward normalization changes everything.** Try commenting out the `returns` normalization line and watch training destabilize. Small preprocessing choices have outsized effects in RL — far more than in supervised learning — because they directly rescale the gradient.

## Extensions

Once the baseline agent works, level up in the direction of the course's payoff:

1. **Upgrade to full PPO.** Replace the single-step REINFORCE update with the clipped objective and GAE from lesson 05; run multiple epochs per batch. Notice how much more stable and sample-efficient it is — this is the exact algorithm that fine-tunes LLMs.
2. **Harder environments.** Try `LunarLander-v3` (continuous-ish control, sparser reward) or `MountainCar` (genuinely sparse reward — a lesson in why exploration matters).
3. **Add an entropy bonus** and sweep its coefficient; watch it trade exploration for final performance.
4. **Connect it to LLMs.** Reflect: the CartPole policy is a tiny network mapping state→action distribution. An LLM is a huge network mapping context→next-token distribution. The training loop you just wrote — sample, score, advantage, policy-gradient step — is *structurally the same loop* that runs GRPO on a reasoning model. You've built the skeleton of modern LLM post-training.

## Key takeaways

- A complete RL agent is small: a policy network, an environment loop, discounted returns, an advantage, and a policy-gradient step. You built one.
- The **actor-critic** structure (shared trunk, policy head + value head) is the practical default and scales up to PPO and LLM training unchanged.
- **Reward collapse** usually means the learning rate is too high — the on-policy instability PPO's clip was designed to fix.
- **Return/advantage normalization**, a **value baseline**, and an **entropy bonus** are the standard variance-and-exploration controls.
- **Seed everything and evaluate over many runs** — RL is stochastic and single runs lie. This discipline transfers directly to LLM RL.
- The loop you wrote is the *same loop* that trains reasoning LLMs with GRPO — only the network and reward change.

## Try it

1. Run the project as-is and reproduce the learning curve. Then raise `lr` to `3e-2` and trigger a reward collapse — observe it, then explain the mechanism.
2. Comment out the return-normalization line. How much less stable is training? Why does rescaling the return matter so much for the gradient?
3. Add an entropy bonus to the loss and tune its coefficient. Find a value that speeds learning and one that harms it, and explain the tradeoff.
4. Implement the PPO extension (clip + GAE, multiple epochs per batch) and compare sample efficiency (episodes-to-solve) against the REINFORCE baseline. Write two sentences connecting your result to why PPO/GRPO is the workhorse for LLMs.
