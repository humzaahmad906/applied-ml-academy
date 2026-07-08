# 01 — The RL Problem

Supervised learning shows a model the right answer for every example. Reinforcement learning never does. Instead it drops an **agent** into a world, lets it act, and hands back a thin trickle of reward — a number that says "that was good" or "that was bad" without ever explaining *why* or telling you what you should have done instead. From that trickle the agent has to figure out a whole strategy. This lesson lays out the pieces of that setup, why it's genuinely harder than supervised learning, and the vocabulary the rest of the course is built on.

## Agent and environment

Everything in RL is a loop between two things. The **agent** is the decision-maker — the thing we're training. The **environment** is everything else: the world the agent acts in and that reacts to what the agent does.

The loop runs in discrete time steps. At each step `t`:

1. The agent observes the current **state** `s_t` — a description of the situation.
2. It picks an **action** `a_t`.
3. The environment responds with a **reward** `r_{t+1}` (a scalar) and a new state `s_{t+1}`.
4. Repeat.

That's it. A chess agent sees the board (state), moves a piece (action), and eventually gets +1 for winning or −1 for losing (reward). A robot sees its joint angles, applies torques, and gets rewarded for staying upright. A language model sees a prompt-so-far, emits a token, and — much later — gets a reward for whether the finished answer was good. The frame is astonishingly general, which is exactly why RL shows up everywhere from robotics to game-playing to aligning LLMs.

```python
# NN — The agent–environment loop (schematic)
state = env.reset()
done = False
while not done:
    action = agent.act(state)              # agent decides
    next_state, reward, done = env.step(action)  # environment responds
    agent.learn(state, action, reward, next_state)
    state = next_state
```

Notice what's *missing* compared to supervised learning: there is no label. Nobody tells the agent "the correct action in state `s_t` was `a*`." There's only a reward, and rewards are **evaluative** (how good was what you did) rather than **instructive** (what you should have done). That single difference is the source of nearly every hard problem in RL.

## The reward, and why it's slippery

The reward is the only signal the agent gets about what we want. It is deceptively small — one number per step — and it carries three difficulties that supervised learning simply doesn't have.

**Delay.** The reward for a good decision often arrives long after the decision. Sacrifice your queen on move 12, win on move 40 — which move deserves the credit? This is the **credit assignment problem**, and it's central. The agent has to learn to connect outcomes to the actions that actually caused them, across gaps of many steps.

**Sparsity.** In many problems the reward is zero almost everywhere and nonzero only at rare moments (you reach the goal, you win, you crash). A random agent may act for thousands of steps and never once see a nonzero reward, which means it never learns anything. Reward sparsity is why some tasks are hard even when the rules are simple.

**The agent's actions change its data.** In supervised learning the dataset is fixed and handed to you. In RL the agent's own behavior determines what states it visits and therefore what it sees next. A cautious agent that never explores a region of the world can never learn that region is valuable. The data distribution is a moving target that the policy itself controls — a feedback loop with no analog in supervised training.

A useful mental anchor is the **reward hypothesis**: any goal we care about can, in principle, be expressed as the maximization of expected cumulative reward. Whether that's *always* true is debatable, but it's the working assumption that lets us treat wildly different tasks with one framework.

## The Markov Decision Process

To reason about this precisely we formalize the environment as a **Markov Decision Process (MDP)**, the mathematical object underneath all of RL. An MDP is five pieces:

- **States** `S` — the set of situations the agent can be in.
- **Actions** `A` — what the agent can do.
- **Transition dynamics** `P(s' | s, a)` — the probability of landing in state `s'` after taking action `a` in state `s`. The world can be stochastic; the same action needn't always lead to the same place.
- **Reward function** `R(s, a)` — the expected reward for taking `a` in `s`.
- **Discount factor** `γ` (gamma), between 0 and 1 — how much we care about the future versus the present. More on this below.

The word **Markov** carries real weight. The **Markov property** says the next state and reward depend *only* on the current state and action, not on the full history of how you got there:

```
P(s_{t+1} | s_t, a_t)  =  P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...)
```

In other words, the state is a *sufficient summary* of the past. This is not a triviality — it's a modeling choice, and much of the art of applying RL is defining a state that actually captures everything relevant. A chess position is Markov (the board tells you everything). A single video frame is *not* Markov for a moving ball (you can't tell which way it's going), which is why agents that see raw frames often stack several together to recover velocity. Get the state wrong and every theorem below quietly breaks.

## Policy, return, and discounting

The agent's behavior is captured by its **policy**, written `π`. A policy is a mapping from states to actions — the agent's strategy. It can be deterministic (`a = π(s)`) or, more commonly, stochastic: `π(a | s)` is a probability distribution over actions. Training an RL agent *is* finding a good policy.

What makes a policy good? Not the immediate reward but the total reward it accumulates over time. We call the cumulative future reward the **return**, `G_t`:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
```

Each future reward is multiplied by `γ` raised to how many steps away it is. This **discounting** does two jobs at once. Mathematically, if the interaction can go on forever, discounting (`γ < 1`) keeps the infinite sum finite and well-defined. Behaviorally, it encodes how far-sighted the agent is:

- `γ = 0`: totally myopic. Only the very next reward matters. `G_t = r_{t+1}`.
- `γ` near `1` (say `0.99`): far-sighted. Rewards far in the future count almost as much as immediate ones.

```python
# NN — Computing discounted return from a reward sequence
def discounted_return(rewards, gamma=0.99):
    G = 0.0
    for r in reversed(rewards):     # walk backward so each step multiplies by gamma once
        G = r + gamma * G
    return G

print(discounted_return([0, 0, 0, 1], gamma=0.9))
# output: 0.7290000000000001   (the reward of 1, four steps out, discounted three times)
```

The backward accumulation trick — `G = r + γ·G` — is worth internalizing now, because the Bellman equation in the next lesson is essentially this same recursion turned into a fixed-point equation.

The agent's objective, stated cleanly, is: **find the policy `π` that maximizes the expected return.** Everything else in the course — value functions, Q-learning, policy gradients, PPO — is a different algorithm for doing exactly that.

## Exploration versus exploitation

Because the agent generates its own data, it faces a dilemma with no counterpart in supervised learning. At any moment it can **exploit** — take the action it currently believes is best — or **explore** — try something else to learn whether a better option exists.

Pure exploitation is a trap: the agent locks onto the first decent strategy it finds and never discovers the great one sitting just out of view. Pure exploration is also a trap: it wanders forever, never cashing in on what it knows. Good learning requires balancing the two, usually exploring a lot early (when the agent knows little) and gradually shifting toward exploitation as its estimates sharpen.

The simplest scheme, which we'll use repeatedly, is **ε-greedy** (epsilon-greedy): with probability `ε` act randomly, otherwise take the best-known action. Decay `ε` over time and you get exactly the "explore early, exploit late" behavior.

```python
# NN — epsilon-greedy action selection
import random

def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(q_values))  # explore: random action
    return max(range(len(q_values)), key=lambda a: q_values[a])  # exploit: best known
```

The exploration–exploitation tradeoff is not a nuisance to be engineered away; it is intrinsic to learning from evaluative feedback. Keep it in mind — it explains many design decisions in the algorithms ahead.

## How this differs from supervised learning

It's worth stating the contrast bluntly, because it reframes everything you already know:

| | Supervised learning | Reinforcement learning |
|---|---|---|
| Signal | Correct label per example | Scalar reward, possibly delayed and sparse |
| Feedback | Instructive ("the answer is X") | Evaluative ("that was worth 0.3") |
| Data | Fixed, i.i.d. dataset given to you | Generated by the agent; distribution shifts with the policy |
| Objective | Minimize prediction error now | Maximize cumulative future reward |
| Core difficulty | Fit the mapping | Credit assignment + exploration |

RL is not a fancier classifier. It's a different problem: sequential decision-making under uncertainty, learning from consequences rather than from answers. That reframing is the whole point of this first lesson.

## Key takeaways

- RL is a loop: an **agent** observes a **state**, takes an **action**, and receives a **reward** and a next state from the **environment**.
- Reward is **evaluative, not instructive** — it says how good, never what was right — and it is often **delayed** (credit assignment) and **sparse**.
- The **MDP** `(S, A, P, R, γ)` formalizes the environment; the **Markov property** says the state summarizes the past, which is a modeling responsibility, not a freebie.
- The **policy** `π(a|s)` is the strategy; the **return** `G_t` is discounted cumulative reward; the objective is to **maximize expected return**.
- **Discounting** (`γ`) keeps infinite sums finite and sets how far-sighted the agent is.
- The **exploration–exploitation tradeoff** is intrinsic: ε-greedy is the simplest way to balance it.

## Try it

1. Implement `discounted_return` above and compute the return for `rewards = [1, 1, 1, 1, 1]` at `γ = 0.5`, `0.9`, and `0.99`. Explain the trend in the three numbers.
2. A reward of `+10` arrives 20 steps in the future. At `γ = 0.9`, what is its present (discounted) value? At `γ = 0.99`? What does the gap tell you about choosing `γ`?
3. Write down the state, action, and reward for two problems you know (e.g., a thermostat, a game you play). For each, argue whether the state you chose is truly Markov — and if not, what you'd add to make it so.
