# 06 — RL for LLMs

This is the lesson the whole course was built for. You now understand the MDP, value functions, policy gradients, advantage, and PPO. It turns out that training a large language model with reinforcement learning uses *exactly* this machinery — the same PPO objective, the same advantage estimates, the same exploration concerns — applied to an unusual environment where the "game" is generating text. This lesson maps every RL concept onto the LLM setting, walks the RLHF pipeline, and then covers the 2025-era shift to **GRPO** and **verifiable rewards** that powers today's reasoning models. It connects directly to the alignment chapter in the LLM Foundation-Model course and the LLM chapters in the GenAI course.

## Language generation as an MDP

The reframing is the key insight. Line up autoregressive text generation with the RL vocabulary:

- **State** `s_t` = the prompt plus all tokens generated so far.
- **Action** `a_t` = the next token to emit (the action space is the entire vocabulary — tens of thousands of actions).
- **Policy** `π_θ(a_t | s_t)` = the language model itself. Its softmax over the vocabulary *is* a stochastic policy. This is the pivotal equivalence: **an LLM is already a policy network.**
- **Transition** = deterministic and trivial — append the chosen token to the context. (The environment dynamics are known and free, which simplifies things versus robotics or games.)
- **Episode** = generating a full response, one token at a time, until an end-of-sequence token.
- **Reward** = a score for the *finished* response — how good the whole answer was.

That last point is where the RL structure earns its keep. The reward lands only at the *end* of the sequence (was the answer helpful? correct?), yet it must be attributed back to individual token choices made hundreds of steps earlier. That is the **credit assignment problem** from lesson 01, at scale — and precisely what policy-gradient methods with advantage estimation are built to handle. Pretraining and SFT are supervised (next-token prediction against a fixed corpus); RL fine-tuning is different in kind, because there is no "correct token" label — only an evaluative reward on the whole generation.

## RLHF: the three-stage pipeline

**RLHF** (Reinforcement Learning from Human Feedback) is the classic recipe — the one behind InstructGPT and the first ChatGPT — for turning a base model into a helpful assistant. It has three stages.

**Stage 1 — Supervised fine-tuning (SFT).** Fine-tune the base model on curated prompt→response demonstrations. This teaches format and basic instruction-following by imitation. It's ordinary supervised learning, and it produces the starting policy for RL.

**Stage 2 — Train a reward model.** We can't write a reward function for "is this a good answer" by hand. So we *learn* one. Humans are shown pairs of model responses to the same prompt and pick the better one. A **reward model** `RM(prompt, response) → scalar` is trained on these preference comparisons to predict which responses humans prefer. This is how fuzzy human judgment becomes the scalar reward RL requires — the reward model is the environment's reward function, learned from data.

**Stage 3 — Optimize the policy with PPO.** Now run PPO, exactly as you learned it:

1. Sample prompts, generate responses with the current policy (the LLM).
2. Score each response with the reward model — that's the reward.
3. Estimate advantages (with GAE), and update the policy with the **clipped PPO objective** to make high-reward responses more likely.
4. A separate value-network critic estimates `V(s)` for the advantage, just as in lesson 05.

Everything transfers. The LLM is the actor; a value head is the critic; the reward model provides reward; PPO's clip keeps the policy from lurching. There is one LLM-specific addition, and it matters enormously.

## The KL-to-reference penalty

Left alone to maximize reward-model score, the policy will **reward-hack**: it discovers weird, repetitive, or degenerate text that the reward model happens to score highly but that is actually terrible. The reward model is only an imperfect proxy for human preference, and any imperfect proxy, optimized hard enough, gets gamed (Goodhart's law in action). The policy also risks "forgetting" its language ability while chasing reward.

The fix is a **KL-divergence penalty** that keeps the trained policy close to the original SFT model (the **reference** policy). At every token, we subtract a penalty proportional to how far the current policy's distribution has drifted from the frozen reference:

```
reward_t = RM_score  −  β · KL( π_θ(·|s_t)  ||  π_ref(·|s_t) )
```

The `β` coefficient sets the leash length. This is a soft trust region layered on top of PPO's clip: PPO's clip limits each *update step*, while the KL penalty anchors the policy to a *known-good reference* over the whole run. It's the RLHF-specific term you won't see in game-playing PPO, and it's why aligned models stay fluent instead of collapsing into reward-hacked gibberish. Note this KL-to-reference is a *regularizer toward a fixed model*, distinct from (and additional to) any KL used inside the trust-region logic.

```python
# NN — Per-token reward in RLHF: reward-model score minus KL drift from reference
def rlhf_reward(rm_score, logp_policy, logp_ref, beta=0.1):
    # logp_* are log-probs of the generated tokens under each model
    kl = (logp_policy - logp_ref)           # per-token KL estimate (policy vs. frozen reference)
    return rm_score - beta * kl             # only the final token carries rm_score; KL applies throughout
```

## GRPO: dropping the critic

PPO for LLMs is expensive. You're holding *four* large models in play at once: the policy, the reference, the reward model, and the critic value network. The critic is especially painful — it's a second network the size of the policy, and training an accurate token-level value function for language is hard and unstable.

**GRPO (Group Relative Policy Optimization)**, introduced by DeepSeek (DeepSeekMath, and made famous by DeepSeek-R1 in January 2025), removes the critic entirely. The insight is a callback to lesson 04's baseline trick: the critic existed only to provide a *baseline* for the advantage. GRPO gets that baseline empirically instead of with a learned network.

For each prompt, GRPO samples a **group** of `G` responses from the current policy (say 8 or 16). It scores them all, then computes each response's advantage *relative to its group*:

```
A_i = ( r_i − mean(r_1..r_G) ) / std(r_1..r_G)
```

The group mean *is* the baseline — a direct, unbiased empirical estimate of the expected reward for that prompt, requiring no value network at all. A response that beats its group's average gets a positive advantage and is reinforced; one that lags gets pushed down. (Dividing by the group's standard deviation normalizes the signal across easy and hard prompts.) These group-relative advantages then feed the *same clipped PPO objective* and the *same KL-to-reference penalty* you already know.

```python
# NN — GRPO group-relative advantage (replaces the critic)
import torch

def grpo_advantages(rewards):                 # rewards: one scalar per response in the group
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std() + 1e-8)  # baseline = group mean; no value network
```

So GRPO = PPO's clip + KL penalty, but with the learned critic replaced by "sample a group and normalize against its mean." Dropping the critic cuts memory and compute roughly in half and removes a major source of instability. Since early 2025 it has become the default starting point for training reasoning and agentic models. *(GRPO is fast-moving research; treat the exact normalization and hyperparameters as current-as-of-2026 rather than settled — several variants exist.)*

## RLVR and verifiable rewards

GRPO pairs naturally with the other big 2025 shift: **RLVR (Reinforcement Learning with Verifiable Rewards)**. The idea is to sidestep the learned reward model — and its reward-hacking problems — entirely, for tasks where correctness can be *checked automatically*:

- **Math:** does the final answer match the known solution? Reward `1` if yes, `0` if no.
- **Code:** do the unit tests pass? Run them and read the result.
- **Structured tasks:** does the output satisfy a verifiable constraint (valid JSON, correct format)?

The reward becomes a simple, ungameble, rule-based `0/1` signal — objective rather than subjective. Because it's computed by a verifier (a math checker, a test harness) rather than a neural network, it cannot be reward-hacked the way a learned reward model can, and it's cheap to compute at scale. RLVR is *what* you reward (verifiable correctness); GRPO is *how* you optimize (group-relative, no critic). Together — verifiable reward, group-relative advantage, KL leash to the reference — they form the recipe behind DeepSeek-R1 and the current generation of reasoning models, which learn to produce long chains of thought purely because correct final answers get rewarded.

A related cousin is **RLAIF (RL from AI Feedback)**: instead of humans labeling preferences (RLHF) or a verifier checking correctness (RLVR), a strong LLM acts as the judge that provides the preference signal. It's a way to scale the *human* out of the loop when tasks aren't cleanly verifiable, trading some quality for enormous throughput.

## The landscape, in one table

| Method | Reward source | Baseline / critic | Best for |
|---|---|---|---|
| **RLHF (PPO)** | Learned reward model (human prefs) | Value-network critic | Helpfulness, open-ended quality |
| **RLAIF** | LLM judge (AI prefs) | Value-network critic or group | Scaling preference data |
| **RLVR (GRPO)** | Automatic verifier (0/1) | Group mean (no critic) | Math, code, reasoning |

All three run the same engine underneath: sample from the policy, score, estimate advantage, take a clipped policy-gradient step, stay near a reference via KL. That engine is everything you learned in lessons 04 and 05. The LLM setting changes the *environment* (text generation), the *reward* (preference model or verifier), and *one regularizer* (KL to reference) — not the fundamentals.

## Key takeaways

- Text generation is an **MDP**: state = context so far, action = next token, **the LLM's softmax is the policy**, reward lands on the finished response — a large-scale **credit assignment** problem.
- **RLHF** has three stages: **SFT** (imitation), a **reward model** (turns human preferences into scalar reward), and **PPO** (optimize the policy against that reward).
- A **KL-to-reference penalty** keeps the policy near the SFT model, preventing **reward hacking** and preserving fluency — the LLM-specific addition to PPO.
- **GRPO** drops the critic: it samples a **group** of responses and uses the **group mean as the baseline**, feeding group-relative advantages into the same clipped objective. Cheaper and more stable; the DeepSeek-R1 recipe.
- **RLVR** replaces the learned reward model with **automatic verifiers** (math answers, unit tests) giving ungameble 0/1 rewards; **RLAIF** uses an LLM as the preference judge.
- Under the hood it's all the **same policy-gradient + advantage + trust-region** machinery from lessons 04–05 — only the environment, reward, and reference regularizer are new.

## Try it

1. Write out, for a specific example ("What is 17 × 23?"), the state, action, episode, and reward under RLVR. What exactly computes the reward, and why can't the model hack it?
2. Explain in your own words why removing the critic (GRPO) is *safe* — connect it back to the baseline argument from lesson 04 (why does subtracting the group mean leave the gradient unbiased?).
3. Set `β = 0` in `rlhf_reward` (no KL penalty) and describe the likely failure mode over a long training run. Then argue what setting `β` too *high* would cost.
4. For each of these tasks, decide whether RLHF, RLVR, or RLAIF fits best and justify: (a) writing an empathetic support reply, (b) solving competition math, (c) generating valid SQL that runs. Cross-reference the alignment chapter in the LLM Foundation-Model course.
