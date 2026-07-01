# 17 — Interview Bank III: Derivations & First Principles

This is the whiteboard round. Not "what is FlashAttention" but "derive the attention backward pass";
not "what does Chinchilla say" but "start from `L(N,D)` and show me `N* ∝ √C`". Frontier labs run
this to find out whether you actually own the math or just narrate around it. The tell is that they
hand you a marker, erase an assumption halfway through, and watch whether your derivation bends or
breaks. Each question below has the full derivation at the depth an interviewer wants to see on the
board, followed by one line on what the interviewer is actually listening for. Read it, then close
it and reproduce every line from a blank page — a half-remembered derivation is worse than none,
because you will confidently write down a wrong sign.

Notation follows the earlier chapters: `N` non-embedding params, `D` tokens, `d = d_model`,
`d_ff`, `n = n_layers`, `L` context, `T = B·L`, `V` vocab. Natural log throughout unless noted.

---

## Part A — The core gradients

**A1. Derive `∂L/∂z` for softmax followed by cross-entropy. Show the clean collapse.**

Let `z ∈ ℝ^V` be the logits, `p = softmax(z)` so `p_i = e^{z_i} / Σ_k e^{z_k}`, and the loss for a
one-hot target `y` (true class `c`) be `L = −log p_c = −Σ_i y_i log p_i`.

First the softmax Jacobian. For `i = j`:

```text
∂p_i/∂z_i = [e^{z_i}·S − e^{z_i}·e^{z_i}] / S²  = p_i − p_i²  = p_i(1 − p_i)     (S = Σ_k e^{z_k})
```

For `i ≠ j`:

```text
∂p_i/∂z_j = [0 − e^{z_i}·e^{z_j}] / S²  = −p_i p_j
```

Both cases fold into one line with the Kronecker delta:

```text
∂p_i/∂z_j = p_i (δ_ij − p_j)
```

Now chain into the loss. `L = −Σ_i y_i log p_i`, so `∂L/∂z_j = −Σ_i (y_i / p_i) · ∂p_i/∂z_j`.
Substitute the Jacobian:

```text
∂L/∂z_j = −Σ_i (y_i / p_i) · p_i (δ_ij − p_j)
        = −Σ_i y_i (δ_ij − p_j)
        = −y_j + p_j · Σ_i y_i
        = p_j − y_j                          (since Σ_i y_i = 1)
```

So `∂L/∂z = p − y`: the softmax probabilities minus the one-hot. Every `1/p_i` cancelled against a
`p_i` in the Jacobian, which is exactly why the gradient is numerically clean and why frameworks fuse
softmax and cross-entropy into one op — you never form the Jacobian, you subtract a one-hot from the
probabilities. Note it holds for any target distribution `y` that sums to 1, not just one-hot.

*What the interviewer listens for:* that you write the `δ_ij` Jacobian and show the `1/p` cancellation
explicitly, rather than just asserting "it's `p − y`."

**A2. Why the `1/√d` scale in attention? Derive it from the variance of the dot product.**

Take a query `q` and key `k`, each `d`-dimensional, with components i.i.d. mean 0, variance 1, and
independent of each other. The unscaled score is `s = q·k = Σ_{i=1}^{d} q_i k_i`.

Mean: `E[s] = Σ_i E[q_i] E[k_i] = 0` by independence and zero mean.

Variance: the terms `q_i k_i` are independent across `i`, so variances add:

```text
Var(s) = Σ_i Var(q_i k_i)
Var(q_i k_i) = E[q_i² k_i²] − E[q_i k_i]²  = E[q_i²]E[k_i²] − 0  = 1·1 = 1
⇒ Var(s) = d
```

So the raw dot product has standard deviation `√d` and grows with dimension. Feed scores of typical
magnitude `√d` into softmax and, as `d` grows, the largest score dominates: the softmax saturates
toward a one-hot, its gradient `p(1−p)` collapses toward 0, and learning stalls. Dividing by `√d`
rescales the score to unit variance (`Var(s/√d) = d/d = 1`), independent of head dimension, keeping
the softmax in its responsive regime. This is why the scale is `1/√(d_k)` (head dim), not `1/√d`
(model dim) — the sum runs over the head dimension.

*What the interviewer listens for:* the variance-adds argument giving exactly `d`, and the connection
to softmax saturation killing the gradient — not just "it normalizes the scores."

**A3. Backprop through RMSNorm. Derive the gradient and show why normalization couples the components.**

RMSNorm is `y_i = g_i · x_i / r`, where `r = √( (1/d) Σ_k x_k² + ε )` and `g` is the
learned gain. Drop `ε` for the derivation. Let the upstream gradient be `∂L/∂y_i ≡ ḡ_i`.

The gain gradient is immediate: `∂L/∂g_i = ḡ_i · x_i / r`.

For `∂L/∂x_j` there are two paths, because `x_j` appears both directly in `y_j` and inside `r` (which
enters *every* `y_i`). Write `y_i = g_i x_i r^{-1}`.

```text
∂y_i/∂x_j = g_i [ δ_ij r^{-1} + x_i · ∂(r^{-1})/∂x_j ]
```

Now `r = (1/d · Σ_k x_k²)^{1/2}`, so `∂r/∂x_j = (1/d) x_j / r`, and `∂(r^{-1})/∂x_j = −r^{-2} ∂r/∂x_j
= −x_j / (d r³)`. Substitute:

```text
∂y_i/∂x_j = g_i [ δ_ij / r − x_i x_j / (d r³) ]
```

Sum over the upstream gradient, `∂L/∂x_j = Σ_i ḡ_i ∂y_i/∂x_j`:

```text
∂L/∂x_j = (1/r) [ ḡ_j g_j − (x_j / (d r²)) · Σ_i ḡ_i g_i x_i ]
```

The first term is the direct path; the second is a projection: it subtracts, from every component
`j`, a share of `Σ_i (ḡ_i g_i) x_i` proportional to `x_j`. That coupling term is the whole point —
you cannot compute `∂L/∂x_j` from `ḡ_j` alone, you need the inner product of the upstream gradient
(gained) with `x` across *all* components, because the normalizer `r` ties them together.

Contrast with LayerNorm: it also subtracts the mean, so its `r` is over the centered vector and its
backward has a *second* coupling term (the mean-subtraction path) — `∂L/∂x` for LayerNorm has both a
"remove the component along `x`" projection and a "remove the mean" projection. RMSNorm drops the
mean-centering, so it drops that second coupling term, which is one reason it is cheaper.

*What the interviewer listens for:* that you find the coupling term via the `x_j` inside `r`, express
it as a projection (an inner product over all components), and know LayerNorm has one extra coupling
term from the mean.

**A4. Why do residual connections help gradient flow? Show the `1 +` in the Jacobian.**

A residual block computes `y = x + F(x)` where `F` is the sublayer (attention or FFN, with its norm).
The Jacobian of the block output with respect to its input is:

```text
∂y/∂x = I + ∂F/∂x
```

Now stack `n` such blocks: `x_{ℓ+1} = x_ℓ + F_ℓ(x_ℓ)`. By the chain rule, the gradient of the loss at
the input of layer `ℓ` is the product of per-layer Jacobians from the top down:

```text
∂L/∂x_ℓ = ∂L/∂x_n · Π_{k=ℓ}^{n−1} ( I + ∂F_k/∂x_k )
```

Expand one factor: `(I + ∂F)` means the gradient has a path that passes through `I` — i.e. straight
through, untouched — in addition to the path through `∂F`. Multiply the `I` terms across all layers
and you get a term that is exactly `∂L/∂x_n` with coefficient 1, unattenuated, no matter how deep the
stack. Without the residual, the block is `y = F(x)`, the Jacobian is just `∂F/∂x`, and the deep
gradient is a product of `n` Jacobians with no identity — if their singular values are below 1 the
product decays geometrically (vanishing gradient), if above 1 it explodes.

Connect to pre-norm: putting the norm *inside* the branch, `y = x + F(RMSNorm(x))`, keeps
the residual path a clean identity `x`, so the `I` in the Jacobian is preserved exactly. Post-norm,
`y = Norm(x + F(x))`, wraps a norm around the addition, so the identity path now passes through the
norm's Jacobian and the `1 +` is no longer clean — which is precisely why deep post-norm stacks need
warmup gymnastics and pre-norm does not.

*What the interviewer listens for:* the `I +` in the per-layer Jacobian and that the product of `I`
terms is the unattenuated highway — plus connecting it to why pre-norm beats post-norm.

---

## Part B — Cost accounting, from scratch

**B1. Derive the per-component Transformer FLOP count and collapse it to `2N` forward / `6N` train.**

Start from the matmul rule: `A ∈ ℝ^{m×n}`, `B ∈ ℝ^{n×p}` costs `2mnp` FLOPs, because each of the
`mp` outputs is a length-`n` dot product = `n` multiplies + `n` adds = `2n` FLOPs.

Walk every parameterized matmul in one layer over `T = B·L` tokens (model dim `d`, SwiGLU `d_ff`):

```text
QKV projections:  (T×d)·(d×3d)           → 2·T·d·3d      = 6 T d²
output proj:      (T×d)·(d×d)            → 2·T·d·d       = 2 T d²
FFN up + gate:    (T×d)·(d×d_ff) twice   → 2·2·T·d·d_ff  = 4 T d d_ff
FFN down:         (T×d_ff)·(d_ff×d)      → 2·T·d_ff·d    = 2 T d d_ff
--------------------------------------------------------------------
per-layer params: 8 T d² + 6 T d d_ff
```

Apply the canonical SwiGLU sizing `d_ff = (8/3)d` (chosen to match the old `4d` parameter
budget). Then `6 T d d_ff = 6·T·d·(8/3)d = 16 T d²`, so per layer is `8 T d² + 16 T d² = 24 T d²`.

Now the parameter count of one layer's matmuls: `4d²` (QKV+out) `+ 3·d·d_ff = 4d² + 8d² = 12d²`. So
per-layer FLOPs `24 T d² = 2 · (12 d²) · T = 2 · N_layer · T`. Sum over layers and add the head and
you get **forward ≈ `2N` FLOPs per token** — every parameter does one multiply-accumulate (2 FLOPs)
per token.

Backward is `2×` forward: for each matmul you compute the gradient w.r.t. its input (to keep
propagating) *and* w.r.t. its weight, and each is a matmul of the same shape class as the forward, so
`2 × 2N = 4N`. Total training = `2N + 4N = 6N` per token, and `C ≈ 6ND` over the run.

Assumptions to state out loud: (1) this counts only parameterized matmuls — it ignores the attention
score/value term `4 B L² d` per layer, negligible at short context but dominant when `L` is large;
(2) it ignores the elementwise ops (norms, activations, the AdamW step), which are `O(N)` and tiny
next to the matmuls; (3) `d_ff = (8/3)d` is what makes the constant exactly 24.

*What the interviewer listens for:* that you derive `12d²`/layer and `24 T d²` independently and show
they satisfy `2N`, then justify the `4N` backward as two same-shape matmuls per forward matmul — and
flag the `L²` term you dropped.

**B2. Derive the `18N` training-memory figure. Where does each piece come from?**

Four consumers in a mixed-precision AdamW step, counting bytes per parameter:

```text
bf16 weights (used in the forward/backward matmuls):   2N
fp32 master copy of weights (the source of truth):     4N
fp32 gradients:                                         4N
AdamW first moment m (fp32):                            4N
AdamW second moment v (fp32):                           4N
------------------------------------------------------------
fixed per-model state:                                18N bytes
```

Why each exists. You do the matmuls in bf16 for speed and memory, but bf16 has only 7 mantissa bits,
so accumulating tiny weight updates into bf16 weights loses them to rounding — hence a **fp32 master
copy** you actually update, from which the bf16 copy is cast each step. Gradients are one per
parameter, kept fp32 for a stable optimizer step. AdamW keeps **two** running moments per parameter
(first moment `m`, second moment `v`), each fp32 → `2 × 4N`. Sum: `2 + 4 + 4 + 4 + 4 = 18` bytes/param.

The lesson the number teaches: mixed precision does **not** halve everything. The fp32 master and the
`8N` of optimizer state dominate and stay fp32; only the working weights (and activations) go to bf16.
A pure-fp32 run without the bf16 copy is `4·(N + N + 2N) = 16N`; the bf16 copy *adds* `2N` on top,
giving 18. And this is all *before* activations, which scale with `B·L·n` and are usually the largest,
most variable consumer — measured, not derived.

*What the interviewer listens for:* naming all four consumers with correct byte widths, and the
insight that the fp32 master + optimizer state is why "mixed precision" doesn't cut memory in half.

**B3. Derive the KV-cache size formula symbolically and show it grows linearly in context.**

At decode, each new token attends against all cached keys and values, so you store `K` and `V` for
every past position, every layer. Count bytes for a context of `S` tokens.

Per token, per layer, you cache one key vector and one value vector, each of dimension equal to
(number of KV heads) × (head dim). Let `h_kv` be KV heads (with GQA, `h_kv < h`), `d_head` the head
dim, `p` bytes per element. Then:

```text
bytes per token per layer = 2 (K and V) · h_kv · d_head · p
```

Multiply by `n` layers and `S` tokens:

```text
KV_bytes(S) = 2 · n · h_kv · d_head · p · S
```

Everything except `S` is fixed by the architecture and dtype, so `KV_bytes ∝ S` — **linear in
context length**. Three structural readings fall out. (1) The `2` is K-plus-V; drop V and you halve
it, which some KV-compression schemes approximate. (2) `h_kv · d_head` is where GQA and MLA attack:
GQA shrinks `h_kv` by the query-to-KV ratio (e.g. 8×), MLA replaces `h_kv · d_head` with a small
latent dimension. (3) `p` is where KV-cache quantization attacks — int8 KV halves it again.

The strategic punchline: weights are a *fixed* cost paid once, but the KV cache grows with `S` and
with concurrent users, so at long context or high concurrency the cache — not the weights — is what
fills HBM, and per-user cache can rival a whole quantized model. That linear-in-`S` growth is the
entire reason long-context serving is a memory problem.

*What the interviewer listens for:* the clean `2·n·h_kv·d_head·p·S` with the `2` and `h_kv` (not `h`)
correct, the explicit "∝ S", and mapping each factor to the technique that attacks it (GQA/MLA/KV-quant).

---

## Part C — Optimizer and positional encoding

**C1. Derive the Adam update from the moment estimates, including bias correction. Why AdamW?**

Adam maintains exponential moving averages of the gradient `g_t` and its square:

```text
m_t = β₁ m_{t−1} + (1 − β₁) g_t          (first moment, estimate of E[g])
v_t = β₂ v_{t−1} + (1 − β₂) g_t²         (second moment, estimate of E[g²])
```

The bias problem: initialize `m_0 = v_0 = 0`. Unroll `m_t = (1−β₁) Σ_{i=1}^{t} β₁^{t−i} g_i`. Take
expectation assuming `g_i ≈ g` roughly stationary:

```text
E[m_t] = (1−β₁) g Σ_{i=1}^{t} β₁^{t−i} = (1−β₁) g · (1 − β₁^t)/(1 − β₁) = g (1 − β₁^t)
```

So `E[m_t] = (1 − β₁^t) · E[g]` — biased toward zero, badly so at small `t` (with `β₁ = 0.9`, the
first step's `m_1` is only `0.1·g`). Divide it out:

```text
m̂_t = m_t / (1 − β₁^t)          v̂_t = v_t / (1 − β₂^t)
```

Now both are unbiased. The update normalizes the step by the root second moment (per-parameter
adaptive learning rate):

```text
θ_t = θ_{t−1} − α · m̂_t / (√v̂_t + ε)
```

Intuition: dividing by `√v̂` gives each parameter a step scaled to its own gradient magnitude, so
noisy/large-gradient directions take smaller steps — Adam is roughly sign-of-gradient with a
smoothed magnitude.

Now **AdamW / decoupled weight decay**. Original Adam adds L2 regularization into the gradient:
`g_t ← g_t + λθ`. But that `λθ` then flows through the `1/√v̂` normalization, so parameters with large
`v̂` get *less* decay than intended — the regularization is coupled to the gradient statistics, which
is wrong. AdamW decouples it: apply the adaptive step to the *pure* gradient, and subtract the decay
directly from the weight:

```text
θ_t = θ_{t−1} − α ( m̂_t / (√v̂_t + ε) + λ θ_{t−1} )
```

Now every parameter decays at the same rate `α λ` regardless of its gradient history, which is the
correct behavior and is why every modern LLM uses AdamW, not Adam.

*What the interviewer listens for:* the unrolled `E[m_t] = (1−β₁^t)E[g]` that *derives* the bias
correction (not just states it), and the precise reason decoupling matters — decay shouldn't pass
through the `1/√v̂` normalizer.

**C2. Prove RoPE encodes *relative* position: show `q_m · k_n` depends only on `m − n`.**

Take one 2D pair of a query at position `m` and key at position `n`. RoPE rotates each by an angle
proportional to its absolute position: `q̃_m = R(mθ) q`, `k̃_n = R(nθ) k`, where `R(φ)` is
the 2D rotation matrix. The attention score for this pair is the dot product `q̃_m · k̃_n = q̃_m^T k̃_n`:

```text
q̃_m^T k̃_n = (R(mθ) q)^T (R(nθ) k) = q^T R(mθ)^T R(nθ) k
```

Two facts about rotations: `R(φ)^T = R(−φ)` (the inverse rotation), and `R(a)R(b) = R(a+b)`
(rotations compose additively). So:

```text
R(mθ)^T R(nθ) = R(−mθ) R(nθ) = R((n − m)θ)
```

Therefore:

```text
q̃_m^T k̃_n = q^T R((n − m)θ) k
```

The score depends on the positions *only* through `n − m`, the relative offset — the absolute `m` and
`n` have vanished. Write it out with `R((n−m)θ) = [[cos((n−m)θ), −sin((n−m)θ)], [sin((n−m)θ),
cos((n−m)θ)]]` and the score is a function of `q`, `k`, and `(m−n)θ` alone. Stack `d/2` such 2D pairs
each at its own frequency `θ_k = Θ^{−(2k−2)/d}` and the full-vector score is `Σ_k q_k^T R((n−m)θ_k)
k_k`, still purely a function of the relative offset. This is why RoPE extrapolates and adds no
parameters: relative position is baked into the geometry of the dot product, not learned.

*What the interviewer listens for:* the three-line collapse using `R^T = R(−φ)` and `R(a)R(b) =
R(a+b)`, landing on `R((n−m)θ)` — and knowing RoPE hits `q,k` only, never `v`.

---

## Part D — Alignment, derived

**D1. Derive the DPO loss from the KL-regularized RLHF objective. Do the whole chain.**

Start from the objective every alignment stage shares: maximize expected reward under a
KL leash to the frozen reference `π_ref`, with coefficient `β`:

```text
max_π  E_{y∼π(·|x)} [ r(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

Step 1 — solve for the optimal `π*`. Write the objective as a single expectation and complete it into
a KL. For a fixed `x`:

```text
E_π[r] − β E_π[log(π/π_ref)]
= −β E_π[ log(π/π_ref) − r/β ]
= −β E_π[ log( π / (π_ref e^{r/β}) ) ]
```

Define `π*(y|x) = (1/Z(x)) π_ref(y|x) e^{r(x,y)/β}`, with `Z(x) = Σ_y π_ref(y|x) e^{r(x,y)/β}` the
normalizer. Then `π_ref e^{r/β} = Z(x) π*`, and:

```text
objective = −β E_π[ log( π / (Z π*) ) ] = −β E_π[ log(π/π*) ] + β log Z(x)
          = −β · KL(π ‖ π*) + β log Z(x)
```

`log Z(x)` is independent of `π`, and KL ≥ 0 with equality iff `π = π*`. So the objective is
maximized exactly at:

```text
π*(y|x) = (1/Z(x)) π_ref(y|x) exp( r(x,y)/β )      ⟺   π* ∝ π_ref · exp(r/β)
```

Step 2 — invert for the reward. Take logs and solve for `r`:

```text
log π* = log π_ref + r/β − log Z(x)
⇒  r(x,y) = β log( π*(y|x) / π_ref(y|x) ) + β log Z(x)
```

The implicit reward of any policy is `β·log(π/π_ref)` plus a prompt-dependent constant.

Step 3 — plug into Bradley-Terry and watch `Z` cancel. The preference model is `P(y_w ≻ y_l | x) =
σ( r(x,y_w) − r(x,y_l) )`. The reward *difference* is:

```text
r(x,y_w) − r(x,y_l) = β log(π/π_ref)|_{y_w} + β log Z − β log(π/π_ref)|_{y_l} − β log Z
                    = β [ log(π(y_w|x)/π_ref(y_w|x)) − log(π(y_l|x)/π_ref(y_l|x)) ]
```

The `β log Z(x)` terms cancel because Bradley-Terry only ever sees differences between two responses
to the *same* prompt `x` — this is the crux, and it is why DPO never has to compute the intractable
`Z(x)`. Step 4 — the loss is the negative log-likelihood of the observed preferences under this model:

```text
L_DPO = − E_{(x,y_w,y_l)} log σ( β [ log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x)) ] )
```

Read it back: DPO is training the policy so its implicit reward `β log(π/π_ref)` ranks winners above
losers under Bradley-Terry — the policy *is* the reward model, so no separate RM and no RL loop.

*What the interviewer listens for:* the KL-completion trick that yields `π* ∝ π_ref e^{r/β}`, the
inversion, and specifically *why* `Z(x)` cancels (differences over the same prompt) — that cancellation
is the whole reason DPO exists.

**D2. Derive GRPO's group-normalized advantage and explain why it replaces the value baseline.**

Start from REINFORCE: `∇_θ E[R] = E[ ∇_θ log π_θ(y|x) · R(x,y) ]`. Raw `R` is
high-variance and, if always positive, pushes *up* on every sampled response — you need a baseline
`b` to reinforce only *relative* quality. The key fact: subtracting any baseline that does not depend
on the action leaves the gradient unbiased, because

```text
E_{y∼π}[ ∇_θ log π_θ(y) · b ] = b · Σ_y π_θ(y) ∇_θ log π_θ(y) = b · ∇_θ Σ_y π_θ(y) = b · ∇_θ 1 = 0
```

(using `π ∇log π = ∇π` and `Σ_y π = 1`). So `∇E[R] = E[ ∇log π · (R − b) ]` for any such `b` — the
baseline changes variance, not the expected gradient. The best baseline is the state value `V(x) =
E_{y∼π}[R]`, giving the advantage `A = R − V`. PPO learns `V` with a separate value network.

GRPO's move: estimate `V(x)` by Monte Carlo from a **group** of `G` responses sampled for the same
prompt, instead of learning it. Sample `{y_1..y_G}`, score `{r_1..r_G}`, and use the group mean as
the baseline, normalizing by the group std to control scale:

```text
A_i = ( r_i − mean(r_1..r_G) ) / ( std(r_1..r_G) + ε )
```

`mean(r_1..r_G)` is exactly a sample estimate of `V(x) = E_y[R]`, so `A_i` is a sample advantage — no
value network needed, which is what drops PPO's second model and roughly halves the memory. Every
token of response `i` carries this one scalar `A_i` into the clipped surrogate.

Two structural consequences fall out of the derivation. (1) When all `G` responses tie
(`std ≈ 0`, all equal reward), `A_i ≈ 0` for all `i` and there is no gradient — "advantage collapse,"
which is why DAPO's dynamic sampling filters all-same-reward groups. (2) The estimate's variance
falls with `G`, so group size trades compute (more rollouts) for a cleaner baseline — the whole
reason it works cheaply is that in the verifiable-reward setting sampling a group is free signal.

*What the interviewer listens for:* the proof that a state-independent baseline is unbiased, then the
recognition that the group mean is a Monte-Carlo estimate of `V(x)` — that identification is the
insight, and it explains both the memory win and the tie/advantage-collapse failure mode.

---

## Part E — Scaling and information theory

**E1. From `L(N,D) = E + A/N^α + B/D^β` and `C = 6ND`, derive the compute-optimal `N*`, `D*`.**

Fix compute `C = 6ND`, so `D = C/(6N)`. Substitute into the loss to get a function of `N` alone:

```text
L(N) = E + A N^{−α} + B (C/(6N))^{−β} = E + A N^{−α} + B (6/C)^β N^{β}
```

Minimize over `N`: set `dL/dN = 0`:

```text
dL/dN = −α A N^{−α−1} + β B (6/C)^β N^{β−1} = 0
⇒ α A N^{−α−1} = β B (6/C)^β N^{β−1}
⇒ N^{β−1+α+1} = N^{α+β} = (α A) / (β B (6/C)^β)
⇒ N^{α+β} = (αA)/(βB) · (C/6)^β
```

Take both sides to the `1/(α+β)`:

```text
N* = [ (αA)/(βB) ]^{1/(α+β)} · (C/6)^{β/(α+β)}   ∝  C^{ β/(α+β) }
```

So `N* ∝ C^a` with **`a = β/(α+β)`**. Then from `D = C/(6N*)`:

```text
D* ∝ C / C^{β/(α+β)} = C^{ 1 − β/(α+β) } = C^{ α/(α+β) }   ⇒  b = α/(α+β)
```

Note `a + b = 1`, as it must (`N·D ∝ C`). With Chinchilla's fitted exponents `α ≈ 0.34, β ≈ 0.28`,
`a = 0.28/0.62 ≈ 0.46` and `b = 0.34/0.62 ≈ 0.54` — both ≈ ½, so **parameters and tokens each scale
as roughly `√C`, growing in lock-step**. The tokens-per-parameter ratio `D*/N* ∝ C^{b−a}` is nearly
constant in `C` (since `a ≈ b`), and plugging Chinchilla's constants through gives the famous **≈ 20
tokens per parameter** at the optimum.

State the assumption: this minimizes *training* loss at fixed training compute. It ignores inference
entirely, which is why deployed models deliberately overtrain a smaller model past this
optimum — the loss bowl is flat near its minimum, so moving to a smaller cheaper-to-serve `N` costs
almost no quality.

*What the interviewer listens for:* the substitution + `dL/dN=0` yielding `N^{α+β}`, the clean
`a = β/(α+β)`, the `a+b=1` sanity check, and knowing why the *training*-optimal answer is the wrong
target for a heavily-served model.

**E2. Show cross-entropy = negative log-likelihood, and relate the loss to perplexity via `exp(L)`.**

Maximum likelihood for a language model maximizes the probability of the training corpus. For a
sequence, that factorizes autoregressively:

```text
P(y_{1:T}) = Π_{t=1}^{T} p_θ(y_t | y_{<t})
```

Maximizing the product = maximizing its log = minimizing the negative log, and averaging per token:

```text
L = − (1/T) Σ_{t=1}^{T} log p_θ(y_t | y_{<t})       (negative log-likelihood, nats/token)
```

This *is* cross-entropy: cross-entropy between the empirical one-hot target distribution `q` and the
model `p` is `H(q,p) = −Σ_i q_i log p_i`, and with `q` one-hot on the true token `y_t` that collapses
to `−log p_θ(y_t | y_<t)`. Average over positions and you have the same `L`. So "cross-entropy loss,"
"negative log-likelihood," and "the MLE objective" are three names for one quantity (given natural
log; base-2 gives bits/token, a factor `log 2` apart).

Perplexity is the exponential of that per-token loss:

```text
PPL = exp(L) = exp( −(1/T) Σ_t log p_θ(y_t|y_<t) ) = ( Π_t p_θ(y_t|y_<t) )^{−1/T}
```

which is the geometric-mean inverse probability the model assigns to the true next token — the
"effective branching factor," the number of equally-likely tokens the model is choosing among. A
perplexity of 1 is perfect prediction (`L = 0`); a perplexity of `V` is uniform-random guessing
(`L = log V`). Because `PPL = exp(L)` is monotone, minimizing loss and minimizing perplexity are the
same thing; perplexity just puts it on an interpretable scale. This is also why the irreducible `E`
in the scaling law is the data's entropy — the floor loss is the entropy, and `exp(E)` is the
irreducible perplexity.

*What the interviewer listens for:* the autoregressive factorization → NLL → cross-entropy chain, the
one-hot collapse, and reading perplexity as effective branching factor with `exp(L)` and the two
anchor points (PPL=1 perfect, PPL=V uniform).

---

## Part F — Numerics

**F1. Why does bf16's exponent range matter more than fp16's precision for training?**

Set up the two formats. fp16 is 1 sign / 5 exponent / 10 mantissa; bf16 is 1 / 8 / 7.
The exponent width sets **dynamic range** (the span from smallest to largest representable
magnitude); the mantissa width sets **precision** (relative spacing between representable values).

Range from the exponent. With `e` exponent bits the largest normal magnitude is ~`2^{2^{e−1}}` and
the smallest normal is ~`2^{−(2^{e−1}−2)}`:

```text
fp16 (5 exp bits): smallest normal ≈ 2^{−14} ≈ 6.1e−5,  max ≈ 6.5e4
bf16 (8 exp bits): smallest normal ≈ 2^{−126} ≈ 1.2e−38, max ≈ 3.4e38   (same as fp32)
```

Precision from the mantissa: fp16 has 10 mantissa bits (relative error ~`2^{−11} ≈ 5e−4`), bf16 has
7 (~`2^{−8} ≈ 4e−3`). So bf16 is *less precise* than fp16 — it trades ~3 bits of mantissa for 3 bits
of exponent.

Why range wins for training. Gradients in deep nets routinely have magnitudes well below `1e−5`. In
fp16, anything under ~`6e−5` **underflows to zero** — the gradient silently vanishes, and the update
never happens. That is the failure fp16 forces you to fix with **loss scaling**: multiply the loss by
a large constant `s` before backward so `s·g` lands inside fp16's range, then divide the gradients by
`s` before the optimizer step (and back off `s` on overflow). bf16 has fp32's exponent range, so
those tiny gradients stay representable — no underflow, no loss scaling needed. Losing 3 mantissa
bits costs a little precision, but training is remarkably tolerant of imprecise-but-present gradients
and intolerant of vanished ones. Concretely, `torch.tensor([1e-8], dtype=torch.float16)` is exactly
`0`, while the bf16 cast preserves it — the one-line demonstration of the whole argument.

*What the interviewer listens for:* the exponent-bits → dynamic-range vs mantissa-bits → precision
split, the specific fp16 underflow floor ~`6e−5`, and that this is exactly what loss scaling patches
and bf16 makes unnecessary.

---

## How to practice

Reproduce each derivation on a blank page, out loud, from the first line to the last — no peeking at
the answer, because the interview *is* the blank page. The failure mode these questions punish is
knowing the *result* (`p − y`, `18N`, `N* ∝ √C`, `π* ∝ π_ref e^{r/β}`) without being able to
*generate* it, and an interviewer flushes that out in two follow-ups. For each one, be able to (1)
state the setup and assumptions before you start, (2) write every intermediate step including the
signs, and (3) say in one sentence what the result *means* and where it would change if an assumption
flipped ("now the norm has a bias," "now the reward is un-verifiable," "now I'm optimizing inference
not training loss"). When you can rederive all of these cold and narrate the meaning as you go, you
own the math instead of the vocabulary — which is the entire thing this round is built to measure.
