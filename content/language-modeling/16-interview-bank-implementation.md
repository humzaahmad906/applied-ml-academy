# 16 — Interview Bank II: Implementation Drills

The first interview bank tests whether you can reason, estimate, and design out loud. This
one tests whether your hands know the code. Frontier-lab ML-coding rounds — the OpenAI, Anthropic,
DeepMind, and Meta screens where you share a blank editor — ask you to *implement X from scratch*:
BPE, attention with a causal mask, a KV-cache decode step, RoPE, an AdamW step, the DPO loss. No
autograd shortcuts, no `F.cross_entropy`, no `nn.MultiheadAttention`. The interviewer watches for a
small set of recurring traps — the off-by-one in the causal mask, the missing max-subtraction in
softmax, weight decay hitting the wrong tensors, the reference model not detached — and whether you
catch them yourself.

Each question below gives a minimal, correct reference solution in the course's LLaMA-style
conventions (RMSNorm, RoPE, SwiGLU, GQA, AdamW), then a sentence or two on what the interviewer is
really probing. The code is written to be read and reproduced on a whiteboard, so it favors clarity
over the last 10% of speed; where a real kernel would differ it is noted. Read the
solution, close it, and rebuild it from the signature. These do not overlap with the first bank — that
one has the FlashAttention walk-through and the Triton-vs-`torch.compile` discussion; here you
write plain PyTorch that runs.

Assume `import torch`, `import torch.nn.functional as F`, and `import math` throughout unless a
question imports more.

---

## Part A — Tokenization

**A1. Encode a byte string by applying a learned, ordered merge list.**

Given the ordered merges from BPE training, encode a single pre-token. The whole content is that
merges apply in *training order* — the earliest merge has priority — and you keep sweeping until no
listed merge applies.

```python
def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def encode_pretoken(text, merges):
    """merges: dict {(a, b): new_id}, insertion-ordered by training priority."""
    ids = list(text.encode("utf-8"))          # base vocab is the 256 byte values
    for pair, new_id in merges.items():       # earliest-created merge wins
        ids = merge(ids, pair, new_id)
    return ids
```

The traps: starting from `list(text)` (characters) instead of `text.encode("utf-8")` (bytes), so
multi-byte characters are wrong; iterating pairs in frequency order instead of insertion order; and
the off-by-one in `merge` where you forget to advance by 2 after a merge and re-consume the new
token. A real encoder pre-tokenizes with the GPT-2 regex first and runs this per chunk, but the
merge logic is what they want to see. The interviewer is checking that you understand merges are an
*ordered* rewrite, not a set.

**A2. Implement the core BPE train-merge loop.**

Learn `num_merges` merges from a corpus: count adjacent pairs, merge the most frequent, record it,
repeat. This is the naive `O(corpus × merges)` version — say so, then mention the incremental-count
speedup.

```python
import collections

def train_bpe(text, num_merges):
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}                                          # (a, b) -> new_id, ordered
    for i in range(num_merges):
        counts = collections.Counter(zip(ids, ids[1:]))  # every adjacent pair
        if not counts:
            break
        # most frequent; ties broken by the lexicographically greater byte-pair
        pair = max(counts, key=lambda p: (counts[p], (vocab[p[0]], vocab[p[1]])))
        new_id = 256 + i
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge(ids, pair, new_id)                    # reuse A1's merge
    return vocab, merges
```

Traps: forgetting deterministic tie-breaking (the course specifies preferring the lexicographically
greater pair, so tests reproduce); assigning ids from 0 instead of 256 and colliding with base
bytes; and not knowing why this is too slow for a real corpus. The senior addition, unprompted:
"naively this recounts the whole corpus every merge — the fast version dedups pre-tokens into a
`{bytes: freq}` table and updates only the pair counts each merge touches, so a merge costs work
proportional to that pair's occurrences, not corpus size." They are probing whether you know the
algorithm *and* its production form.

---

## Part B — Attention

**B1. Scaled dot-product attention with a causal mask, in plain PyTorch.**

The primitive. Score, scale by `1/sqrt(d_k)`, causal-mask, numerically-stable softmax, weight the
values. Write the softmax by hand to show you know the max-subtraction.

```python
def sdpa_causal(q, k, v):
    # q, k, v: (..., L, d_head)
    d_head = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_head)     # (..., L, L)

    L = q.shape[-2]
    # True where a query (row i) may attend to a key (col j): j <= i
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal, float("-inf"))

    # numerically-stable softmax: subtract the row max before exp
    scores = scores - scores.amax(dim=-1, keepdim=True)
    weights = scores.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)     # -inf -> 0 after exp
    return weights @ v                                        # (..., L, d_head)
```

Traps, in the order interviewers pounce: dividing by `d_head` instead of `sqrt(d_head)`; the
off-by-one in the mask — `torch.tril` keeps the diagonal so query `i` sees key `i` (itself),
whereas `diagonal=-1` would wrongly forbid a token from attending to its own position; and
forgetting the `amax` subtraction so `exp` of a large score overflows to `inf` and you get `NaN`.
The `-inf` scores exponentiate to exactly 0, so masked keys drop out cleanly and the row still
normalizes. Being able to explain *why* the mask keeps the diagonal is the tell.

**B2. Multi-head attention forward, then the GQA variant.**

Project to Q/K/V, reshape into heads, run per-head SDPA, concat, output-project. Then the
grouped-query variant where several query heads share one KV head.

```python
class MHA(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = torch.nn.Linear(d_model, d_model, bias=False)   # no bias, modern default
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                          # x: (B, L, d_model)
        B, L, _ = x.shape
        # (B, L, n_heads, d_head) -> (B, n_heads, L, d_head)
        q = self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        out = sdpa_causal(q, k, v)                 # broadcasts over B and heads
        out = out.transpose(1, 2).reshape(B, L, -1)   # concat heads
        return self.wo(out)
```

GQA changes only the K/V head count and how they are shared. With `n_kv_heads < n_heads`, each KV
head serves `n_heads // n_kv_heads` query heads; you expand the KV heads to line up before SDPA:

```python
class GQA(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_head = d_model // n_heads
        self.rep = n_heads // n_kv_heads            # query heads per KV head
        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wv = torch.nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        # repeat each KV head `rep` times to match query heads
        k = k.repeat_interleave(self.rep, dim=1)   # (B, n_heads, L, d_head)
        v = v.repeat_interleave(self.rep, dim=1)
        out = sdpa_causal(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)
```

Traps: using `.view` after a `.transpose` (non-contiguous — you need `.reshape` or a
`.contiguous()`); getting the reshape-then-transpose order wrong so heads and sequence interleave
incorrectly; and in GQA, the difference between `repeat` (tiles the whole tensor, wrong grouping)
and `repeat_interleave` (repeats each head adjacently, correct grouping). Point out that the K/V
projections are now smaller — that shrunken KV is the entire inference-memory reason GQA exists.

**B3. One autoregressive decode step with a KV cache.**

The step that runs at generation time: a single new token, append its K/V to the cache, attend the
one query against the full cache, return next-token logits. No causal mask is needed — the cache
contains only past positions, all of which the new query is allowed to see.

```python
def decode_step(x_tok, wq, wk, wv, wo, k_cache, v_cache, n_heads):
    # x_tok: (B, 1, d_model) — embedding of the single new token
    # k_cache, v_cache: (B, n_heads, L_past, d_head) — or None on the first step
    B, _, d_model = x_tok.shape
    d_head = d_model // n_heads

    q = wq(x_tok).view(B, 1, n_heads, d_head).transpose(1, 2)   # (B, H, 1, d_head)
    k = wk(x_tok).view(B, 1, n_heads, d_head).transpose(1, 2)
    v = wv(x_tok).view(B, 1, n_heads, d_head).transpose(1, 2)

    if k_cache is not None:
        k = torch.cat([k_cache, k], dim=2)         # append along the time axis
        v = torch.cat([v_cache, v], dim=2)

    scores = q @ k.transpose(-1, -2) / math.sqrt(d_head)        # (B, H, 1, L_total)
    scores = scores - scores.amax(dim=-1, keepdim=True)
    weights = scores.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    out = weights @ v                              # (B, H, 1, d_head)
    out = out.transpose(1, 2).reshape(B, 1, d_model)
    return wo(out), k, v                           # return updated caches to store
```

Traps: applying a causal mask here (unnecessary and wrong — the query is at the newest position and
sees the entire cache); recomputing K/V for the whole context instead of only the new token (that
throws away the entire point of the cache); and forgetting to return the grown `k`/`v` so the caller
can persist them. The scale and stable-softmax carry over from B1. Interviewers use this to check
you understand *why* decode is cheap: one token's projections plus a read of the cache, not a
full re-attention. The KV cache is what makes per-step decode linear, not quadratic.

---

## Part C — The block components

**C1. RMSNorm forward.**

Divide by the root-mean-square, apply a learned per-dimension gain, no bias, no mean-subtraction.
The one non-obvious detail is the fp32 upcast around the square.

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(d))    # learned gain, init to 1
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)                        # upcast: squaring bf16 loses precision
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.g).to(in_dtype)         # downcast back at the end
```

Traps: subtracting the mean (that is LayerNorm — RMSNorm deliberately drops re-centering and the
bias); adding a bias term; putting `eps` outside the `sqrt` or forgetting it entirely; and — the one
the course explicitly flags — not upcasting to fp32 before squaring, so bf16 activations overflow or
lose precision. Note the gain initializes to ones so the layer starts as (near) identity.

**C2. Apply RoPE to q and k.**

Rotate query and key pairs by a position-dependent angle so the attention dot product depends only
on relative offset. Rotates `q` and `k`, never `v`, and is applied before the score computation.

```python
def build_rope_cache(seq_len, d_head, theta=10000.0, device=None):
    # inverse frequencies for each of the d_head/2 pairs
    k = torch.arange(0, d_head, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (k / d_head))          # (d_head/2,)
    pos = torch.arange(seq_len, device=device).float()  # (seq_len,)
    angles = torch.outer(pos, inv_freq)                # (seq_len, d_head/2)
    return angles.cos(), angles.sin()                  # each (seq_len, d_head/2)

def apply_rope(x, cos, sin):
    # x: (..., seq_len, d_head); split into even/odd pairs and rotate each 2D pair
    x1, x2 = x[..., 0::2], x[..., 1::2]                 # (..., seq_len, d_head/2)
    # reshape cos/sin to broadcast over batch and head dims
    while cos.dim() < x1.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = rot1, rot2         # re-interleave the pairs
    return out
```

Traps: applying RoPE to `v` (only `q` and `k` are rotated); rotating after the score instead of
before; getting the pair convention wrong — this one pairs adjacent dims `(0,1), (2,3), ...`, and
some implementations instead split the vector in half `(0, d/2), (1, d/2+1), ...`, which is a
different (but internally consistent) convention you must apply identically to q and k; and, at
decode time, indexing the cache at the token's *absolute* position, not position 0. The cos/sin are
precomputed and shared across layers, added as a non-persistent buffer.

**C3. SwiGLU feed-forward forward.**

Three matrices — gate, up, down — with SiLU on the gate branch, elementwise-gated, no bias.

```python
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False)   # up
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))        # SiLU(W1 x) ⊙ (W3 x)
```

If asked to write SiLU by hand: `silu(x) = x * torch.sigmoid(x)`. Traps: only two matrices (that is
the old ReLU MLP — SwiGLU has three because the gate is a separate projection); putting the
activation on the wrong branch or on both; adding biases; and not knowing the sizing rule —
`d_ff ≈ (8/3)·d_model` rounded to a multiple of 64, chosen so the three-matrix parameter count
matches the old two-matrix `4d` design. Naming that `8/3` rule unprompted is the depth signal.

---

## Part D — Loss and optimization

**D1. Cross-entropy from logits, with the log-sum-exp stable form.**

Implement the next-token loss without calling `F.cross_entropy`. The whole trick is computing
`log_softmax` via log-sum-exp with the max subtracted out.

```python
def cross_entropy(logits, targets):
    # logits: (N, V), targets: (N,) integer class ids
    m = logits.amax(dim=-1, keepdim=True)              # (N, 1) — for stability
    shifted = logits - m
    logsumexp = shifted.exp().sum(dim=-1).log() + m.squeeze(-1)   # (N,)
    # log p(target) = logit[target] - logsumexp
    tgt_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    nll = logsumexp - tgt_logit                        # -log p(target)
    return nll.mean()
```

Traps: `exp`-ing the raw logits and overflowing (the missing max-subtraction — same trap as
softmax, and the single most common one here); computing full probabilities and indexing them
rather than working in log-space; using Python indexing `logits[targets]` instead of `gather`, which
breaks in the batched case; and averaging over the wrong axis. Note `logsumexp - tgt_logit` is
exactly `-log_softmax(logits)[target]` written stably. If they ask about ignore-index for padding,
mask those positions out of the mean.

**D2. An AdamW optimizer step from the update rule.**

No `torch.optim`. Maintain first/second moment estimates, apply bias correction, and — the point of
the "W" — apply *decoupled* weight decay directly to the parameters, not through the gradient.

```python
def adamw_step(params, grads, state, lr, betas=(0.9, 0.95),
               eps=1e-8, weight_decay=0.1):
    b1, b2 = betas
    state["t"] = state.get("t", 0) + 1
    t = state["t"]
    for i, (p, g) in enumerate(zip(params, grads)):
        m = state.setdefault(f"m{i}", torch.zeros_like(p))
        v = state.setdefault(f"v{i}", torch.zeros_like(p))
        m.mul_(b1).add_(g, alpha=1 - b1)               # m = b1*m + (1-b1)*g
        v.mul_(b2).addcmul_(g, g, value=1 - b2)        # v = b2*v + (1-b2)*g^2
        m_hat = m / (1 - b1 ** t)                      # bias correction
        v_hat = v / (1 - b2 ** t)
        # decoupled weight decay: shrink the weight itself, independent of the gradient
        p.mul_(1 - lr * weight_decay)
        p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)   # p -= lr * m_hat / (sqrt(v_hat)+eps)
```

Traps: folding weight decay into the gradient (`g += wd * p`) — that is Adam-with-L2, the exact
thing AdamW was invented to fix; the decoupled form multiplies the parameter by `(1 - lr·wd)`
separately. Forgetting bias correction (matters most in the first few hundred steps when the moments
are still warming up from zero). Using betas `(0.9, 0.999)` when LLM training conventionally uses
`(0.9, 0.95)`. And, the subtle one interviewers love: weight decay should *not* be applied to
1-D parameters — biases, and the RMSNorm/LayerNorm gains — only to the 2-D weight matrices; decaying
the norm gains toward zero hurts. Say that even if the toy signature above decays everything.

**D3. Gradient clipping by global norm.**

Compute the ℓ2 norm across *all* gradients jointly and, if it exceeds a threshold, scale every
gradient down by the same factor.

```python
def clip_grad_norm(params, max_norm, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:                                # only scale down, never up
        for g in grads:
            g.mul_(clip_coef)
    return total_norm
```

Traps: clipping each gradient's norm independently instead of the *global* norm over all parameters
concatenated (independent clipping changes the update direction; global clipping preserves it);
scaling up when the norm is already below the threshold (guard with `< 1.0`); missing the `eps` so a
zero-gradient step divides by zero; and forgetting `.detach()` so the norm computation drags into
the graph. This runs after `.backward()` and before `optimizer.step()`.

---

## Part E — Sampling

**E1. Temperature, top-k, and top-p (nucleus) sampling from logits.**

Turn a logit vector into a sampled token. Temperature sharpens or flattens; top-k keeps the k
largest; top-p keeps the smallest set whose cumulative probability reaches p.

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    # logits: (V,) for a single position
    if temperature == 0:                               # greedy: just argmax
        return int(logits.argmax())
    logits = logits / temperature                      # scale before softmax

    if top_k is not None:
        kth = torch.topk(logits, top_k).values[-1]     # k-th largest value
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        # keep tokens up to and INCLUDING the one that crosses p
        remove = cumprobs - probs > top_p              # shift so the crossing token stays
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
```

Traps: applying temperature *after* the softmax instead of to the logits; the top-p boundary — you
must keep the token that *crosses* the threshold, which is why the mask is `cumprobs - probs > p`
(compare the cumulative sum *excluding* the current token) and not `cumprobs > p`, or you drop the
token that took you over p and can even mask the single most-likely token when its probability
already exceeds p; forgetting to scatter the sorted, filtered logits back to original vocab order;
and using `argmax` after masking instead of `multinomial`, which makes top-k/top-p pointless.
Greedy is the `temperature == 0` shortcut. Mention **min-p** as the newer alternative: keep tokens
whose probability is at least `min_p × p_max` (a floor relative to the top token), which adapts to
how peaked the distribution is instead of using a fixed count or mass.

---

## Part F — Mixture of Experts

**F1. A top-k MoE router: gate, select, weighted-combine.**

Given a token and a set of experts, softmax the router logits, pick the top-k experts, renormalize
their weights, and combine their outputs. Write the readable per-token version and note the
production form.

```python
class MoE(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = torch.nn.Linear(d_model, n_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            [SwiGLU(d_model, d_ff) for _ in range(n_experts)]
        )

    def forward(self, x):                              # x: (N, d_model), N tokens
        gate_logits = self.gate(x)                     # (N, n_experts)
        weights, idx = torch.topk(gate_logits, self.top_k, dim=-1)  # (N, top_k)
        weights = weights.softmax(dim=-1)              # softmax OVER the chosen k
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for e in range(len(self.experts)):
                mask = idx[:, slot] == e               # tokens routing to expert e here
                if mask.any():
                    out[mask] += weights[mask, slot].unsqueeze(-1) * self.experts[e](x[mask])
        return out
```

Traps: softmaxing over *all* experts and then selecting (the course convention — and DeepSeek/Mixtral
— softmaxes over the top-k *after* selection so the k weights sum to 1; softmax-then-select leaves
the weights summing to less than 1); combining unweighted expert outputs; and running every expert
on every token (the entire point of MoE is that each token hits only `top_k` experts, so the
production kernel gathers tokens per expert rather than masking, and adds a load-balancing auxiliary
loss so the router does not collapse onto a few experts). Mention that aux loss unprompted.

---

## Part G — Alignment losses

**G1. The DPO loss from policy and reference log-probs.**

Given sequence log-probabilities under the policy and the frozen reference for chosen and rejected
responses, compute the DPO loss. The reference must be detached.

```python
def dpo_loss(logp_pol_chosen, logp_pol_rejected,
             logp_ref_chosen, logp_ref_rejected, beta=0.1):
    # each argument is a (B,) tensor of summed log-probs over the response tokens
    pol_logratio = logp_pol_chosen - logp_pol_rejected
    ref_logratio = logp_ref_chosen - logp_ref_rejected     # from the frozen reference
    logits = beta * (pol_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()                    # -log σ(beta * (...))
```

Traps: not detaching the reference — the reference model is frozen, so its log-probs must carry no
gradient (compute them under `torch.no_grad()` when you produce them, or `.detach()` here);
implementing `-log σ(x)` as `-torch.log(torch.sigmoid(x))` and hitting numerical instability
instead of the stable `F.logsigmoid`; getting the sign wrong so you push the rejected response *up*;
and summing log-probs over the wrong tokens — it must be the response tokens only, prompt masked.
The interviewer wants to hear that `beta·log(π/π_ref)` is the *implicit reward* and that the
partition function cancels because Bradley-Terry only sees the difference between the two responses.

**G2. GRPO: group-normalized advantage and the clipped surrogate.**

Two pieces. First, turn a group of rewards for the same prompt into per-response advantages by
subtracting the group mean and dividing by the group std. Second, the PPO-style clipped per-token
loss using those advantages.

```python
def grpo_advantages(rewards, eps=1e-6):
    # rewards: (G,) — the G sampled responses for ONE prompt
    return (rewards - rewards.mean()) / (rewards.std() + eps)   # per-response scalar

def grpo_loss(logp_pol, logp_old, advantages, clip=0.2):
    # logp_pol, logp_old: (N, T) per-token log-probs; advantages: (N, 1) broadcast per token
    ratio = (logp_pol - logp_old.detach()).exp()               # π_θ / π_θ_old
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages
    per_token = -torch.min(unclipped, clipped)                 # PPO-clip surrogate
    return per_token.mean()
```

Traps: not detaching `logp_old` (and the reference, if a KL term is present) so gradients leak into
the rollout policy — the course flags this as load-bearing; computing the ratio as a division of
probabilities instead of `exp(logp - logp_old)`, which is less stable; forgetting the `+ eps` in the
std so a group where every response scored identically (advantage should be ~0) divides by zero
instead; taking `max` instead of `min` of the two surrogate terms (the `min` is what makes the clip
a *pessimistic* bound); and applying the advantage per-response when it must broadcast to every
token of that response. Note that when all rewards in a group tie, the advantage is zero and there
is no gradient — that is "advantage collapse," which dynamic sampling filters.

---

## Part H — Tying it together

**H1. A single causal-LM training step.**

Wire the pieces into one step: forward, shift labels, masked/mean cross-entropy, backward, clip,
optimizer step, zero grads. This is the drill that checks you can assemble the whole loop.

```python
def train_step(model, batch, optimizer, max_grad_norm=1.0):
    # batch: (B, L+1) token ids — inputs and targets come from the same sequence
    inputs = batch[:, :-1]                              # (B, L)
    targets = batch[:, 1:]                              # (B, L) — shifted by one

    logits = model(inputs)                              # (B, L, V)
    loss = cross_entropy(
        logits.reshape(-1, logits.size(-1)),           # (B*L, V)
        targets.reshape(-1),                           # (B*L,)
    )

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(model.parameters(), max_grad_norm)  # global-norm clip, after backward
    optimizer.step()
    return loss.item()
```

Traps: the label shift — the target for position `i` is the token at `i+1`, so you feed
`batch[:, :-1]` and score against `batch[:, 1:]`; getting this off by one silently trains the model
to copy the current token, and the loss looks plausible while the model learns nothing useful.
Others: flattening logits and targets inconsistently before cross-entropy; calling `zero_grad`
*after* `backward` (wipes the gradients you just computed) instead of before; clipping before
`backward` when there are no gradients yet; and, in an accumulation setup, dividing the loss by the
accumulation steps and only stepping every N micro-batches. The ordering — forward, loss, zero,
backward, clip, step — is the muscle memory they are checking.

---

## How to practice

Whiteboard these from memory. Pick a question, write the signature, close the reference, and produce
the body without running it — then run it against a tiny random tensor to catch the trap you missed
(the off-by-one mask, the un-subtracted max, the un-detached reference). The point is not to memorize
the reference; it is that after a dozen reps your hand writes the causal mask with the diagonal
included, the softmax with the max pulled out, and the AdamW step with weight decay decoupled,
*without* stopping to think — because in the round you want your attention on the interviewer's
follow-up ("now make it GQA," "now add a KV cache," "now the reward ties across the group"), not on
the mechanics. Build one small `nn.Module` decoder from C1–C3 plus B2, run H1 on a few hundred steps
of a toy corpus, and watch the loss fall; a loop you have made train once is worth more than ten you
have only read.
