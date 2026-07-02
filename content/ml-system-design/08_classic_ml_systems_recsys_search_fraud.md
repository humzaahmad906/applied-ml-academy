# Module 08 — Classic ML Systems: Recommendations, Search, Fraud

## Why this module matters

Despite the LLM era, the single most common interview question at large consumer companies is still "design the feed / recommendations / People You May Know / ads ranking." These systems run the revenue. They also have the most mature, well-documented architecture in all of ML — the **multi-stage funnel** — and that architecture turns out to be the template for RAG and agent-memory systems too. Master it once, reuse it everywhere. The 2026 twist: generative/sequence-based recommenders and LLM-augmented search are now fair-game follow-ups.

## 1. The multi-stage funnel (the one diagram to internalize)

Corpus of 10⁷–10⁹ items → **(1) Candidate generation / retrieval**: several cheap sources each pull hundreds of candidates in ~10 ms (two-tower ANN, collaborative "users-also-liked", follows/subscriptions, fresh/trending, geo) → **(2) Ranking**: one model scores the merged ~1–5k candidates with full features, ~10–20 ms → **(3) Re-ranking / policy**: diversity (MMR / determinantal methods), freshness boosts, deduplication, business and integrity rules, exploration injection → top ~50 served. Why stages: you cannot run an expressive model over a billion items per request; the funnel spends compute proportional to candidate quality. (Recognize this shape from the retrieval chapter — retrieve→rerank is the same idea.)

## 2. Candidate generation — the two-tower workhorse

- **Architecture:** user tower (profile + behavior sequence → embedding) and item tower (content + stats → embedding), trained so that positive (user, item) pairs score high by dot product. Item embeddings are precomputed into an **ANN index** (the same HNSW/IVF machinery from the retrieval chapter); user embedding computed per request; retrieval = ANN lookup. The towers *must not* see cross-features (anything combining user×item) — that's exactly what makes precomputation and ANN possible, and articulating this constraint is a classic interview checkpoint.
- **Training:** sampled softmax with **in-batch negatives** (other items in the batch serve as negatives — free, but biased toward popular items since popular items appear in batches more often) corrected via **logQ correction** (subtract log of item's sampling probability from its logit); add **hard negatives** (impressed-but-not-clicked, or mined near-misses) to sharpen the boundary. This is the standard two-tower training recipe used in large-scale industrial retrieval.
- **Multiple generators are the norm** — a learned two-tower plus heuristic sources (recency, social graph, trending) merged downstream; resilience and coverage beat elegance.

## 3. Ranking

- **Features:** user (demographics, long-term history aggregates), item (stats, content embeddings), context (time, device, surface), and crucially **cross/interaction features** — which the ranker, scoring only thousands of items, can afford.
- **Models:** GBDTs (still strong, especially tabular-heavy and at smaller shops) → deep CTR models: Wide&Deep → **DCN-v2** (learned explicit feature crosses) → **DLRM** (embedding tables + interaction layer; Meta's workhorse, and a systems topic itself — TB-scale embedding tables sharded across GPUs/parameter servers). **Multi-task heads** predict click, like, share, watch-time, hide; combined by a tuned value formula (`w₁·P(click)+w₂·E[watch]−w₃·P(hide)`) — *the value formula encodes product strategy*, mention it. Architectures for task conflict: shared-bottom → **MMoE** → **PLE**.
- **Calibration matters** (ads especially): predicted probabilities must match observed rates (Platt/isotonic post-hoc), because bids and value formulas consume probabilities, not ranks.
- **Position & presentation bias:** users click what's shown high regardless of relevance; train with position as a feature (zeroed at serving), inverse-propensity weighting, or randomization traffic; otherwise the model learns the old ranker's biases, not relevance.
- **Sequence & generative rankers (the frontier now in production):** model the user's action *sequence* directly with transformers — the SASRec/BERT4Rec lineage → **generative recommenders** built on architectures like **HSTU**: reformulate recommendation as sequential transduction over interaction streams, with trillion-parameter-scale models replacing stage-separate DLRMs and showing LLM-like scaling laws — deployed at scale with double-digit engagement gains. Companion idea: **semantic IDs** — quantize item content embeddings into discrete tokens so the recommender "speaks item language," fixing cold-start and enabling generative retrieval. Know these as the answer to "how would you push this system to the 2026 frontier?"

## 4. Cold start, exploration, freshness

New items: content-based embeddings (text/image towers — now often LLM/VLM-derived), semantic IDs, creator priors, and a guaranteed exploration budget (epsilon or UCB/Thompson on impression-starved items). New users: onboarding signals, popularity-by-cohort, contextual bandits early. **Feedback-loop integrity:** the system trains on its own exposures — without logged propensities and exploration traffic, popularity bias compounds and the catalog's tail starves; this is a favorite senior-level probe.

## 5. Search and fraud variations (same skeleton, different physics)

- **Search** = the funnel with a query: query understanding (spell/synonym/intent; now LLM-assisted rewriting), retrieval = BM25 + dense hybrid (see the retrieval chapter), ranking = LTR objectives (pairwise/listwise, nDCG) with query-document features, plus strict latency budgets. LLM era adds: query→structured-filter parsing, LLM-labeled relevance data (replacing some human rating programs), and RAG-style answer synthesis *on top of* — not instead of — the retrieval stack.
- **Fraud/integrity** = the funnel inverted under adversarial drift: extreme class imbalance (≪1% positives — PR-AUC and recall@fixed-FPR, never accuracy), real-time features (the velocity counters from the data-engineering chapter), **GNNs** over entity graphs (devices/cards/addresses linking fraud rings — relational structure is the signal individual features miss), delayed and adversarially-shifting labels (frequent retraining, champion/challenger), and a decision layer: approve / step-up-auth (friction as a middle action) / block / human review queue sized to analyst capacity.

## Foundations Box: position bias

**Feature engineering.** Represent position as both an integer feature and a one-hot over six buckets: pos=1 | pos=2 | pos=3 | pos in [4–5] | pos in [6–10] | pos≥11. The bucket encoding lets the model learn the steep top-of-page click falloff without assuming linearity — the gap from position 1 to 2 dwarfs the gap from 8 to 9, so bucket resolution is coarser at depth. Both representations together (integer + one-hot) are useful: the integer captures ordinal distance, the buckets capture the non-linear shape.

**The serving contract.** During training, position comes from the log. At serving, **all candidates must receive the same constant position value** — typically bucket "1" (as if every item is scored at the top slot). This makes the ranker output "relevance independent of where this item will appear," which is the signal you want. Knowing and naming this serving contract precisely is a senior signal in system design interviews.

**The common skew failure.** Two mistakes recur in production: (1) the serving code passes the candidate's actual rank in the current ordering (1 for first, 2 for second, …), so the model scores each candidate as if it is already placed — a self-referential loop; (2) the code passes 0 for all candidates because a new engineer left the field at its default, and 0 is outside the training distribution. Both corrupt scores quietly; loss curves will not tell you.

```python
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SERVING_POSITION: int = 1  # all candidates scored as if at position 1


def position_to_bucket(positions: torch.Tensor) -> torch.Tensor:
    """
    Map 1-based integer positions to a 6-class one-hot bucket tensor.
    Buckets: pos=1→0, pos=2→1, pos=3→2, pos in [4,5]→3, pos in [6,10]→4, pos≥11→5.
    positions: [B] int64.  Returns [B, 6] float32.
    """
    boundaries = torch.tensor([2, 3, 4, 6, 11], dtype=torch.int64, device=positions.device)
    bucket = torch.bucketize(positions, boundaries, right=False)
    return F.one_hot(bucket, num_classes=6).float()


def serving_position_features(n: int, device: torch.device) -> torch.Tensor:
    """Return position features for n candidates, all fixed to SERVING_POSITION."""
    positions = torch.full((n,), SERVING_POSITION, dtype=torch.int64, device=device)
    return position_to_bucket(positions)


def assert_serving_position_constant(positions: torch.Tensor) -> None:
    """Call in serving integration tests; raises AssertionError if values differ."""
    unique = positions.unique()
    if unique.numel() != 1:
        logger.error("Position skew at serving: found values %s", unique.tolist())
        raise AssertionError(
            f"All candidates must receive the same position at serving; got {unique.tolist()}"
        )
```

**Validation cadence.** Log the position tensor for the first N serving requests after each deploy; call `assert_serving_position_constant` in your serving integration test. The failure typically arrives from a downstream team wiring the re-ranker output index directly into the feature builder — caught by the assertion, invisible to offline metrics.

---

## Foundations Box: in-batch negatives + logQ correction

**The formula.** In-batch negatives treat each positive item in a training batch as a negative for all other queries in that batch — free and fast, but biased: popular items appear in batches at rate proportional to their corpus frequency Q(item), so they are over-represented as negatives. The **logQ correction** subtracts `log Q(item_j)` from every logit in column j before softmax, recovering an unbiased estimate of the full-corpus cross-entropy:

```text
corrected_logit[i, j] = sim(query_i, item_j) − log Q(item_j)
```

The diagonal (query_i paired with its own positive item_i) also gets the column correction applied — that is correct, because the positive item also appears as an in-batch negative in other rows and its contribution to the partition function should be de-biased there too.

```python
import torch
import torch.nn.functional as F


def sampled_softmax_with_logq(
    query_emb: torch.Tensor,  # [B, D]
    item_emb: torch.Tensor,   # [B, D] — one positive per query; off-diagonal entries are negatives
    log_q: torch.Tensor,      # [B]    — log(sampling probability) per item in the batch
) -> torch.Tensor:
    """
    In-batch sampled softmax with logQ bias correction.
    Subtracts log_q[j] from every logit in column j before cross-entropy.
    log_q[j] = log(freq(item_j) / total_training_events), clipped to avoid -inf.
    """
    logits = torch.matmul(query_emb, item_emb.T)  # [B, B]
    logits = logits - log_q.unsqueeze(0)           # broadcast: col j -= log_q[j]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
```

**Computing Q.** Count each item's frequency across all positive events in one training epoch; normalize by total events; clamp at a floor before taking log to avoid `-inf` for rare items.

```python
import numpy as np


def compute_log_q(
    item_ids: np.ndarray,
    total_events: int,
    floor: float = 1e-9,
) -> dict[int, float]:
    """
    item_ids: flat array of all positive item IDs seen in training (one entry per event).
    Returns a dict mapping item_id -> log Q(item).  Recompute whenever catalog or data changes.
    """
    unique, counts = np.unique(item_ids, return_counts=True)
    q = np.clip(counts.astype(np.float64) / total_events, floor, 1.0)
    return dict(zip(unique.tolist(), np.log(q).tolist()))
```

**What breaks when Q is stale.** A catalog expansion adds one million new items. The Q table is not recomputed. The new items accumulate positive training events — they are popular new launches — but their Q entry is either absent or near-zero (from before launch). The correction term `−log Q` is therefore very large, so the corrected logit for any new-popular item used as a negative is enormous — the model exerts maximum gradient force to push it away from every query. New popular items are actively suppressed at retrieval even though they are genuinely relevant. The fix is mechanical but must be enforced: make the Q table a first-class pipeline artifact, written by the same step that produces training pairs, with a freshness assertion before training starts.

---

## Foundations Box: calibration

**Why AUC hides calibration failure.** AUC measures only whether positives rank above negatives — not whether the absolute probability values are correct. A model that predicts 0.0010 for every positive and 0.0009 for every negative achieves perfect AUC and is completely miscalibrated. In a pure ranking system this is harmless. In any system that applies a probability threshold, aggregates expected value, or feeds a bid formula, it is a silent catastrophe.

**Why calibration breaks in recsys/ads.** The most common causes: (1) negative subsampling for training efficiency — subsampling 1% of negatives shifts the model's implicit base rate to roughly 50× the true rate; correct with the log-odds offset `log(1/k)` where k is the subsample ratio; (2) a preprocessing change (deduplication, time-window shift) that changes the effective positive rate in training data without touching the eval set; (3) distribution shift between training period and serving period without recalibration.

**The cascade problem.** Multi-stage systems chain calibrated probabilities. The CTR model feeds a bid formula or a hard threshold into the re-ranker or monetization layer. If CTR shifts 3× after a retrain, bids shift 3×, revenue craters — while AUC stays flat. Volunteering calibration as a deployment gate metric alongside AUC is a senior signal; most candidates think exclusively in ranking metrics.

**Fix.** Isotonic regression on a held-out calibration set — not the training set, not the eval/test set. Platt scaling (logistic) works for monotone miscalibration; isotonic handles non-monotone distortions and is the practical default. Save the calibrator as a pipeline artifact alongside the model checkpoint.

```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


def calibration_check(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Mean absolute calibration error (MACE) + per-bin (predicted_mean, actual_fraction) pairs.
    MACE > 0.015 on a held-out set is a deployment gate failure for most production systems.
    Bins are quantile-stratified so each bin has equal sample count.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="quantile"
    )
    mace = float(np.mean(np.abs(frac_pos - mean_pred)))
    return {
        "mace": mace,
        "bins": [(float(p), float(f)) for p, f in zip(mean_pred, frac_pos)],
    }


def fit_isotonic_calibrator(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
) -> IsotonicRegression:
    """Fit on the held-out calibration split only; apply to serving outputs."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred_cal, y_true_cal)
    return iso
```

**Reliability diagram as a deployment gate.** After every retrain, run `calibration_check` on the held-out calibration set and log MACE to your experiment tracker. A sudden MACE spike — even when AUC is stable — is the fastest signal of a base-rate shift in the training distribution. Block the deploy until either the miscalibration source is fixed or an isotonic calibrator is fitted, validated, and bundled with the serving artifact.

---

## War Story: calibration break after retrain

**Symptom.** A consumer platform ran weekly CTR model retrains. The Monday deploy caused ad revenue to drop roughly 20% within two hours of cutover. On-call checked the eval dashboard: AUC on the held-out set was 0.762, marginally better than the prior week's 0.758. Nothing in the standard eval looked wrong.

**Debug.** The monetization team noticed bid prices had collapsed across every ad unit simultaneously — not isolated to specific campaigns. The bidder computed expected value as `predicted_CTR × eCPM`; lower CTR → lower expected value → lower bids → fewer auctions won. Pulling the raw probability distributions on a fixed held-out batch: the old model output in the range [0.010, 0.120]; the new model in [0.002, 0.018] — roughly a 5× compression toward zero. Ranking order was largely preserved (explaining the good AUC), but absolute magnitudes had shifted.

Running `calibration_check` on the held-out labels: old model MACE ≈ 0.009; new model MACE ≈ 0.042. Calibration had broken. AUC was stable because rank order was intact.

**Root cause.** The data engineering team had that week landed a deduplication job removing repeated impressions from the same user session — a legitimate data quality improvement. It cut the training set by roughly 35% and, critically, removed a disproportionate share of clicks (users who click once tend to click again in the same session), reducing the effective positive rate in training from roughly 6% to roughly 3.5%. The held-out eval set predated the deduplication job and was never deduplicated — so AUC saw correct rank order while absolute probabilities diverged from the real-world base rate.

**Fix.** Rollback within 30 minutes. Post-incident: (1) add a negative-weighting correction to restore the base rate shifted by deduplication; (2) gate deploys on MACE < 0.015 alongside AUC — this alone would have blocked the bad model; (3) add a preprocessing fingerprint check asserting that training and eval pipelines apply the same transformations to a shared fixture batch before any training run starts.

**Prevention pattern.** Calibration is the first quantity to break when the training distribution shifts. Any deploy gate that only checks AUC will miss it. Treat MACE as a tier-1 eval metric, log it on every run, and alert on relative degradation (e.g., >50% worse than the rolling baseline) — not just absolute threshold crossings.

## References

- The multi-stage funnel is the one diagram to internalize; the classic deep-recommender line of work (two-tower retrieval with sampling-bias correction, embedding-based retrieval, DLRM-style embedding+interaction models, DCN-v2 explicit crosses, MMoE/PLE multi-task heads) is the reference architecture for the ranking stage.
- Sequence recommenders (SASRec/BERT4Rec) and their generative-recommender successors (HSTU-style transduction, semantic IDs via residual quantization) are the frontier direction; study them to answer "how would you push this to the 2026 frontier?"
- Public engineering write-ups of production ranking stacks map the funnel concretely — read a few from large consumer platforms to see how candidate generation, ranking, and policy layers are wired in practice.
- The recommender embedding-infrastructure and classical-baseline libraries are the fastest way to build the Project below hands-on.

## Project 08 — Two-stage recommender on MovieLens-25M

(1) **Retrieval:** train a two-tower model (user history → mean/GRU-pooled tower; item id+genre tower) with in-batch negatives; add logQ correction and measure its effect on tail-item recall — this single ablation teaches more than any blog post. Export item embeddings → FAISS HNSW; report recall@100 vs a popularity baseline. (2) **Ranking:** for top-200 retrieved, train LightGBM and a small DCN-v2 on engineered features (user stats, item stats, genre crosses, recency); compare nDCG@10. (3) **Re-rank:** add an MMR diversity pass; quantify the relevance-diversity tradeoff curve. (4) **Bias study:** simulate position bias in synthetic logs and show the ranker learns it; fix with position-as-feature. Stretch: replace the user tower with a SASRec-style sequence encoder and report the delta — then you can speak to "sequence models in recsys" from experience.

## Interview Q&A

**Q1. Why can't the retrieval stage use cross-features, and what does that imply?**
**A.** Retrieval must score 10⁸+ items in ~10 ms, which is only possible if item representations are precomputed offline and searched with ANN. Precomputation requires item embeddings to be *independent of the user/query* — any feature combining user×item (e.g., "user's CTR on this creator") can only be computed per-pair at request time, destroying precomputability. Hence the two-tower factorization: all user-item interaction is compressed into one dot product. Implications: (1) retrieval is deliberately coarse — its job is recall, not precision; nuanced preference modeling is *delegated to the ranker*, which sees only thousands of candidates and can afford full cross-features; (2) the dot-product bottleneck loses expressiveness, partially recovered by better towers (sequence encoders) or late-interaction-style multi-vector retrieval; (3) evaluation differs by stage — recall@k for retrieval, nDCG/calibrated CTR for ranking. Volunteering this division of labor is exactly what the question is testing.

**Q2. What goes wrong with naive in-batch negatives, and how is it fixed?**
**A.** In-batch negatives sample negatives proportional to item *popularity* (popular items appear in more training pairs, hence more batches). The model therefore over-penalizes popular items as negatives, learning to suppress them, and never sees genuinely hard negatives among unpopular items — the gradient signal conflates "irrelevant" with "popular elsewhere." Fix one: **logQ correction** — subtract log Q(item) (its sampling probability, ≈ popularity frequency) from each negative's logit before softmax, recovering an unbiased estimate of the full softmax. Fix two: mix in **uniform random negatives** (covers the tail) and **mined hard negatives** — impressed-but-skipped items, or near-threshold ANN retrievals from the previous model generation — being careful not to mine *false* negatives (relevant items the user simply didn't see). The combination — in-batch + logQ + a slice of hard negatives — is the production standard, and knowing the failure mode behind each piece is the senior signal.

**Q3. How do you handle position bias in ranking training data?**
**A.** Logged clicks confound relevance with exposure: position 1 gets clicked more regardless of quality, so a model trained naively on clicks learns to imitate the previous ranker's ordering (an echo chamber). Approaches: (1) **position-as-feature** — train with the logged position as an input, then serve with it fixed to a constant; simple, widely used, works when position effects are roughly separable; (2) **inverse propensity scoring** — weight examples by 1/P(examined | position), with propensities estimated from randomization or models; unbiased in principle, high-variance in practice (clipped weights); (3) **small randomization traffic** — swap adjacent positions for a tiny % of sessions to directly estimate propensities and provide clean training signal; (4) two-tower-of-bias architectures (PAL-style) separating relevance and examination. Also name the sibling biases: presentation bias (thumbnail/snippet quality) and trust bias. The practical answer: position-as-feature plus a sliver of randomization to keep propensity estimates honest, monitored by comparing model ranks against randomized-slot outcomes.

**Q4. Design "People You May Know" for a professional network.**
**A.** Frame: grow the connection graph; online metric = invitations *accepted* (not sent — sent optimizes spam), guardrails = invite-decline/report rates. **Candidate generation** is graph-native: friends-of-friends (triangle closing — the workhorse), shared employer/school/group, contact-import matches, and an embedding source — graph embeddings (node2vec/GraphSAGE lineage) via ANN for beyond-2-hop discovery; each source tagged for ranker feature use. **Ranking:** GBDT or deep model on pair features — mutual-connection count and *strength*, employment/education overlap windows, profile similarity, interaction history (profile views), recency, plus both users' invite-acceptance propensities; predict P(send)×P(accept | send), optimizing the product. **Policy layer:** suppress declined/dismissed pairs, rate-limit appearances, integrity filters (fake-account scores), diversity across candidate sources. **Cold start:** new users get contact-import + employer/school candidates immediately (highest-precision sources). **Loop integrity:** log impressions with propensities; reserve exploration slots. **Serving:** precompute top-N lists daily for most users (this surface tolerates staleness), with online freshness re-rank using same-day signals — a nice example of the batch-vs-realtime hybrid from the foundations chapter.

**Q5. What are generative recommenders (HSTU/semantic IDs), and when would you actually adopt them?**
**A.** The reformulation: instead of stage-separated models over engineered features, treat the user's full interaction stream as a token sequence and train a large transformer-style sequential transduction model to predict next actions — HSTU-style architectures are the flagship: a modified attention design handling high-cardinality, non-stationary action vocabularies, scaled to trillion-parameter regimes, exhibiting LLM-like scaling laws, and deployed with double-digit online engagement gains while replacing chunks of the legacy DLRM stack. **Semantic IDs** complement it: quantize item *content* embeddings (RQ-VAE) into short discrete codes replacing arbitrary item IDs, so new items inherit meaningful tokens (cold-start fix) and retrieval can be *generative* (decode the next item's code directly). Adoption calculus: pro — strongest known scaling path, sequence modeling captures temporal dynamics feature pipelines miss, collapses feature-engineering debt; con — serving cost (autoregressive inference per request vs a dot product + MLP — needs the full inference/serving toolbox: caching, quantization, distillation), training-data scale requirements (billions of interactions before the scaling advantage shows), and organizational rewiring of a revenue-critical stack. Honest answer: at the largest consumer scale it's the trajectory, and it has been proven in production; at mid-scale, a SASRec-style sequence encoder *inside* the ranker captures much of the gain at a fraction of the risk — and proposing that staged migration is the senior answer.
