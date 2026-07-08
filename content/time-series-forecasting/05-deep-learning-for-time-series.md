# 05 — Deep Learning for Time Series

Deep learning arrived late to time-series forecasting and, for a while, underwhelmed. On the single univariate series that dominate business forecasting, a neural network was usually beaten by a well-tuned ETS or ARIMA (lesson 02) — a genuinely humbling result that the M-competitions confirmed repeatedly. But the picture has changed. On problems with *many related series*, *rich covariates*, and *long horizons*, and now with the arrival of pretrained **foundation models**, deep learning has become a serious and sometimes dominant tool. This lesson is a map of the landscape: what the architectures are, what each is good for, and — the question that saves you the most time — *when it's even worth reaching for them.*

## RNNs and LSTMs: the starting point

The first neural approach to sequences was the **recurrent neural network (RNN)**: process the series one step at a time, carrying a hidden state that summarizes everything seen so far. Plain RNNs struggled to remember long-range dependencies (the vanishing-gradient problem), so the **LSTM** (Long Short-Term Memory) and its lighter cousin the **GRU** added gating mechanisms that let the network hold information over longer spans.

LSTMs were the workhorse of neural forecasting through the late 2010s and still appear inside newer models. But as standalone forecasters they've largely been superseded: they train slowly (the step-by-step recurrence resists parallelization), and the architectures below simply forecast better. Know what an LSTM is — it's foundational vocabulary and it lives inside models like the TFT — but you'll rarely build one from scratch for forecasting today.

## The modern specialist architectures

These are models designed and trained *for your specific dataset* — you supply the series, they learn its patterns. They're the sweet spot when you have substantial data but no need for (or no trust in) a giant pretrained model.

**N-BEATS** (2019/2020) was a landmark: a pure deep architecture — stacks of fully-connected blocks, no recurrence, no attention — that beat the statistical benchmarks on the M4 competition. It works by learning basis expansions (trend and seasonality blocks) that make it partly interpretable, and it forecasts the whole horizon at once (multi-output, sidestepping the recursive-vs-direct dilemma of lesson 04).

**N-HiTS** (2022/2023) extends N-BEATS for *long-horizon* forecasting. Its trick is **hierarchical interpolation and multi-rate sampling**: different stacks operate at different time resolutions (some capture slow trend, others fast fluctuation) and their outputs are combined. This makes it both more accurate and dramatically cheaper on long horizons than earlier models.

**Temporal Fusion Transformer (TFT)** (Google, 2019/2021) is the model to reach for when you have **rich covariates**. It combines an LSTM encoder with attention and is explicitly designed to handle three kinds of inputs at once: *static* features (which store is this?), *known-future* inputs (next week's holidays, planned prices), and *observed-past* inputs. It's also notably interpretable — its attention weights and variable-selection networks tell you *which* inputs drove a forecast, which matters when a human must trust it.

**PatchTST** (2023) brought the transformer's "patching" idea (borrowed from vision) to time series: chop the series into subseries-level **patches** and treat each patch as a token, rather than each individual timestep. This retains local structure, slashes the attention cost, and lets the model see much longer histories. It also uses **channel independence** — each variable modeled separately with shared weights. PatchTST set strong long-horizon benchmarks and its patching idea is now everywhere, including in the foundation models below.

A practical note: nearly all of these are available in **`neuralforecast`** (Nixtla) and **`darts`** with a consistent API, so you can swap N-HiTS for TFT for PatchTST by changing one class name and benchmark them fairly.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, PatchTST

nf = NeuralForecast(
    models=[NHITS(input_size=28, h=7), PatchTST(input_size=28, h=7)],
    freq="D",
)
nf.fit(df)                 # df has columns: unique_id, ds (date), y
forecasts = nf.predict()   # forecasts the next h=7 steps for each series
```

## Foundation models for time series

The newest and most exciting development, borrowed straight from the LLM playbook: **pretrain one large model on billions of time points from many domains, then forecast a brand-new series with little or no training** — so-called **zero-shot** forecasting. You hand the model your history and it forecasts the future without you fitting anything. This is a genuine paradigm shift, turning forecasting from a *model-training* problem into more of a *model-selection* one.

The main players as of 2026 (this space moves fast — treat specifics below as a snapshot to verify, not gospel):

- **TimesFM** (Google) — a decoder-only transformer using patching, pretrained on ~100 billion real-world time points. Strong zero-shot performance; open-weights. Later versions (2.x) have extended context and accuracy.
- **Chronos** (Amazon) — tokenizes the series by scaling and quantizing values into a fixed vocabulary, then applies a T5-style language-model architecture to "predict the next token." The **Chronos-Bolt** variant is reported to be dramatically faster and more memory-efficient than the original, and a **Chronos-2** generation has been announced with stronger zero-shot and probabilistic results. Well-documented, strong community support.
- **Moirai** (Salesforce) — a masked-encoder model trained on a large multi-domain archive (reported ~231B observations), built for multivariate, any-frequency forecasting. A mixture-of-experts variant, **Moirai-MoE**, extends it.
- **TimeGPT** (Nixtla) — one of the first *commercial* TS foundation models, offered as a hosted API rather than open weights; production-oriented.
- **Lag-Llama** and **Timer/Timer-XL** are notable open research models in the same vein.

```python
# Chronos zero-shot: no training step, just a pretrained pipeline
# (illustrative API — check the current library version for exact calls)
from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base")
forecast = pipeline.predict(context=torch.tensor(history), prediction_length=7)
```

**A caveat you must carry.** The exact model names, version numbers, benchmark claims, and API signatures above are the fastest-moving part of this entire course. Vendors release new generations every few months and reshuffle the leaderboard each time. Verify the current best model, its license, and its API against the official repositories before relying on any specific claim here. What is durable is the *idea*: pretrained, zero-shot TS forecasting is now real and worth trying early on a new problem, because it can give a strong forecast in minutes with no training.

## When is deep learning actually worth it?

This is the judgment that matters most, because the default temptation — reach for the newest, biggest model — is usually wrong. Deep learning (specialist or foundation) earns its place when:

- **You have many related series.** Thousands of stores, SKUs, or sensors let a single global model learn shared patterns and transfer strength between series — the regime where deep models clearly beat per-series ARIMA.
- **You have rich covariates.** Known-future inputs (prices, promotions, weather forecasts, holidays) that a TFT can exploit and a univariate ARIMA cannot.
- **The horizon is long.** N-HiTS and PatchTST are built for long-horizon forecasting where classical methods degrade.
- **You have the data volume.** Deep models are data-hungry; a single series with 200 points will not support a transformer, and a classical model will win.
- **A foundation model is a cheap first swing.** Because zero-shot needs no training, trying TimesFM or Chronos on a new problem costs minutes and gives a strong reference point — increasingly a smart *early* move, not a last resort.

And deep learning is **not** worth it when: you have a single, short, univariate series (use ETS/ARIMA); you need airtight calibrated uncertainty and full interpretability on a small problem (classical wins); or you haven't yet built and beaten a simple baseline. The iron rule from lesson 02 stands — **always establish a classical baseline first.** A deep model that can't beat a well-tuned ETS is not a result worth shipping, and discovering that early saves you weeks.

## Key takeaways

- RNNs/LSTMs pioneered neural forecasting and remain vocabulary (and live inside newer models), but rarely the right standalone choice today.
- Specialist architectures each have a niche: N-BEATS/N-HiTS for accurate long-horizon multi-output forecasting, TFT for rich covariates and interpretability, PatchTST for long context via patching.
- Foundation models (TimesFM, Chronos, Moirai, TimeGPT) enable zero-shot forecasting — pretrained on billions of points, they forecast a new series with little or no training.
- The specific foundation-model names, versions, and benchmarks change every few months — verify against official sources; only the zero-shot *idea* is stable.
- Deep learning wins with many related series, rich covariates, long horizons, and enough data — but always beat a classical baseline first, and try a zero-shot foundation model early since it's nearly free.

## Try it

1. Install `neuralforecast`, format a series into its `unique_id`/`ds`/`y` layout, and fit both `NHITS` and `PatchTST` for a 7-day horizon. Compare their test errors to your LightGBM model from lesson 04.
2. Try a zero-shot foundation model (e.g. Chronos via its pipeline) on the *same* series with no training. How does its zero-shot forecast compare to your trained models — and how long did each take to produce a forecast?
3. Write a short paragraph arguing either for or against using a deep model on *your* specific dataset, citing the "when is it worth it?" criteria. Be honest about series count, covariates, horizon, and data volume.
