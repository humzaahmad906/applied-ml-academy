# The LLM / VLM / RAG / Agents Knowledge Map

**Goal of this guide:** make you able to open *any* new paper in these four areas and understand it without needing to look anything up. Not "be a beginner who knows the words" — be the person who reads the architecture diagram, sees "we replace MHA with GQA and add a shared expert," and immediately knows what changed, why, and what the tradeoff is.

This is written for someone with a working ML/engineering background. It does not re-explain backprop in baby terms, but it does make sure every concept a frontier paper assumes is covered somewhere in here, from first principles, with the math when the math is load-bearing.

## The chapters

| Chapter | Covers | Read when |
|---|---|---|
| Foundations | Tokenization, embeddings, the Transformer from scratch, attention math, normalization, activations, positional encoding (RoPE in depth) | First. Everything else assumes this. |
| Large Language Models | Decoder-only architecture, modern attention/FFN variants (GQA/MQA/MLA, MoE, hybrid linear-attention/SSM), pretraining, scaling laws, data, post-training (SFT/RLHF/DPO/GRPO/RLVR), reasoning models, decoding, long context | Second. The core. |
| Inference & Efficiency | KV cache, FlashAttention, PagedAttention/vLLM, quantization (PTQ/QAT, GPTQ/AWQ/GGUF/MLX), distillation, pruning, speculative decoding, serving metrics, on-device | After the LLM chapter, or whenever a paper is about *making it fast/small*. |
| Vision-Language Models | ViT, contrastive encoders (CLIP/SigLIP/DINO), fusion architectures, projectors (MLP/Q-Former/Perceiver), training stages, native/early fusion, visual tokenization, document understanding | After the LLM chapter. |
| Retrieval-Augmented Generation | Embedding retrieval, ANN/vector DBs, chunking, hybrid + reranking, advanced RAG (HyDE/RAPTOR/GraphRAG/HippoRAG), agentic RAG, evaluation | Any time. Depends lightly on the foundations and LLM chapters. |
| Agentic Systems | Agent loop, ReAct, planning, memory, tool use / function calling / MCP, multi-agent, context engineering, eval, failure modes | After the LLM chapter and ideally after RAG. |
| Reading Papers & Glossary | How to read an ML paper fast, the canonical-papers reading list, a dense glossary, benchmarks, how to stay current | Reference. Skim early, return often. |
| The Progression Timeline | The full chained lineage 2017→2026: every important paper/architecture, what it introduced, what it fixed, and what it motivated next — main spine + 5 parallel tracks (efficiency, reasoning, multimodal, RAG, agents). Folds in the pre-transformer prehistory and other assumed concepts. | After the core chapters to see how it connects historically — or read it *first* as a narrative map, then dive into the chapters for depth. |

## The dependency graph (what you actually need before what)

```text
                 Linear algebra + probability + calculus (assumed)
                                  |
                          [FOUNDATIONS]
              tokenization -> embeddings -> attention -> transformer
                       -> normalization/activations -> RoPE
                                  |
        +-------------------------+--------------------------+
        |                         |                          |
     [LLMs]                    [VLMs]                    (shared)
  arch variants            vision encoders
  pretraining              fusion + projectors
  post-training (RL)       multimodal training
  reasoning                visual tokenization
        |                         |
        +-----------+-------------+
                    |
        [INFERENCE & EFFICIENCY]   <- applies to both LLMs and VLMs
        KV cache, quant, serving, on-device
                    |
        +-----------+-----------+
        |                       |
      [RAG]                 [AGENTS]
   retrieval +            loop + tools +
   reranking +            planning + memory
   graph/agentic          context engineering
        \                     /
         \                   /
          (agentic RAG sits in the overlap)
                    |
        [READING PAPERS + GLOSSARY]
```

## The mental model that ties everything together

Almost every system in this space is one or more of these four primitives, composed:

1. **A sequence model.** Something that maps a sequence of tokens to a probability distribution over the next token. *Everything* — LLM, VLM, RAG generator, agent policy — is, at the core, this. If you deeply understand "next-token prediction over a transformer with attention + FFN," 80% of every paper is a variation on that theme.

2. **A representation/encoder.** Something that maps raw input (text span, image patch, document) into a vector. Embeddings for retrieval, ViT patches for vision, the residual stream itself. Retrieval, multimodality, and memory are all "get the right vectors close together."

3. **A training signal.** What gradient is flowing. Self-supervised next-token loss (pretraining), imitation (SFT), preference (DPO/RLHF), or verifiable reward (RLVR/GRPO). When you read a "new training method" paper, the only real questions are: *what is the loss, where does the signal come from, and what does it optimize that the previous one didn't.*

4. **A control loop.** Something outside the model that decides what to feed it next: retrieval (RAG), tool calls and planning (agents), decoding strategy, test-time search. The model is a function; the loop is the program calling it.

When you read a paper, classify its contribution into these four buckets first. It's almost always a change to exactly one of them, holding the other three roughly fixed. That framing alone removes most of the intimidation.

## How to use this for paper-reading specifically

For any paper, after applying the reading method covered in the reading-papers chapter, run this checklist:

- **Which primitive does this touch?** (sequence model / encoder / training signal / control loop)
- **What's the baseline they're beating, and on what axis?** (quality / speed / memory / cost / context length / data efficiency)
- **What's the one core idea in one sentence?** If you can't compress it, you haven't understood it yet.
- **What did they trade away?** Every gain has a cost. If the paper doesn't say, find it yourself — that's usually where the real understanding is.
- **Would the idea survive at a different scale / on different hardware / with a different base model?** Most don't. Knowing which do is the senior-level skill.

## A note on freshness (state of the field as of early-mid 2026)

The field's center of gravity has moved from raw parameter-scaling toward **efficiency** (linear-attention/SSM hybrids going mainstream, e.g. Qwen3-Next/Qwen3.5, Kimi Linear), **post-training as the main quality lever** (RLVR/GRPO-style reasoning training is now standard, post-DeepSeek-R1), **test-time compute / reasoning models** as a first-class axis, and **agentic systems** as the dominant application pattern. The foundations are stable and will not date. The specific model names in the LLM and VLM chapters will. The frameworks for thinking about them won't. This guide is built so the durable parts carry the weight.
