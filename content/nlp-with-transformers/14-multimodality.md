# 14 — Multimodality: When the Model Also Sees and Hears

Most real documents aren't clean text. They're scanned invoices with stamps in the margin, screenshots
of dashboards, product photos with text baked into the pixels, and voicemails. The moment your NLP
system touches the actual world, "text in, text out" stops being enough, and the question becomes: how
do you get a transformer that was built for tokens to also consume an image or a waveform? This lesson
is the NLP engineer's map of multimodality — the fusion architectures that dominate in 2026, how these
models are trained, and the one application (document understanding) where multimodal NLP pays the
rent. For the CV-heavy engineering detail — vision encoders, resolution tiling, throughput — go deep
in [VLMs](../vlm-guide/04_vlms.md); here we stay on the language side of the bridge.

## Why bother, and what "multimodal" buys you

A text-only model reading an invoice depends on an upstream OCR system, and every OCR error becomes a
silent input corruption you can't recover from. A model that sees the pixels can use *layout* — that
the number top-right is a total, that this block is a table — which OCR throws away. The payoff is
grounding: the model's answer is conditioned on what's actually on the page, not on a lossy
transcription. The same argument extends to charts, UI screenshots, and audio. Multimodality is less
about flashy demos than about removing a lossy preprocessing step from your pipeline.

## Three ways to fuse a modality into a transformer

Every vision-language model (VLM) answers one question: at what point do image information and text
information meet? Three answers dominate, and the differences drive real engineering tradeoffs.

### Late fusion: LLaVA-style (the default)

The dominant, cheapest pattern. Keep a pretrained vision encoder (a CLIP/SigLIP ViT) and a pretrained
LLM, both frozen or nearly so, and glue them with a small **adapter** (a linear layer or a 2-layer MLP,
sometimes a resampler like a Q-Former). The encoder turns an image into a grid of patch embeddings; the
adapter projects those into the LLM's token embedding space; you *prepend them as if they were text
tokens*. The LLM then attends over `[image tokens] + [text tokens]` with its ordinary causal attention.

This is **late fusion**: each modality is encoded separately and they meet only inside the LLM's
attention, near the "end" of the vision pipeline. LLaVA (Liu et al., 2023) made this the standard
because it's absurdly efficient — you train mostly the adapter, reusing two frozen giants. The cost:
the vision encoder is a bottleneck (whatever it discards, the LLM never sees), and generation is
text-only — the model can describe an image but can't produce one.

### Early fusion: Chameleon-style

Instead of bolting vision onto a text model, tokenize *everything* into one vocabulary and train one
transformer over the mixed stream. An image is passed through a **VQ tokenizer** (a learned codebook,
VQ-VAE/VQGAN lineage) that maps patches to discrete codes, so a picture becomes a sequence of "image
tokens" living in the same vocabulary as text. The model does next-token prediction over interleaved
text and image tokens from the first layer.

This is **early fusion** (Chameleon, Meta, 2024). Because there's no modality boundary, the model can
freely mix — read an image and *generate* one, interleave text and pictures. The cost is real: image
tokenization is lossy (a fixed codebook can't reconstruct fine detail), training is less stable
(competing modality losses), and you can't reuse a strong pretrained LLM as-is — the vocabulary changed.

### Hybrid: Transfusion

The newest line refuses the choice. **Transfusion** (Zhou et al., 2024) runs one transformer that does
**next-token prediction on text and diffusion on images simultaneously** — discrete language modeling
loss on text tokens, continuous diffusion loss on image patches, in the same network. It avoids the
lossy VQ tokenizer (images stay continuous) while still being a single any-to-any model. This
hybrid — autoregressive for language, diffusion for pixels — is where unified generation is heading in
2026, and it's worth being able to contrast against the other two in an interview.

The one-line summary: **late fusion** (adapter, reuse both giants, text-out only, cheapest);
**early fusion** (one vocab, any-to-any, lossy tokens, harder to train); **hybrid/Transfusion**
(diffusion for images inside one model, best generation quality, newest).

A useful way to hold the tradeoff: fusion point trades *reuse* against *capability*. The later you
fuse, the more you can reuse strong off-the-shelf components (a frozen ViT, a frozen LLM) and the
cheaper the project — but the modality boundary caps what the model can express, and it can't generate
the other modality. The earlier you fuse, the more unified and generative the model, at the cost of
training from a weaker starting point and eating tokenization loss. In 2026 the vast majority of
production VLMs are still late-fusion, because "reuse two frozen giants" is a budget most teams can
actually afford; early/hybrid designs show up when generation across modalities is the product.

## Image tokenization, briefly

Two senses of "image token" trip people up. In late fusion, image tokens are *continuous* patch
embeddings projected into the LLM space — real-valued vectors, never in the text vocabulary. In early
fusion, image tokens are *discrete* codebook indices — genuine vocabulary entries. The count matters
for cost: a single high-res image easily becomes 256–2500+ tokens, and since attention is $O(n^2)$
(see [transformer architecture](04-transformer-architecture.md)), image tokens dominate the context
budget. Resolution handling (tiling a big image into crops, then adding a downsampled global view) is
the main lever VLMs pull to control that cost — the engineering detail lives in the
[VLM guide](../vlm-guide/04_vlms.md).

## How a VLM is actually trained

Late-fusion VLMs train in stages, and the staging is the exam question:

1. **Adapter warmup (alignment).** Freeze the vision encoder and the LLM; train *only* the adapter on
   image-caption pairs. Cheap, and it teaches the projection to land image features in a place the LLM
   can read. Loss is next-token prediction on the caption.
2. **Vision-language pretraining.** Unfreeze the LLM (and sometimes the encoder), train on large
   interleaved image-text corpora. This is where broad visual knowledge enters the language model.
3. **Visual instruction tuning.** SFT on curated `(image, question, answer)` conversations — the step
   that turns a caption-continuation model into an assistant that answers questions about images. LLaVA's
   key trick was *generating* this instruction data with a text-only LLM from image annotations.
4. **Preference / RLHF (increasingly).** DPO or RLHF against multimodal preference pairs to reduce
   visual hallucination (confidently describing objects that aren't there) and improve helpfulness.

The pattern mirrors text post-training ([post-training](07-post-training.md)): a cheap alignment stage,
a knowledge stage, an instruction stage, a preference stage. What's modality-specific is that most of
the *capability* is inherited from the two frozen pretrained parts — you're mostly teaching them to
talk to each other.

## Benchmarks and their gaps

The canon: **MMMU** (college-level multi-discipline visual reasoning — the "hard" general benchmark),
**DocVQA** (question answering over document images — the one that predicts document-AI performance),
**ChartQA**, **TextVQA** (reading text in natural images), and **MathVista**. Read them with the same
suspicion you bring to text benchmarks (see [evaluation](10-evaluation.md)): they saturate,
they leak into training sets, and a high MMMU number tells you little about whether the model can read
*your* messy invoices. The specific multimodal trap is **language-prior shortcutting** — a model
answering a visual question from text priors alone without truly looking (answering "what color is the
banana?" with "yellow" regardless of the image). Always keep a small held-out set of *your* documents.

## Document AI: the killer job application

For an NLP engineer, document understanding is where multimodality earns its keep, because businesses
run on documents — invoices, contracts, forms, statements, medical records — and turning them into
structured data is a durable, well-paid problem.

The old pipeline was **OCR → layout model → text model**, with LayoutLM (Xu et al., 2020) as the
archetype: OCR gives words + bounding boxes, and the model fuses text, 2D position, and image patches so
it can learn that "the number in the box labeled Total, bottom-right" is the amount due. It works, but
OCR errors propagate and the stages are brittle.

The 2026 direction is **OCR-free document understanding** — a VLM reads the page pixels directly and
emits structured output (JSON, key-value pairs, or an answer), collapsing the pipeline. Donut
(Kim et al., 2021) pioneered the OCR-free encoder-decoder; modern general VLMs (Qw2.5-VL-class and
similar) now do competitive document extraction zero-shot. The engineering reality is nuanced: OCR-free
is simpler and captures layout natively, but for dense pages a strong OCR front-end still often wins on
pure text accuracy and is cheaper per page. The senior answer in an interview is "measure both on the
actual document distribution; pick per SLA on accuracy, latency, and cost" — not "OCR-free is the
future so use it." Extraction quality is judged with field-level precision/recall and normalized edit
distance, not ROUGE.

## Speech: the Whisper pattern

Audio slots into the same mental model. **Whisper** (Radford et al., 2022) is an encoder-decoder
transformer trained on 680k hours of weakly-supervised web audio: the audio is turned into a
**log-mel spectrogram** (a time-frequency image), a convolutional stem downsamples it, a transformer
encoder produces audio features, and a text decoder cross-attends to them to emit the transcript. Its
robustness comes from data scale and diversity, not architectural novelty — the same "big weakly-labeled
corpus" lesson as text pretraining.

Two integration patterns matter for jobs. **Cascade**: Whisper transcribes, then your text LLM does the
work — simple, debuggable, and still the default for most production ASR-to-NLP. **Native audio tokens**:
newer speech-LLMs discretize audio into tokens (à la early fusion) so one model listens and responds,
enabling low-latency voice agents and preserving prosody the transcript discards. Cascade for accuracy
and control; native for latency and full-duplex conversation.

## Any-to-any, and what an NLP engineer actually owns

The trajectory is one model that ingests and emits text, images, and audio — GPT-4o-class and
Gemini-class systems already blur the boundaries, and Transfusion-style unification points further. You
should track it, but be clear on your lane. A **CV engineer** owns the vision encoder, augmentation,
resolution/throughput, and pixel-level metrics. As an **NLP engineer** you own: the fusion boundary and
prompt format, the instruction data and chat templates that make the model useful, the *language* side
of failures (hallucinated objects, layout misreads, extraction schema), and the evaluation on your
document/audio distribution. You rarely train the ViT; you very often decide *which* VLM, how to prompt
it for structured output, and how to measure whether it's good enough to ship. For the deeper VLM
engineering stack, work through the [VLM guide](../vlm-guide/04_vlms.md).

## What interviews ask here

- Contrast late fusion, early fusion, and Transfusion. Adapter + frozen giants (text-out); one shared
  vocab via VQ (any-to-any, lossy); diffusion-for-images hybrid (newest, best generation).
- Why isn't an attention/OCR pipeline enough for documents? OCR errors propagate and layout is lost;
  pixel-reading VLMs use position natively. Name the OCR-free vs OCR-front-end tradeoff.
- Walk through VLM training stages. Adapter alignment → VL pretraining → visual instruction tuning →
  multimodal preference tuning.
- What is visual hallucination and how do you reduce it? Confidently describing absent objects; mitigate
  with multimodal DPO/RLHF and grounded prompting; measure on held-out data.
- How does Whisper turn audio into something a transformer eats? Log-mel spectrogram → conv stem →
  encoder features → text decoder cross-attends. Cascade vs native-audio-token integration.
- Why do image tokens dominate cost? Hundreds to thousands of tokens per image against $O(n^2)$
  attention; resolution tiling is the lever.

## Where this shows up on the job

- Building a document extraction pipeline (invoices, forms, contracts) — choosing OCR-free VLM vs an
  OCR-front-end model and defending it with field-level precision/recall on real pages.
- Standing up a voice feature — deciding cascade (Whisper → LLM) vs a native speech model on the
  latency/accuracy tradeoff.
- Selecting and prompting a VLM for structured output, and building the held-out eval that tells you
  whether it clears your SLA — the part CV engineers won't do for you.
