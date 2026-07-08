# 07 — Vision Transformers

For most of this course, "computer vision" has meant convolutions. But around 2020 a different idea arrived from the world of language models and started winning: what if we treated an image less like a grid to convolve and more like a *sentence* of visual words to attend over? That idea is the **Vision Transformer (ViT)**, and it now underpins a large share of state-of-the-art vision — including the vision-language models you'll study next. This lesson explains how ViT works, how it differs from a CNN, and when to reach for each. It's the deliberate bridge between this course and the GenAI / VLM material.

## The core move: patches as tokens

A transformer was invented for text, where the input is a sequence of tokens (words) and the model uses **self-attention** to let every token look at every other token and decide what's relevant. Images aren't sequences of words — so ViT's key trick is to *turn* an image into a sequence.

It chops the image into a grid of small fixed-size **patches** — typically 16×16 pixels each. A 224×224 image becomes a 14×14 grid, i.e. 196 patches. Each patch is flattened and passed through a linear layer to become a vector (a "patch embedding"), exactly analogous to a word embedding. Now the image *is* a sequence of 196 tokens, and a standard transformer can process it.

```python
import torch

img = torch.randn(1, 3, 224, 224)          # one RGB image
patch = 16
# unfold into non-overlapping 16x16 patches
patches = img.unfold(2, patch, patch).unfold(3, patch, patch)
patches = patches.reshape(1, 3, 14*14, patch*patch)
print(patches.shape)     # torch.Size([1, 3, 196, 256]) -> 196 patches
# each patch (3*16*16 = 768 numbers) is then linearly projected to a token
```

Two more ingredients complete the setup, both borrowed straight from language transformers. Because attention itself has no notion of order, ViT adds a **positional embedding** to each patch token so the model knows patch (0,0) sits top-left. And it prepends a special learnable **[CLS] token** whose final output vector serves as the summary of the whole image, fed to a classification head. From there it's a stack of standard transformer blocks — multi-head self-attention plus a small MLP, each wrapped in the residual connections and layer normalization you met with ResNet.

## Self-attention: global from layer one

The defining difference from a CNN lives in that attention step. Recall that a conv filter only sees a small local patch, and a neuron needs *many* layers of stacked convolutions before its receptive field spans the whole image. Self-attention has no such limit: in a single layer, **every patch can directly attend to every other patch**, no matter how far apart. A patch in the top-left corner can relate to one in the bottom-right immediately.

This gives ViT a **global receptive field from the very first layer** — a structural advantage for tasks where distant parts of an image relate to each other (understanding a whole scene, relating an object to its context). A CNN gets there eventually through depth; a ViT starts there. The cost is that attention compares every patch to every other, so compute grows with the *square* of the number of patches — which is why ViTs use coarse 16×16 patches rather than per-pixel tokens.

## The trade-off: inductive bias vs. data

Here is the single most important idea for choosing between them, and it comes straight back to a theme from lesson 2.

A CNN has **strong inductive biases** baked in — locality (nearby pixels relate) and translation equivariance (a feature detector works anywhere). These are *assumptions* about how images behave, and they happen to be true, so a CNN doesn't have to learn them from data. That's why CNNs learn efficiently even from modest datasets.

A ViT has **almost none** of this built in. It doesn't assume nearby patches are related or that a pattern means the same thing wherever it appears — it must *learn* all of that from examples. This has a stark consequence:

- **On small-to-medium datasets, CNNs usually win.** The ViT, lacking helpful assumptions, overfits or fails to generalize without enough data to learn structure from scratch.
- **On very large datasets, ViTs win.** Given enough data (hundreds of millions of images, or heavy pretraining), the ViT's flexibility becomes an asset: unconstrained by convolutional assumptions, it learns richer, more global representations and surpasses comparable CNNs.

In one sentence: **a CNN's built-in assumptions are a gift when data is scarce and a ceiling when data is abundant; a ViT is the reverse.** This is why the original ViT only beat CNNs after pretraining on enormous datasets, and why for a typical few-thousand-image project a fine-tuned ResNet is still often the smarter, cheaper choice.

## Hybrids and the practical middle ground

The field didn't stay at "CNN vs. ViT" for long — it borrowed from both. **Hybrid** designs use convolutions early (to cheaply capture local detail and inject some helpful bias) and attention later (for global reasoning). Architectures like the **Swin Transformer** compute attention within local windows that shift between layers, recovering CNN-like efficiency and locality while keeping attention's power. And as we saw, **ConvNeXt** went the other way — a pure CNN redesigned with lessons from transformers — proving the two families are converging.

Crucially, in practice you use ViTs the same way you use CNNs: **load a pretrained one and fine-tune.** Because ViTs are data-hungry, using a model already pretrained on a massive dataset is not optional — it's the whole point. Self-supervised pretrained ViTs like **DINOv2** produce general-purpose visual features that transfer remarkably well, often used frozen as a feature extractor. The transfer-learning skills from lesson 5 carry over directly; only the backbone changes.

## The bridge to vision-language models

This lesson is also your on-ramp to the next course. Modern **vision-language models** (VLMs) — the systems that let a chatbot "see" an image — almost universally use a ViT (or a ViT-derived encoder like CLIP's) to turn an image into a sequence of patch tokens, then feed those tokens *into a language model* alongside text tokens. The reason that works so cleanly is exactly what this lesson established: a ViT already represents an image as a **sequence of tokens**, the same data structure a language transformer consumes. The patches-as-tokens idea is what makes gluing vision and language together natural. When the VLM course talks about "image tokens" or a "vision encoder," it means precisely the ViT you now understand.

## Why this matters for ML

Vision transformers are no longer exotic — they are the backbone of frontier vision systems, from foundation models like DINOv2 and SAM to every multimodal LLM. Understanding ViT tells you why some models need enormous pretraining, why "just use a CNN" is still the right call on a small dataset, and how images get fed into language models. It also crystallizes a concept — inductive bias, and its trade-off against data scale — that governs architecture choices across all of deep learning, not just vision. This is the knowledge that lets you read a modern model card and understand *why* it's built the way it is.

## Key takeaways

- A **ViT** splits an image into fixed-size **patches**, embeds each as a token, adds positional embeddings and a [CLS] token, and runs a standard transformer — treating an image as a sequence.
- **Self-attention** lets every patch attend to every other in one layer, giving a **global receptive field immediately**, unlike a CNN that needs depth to see the whole image (at a cost quadratic in patch count).
- CNNs have strong **inductive biases** (locality, translation equivariance) and win on **small/medium data**; ViTs have few biases and win on **very large data or with heavy pretraining**.
- **Hybrids** (Swin, and CNNs like ConvNeXt) blend both; in practice you always **fine-tune a pretrained ViT** (e.g. DINOv2) rather than train from scratch.
- ViT's patches-as-tokens design is exactly what lets **vision-language models** feed images into language models — the bridge to the VLM course.

## Try it

Load a pretrained ViT from torchvision (`vit_b_16(weights="DEFAULT")`) and a ResNet-50, and compare their parameter counts and, if you can, inference speed on one image. Then reason through a scenario: you have 2,000 labeled X-ray images and need a classifier. Which architecture would you start with and why? Now change the scenario to 50 million images with heavy pretraining available — does your answer change? Write two sentences justifying each choice in terms of inductive bias and data.
