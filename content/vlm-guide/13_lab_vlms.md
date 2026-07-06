# Lab 4 — Vision-Language Models: CLIP Similarity and a Small VLM

This lab makes the [VLMs chapter](04_vlms.md) concrete. Part A uses CLIP to build an image–text similarity explorer — zero-shot classification, similarity matrix, shared embedding space intuition. Part B loads SmolVLM to caption an image and answer a visual question, tracing how the projector maps patch tokens into the language model's input space.

## Setup

```bash
pip install torch transformers accelerate Pillow requests
# alternatives: pip install open_clip_torch
#               ollama pull llava:7b  (see Stacks section)
#               pip install mlx-vlm  (Apple Silicon only)
```

Models:
- `openai/clip-vit-base-patch32` — ~600 MB, vision + text encoders
- `HuggingFaceTB/SmolVLM-Instruct` — ~2 GB; fallback `Qwen/Qwen2-VL-2B-Instruct` (~4.5 GB)

Hardware: CLIP runs fine on CPU (seconds per batch). SmolVLM needs ~3 GB RAM; on MPS/CUDA captioning takes under 5 seconds.

---

## Part A — Contrastive Image–Text Similarity with CLIP

### 1. Setup, seeds, sample images

```python
import random
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"device: {device}")

# ── fetch three sample images (Wikimedia, stable URLs) ────────────────────────
_URLS = {
    "cat":    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "dog":    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
    "bridge": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/320px-GoldenGateBridge-001.jpg",
}

def _fetch(url: str) -> Image.Image:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

images, img_names = [], []
for name, url in _URLS.items():
    try:
        images.append(_fetch(url))
    except requests.RequestException:
        # solid-color fallback so the rest of the lab still runs offline
        images.append(Image.new("RGB", (224, 224), color=(100, 150, 200)))
    img_names.append(name)
    print(f"  {name}: {images[-1].size}")
```

### 2. Load CLIP and embed images + captions

```python
clip_id   = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_id)
model     = CLIPModel.from_pretrained(clip_id).to(device)
model.eval()
print(f"CLIP loaded  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f} M params)")

# ── image features ────────────────────────────────────────────────────────────
with torch.inference_mode():
    img_in    = processor(images=images, return_tensors="pt", padding=True).to(device)
    img_feats = model.get_image_features(**img_in)               # [3, 512]
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True) # L2-normalize

# ── caption features ──────────────────────────────────────────────────────────
captions = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a famous bridge",
    "a photo of a mountain landscape",
    "a painting of a sunset over the ocean",
]

with torch.inference_mode():
    txt_in    = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
    txt_feats = model.get_text_features(**txt_in)                # [5, 512]
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
```

CLIP trains a ViT vision encoder and a Transformer text encoder jointly. For a batch of N (image, caption) pairs the loss pulls together the N diagonal pairs (the true matches) and pushes apart the N²−N off-diagonal pairs. After training both modalities land in the same 512-D space: dot product = semantic similarity. L2-normalizing first makes dot product equal cosine similarity.

### 3. Similarity matrix

```python
sim = (img_feats @ txt_feats.T).cpu().numpy()    # [3, 5]

short = [c.replace("a photo of ", "").replace("a painting of ", "")[:18] for c in captions]
print(f"\n{'':>8}  " + "  ".join(f"{s:>18}" for s in short))
for i, name in enumerate(img_names):
    row = "  ".join(f"{sim[i, j]:>18.3f}" for j in range(len(captions)))
    print(f"{name:>8}  {row}")
```

The diagonal (matching pairs) should show the highest values. Off-diagonal scores reveal semantic proximity — "cat" and "dog" will be closer to each other than either is to "bridge," because the shared embedding space encodes category-level similarity.

### 4. Zero-shot classification

```python
classes = ["a photo of a cat", "a photo of a dog", "a photo of a bridge"]

with torch.inference_mode():
    cls_in    = processor(text=classes, return_tensors="pt", padding=True).to(device)
    cls_feats = model.get_text_features(**cls_in)
    cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

# logit_scale is a learned temperature parameter; exp() typically ~100
logit_scale = model.logit_scale.exp()
logits = (logit_scale * img_feats @ cls_feats.T)   # [3, 3]
probs  = logits.softmax(dim=-1).cpu().numpy()

cls_names = ["cat", "dog", "bridge"]
print("\nZero-shot classification:")
for i, name in enumerate(img_names):
    pred    = cls_names[int(probs[i].argmax())]
    bar     = "  ".join(f"{cls_names[j]}: {probs[i, j]:.1%}" for j in range(3))
    correct = "OK" if pred == name else "WRONG"
    print(f"  {name:>8}: {bar}  → {pred} [{correct}]")
```

No fine-tuning, no training examples — zero-shot works because CLIP saw hundreds of millions of (image, caption) pairs and learned which visual concepts map to which text. The logit scale (temperature inverse) sharpens or softens the probability distribution; CLIP learns it jointly with the encoders.

---

## Part B — SmolVLM for Captioning and VQA

### 1. Load the model

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

vlm_id    = "HuggingFaceTB/SmolVLM-Instruct"
print(f"\nLoading {vlm_id} (~2 GB) ...")
vlm_proc  = AutoProcessor.from_pretrained(vlm_id)
vlm_model = AutoModelForVision2Seq.from_pretrained(
    vlm_id,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager",    # swap to "flash_attention_2" if installed
).to(device)
vlm_model.eval()
print(f"SmolVLM loaded  ({sum(p.numel() for p in vlm_model.parameters()) / 1e6:.0f} M params)")
```

### 2. Caption and VQA via the processor + generate path

```python
def vlm_ask(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    # apply_chat_template builds the text prompt; {"type": "image"} is a placeholder
    # the actual pixel data is passed separately in the processor() call below
    prompt = vlm_proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_proc(
        text=prompt, images=[image], return_tensors="pt", padding=True
    ).to(device)

    with torch.inference_mode():
        out_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    n_prompt = inputs["input_ids"].shape[1]        # strip the prompt tokens
    return vlm_proc.decode(out_ids[0, n_prompt:], skip_special_tokens=True).strip()


test_image = images[0]    # cat

print("\n-- Captioning --")
print(vlm_ask(test_image, "Describe this image in one concise sentence."))

print("\n-- Visual Question Answering --")
print(vlm_ask(test_image, "What color is the main animal in this image?"))
```

### 3. Projector architecture — map to code

SmolVLM is a three-stage system. Every line in `vlm_ask` corresponds to one stage:

1. **Vision encoder (SigLIP ViT):** `processor(images=[image], ...)` extracts and normalizes patch tokens — the `pixel_values` key in `inputs`. The ViT produces `[1, num_patches, vision_dim]`.
2. **Pixel-shuffle projector (MLP):** SmolVLM uses a pixel-shuffle operation to compress 2×2 patches into one, then an MLP to project from `vision_dim` to `lm_dim`. This produces `[1, n_visual_tokens, lm_dim]` — visual tokens in the same dimension as word embeddings. This is what makes cross-modal attention possible: both live in the same vector space.
3. **Language model (SmolLM2 decoder):** `inputs["input_ids"]` carries the text tokens; the model concatenates visual tokens + text tokens and runs standard causal attention over the combined sequence. The `{"type": "image"}` placeholder in the chat template marks where the visual tokens are inserted.

The `inputs.keys()` you see (`input_ids`, `attention_mask`, `pixel_values`) map directly to these three stages. The model never sees raw pixels; it sees projected patch embeddings.

---

## Stacks & alternatives

### open_clip vs HF CLIP

```python
# pip install open_clip_torch
import open_clip

model_oc, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model_oc.eval()

img_t = preprocess(test_image).unsqueeze(0)           # [1, 3, 224, 224]
txt_t = tokenizer(["a cat", "a dog", "a bridge"])     # [3, 77]
with torch.no_grad():
    img_f = model_oc.encode_image(img_t)
    txt_f = model_oc.encode_text(txt_t)
    img_f /= img_f.norm(dim=-1, keepdim=True)
    txt_f /= txt_f.norm(dim=-1, keepdim=True)
    print((img_f @ txt_f.T))
```

Use `open_clip` when you need SigLIP, EVA-CLIP, CoCa, MetaCLIP, or any model not on HF Hub, or when you want explicit control over preprocessing pipelines. HF CLIP is more convenient for pipelines already in the `transformers` ecosystem. Both produce identical embeddings for the same checkpoint.

### Ollama vision models

```bash
ollama pull llava:7b      # or llava-llama3, moondream, qwen2-vl
```

```python
import base64, ollama

img_b64 = base64.b64encode(open("cat.jpg", "rb").read()).decode()
resp = ollama.chat(
    model="llava:7b",
    messages=[{"role": "user", "content": "What is in this image?", "images": [img_b64]}],
)
print(resp["message"]["content"])
```

Reach for Ollama when you want zero-boilerplate local serving, a shared model server across multiple scripts or processes, or quick interactive experiments. It handles quantization automatically and exposes an OpenAI-compatible HTTP endpoint. No GPU driver setup beyond CUDA/Metal.

### MLX-VLM (Apple Silicon)

```bash
pip install mlx-vlm
python -m mlx_vlm.generate \
    --model mlx-community/SmolVLM-Instruct-4bit \
    --image cat.jpg \
    --prompt "Describe this image."
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/SmolVLM-Instruct-4bit")
prompt = apply_chat_template(processor, model.config, "What animal is this?", num_images=1)
output = generate(model, processor, "cat.jpg", prompt, max_tokens=100, verbose=False)
print(output)
```

Use MLX-VLM on Apple Silicon when throughput matters. PyTorch-MPS has CPU fallbacks that silently kill performance on ops not fully implemented in Metal. MLX runs entirely on the unified memory ANE/GPU. The 4-bit quantized SmolVLM stays well under 4 GB of unified memory.

---

## What you built

- Embedded images and captions in CLIP's shared 512-D space and printed the full cross-modal similarity matrix
- Built a zero-shot image classifier using text class names — no fine-tuning, no training data
- Loaded SmolVLM and ran captioning and VQA via the processor + `generate` path
- Traced the three-stage projector pipeline: `pixel_values` → SigLIP → pixel-shuffle MLP → visual tokens concatenated with text tokens
- Surveyed three alternative stacks: open_clip for model variety, Ollama for zero-boilerplate serving, MLX-VLM for Apple Silicon throughput

## Build it further

Build a **zero-shot product-category classifier**: collect 20+ product images (Wikipedia or a local folder), define 10 category templates (`"a photo of {category}"`), and run CLIP zero-shot classification on all of them. Compute precision@1 and log a confusion matrix to wandb. Then swap the backbone to `google/siglip-base-patch16-224` (use HF `AutoModel`) and produce a two-row comparison table: CLIP ViT-B/32 vs SigLIP-B/16 accuracy on your product set. Note which categories each model gets wrong.
