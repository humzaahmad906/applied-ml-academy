# 13 — Diffusion Models

The [previous module](12-generative-models-vae-gan.md) left you with two flawed champions. VAEs are stable to train and give you a clean latent space, but their samples come out blurry — the reconstruction-plus-KL objective rewards hedging, so the model paints a smeared average of every plausible pixel. GANs are sharp, but training is a knife-edge: the minimax game collapses, oscillates, or drops whole modes of the data. For years the field wanted the best of both — stable training *and* sharp samples. Diffusion models are the answer that won, and by 2026 they are what actually powers the image, audio, and video generators you use. This module builds the mechanism from the same denoising intuition you already met in U-Net, then shows how it scaled into Stable Diffusion.

## From noising to denoising

The core idea is almost suspiciously simple. Take a real image and gradually destroy it by adding a little Gaussian noise, step after step, until after enough steps it is indistinguishable from pure static. This is the **forward process**, and it needs no learning at all — it is just a fixed recipe for adding noise. If you can add noise in a known, controlled way, then reversing one small step — going from a slightly noisier image to a slightly cleaner one — is a well-defined prediction problem. A model that learns to undo one step of noising can be run repeatedly, starting from pure static, to walk backward into a clean image that never existed.

The forward process has a convenient closed form. Rather than actually adding noise step by step, you can jump directly to any noise level `t` in one shot:

```python
import torch
torch.manual_seed(0)

# a cosine/linear schedule of "how much signal survives" at each step
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)   # cumulative signal retention

def noise_image(x0, t):
    # x0: a clean image in [-1, 1]; t: an integer timestep
    eps = torch.randn_like(x0)
    a = alpha_bar[t].sqrt()
    b = (1 - alpha_bar[t]).sqrt()
    xt = a * x0 + b * eps      # noisy image at level t
    return xt, eps             # we keep eps as the training target
```

At small `t` the image is barely touched; at `t = T` it is essentially `eps`, pure noise. Notice we return `eps` — the exact noise we added. That, it turns out, is what we ask the network to predict.

Two properties make this schedule work. First, the closed form means training never has to simulate the slow step-by-step corruption — one draw of `eps` gives you a training example at any noise level instantly. Second, the noise levels overlap: a network that has learned to clean up level 400 has seen images that look a lot like slightly-noisier level 401, so the learned denoiser is smooth across `t`. That smoothness is what lets sampling chain many small, reliable steps together instead of attempting one impossible leap from static to a photo.

## The training objective: predict the noise

Here is the whole trick, and the reason diffusion is so much easier to train than a GAN. You do not ask the model to output an image, or a probability, or to win an adversarial game. You show it a noisy image `x_t` and its timestep `t`, and you ask it to predict the noise that was added. That is a plain regression problem with a mean-squared-error loss. The **DDPM** objective (Ho et al., 2020), after the full variational derivation is simplified away, reduces to:

$$
L_{\text{simple}} = \mathbb{E}_{x_0,\;\epsilon \sim \mathcal{N}(0, I),\;t}
\left[\,\big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\big) \big\rVert^2 \right]
$$

Read it slowly. Inside the network `epsilon_theta` is exactly the noisy image our `noise_image` function produced. The target is `epsilon`, the noise we know we added. The expectation says: average this over random clean images, random noise draws, and random timesteps. There is a fuller derivation from the ELBO that shows why predicting the noise is equivalent to predicting the clean image or the score (the gradient of log-density) up to a reweighting — but the practical takeaway is that a stable MSE loss on noise is all you need. The network itself is a U-Net: the encoder-decoder-with-skips shape from the [segmentation module](06-detection-and-segmentation.md), taking a noisy image and a timestep embedding and returning a noise-shaped tensor.

```python
# training step: pick random timesteps, predict the noise, MSE loss
def train_step(model, x0):
    t = torch.randint(0, T, (x0.size(0),), device=x0.device)
    xt, eps = noise_image(x0, t)
    eps_pred = model(xt, t)             # U-Net: (image, t) -> noise
    return torch.nn.functional.mse_loss(eps_pred, eps)
```

No discriminator, no balancing act, no mode collapse. The loss goes down and the samples get better. That reliability is most of the story.

One detail the loss hides: the network must know *how noisy* its input is, so the timestep `t` is fed in alongside the image. It is encoded with a sinusoidal embedding — the same positional-encoding trick you saw with transformers — and mixed into every block of the U-Net:

```python
import math

def timestep_embedding(t, dim=128):
    # map an integer timestep to a smooth high-dimensional vector
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)
```

Without this signal the network would have to guess the noise level from the image alone; with it, the same weights can specialize their behavior across the whole schedule.

Why does predicting noise work so well? The full derivation starts from the ELBO — the same evidence lower bound that gave the VAE its objective last module — and shows that maximizing the likelihood of the data under the reverse process reduces, after algebra, to matching the true noise at each step. Predicting the noise `epsilon`, predicting the clean image `x_0`, and predicting the *score* (the gradient of the log data density, which points toward regions of higher probability) all turn out to be the same task up to a rescaling. You do not need to carry that derivation in your head to use diffusion, but it explains why such a simple loss is principled rather than a lucky hack: each denoising step is nudging the sample uphill toward the data distribution.

## Sampling, and why it is slow

Generation runs the learned reverse step over and over. You start from `x_T`, pure Gaussian noise, and at each step use the predicted noise to estimate a slightly cleaner image, then repeat down to `x_0`. Concretely, the DDPM reverse update subtracts the predicted noise, rescales, and adds a little fresh noise back (except on the last step):

```python
@torch.no_grad()
def sample(model, shape, device):
    x = torch.randn(shape, device=device)          # start from pure noise
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0.0   # no noise on the final step
        eps = model(x, torch.full((shape[0],), t, device=device))
        a, ab, b = alphas[t], alpha_bar[t], betas[t]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
        x = mean + b.sqrt() * z                     # one step cleaner
    return x                                        # a fresh image
```

The catch is right there in the loop: the original DDPM sampler takes hundreds or a thousand sequential forward passes through the U-Net to make one image. That is far slower than a GAN, which generates in a single pass. Slow sampling was diffusion's biggest practical liability, and shrinking it has been a major research thread.

**DDIM** (Song et al., 2020) reinterprets sampling as a deterministic process that can skip steps, cutting a thousand steps to twenty or fifty with little quality loss. Since then the step count has kept falling: **consistency models** and **flow matching** train the network so that a handful of steps — sometimes a single one — produces a clean sample, often by distilling a slow, many-step teacher into a fast student. In 2026 this is why interactive, near-real-time image generation exists at all.

## Latent diffusion: making it practical

Running a U-Net over full-resolution pixels for fifty steps is expensive, and most of that compute is spent modeling imperceptible pixel-level detail. **Latent diffusion** (Rombach et al., 2022 — the basis of Stable Diffusion) makes the key move: first train a VAE to compress images into a much smaller latent grid, then run the entire diffusion process *in that latent space*. A 512×512 image might become a 64×64×4 latent — a ~48× reduction in the number of values the U-Net has to denoise. The VAE's decoder turns the final clean latent back into pixels.

This is exactly where the VAE from the previous module earns its keep: not as a generator (its samples were blurry), but as a learned compressor whose latent space is smooth enough for diffusion to model. Latent diffusion is the single change that turned diffusion from a research curiosity into something that runs on a consumer GPU.

It is worth being precise about the division of labor, because it is easy to blur. The VAE is trained once, on its own, purely to compress and reconstruct — it is frozen afterward. The diffusion U-Net is then trained to generate latents, never touching pixels. At inference you chain them: sample a latent with the diffusion loop, then decode it once through the VAE. The blurriness that plagued the VAE as a standalone generator does not hurt here, because the VAE is only asked to reconstruct latents that the diffusion model has already made realistic — it never has to invent detail from a prior.

## Conditioning and classifier-free guidance

An unconditional model makes *some* image; you want *this* image — "a red bicycle in the rain." Text conditioning works by encoding the prompt (with a text encoder such as CLIP, covered in the [vision-and-language module](15-vision-and-language.md)) and injecting those embeddings into the U-Net through **cross-attention**, so every denoising step can attend to the words.

But conditioning alone produces weak, prompt-ignoring samples. The fix that made text-to-image genuinely controllable is **classifier-free guidance** (Ho & Salimans, 2022). During training you randomly drop the conditioning some fraction of the time, so the same network learns both a conditional predictor `epsilon_theta(x_t, c)` and an unconditional one `epsilon_theta(x_t, empty)`. At sampling time you run the network twice — once with the prompt, once without — and push the prediction in the direction that the prompt adds:

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s\,\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big)
$$

The **guidance scale** `s` controls how hard you pull toward the prompt. At `s = 1` you get ordinary conditional sampling; at `s = 7` or so (a common default) images follow the prompt much more faithfully at some cost to diversity; crank it too high and images oversaturate and distort. That single scalar is the knob every image generator exposes, and now you know it costs two forward passes per step.

```python
def guided_eps(model, xt, t, c_embed, null_embed, scale=7.5):
    eps_c = model(xt, t, c_embed)       # conditioned on the prompt
    eps_u = model(xt, t, null_embed)    # unconditioned
    return eps_u + scale * (eps_c - eps_u)
```

Beyond text, **ControlNet** and lightweight **adapters** (LoRA-style) add spatial or stylistic control — an edge map, a pose skeleton, a depth map — by attaching a trainable branch to a frozen base model, so you can steer composition without retraining the whole thing. This adapter ecosystem, layered on open weights, is most of why Stable Diffusion spawned such a large community.

## Why diffusion beat GANs

Put the pieces together and lay the three generative families side by side:

- **VAE** — stable training, fast single-pass sampling, meaningful latent space, but *blurry* samples because the objective rewards averaging.
- **GAN** — sharp samples and fast single-pass sampling, but *unstable* training and *mode collapse* that drops parts of the data.
- **Diffusion** — stable MSE training, *full mode coverage*, sharp samples, and easy conditioning; the one cost is slow multi-step sampling.

Diffusion trains with a stable regression loss instead of a fragile adversarial game, so it scales cleanly to enormous datasets without collapsing. It covers the full data distribution rather than dropping modes, giving genuine sample diversity. Latent diffusion made it cheap enough to run widely, and classifier-free guidance made it controllable. The one thing GANs still win is raw sampling speed — but consistency and flow-matching methods have largely closed even that gap. By 2026, essentially every frontier image and video generator, and Sora-class video models, are diffusion or diffusion-adjacent. FID and the precision/recall metrics you met last module confirmed the shift quantitatively, but the deeper reason is that a regression loss simply trains more reliably than a saddle-point game.

## Why this matters for ML

Diffusion is where a large share of applied generative work lives right now: product image generation, inpainting and editing tools, synthetic data for training other models, texture and asset creation for games and film, and the image branch of multimodal assistants. You will rarely train a base model from scratch — that costs a small fortune in compute — but you will constantly fine-tune one, attach a ControlNet, train a LoRA on a client's style, or wire a Stable Diffusion pipeline into a product. Knowing that the loss is "predict the noise," that guidance is a two-pass trick with one scalar, and that everything happens in a VAE latent tells you exactly which knobs exist and what they cost.

## Key takeaways

- Diffusion **adds noise in a fixed forward process**, then learns to **reverse it one step at a time**; sampling starts from pure noise and denoises down to an image.
- The training objective is a plain **MSE loss that predicts the added noise** — stable to train, no adversarial game, no mode collapse.
- Sampling is **slow** (many sequential U-Net passes); DDIM, consistency models, and flow matching cut it from ~1000 steps toward a handful.
- **Latent diffusion** runs the process inside a VAE's compressed latent space, which is what made diffusion cheap enough to be practical.
- Text conditioning uses **cross-attention**; **classifier-free guidance** runs two passes and interpolates by a **guidance scale** `s` to control prompt adherence.
- Diffusion beat GANs on **training stability, mode coverage, and controllability**, giving up only sampling speed — a gap now largely closed.

## Try it

Load a pretrained latent diffusion pipeline (e.g. `diffusers`' `StableDiffusionPipeline`), seed it, and generate the same prompt at guidance scales 1, 3, 7.5, and 15. Watch how prompt adherence, diversity, and saturation change — you are seeing the equation above in action. Then set the number of inference steps to 5, 20, and 50 and compare quality against wall-clock time. Finally, implement `noise_image` above on a real image and display `x_t` at `t` = 100, 400, and 900 to see the forward process destroy structure.

Next, in [3D vision](14-3d-vision.md), we leave the flat image plane entirely. The recurring lesson there echoes this one — that the *representation* you optimize decides everything — and you will see diffusion and volume rendering reappear as tools for generating and reconstructing three-dimensional scenes.
