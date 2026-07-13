# 12 — Generative Models: VAEs and GANs

The [previous module](11-self-supervised-learning.md) trained networks to *understand* images without labels — to compress a photo into features that capture what is in it. This one flips the arrow. Instead of mapping an image to a representation, we want to map a representation back out to a brand-new image that never existed. That is **generative modeling**: learning the distribution of real images well enough to sample fresh ones. It is a genuinely harder problem than classification, and the two families in this module — the VAE and the GAN — are the ideas that made it work before diffusion took over, and whose machinery diffusion still runs on.

## The generative modeling problem

A dataset of images is really a sample from some unknown probability distribution $p_{\text{data}}(x)$ over "images that look real." Generation means learning a model $p_\theta(x)$ close to it, and then drawing new samples from that model. The catch is dimensionality: a 256×256 RGB image lives in a space of nearly 200,000 dimensions, and the region of it that looks like a real photo is a vanishingly thin sliver. Learning where that sliver is, and how to sample from it, is the whole game.

Approaches split by *what they let you compute*:

- **Likelihood-based** models define $p_\theta(x)$ explicitly and train by maximizing the probability of the data. Autoregressive models and VAEs (and later, normalizing flows and diffusion) live here. You can, at least in principle, ask "how likely is this image?"
- **Implicit** models never write down $p_\theta(x)$ at all. They only give you a way to *sample* — a function from noise to an image — and are trained by a signal other than likelihood. GANs are the archetype.

Keep this split in mind; it explains almost every tradeoff that follows, including why the two families fail in opposite ways.

## Autoregressive image models, briefly

The most direct likelihood approach is to generate the image one pixel at a time, each pixel conditioned on all the pixels before it — a chain-rule factorization $p(x) = \prod_i p(x_i \mid x_{<i})$. **PixelRNN** and **PixelCNN** did exactly this and gave sharp, exact-likelihood samples. Their problem was speed: generating an image is a sequential loop over thousands of pixels, hopelessly slow at high resolution, and pixels are a clumsy unit to model one at a time.

The idea did not die — it *changed units*. Modern autoregressive image models first compress an image into a short grid of discrete **tokens** (using a VAE-like quantizing autoencoder), then run a [transformer](07-vision-transformers.md) over those tokens exactly as a language model runs over word tokens. This is the same tokens-of-anything recipe you meet in the [NLP course](../nlp-with-transformers/04-transformer-architecture.md), and it is why image and text generation have converged on shared architecture. We will not dwell on it, but note the lineage: autoregression came back the moment we stopped modeling raw pixels.

## VAE: encoder, decoder, and a latent space

The **Variational Autoencoder** takes the encoder-decoder shape you already know from segmentation's [U-Net](06-detection-and-segmentation.md) and makes it generative. An encoder maps an image $x$ to a distribution over a low-dimensional **latent** $z$; a decoder maps $z$ back to an image. The twist versus an ordinary autoencoder is that we force the latent space to follow a known prior — a standard Gaussian $p(z) = \mathcal{N}(0, I)$ — so that after training we can *sample* $z$ from that prior and decode it into a new image, no input required.

Why not just maximize $\log p_\theta(x)$ directly? Because computing it requires integrating over every possible latent, $p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz$, which is intractable. The VAE's answer is to optimize a tractable *lower bound* instead.

## Deriving the ELBO

Introduce the encoder $q_\phi(z \mid x)$ as an approximation to the true posterior. Starting from the log-likelihood and multiplying inside by $q_\phi/q_\phi$:

$$
\begin{aligned}
\log p_\theta(x)
&= \log \int p_\theta(x \mid z)\,p(z)\,dz
= \log \mathbb{E}_{q_\phi(z\mid x)}\!\left[\frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right] \\[4pt]
&\ge \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right]
&&\text{(Jensen's inequality)} \\[4pt]
&= \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}}
- \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularizer}}
\end{aligned}
$$

That last line is the **Evidence Lower BOund (ELBO)**, and maximizing it maximizes a floor under the true log-likelihood. It reads as two forces in tension:

- The **reconstruction** term wants the decoder to rebuild $x$ accurately from its latent — pushing the encoder to pack in as much image-specific information as possible.
- The **KL** term is a leash: it pulls each per-image latent distribution $q_\phi(z \mid x)$ back toward the shared Gaussian prior. This is what keeps the latent space smooth and samplable — without it the encoder would scatter each image to its own isolated spike and sampling the prior would decode to garbage.

Training maximizes the ELBO, or equivalently minimizes its negative: reconstruction loss plus KL penalty.

## The reparameterization trick

There is one obstacle to training this with backprop. The forward pass *samples* $z$ from $q_\phi(z \mid x)$, and you cannot backpropagate through a random draw — the sampling node has no gradient. The **reparameterization trick** fixes it by moving the randomness outside the learnable path: instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ directly, sample a fixed noise $\epsilon \sim \mathcal{N}(0, I)$ and compute $z = \mu + \sigma \odot \epsilon$. Now $\mu$ and $\sigma$ are deterministic outputs of the encoder that gradients flow through cleanly, and all the stochasticity sits in $\epsilon$, which needs no gradient.

```python
import torch
import torch.nn.functional as F

def vae_step(x, encoder, decoder):
    mu, logvar = encoder(x).chunk(2, dim=-1)   # encoder outputs mean and log-variance
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)                # the externalized randomness
    z = mu + std * eps                         # reparameterized sample
    x_hat = decoder(z)

    recon = F.mse_loss(x_hat, x, reduction="sum")
    # closed-form KL between N(mu, sigma^2) and N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

torch.manual_seed(0)
enc = torch.nn.Linear(64, 2 * 8)               # -> mu, logvar of an 8-d latent
dec = torch.nn.Linear(8, 64)
print(round(vae_step(torch.randn(4, 64), enc, dec).item(), 2))
```

Note that the KL against a unit Gaussian has a closed form, so only the reconstruction term needs sampling. That single line — `z = mu + std * eps` — is what makes the whole model differentiable.

## Why VAE samples look blurry

VAEs are stable and easy to train, but their samples tend to look **soft and blurry**, and the ELBO tells you why. The reconstruction term is typically a pixel-wise loss (Gaussian likelihood is just MSE). When the decoder is unsure whether an edge falls on pixel 40 or 41, the loss-minimizing move is to *average* — predict a faint smear across both rather than commit to one and risk a large squared error. Pixel-wise losses reward hedging, and hedging looks like blur. This is the VAE's characteristic weakness and the direct motivation for the next approach, which replaces the pixel loss with a learned judge of realism.

## GAN: the minimax game

The **Generative Adversarial Network** throws out likelihood entirely. It pits two networks against each other. A **generator** $G$ maps noise $z$ to a fake image. A **discriminator** $D$ is a classifier trained to tell real images from $G$'s fakes. The generator's goal is to fool the discriminator; the discriminator's goal is to not be fooled. As $D$ gets better at spotting fakes, it hands $G$ an ever-sharper gradient toward realism. There is no pixel-wise target and no averaging — so, done right, no blur.

Formally it is a two-player minimax game over a single value function:

$$
\min_G \max_D \;\; \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big]
\;+\; \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]
$$

```python
def gan_step(real, G, D, opt_g, opt_d, z_dim=64):
    bs = real.size(0)
    z = torch.randn(bs, z_dim)

    # --- discriminator: real should score 1, fake should score 0 ---
    fake = G(z).detach()                       # detach so G is not updated here
    loss_d = F.binary_cross_entropy(D(real), torch.ones(bs, 1)) \
           + F.binary_cross_entropy(D(fake),  torch.zeros(bs, 1))
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()

    # --- generator: wants D to call its fakes real ---
    fake = G(z)
    loss_g = F.binary_cross_entropy(D(fake), torch.ones(bs, 1))
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    return loss_d.item(), loss_g.item()
```

Two implementation details in that sketch carry real weight. The `detach()` keeps the generator out of the discriminator's update. And the generator is trained to *maximize* $\log D(G(z))$ (target label 1) rather than minimize $\log(1 - D(G(z)))$ — the "non-saturating" form — because the original term gives near-zero gradient exactly when the generator is losing badly and needs the signal most.

## Instability and mode collapse

GANs are famously hard to train, and the reason is structural: there is no single loss going down. The two networks are chasing a moving equilibrium, and if either gets too strong the other's gradient vanishes. If the discriminator becomes perfect, it outputs 0 for every fake and the generator learns nothing; if the generator races ahead, the discriminator's signal is uninformative. You are balancing two learners, not minimizing one objective, which is why GAN training feels more like tuning a control loop than fitting a model.

The signature failure is **mode collapse**. Because the generator is only rewarded for fooling the discriminator, it can win by producing a handful of very convincing images — or even one — and ignoring the rest of the data's diversity. It found a spot the discriminator cannot currently punish and parks there. The samples look real *individually* but the model has quietly stopped covering the true distribution. This is the mirror image of the VAE's failure: a VAE covers the distribution but blurs each sample; a collapsed GAN makes each sample sharp but covers only a fraction of the distribution.

## The DCGAN to StyleGAN arc

Progress in GANs was largely a story of taming that instability. **DCGAN** established the architectural recipe that made GANs train at all reliably — all-convolutional generator and discriminator, batch normalization, strided convolutions instead of pooling. From there the frontier pushed resolution and control: progressive growing trained from low to high resolution, and **StyleGAN** restructured the generator to inject the latent at every layer through adaptive normalization, giving both photorealistic faces and disentangled control over coarse-to-fine attributes (pose, then features, then fine texture). StyleGAN was the high-water mark of GAN image quality and still shows up wherever fast, single-pass generation matters.

A parallel line made GANs *controllable* rather than just unconditional. **Conditional GANs** feed a label or an input image to both networks so you can steer the output: **pix2pix** learns image-to-image translation from paired data (sketch to photo, map to satellite), and **CycleGAN** does it without pairs by adding a cycle-consistency loss that requires translating there-and-back to recover the original. These image-to-image GANs remain practical workhorses for style transfer, domain adaptation, and paired-data-scarce translation tasks.

## Evaluating generative models, and their lies

How do you score a model whose whole job is to produce images with no correct answer? You cannot use accuracy. The field's standard is the **Fréchet Inception Distance (FID)**: run both real images and generated images through a pretrained network, model each set's feature activations as a Gaussian, and measure the Fréchet distance between the two Gaussians. Lower is better; it rewards samples that are both realistic and as varied as the real data.

FID is useful and it is also routinely misleading. It compresses "quality" and "diversity" into one scalar, so a model that generates gorgeous images of narrow variety can post a similar FID to one that generates so-so images of full variety — the number hides *which* failure you have. It depends on the feature network and preprocessing, so FID values are only comparable within the exact same evaluation pipeline; a "better FID" from another codebase may mean nothing. And it is biased by sample count. **Precision and recall for generative models** were introduced precisely to split what FID merges: *precision* asks how many generated samples fall within the real distribution (quality), *recall* asks how much of the real distribution the samples cover (diversity) — and mode collapse shows up cleanly as high precision with low recall. The durable lesson: never trust a single generative metric, and always look at samples with your own eyes.

## Where this shows up

VAEs and GANs are still working parts, not museum pieces. The VAE's encoder-decoder is the compression stage inside modern latent-diffusion image generators — the model diffuses in a VAE's latent space, not in pixels, which is what made high-resolution generation affordable. GANs remain the tool of choice when you need *fast* single-pass generation or upscaling: real-time super-resolution, face and avatar synthesis, texture generation for games, and data augmentation for domains with too few real images. And the FID-versus-samples discipline is exactly the judgment call product teams make when deciding whether a generative feature is good enough to ship.

More broadly, understanding these two failure modes — blur from pixel-wise averaging, and collapse from an objective that rewards fooling rather than covering — is what lets you diagnose *any* generator quickly, including the diffusion systems that now dominate. The vocabulary of latent spaces, adversarial judges, and distribution coverage carries straight over.

## Key takeaways

- Generative modeling learns the distribution of real images to sample new ones; methods split into **likelihood-based** (autoregressive, VAE) and **implicit** (GAN).
- Autoregressive pixel models were exact but slow; they returned by modeling discrete **tokens** with a transformer instead of raw pixels.
- The **VAE** maximizes the **ELBO** = reconstruction − KL; the KL leashes the latent to a Gaussian prior so you can sample it, and the **reparameterization trick** ($z = \mu + \sigma \odot \epsilon$) makes sampling differentiable.
- Pixel-wise reconstruction rewards hedging, which is why **VAE samples look blurry**.
- The **GAN** is a minimax game between generator and discriminator; it produces sharp samples but suffers **instability** and **mode collapse** (sharp but low-diversity) — the opposite failure to the VAE.
- Evaluate with **FID**, but know its lies: it merges quality and diversity, depends on the pipeline, and needs **precision/recall** to separate the two — and never skip looking at real samples.

## Try it

Build a tiny VAE on MNIST: a two-layer encoder to an 8-dimensional latent, the reparameterization trick, a two-layer decoder, and the reconstruction-plus-KL loss above. Train briefly, then sample `z ~ N(0, I)` and decode — the digits will be recognizable but soft. Next, drop the KL term (set its weight to 0) and watch reconstruction sharpen while prior samples turn to noise, seeing the ELBO's tension directly. If you have time, train the small GAN sketch on the same data and compare which failure you hit first: blur or collapse.

The VAE gave us a samplable latent space; the GAN gave us a learned notion of realism instead of a pixel loss. The [next module](13-diffusion-models.md) shows how diffusion models keep the best of both — a stable, likelihood-flavored training objective *and* sharp, diverse samples — by learning to reverse a gradual noising process, and how they reuse the VAE latent and the U-Net you already know to become the dominant image generators of the era.
