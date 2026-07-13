# 09 — Visualizing and Attacking CNNs

By now you can build a CNN, train it, transfer a pretrained backbone to a new task, and read the outputs of detectors and segmenters. You trust these models because their accuracy numbers are good. But a good number tells you *that* a model works, not *how* — and it certainly doesn't tell you *when it will fail*. This lesson opens the box. We'll look at what the filters and features you trained actually respond to, then show that the same gradients that trained the network can be turned against it to build inputs that fool it completely. Both halves use one idea you already know — backpropagation — pointed in unusual directions.

## What does a filter respond to?

Recall from the [convolution lesson](02-the-convolution-operation.md) that a CNN is a stack of learned filters. The first-layer filters act directly on pixels, so you can just *look* at them: reshape each filter to an image and display it. Do this to almost any ImageNet-trained network and you see the same thing — oriented edges, color blobs, and small gradients. The network rediscovers edge detectors because edges are the most reusable primitive for images, the same way a compression scheme rediscovers frequency bases.

```python
import torch, torchvision
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
w = model.conv1.weight.data        # first conv: (64, 3, 7, 7)
print(w.shape)                     # torch.Size([64, 3, 7, 7])

# normalize each 7x7x3 filter to [0,1] so it can be shown as an RGB image
grid = torchvision.utils.make_grid(w, nrow=8, normalize=True, padding=1)
print(grid.shape)                  # a single image tensor you can imshow
```

Looking at raw weights only works for the first layer, because only there do the weights live in pixel space. A layer-3 filter acts on the *features* of layer 2, not on pixels, so its weights are uninterpretable as an image. For deeper units we need a different trick.

## Feature visualization by optimization

Here is the key inversion of thinking. Training holds the input fixed and adjusts the weights to reduce a loss. To see what a *neuron* wants, we hold the weights fixed and adjust the **input** to maximize that neuron's activation. Start from noise, do gradient *ascent* on the image, and you synthesize the picture that lights up that unit the most.

```python
import torch
torch.manual_seed(0)

img = torch.randn(1, 3, 224, 224, requires_grad=True)
opt = torch.optim.Adam([img], lr=0.05)

activations = {}
model.layer3.register_forward_hook(
    lambda m, i, o: activations.__setitem__("feat", o))

for step in range(30):
    opt.zero_grad()
    model(img)
    # maximize the mean response of channel 12 in layer3
    loss = -activations["feat"][0, 12].mean()
    loss.backward()
    opt.step()
    img.data.clamp_(-2, 2)         # keep pixels in a sane range
```

With regularization (blur, jitter, clamping) these optimized images become the psychedelic textures you have seen: one channel turns out to love dog snouts, another honeycomb patterns, another text. This is how we learned that CNNs build a rough hierarchy — edges, then textures, then parts, then object-like patterns — without anyone ever programming that hierarchy in.

## Feature inversion: what does a layer keep?

Feature visualization asks what a unit *wants*; the complementary question is what a whole layer *remembers*. **Feature inversion** (Mahendran and Vedaldi) freezes the target representation for a real image, then optimizes a fresh image so that *its* features at that layer match — plus a smoothness prior so the result stays image-like. Invert from an early layer and you recover something close to the original photo; invert from a deep layer and you get a blurry, rearranged image that keeps the object's identity and rough layout but discards exact color and position. That gap is the layer's invariance made visible: it shows you precisely which information the network chose to throw away on the road to a label.

```python
target = activations_at(model, real_img, "layer2").detach()   # frozen features
guess = torch.randn_like(real_img, requires_grad=True)
opt = torch.optim.Adam([guess], lr=0.05)
for _ in range(200):
    opt.zero_grad()
    feat = activations_at(model, guess, "layer2")
    loss = F.mse_loss(feat, target) + 1e-3 * total_variation(guess)  # match + smooth
    loss.backward(); opt.step()
```

The same "match a target in feature space" pattern reappears at the end of this lesson in style transfer, and again in the guidance tricks of the generative-models modules — worth filing away now.

## Saliency: which pixels moved the decision?

Feature visualization asks what a unit likes *in general*. A different, more practical question is: for *this* image and *this* prediction, which pixels mattered? The simplest answer is a **saliency map** — the gradient of the class score with respect to the input pixels. Large gradient magnitude means a small change to that pixel would move the score a lot, so that pixel is "important."

```python
x = torch.randn(1, 3, 224, 224, requires_grad=True)   # use a real image here
scores = model(x)
top = scores.argmax(dim=1)
scores[0, top].backward()
saliency = x.grad.abs().amax(dim=1)[0]                 # (224, 224) heatmap
```

Raw gradient saliency is noisy. The workhorse in practice is **Grad-CAM**, which weights the last convolutional feature maps by how much each contributes to the class score, giving a coarse but robust heatmap that lands on the actual object. Saliency methods are the standard tool for the sanity check "is my classifier looking at the animal, or at the watermark in the corner?" — and that failure mode is real and common. A model that scores 98% by reading a dataset artifact is a model that will collapse in production.

## Adversarial examples: gradients turned into weapons

The saliency gradient told us which pixels the model is sensitive to. If we are sensitive to pixels, we can *exploit* that sensitivity. Take a correctly classified image, compute the gradient of the loss with respect to the input, and step the input in the direction that *increases* the loss. A tiny, visually invisible step is enough to flip the prediction. This is the **Fast Gradient Sign Method (FGSM)** of Goodfellow et al. — one gradient sign, one step:

$$x_{\text{adv}} = x + \epsilon \cdot \operatorname{sign}\big(\nabla_x\, \mathcal{L}(f(x), y)\big)$$

The `sign` is what makes it cheap and effective: every pixel is pushed by the same small amount $\epsilon$ in whichever direction hurts, so the perturbation's max size is bounded but it touches all pixels at once.

```python
import torch, torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
for p in model.parameters():
    p.requires_grad_(False)        # we attack the input, not the weights

# a normalized input image tensor (1,3,224,224); random stand-in here
torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)
y = model(x).argmax(dim=1)         # treat the clean prediction as the label

x.requires_grad_(True)
loss = F.cross_entropy(model(x), y)
loss.backward()

eps = 0.01
x_adv = (x + eps * x.grad.sign()).detach()

clean = model(x).argmax(dim=1).item()
adv   = model(x_adv).argmax(dim=1).item()
print("clean:", clean, " adversarial:", adv)   # frequently differ
print("max pixel change:", (x_adv - x).abs().max().item())  # == eps
```

The perturbation is bounded by `eps` per pixel — small enough that clean and adversarial images look identical to you, yet the label changes. Iterating this step with projection back into the `eps`-ball gives **PGD**, a stronger attack and the standard one for evaluating robustness.

## Why they exist

The tempting story is that adversarial examples are weird corner cases the network never saw. The better-supported explanation is the opposite: they are a consequence of models being *too linear* in high dimensions. A 224×224×3 input has ~150,000 dimensions; a nudge of `eps` in each dimension is imperceptible individually but sums to a large dot product with the weight vector. The model responds to that accumulated push exactly as its (locally near-linear) decision boundary says it should. There is a further, uncomfortable finding — that adversarial perturbations often track *genuinely predictive but non-robust features* in the data, features the model was right to use for accuracy but which are brittle. Adversarial examples are less a bug in one network than a property of how these models carve up high-dimensional space.

Two consequences matter in practice. First, **transferability**: an adversarial image crafted against one model often fools a *different* model it was never computed on, because different networks learn similar non-robust features. That means an attacker does not need your weights to attack you. Second, **defenses are hard**. Adversarial training (training on PGD examples) is the only defense that has broadly held up, and it costs a large accuracy drop on clean inputs plus much slower training. A long list of published defenses were later broken by adapting the attack. Treat any "we solved adversarial robustness" claim with suspicion; the honest state in 2026 is that robustness is bought, expensively, not solved.

## DeepDream and style transfer: the same optimization, aimed differently

Once you internalize "hold weights fixed, optimize the image," a family of generative tricks falls out for free.

**DeepDream** runs feature visualization on a real photo instead of noise: pick a layer, and do gradient ascent to *amplify whatever it already detects*. The network hallucinates more of what it thinks it sees — eyes in clouds, dogs in leaves — because you are asking it to exaggerate its own activations.

**Neural style transfer** (Gatys et al.) optimizes an image to satisfy two losses at once: a **content** loss keeping its deep features close to a content photo, and a **style** loss matching the *Gram matrices* (feature correlations) of a style painting at several layers. The Gram matrix throws away where features occur and keeps how often they co-occur — which turns out to be a good mathematical proxy for "texture" and "style."

$$\mathcal{L} = \alpha\, \mathcal{L}_{\text{content}} + \beta\, \mathcal{L}_{\text{style}}, \qquad G^l_{ij} = \sum_k F^l_{ik} F^l_{jk}$$

Both are the mirror image of adversarial attacks: adversarial optimization changes the image to break the network's output; style transfer changes the image to match a target in the network's *feature space*. Same machinery, opposite intent. This "optimize in feature space" idea is also the conceptual ancestor of the guidance tricks you will meet in the generative-models lessons later in this tier.

## Why this matters for ML

When you ship a vision model, the accuracy on your test set is a floor, not a guarantee. Visualization tools (Grad-CAM especially) are how you catch a model that is right for the wrong reason before a customer does — the classic "it detected the ruler, not the melanoma." Adversarial fragility is a genuine security surface anywhere a vision model gates a decision an adversary wants to change: content moderation, fraud and document checks, face-based access, autonomous perception. You are unlikely to run PGD in a normal product loop, but you are very likely to be asked "can this be fooled, and how do we know it's looking at the right thing?" — and the answer to both lives in this lesson.

## Key takeaways

- First-layer filters are viewable as images and reliably learn edges/colors; deeper units are seen by **optimizing an input** to maximize their activation (gradient ascent on the image).
- **Saliency maps** and **Grad-CAM** show which pixels drove a specific prediction — the standard check for "right answer, wrong reason."
- **FGSM** flips a prediction with an imperceptible perturbation: step the input by `eps * sign(grad of the loss)`; iterating it (PGD) is stronger.
- Adversarial examples come from near-linear behavior in high dimensions and non-robust-but-predictive features; they **transfer** across models, and only adversarial training defends reliably, at a real cost.
- **DeepDream** and **style transfer** are the same "optimize the image, freeze the weights" idea aimed at feature-space targets instead of at breaking the output.

## Try it

Grab a real photo, preprocess it with `ResNet18_Weights.DEFAULT.transforms()`, and confirm the top-1 prediction. Run the FGSM block above on it (use the true clean prediction as the label) and find the smallest `eps` that flips the label — plot label vs. `eps`. Then compute a Grad-CAM heatmap for the clean image's predicted class and overlay it, and check whether the model was looking at the object or at the background. Write two sentences: one on how small `eps` had to be, one on whether the saliency surprised you.

Next up, [video understanding](10-video-understanding.md): we add a time axis to the tensor and find that everything gets more expensive, fast.
