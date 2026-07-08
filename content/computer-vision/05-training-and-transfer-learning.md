# 05 — Training and Transfer Learning

Knowing how a CNN is built is one thing; getting one to actually reach good accuracy on your data is another. This lesson is the practical core of the course. It covers the two things that make the difference in real projects: **data augmentation** (squeezing more out of the data you have) and **transfer learning** (starting from a model someone else already trained on millions of images). The headline, which is worth saying up front: for almost any real vision task, you should *not* train a CNN from scratch — you should fine-tune a pretrained one.

## Training a CNN from scratch: the baseline

Let's first train the small CNN from lesson 3 on a real dataset — CIFAR-10, 60,000 tiny 32×32 color images across 10 classes — using torchvision to fetch and load it. This is the standard PyTorch training loop you've seen before, now on images.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

train_tf = T.Compose([
    T.ToTensor(),                         # PIL -> (C,H,W) float in [0,1]
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # center per channel
])

train = torchvision.datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=train_tf)
loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallCNN(num_classes=10).to(device)   # from lesson 3, adapted to 3 channels
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: loss {loss.item():.3f}")
# epoch 0: loss 1.512
# ...
# epoch 9: loss 0.842
```

Nothing here is new relative to the deep-learning course: forward pass, cross-entropy loss on the 10 logits, backward, optimizer step. The only vision-specific parts are the `transforms` pipeline and the fact that batches are 4-D image tensors. Trained this way, the small CNN reaches maybe 70% accuracy on CIFAR-10 — decent, but nowhere near the >95% a good model achieves. Two upgrades close that gap.

## Data augmentation: free extra data

A model overfits when it memorizes the exact training images instead of learning general features. **Data augmentation** fights this by randomly perturbing each training image every time it's loaded — flipping it, cropping it, shifting its colors — so the network almost never sees the identical image twice. It learns that a cat flipped horizontally is still a cat, which is exactly the kind of invariance we want.

```python
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),      # random shift, padded then cropped
    T.RandomHorizontalFlip(),         # 50% chance to mirror
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

Two rules keep augmentation honest. **First, augment training data only** — never the validation or test set, which must stay a fixed, realistic yardstick. **Second, augmentations must preserve the label.** A horizontal flip is fine for a cat, but flipping a photo of the digit "2" or a road sign can change its meaning — so choose transforms that make sense for *your* task. Applied well, augmentation alone can add several points of accuracy for free, and it's standard in every serious training run.

## Transfer learning: stand on a giant's shoulders

Here's the idea that changes how you approach nearly every real problem. A model like ResNet-50 was trained on ImageNet's 1.2 million images, and in doing so its early and middle layers learned *general* visual features — edges, textures, shapes, object parts — that are useful for almost any image task, not just ImageNet's 1,000 classes. Rather than relearn all of that from your small dataset, you **reuse** those learned features and only retrain the final classifier for your classes. This is **transfer learning**, and it's the practical default: it needs far less data, trains far faster, and usually reaches far higher accuracy than training from scratch.

The mechanics: load a pretrained backbone, replace its classification head with one sized for your number of classes, and train. Modern torchvision uses a `weights=` enum (the old `pretrained=True` is deprecated) and — importantly — bundles the exact preprocessing each model expects:

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT          # best available ImageNet weights
model = resnet50(weights=weights)

# use the SAME preprocessing the model was trained with
preprocess = weights.transforms()           # resize, center-crop, normalize

# swap the 1000-class ImageNet head for your own (e.g. 10 classes)
import torch.nn as nn
model.fc = nn.Linear(model.fc.in_features, 10)
```

That `weights.transforms()` call matters: it applies the correct resize and the ImageNet normalization stats from lesson 1. Feed a pretrained model differently-normalized inputs and its features misalign — a classic silent bug that just yields mediocre accuracy with no error message.

## Feature extraction vs. fine-tuning

There are two ways to train after swapping the head, and the right one depends on how much data you have.

**Feature extraction (freeze the backbone).** Freeze all the pretrained layers so their weights don't update, and train *only* the new head. The backbone acts as a fixed feature extractor. This is fast, needs very little data, and rarely overfits — the right choice when your dataset is small or very similar to ImageNet.

```python
for param in model.parameters():
    param.requires_grad = False           # freeze everything
model.fc = nn.Linear(model.fc.in_features, 10)  # new head is trainable
# only model.fc's parameters will now receive gradients
```

**Fine-tuning (unfreeze some or all).** Let the backbone's weights update too, usually with a **small learning rate** so you gently adjust the pretrained features rather than smashing them. This can reach higher accuracy when you have more data or your images differ a lot from ImageNet (medical scans, satellite imagery). A common recipe: train the head first with the backbone frozen, then unfreeze and fine-tune the whole thing at a low learning rate.

A rough decision guide:

- **Small data, similar to ImageNet** → feature extraction (freeze).
- **More data, or a different domain** → fine-tune, low learning rate.
- **Lots of data, very different domain** → fine-tune more layers, or (rarely) train from scratch.

The reason fine-tuning needs a *small* learning rate deserves a name: **catastrophic forgetting**. The pretrained weights encode millions of images' worth of visual knowledge, and a large learning-rate update can overwrite that knowledge with a few gradient steps from your small dataset — leaving you worse off than if you'd frozen the backbone. A gentle learning rate nudges those features toward your task without erasing what they already know. A common refinement is *discriminative* learning rates: a tiny rate for the early, general layers and a larger one for the later, more task-specific layers and the new head, since the early layers need the least adjustment.

## Watching the training and validation curves

Whichever strategy you pick, judge it by tracking training and validation accuracy together, epoch by epoch — the same habit from classical ML. If both rise together, you're learning; if training accuracy keeps climbing while validation flattens or falls, you're overfitting and should add augmentation, freeze more layers, or stop early. Because transfer learning starts from strong features, you'll often see validation accuracy jump high within the very first epoch — a signature of a good pretrained backbone doing most of the work before you've barely trained. If it *doesn't* jump, that's an early warning that your preprocessing or normalization is likely mismatched.

## Why this matters for ML

Transfer learning is the single highest-leverage technique in applied computer vision. It's the reason a team with a few thousand labeled images can build a production-grade classifier in an afternoon, where from-scratch training would demand a huge dataset and days of compute. Knowing *when* to freeze versus fine-tune, and knowing to reuse the model's original preprocessing, separates results that work from results that quietly underperform. This same pretrain-then-fine-tune pattern is exactly how large language models and vision-language models are adapted too — you're learning the dominant paradigm of modern ML, applied to images.

## Key takeaways

- Training a CNN from scratch works but needs lots of data and compute, and often underperforms — treat it as a baseline, not the goal.
- **Data augmentation** (random flips, crops, color jitter) creates variety, fights overfitting, and is nearly free accuracy — apply it to **training data only**, and only label-preserving transforms.
- **Transfer learning** reuses a model pretrained on ImageNet, replacing just the classification head for your classes — the practical default: less data, faster, more accurate.
- Use the model's bundled `weights.transforms()` so preprocessing matches training exactly; mismatched normalization silently hurts accuracy.
- **Freeze** the backbone (feature extraction) for small/similar data; **fine-tune** at a low learning rate for larger or different-domain data.

## Try it

Fine-tune a pretrained model on CIFAR-10 (or any small image dataset). Load `resnet18(weights="DEFAULT")`, replace `model.fc` with a 10-class linear layer, and train two versions: one with the backbone frozen (only the head trains), and one fully fine-tuned at `lr=1e-4`. Compare their validation accuracy after 3 epochs, and note which trained faster. Then remove your augmentation transforms and retrain the better model — how much accuracy do you lose without augmentation?
