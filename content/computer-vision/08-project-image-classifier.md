# 08 — Project: Build an Image Classifier

You now have every piece: images as tensors, convolutions, CNN architectures, transfer learning, and an awareness of detection, segmentation, and transformers. This capstone puts them together into the workflow you'd actually run on a real classification problem — load and split data, augment it, transfer-learn from a pretrained backbone, evaluate honestly, and avoid the pitfalls that silently sink beginner projects. Treat it as a template you can reuse for any image-classification task, not just the demo dataset.

## The goal and the plan

The task: classify CIFAR-10 images into their 10 categories, using a pretrained ResNet fine-tuned to our classes. We use CIFAR-10 because torchvision downloads it in one line, but every step transfers to your own folder of labeled images. The plan, in order:

1. Load data and hold out a validation set.
2. Define transforms — augmentation for training, plain preprocessing for validation.
3. Load a pretrained backbone and swap its head.
4. Train, watching train *and* validation metrics.
5. Evaluate on held-out data and inspect the errors.

## Step 1 — Data and a clean split

The cardinal rule from the ML foundations course applies with full force: **split before you do anything else, and never let test data touch training.** CIFAR-10 ships with a train/test split; we carve a validation set out of the training portion for tuning, and we keep the test set sealed until the very end.

```python
import torch, torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
test_set   = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

n_val = 5000
n_train = len(full_train) - n_val
train_idx, val_idx = random_split(range(len(full_train)),
                                  [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))
```

Note the seeded generator — fixing the seed makes the split reproducible, so results are comparable across runs. (Set seeds on `torch`, `numpy`, and `random` at the top of any real training script.)

## Step 2 — Transforms: augment train, not validation

Training data gets augmentation to fight overfitting; validation and test data get *only* the deterministic preprocessing the pretrained model expects. Mixing these up — augmenting the validation set — makes your metrics noisy and dishonest.

```python
train_tf = T.Compose([
    T.Resize(224), T.RandomCrop(224, padding=8),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])
eval_tf = T.Compose([
    T.Resize(224), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

We resize to 224 and use ImageNet normalization because we're fine-tuning an ImageNet-pretrained ResNet — its features expect inputs preprocessed the way it was trained (lesson 5). Getting these stats wrong is the most common silent bug in transfer learning.

## Step 3 — Pretrained backbone, new head

Load ResNet-18 with pretrained weights and replace the 1000-class ImageNet head with a 10-class one. We'll fine-tune the whole network at a modest learning rate — CIFAR differs enough from ImageNet's photo scale that full fine-tuning beats freezing here.

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)   # 10 CIFAR classes
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
```

## Step 4 — The training loop, watching for overfitting

Train, but after every epoch evaluate on the validation set. The gap between training and validation accuracy is your overfitting gauge: if training accuracy climbs while validation stalls or drops, the model is memorizing.

```python
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

for epoch in range(5):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,  train=False)
    print(f"epoch {epoch}: train_acc {tr_acc:.3f}  val_acc {va_acc:.3f}")
# epoch 0: train_acc 0.812  val_acc 0.889
# epoch 4: train_acc 0.961  val_acc 0.951
```

A fine-tuned ResNet-18 comfortably clears 95% validation accuracy on CIFAR-10 in a few epochs — far beyond the ~70% our from-scratch small CNN reached in lesson 5, and with less training. That gap *is* the value of transfer learning, made concrete.

## Step 5 — Evaluate honestly and look at the mistakes

A single accuracy number hides a lot. Once you've picked your final model using the validation set, run it **once** on the sealed test set for your reported number, then dig into *where* it fails with a confusion matrix — which classes get mistaken for which.

```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

model.eval()
preds, targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        p = model(x.to(device)).argmax(1).cpu()
        preds.append(p); targets.append(y)
preds = torch.cat(preds).numpy(); targets = torch.cat(targets).numpy()

print(confusion_matrix(targets, preds))
print(classification_report(targets, preds, digits=3))
```

The confusion matrix on CIFAR-10 typically shows cats and dogs mixed up, and automobiles and trucks confused — semantically similar classes, which is reassuring (the model's errors are *sensible*). Per-class precision and recall reveal whether one class is dragging down the average, which overall accuracy would hide — exactly the evaluation lesson from ML foundations, now on images. Beyond the numbers, pull up the actual misclassified images and look at them: you'll often find mislabeled data, ambiguous pictures, or a systematic weakness (say, dark images) that no metric names for you. This eyeball step is one of the highest-value habits in applied vision.

## Step 6 — Save the model for reuse

A trained model is only useful if you can reload it without retraining. Save the learned weights (the `state_dict`), not the whole Python object, so the checkpoint survives code changes:

```python
torch.save(model.state_dict(), "cifar_resnet18.pt")

# later, to reload for inference:
reloaded = resnet18()
reloaded.fc = nn.Linear(reloaded.fc.in_features, 10)  # rebuild the same head
reloaded.load_state_dict(torch.load("cifar_resnet18.pt"))
reloaded.eval()   # disable dropout/batchnorm updates for inference
```

Two easy-to-forget steps: you must reconstruct the model with the *same* architecture and head before loading the weights, and you must call `.eval()` before inference so layers like batch norm behave deterministically rather than in training mode.

## Common pitfalls

The bugs that quietly ruin image-classification projects, and how to catch them:

- **Wrong or missing normalization.** Feeding a pretrained model raw `[0,255]` pixels or the wrong mean/std yields mediocre accuracy with no error. Use the weights' own `transforms()` or the correct ImageNet stats.
- **Augmenting the validation/test set.** Makes metrics random and inflated-looking. Augment training only.
- **Data leakage across the split.** Duplicate or near-duplicate images landing in both train and test give a falsely high score. Split first, deduplicate if needed.
- **Learning rate too high when fine-tuning.** A large LR smashes the pretrained features you're trying to reuse. Start small (1e-4 or lower) for fine-tuning.
- **Class imbalance ignored.** If 90% of images are one class, 90% accuracy is worthless. Check per-class recall, not just overall accuracy.
- **Judging on the training set.** Training accuracy always looks good. The validation and (once) test numbers are the only ones that mean anything.

## Why this matters for ML

This workflow — split, augment, transfer-learn, watch the train/val gap, evaluate on sealed data, inspect errors — is not specific to CIFAR or even to images. It is *the* applied deep-learning loop, and running it end to end is what separates someone who can recite architectures from someone who can ship a working model. Every pitfall above is one that real practitioners hit; internalizing the checklist here will save you from silently wrong results far more often than knowing one more architecture would.

## Key takeaways

- The real workflow: **split → augment (train only) → transfer-learn from a pretrained backbone → watch train vs. val → evaluate once on sealed test → inspect errors.**
- **Transfer learning wins decisively**: a fine-tuned ResNet-18 clears 95% on CIFAR-10 in a few epochs, far above a from-scratch small CNN.
- Track the **train/validation gap** every epoch — a widening gap is overfitting; that's what augmentation and a small fine-tuning learning rate combat.
- Never trust a single accuracy number — use a **confusion matrix and per-class metrics**, and check that errors are between genuinely similar classes.
- Most failures are **data/preprocessing bugs**, not model bugs: wrong normalization, leaked splits, augmented validation, or a too-high fine-tuning learning rate.

## Try it

Run this project end to end on CIFAR-10, then adapt it to your own images: put labeled photos in class-named folders and load them with `torchvision.datasets.ImageFolder` instead of `CIFAR10` — every other step stays identical. Report your test accuracy and confusion matrix. Then deliberately introduce one pitfall (e.g. drop normalization, or set `lr=1e-2`), observe how much accuracy you lose, and explain why. Finally, swap the ResNet-18 backbone for a pretrained ViT (`vit_b_16`) and compare results and training time on your dataset.
