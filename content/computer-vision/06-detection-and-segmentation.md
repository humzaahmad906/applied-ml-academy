# 06 — Detection and Segmentation

Everything so far has answered one question about an image: *what is this?* — a single label for the whole picture. But most real vision problems ask more. A self-driving car needs to know there are three pedestrians and *where* each one is. A medical tool needs to outline the exact shape of a tumor. These are the tasks of **object detection** and **segmentation**, and they build directly on the CNN features you already understand. This lesson maps the landscape so that when you reach for a detection or segmentation model, you know what its outputs mean and how they're judged.

## The three levels of "where"

It helps to line up the tasks by how precisely they localize objects:

- **Classification** — one label for the whole image. "This is a cat."
- **Object detection** — a label *and* a bounding box for every object. "A cat here, a dog there," each in its own rectangle.
- **Semantic segmentation** — a label for *every pixel*. Each pixel is marked "cat," "road," "sky," etc. But all cats are lumped into one "cat" region.
- **Instance segmentation** — a per-pixel mask *per object*. Not just "these pixels are cat," but "these pixels are cat #1, those are cat #2."

Each step localizes more finely, and each needs richer outputs and slightly different machinery — but all of them run on top of a CNN (or transformer) **backbone** that extracts features, exactly the kind you learned to build and fine-tune.

## Object detection: boxes and confidence

A detector's output for an image is a list of predictions, each being a **bounding box** (four numbers — the box's location and size), a **class label**, and a **confidence score**. The two questions that define the field are: how do we measure whether a predicted box is correct, and how do we avoid predicting the same object five times?

**IoU (Intersection over Union)** measures how well a predicted box matches the true box. It's the area where the two boxes overlap divided by the area they jointly cover — 1.0 for a perfect match, 0 for no overlap. A prediction is usually counted as correct if its IoU with a ground-truth box exceeds a threshold like 0.5.

```python
def iou(box_a, box_b):
    # boxes are (x1, y1, x2, y2)
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter)

print(round(iou((0,0,2,2), (1,1,3,3)), 3))   # 0.143
```

**NMS (Non-Maximum Suppression)** solves the duplicate problem. A detector typically fires many overlapping boxes around one object. NMS keeps the highest-confidence box, then discards every other box that overlaps it too much (high IoU), and repeats. The result is one clean box per object. It's a post-processing step, not something the network learns, and it runs after almost every detector.

## One-stage vs. two-stage detectors

Detectors come in two broad families, trading speed against precision.

**Two-stage detectors** (the **R-CNN** family, culminating in Faster R-CNN) work in two passes: first a "region proposal" step suggests areas that *might* contain objects, then a second network classifies and refines a box for each proposal. Two passes make them accurate but slower.

**One-stage detectors** (the **YOLO** — "You Only Look Once" — family, and SSD) skip the proposal step and predict all boxes and classes in a single forward pass over a grid laid on the image. That makes them fast enough for real-time video, historically at a small accuracy cost — a gap that has largely closed in recent YOLO versions, which is why YOLO is the common default for practical, real-time detection today.

How do you score a detector overall, across all its boxes and classes? The standard metric is **mAP (mean Average Precision)**. For each class you sweep the confidence threshold and trace how precision (of the boxes I predicted, how many were right) trades off against recall (of the true objects, how many I found); the area under that curve is the Average Precision for that class, and mAP averages it over all classes — usually reported at an IoU threshold like 0.5 (written `mAP@0.5`) or averaged over several thresholds. The one thing to remember: a bare "accuracy" number is meaningless for detection, because there's no single label per image — mAP is the currency the whole field reports and compares in.

You rarely build these from scratch. torchvision ships pretrained detectors you can use directly:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn, \
    FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).eval()

# outputs: a list of dicts with 'boxes', 'labels', 'scores' per image
```

## Segmentation: labeling every pixel

Segmentation pushes localization to the pixel. The key architectural idea is the **encoder-decoder**: a CNN encoder downsamples the image into rich, low-resolution features (the channels-grow-space-shrinks pattern you know), then a decoder *upsamples* those features back to full resolution, producing a per-pixel class map. The classic **U-Net** adds skip connections from encoder to decoder so fine spatial detail lost during downsampling is restored — hugely influential, especially in medical imaging (and the same U-Net shape reappears at the heart of image-generating diffusion models).

Recall the distinction:

- **Semantic segmentation** classifies each pixel but doesn't separate objects of the same class — a crowd becomes one "person" blob.
- **Instance segmentation** produces a separate mask per object — every person is individually outlined. **Mask R-CNN** is the classic approach: it's Faster R-CNN plus an extra branch that predicts a pixel mask inside each detected box.

## The current landscape (2026)

Two shifts are worth knowing so you're not surprised by modern tooling. First, **transformer-based models** (DETR for detection, and transformer-backbone segmenters) have become strong alternatives to the CNN-based classics — the vision transformer of the next lesson increasingly serves as the backbone under detection and segmentation heads. Second, **foundation models** have arrived: Meta's **Segment Anything Model (SAM, now SAM 2)** can segment essentially any object you point at, or even track it through video, without task-specific training — a "segment anything" prompt-driven tool. Combined with open-vocabulary detectors that find objects from a text description, the frontier is moving toward general, promptable vision systems rather than models trained for one fixed label set. For most applied work, though, a pretrained YOLO or Faster R-CNN for detection, and a U-Net or Mask R-CNN for segmentation, remain the dependable workhorses.

## Why this matters for ML

Classification is where you learn the concepts, but detection and segmentation are where a great deal of real computer vision *value* lives — inspection systems, medical imaging, autonomous machines, document understanding. You may rarely implement these architectures from scratch, but you constantly need to read their outputs correctly (what a box, score, and mask mean), evaluate them honestly (IoU, and why an accuracy number alone is meaningless for detection), and pick the right tool (real-time one-stage vs. high-precision two-stage; semantic vs. instance). This lesson gives you that literacy and shows that all of it still rests on the CNN backbones from earlier in the course.

## Key takeaways

- Vision tasks localize at increasing precision: **classification** (whole image) → **detection** (boxes) → **semantic segmentation** (per-pixel class) → **instance segmentation** (per-object mask).
- Detection outputs boxes + labels + scores; **IoU** measures box overlap quality, and **NMS** removes duplicate boxes for the same object.
- **Two-stage** detectors (Faster R-CNN) are accurate but slower; **one-stage** detectors (YOLO) are fast enough for real time and now nearly as accurate — the common default.
- Segmentation uses an **encoder-decoder** (e.g. U-Net) to label every pixel; **semantic** merges same-class objects while **instance** (Mask R-CNN) separates them.
- The 2026 frontier adds transformer backbones and promptable **foundation models** (SAM/SAM 2, open-vocabulary detection), but pretrained CNN-based models remain reliable defaults.

## Try it

Load `fasterrcnn_resnet50_fpn(weights="DEFAULT")`, put it in `.eval()` mode, and run it on any photo containing a few objects (use `weights.transforms()` to preprocess). Print the returned `boxes`, `labels`, and `scores`, then keep only detections with score > 0.5 and draw the boxes. Next, implement NMS by hand using the `iou` function above: given several overlapping boxes with scores, keep the highest and suppress any with IoU > 0.5 against it. Verify your result matches `torchvision.ops.nms`.
