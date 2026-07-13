# 10 — Video Understanding

Every model so far has taken a single still image: a tensor of shape `(N, C, H, W)`, as you learned back in the [tensors lesson](01-images-as-tensors.md). But most of the visual world arrives as *video* — a stream of frames unfolding in time. A single frame can tell you there is a person and a door; only a sequence tells you whether they are opening it or closing it. Adding one axis, time, sounds harmless. It is not. It changes what the model must represent, and it multiplies the compute bill by the number of frames. This lesson is about that new axis: how to feed it to a network, the architectures that exploit it, and the hard-nosed engineering that makes video ship at all.

## Video as a tensor, and the sampling problem

A clip is a stack of frames, so it gets one more dimension than an image: `(N, C, T, H, W)`, where `T` is the number of frames. A ten-second phone clip at 30 fps is 300 frames — feeding all of them at 224×224 is a tensor of shape `(1, 3, 300, 224, 224)`, roughly 45 million values for *one* example. You cannot train on that at any reasonable batch size.

Note the axis order: PyTorch's video models expect `(N, C, T, H, W)`, with channels before time, mirroring the channels-first convention you met for images. Some data tools hand you `(N, T, C, H, W)` instead, so a `permute` at the dataloader boundary is a common first fix when a video model throws a shape error.

So the first design decision in video is not architectural, it is **sampling**: which frames do you even look at? The workhorse strategy is *sparse sampling* — split the clip into `T` equal segments and take one frame (or a short snippet) from each. Sixteen or thirty-two frames usually capture the action while keeping the tensor tractable. Neighboring frames are highly redundant anyway; nothing at 30 fps changes much in 1/30th of a second, so throwing most of them away costs little and saves enormous compute.

```python
import torch

def sample_clip(video, num_frames=16):
    # video: (C, T_full, H, W); return (C, num_frames, H, W)
    T_full = video.shape[1]
    idx = torch.linspace(0, T_full - 1, num_frames).long()   # evenly spaced
    return video[:, idx]

clip = torch.randn(3, 300, 224, 224)
print(sample_clip(clip).shape)      # torch.Size([3, 16, 224, 224])
```

Everything downstream inherits this choice. Sample too sparsely and you miss fast actions; too densely and you run out of memory. Frame sampling is the single most consequential knob in a video pipeline.

There is a second choice hiding in "which frames": *uniform* sampling (evenly spaced, as above) is simple and reproducible, but it can straddle a boundary between two shots and average across a scene cut. Production systems often run a cheap shot-boundary detector first and sample *within* a shot, so a single clip does not blend two unrelated scenes. Keep this in mind when your labels are clean but accuracy is stubbornly low — the sampler may be handing the model incoherent clips.

## The baseline you must beat: frames, then pool

Before reaching for anything temporal, know the baseline that embarrasses many fancy models: run your ordinary 2D CNN on each sampled frame independently, then **average the per-frame predictions**. This is the core of the Temporal Segment Network (TSN) idea, and for a surprising number of tasks it is nearly as good as a full spatio-temporal model — because a lot of "actions" are given away by a single frame (a swimming pool, a basketball court, a birthday cake). It reuses your entire image pipeline unchanged and costs almost nothing extra.

```python
def frame_pool_predict(model2d, clip):     # clip: (C, T, H, W)
    frames = clip.permute(1, 0, 2, 3)      # (T, C, H, W) -> batch of frames
    logits = model2d(frames)               # (T, num_classes)
    return logits.mean(dim=0)              # average over time
```

The rule of thumb: always run this baseline first. If a temporal model can't beat "guess from single frames," the task doesn't actually require motion, and the extra compute buys you nothing. Motion modeling earns its cost only when *how* things move — not just what is present — determines the label (opening vs. closing a door, a real vs. faked fall).

## From 2D to 3D convolutions

The obvious way to model time is to extend the convolution you already know. A 2D conv slides a `(k_h, k_w)` kernel over height and width; a **3D convolution** slides a `(k_t, k_h, k_w)` kernel over time, height, and width at once, so a single filter can respond to motion — an edge that moves, a hand that rises. **C3D** was the early proof this works, stacking 3×3×3 convolutions over short clips.

3D convs have two costs. They add a whole dimension of parameters and FLOPs, and — worse for practitioners — you cannot initialize them from an ImageNet model, because ImageNet backbones are 2D. Training a large 3D CNN from scratch on video needs a very large video dataset, which for years nobody had.

## The I3D inflation trick

The fix is clever enough to be worth understanding in full, because the same "reuse a 2D model in 3D" idea recurs. **I3D** (Carreira and Zisserman, "Inflated 3D ConvNets") takes a *proven, pretrained 2D* architecture and **inflates** every 2D kernel into 3D by adding a time dimension. A 2D kernel of shape `(out, in, h, w)` becomes a 3D kernel `(out, in, t, h, w)` by copying the 2D weights across all `t` time slices and dividing by `t`, so that feeding a static (constant-over-time) video reproduces exactly the 2D model's response.

```python
import torch

w2d = torch.randn(64, 3, 7, 7)          # a pretrained 2D conv kernel
t = 3
# repeat across the new time axis, then scale so a static clip matches 2D
w3d = w2d.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
print(w2d.shape, "->", w3d.shape)       # (64,3,7,7) -> (64,3,3,7,7)
```

Why it matters: inflation lets a 3D network *start* from all the visual knowledge baked into an ImageNet backbone, instead of learning "what a dog looks like" and "what running looks like" simultaneously from scratch. The network only has to *fine-tune* the temporal part. I3D pretrained this way on the large Kinetics action dataset became the standard video backbone for years, and inflation remains the default way to bootstrap a 3D model from a strong 2D one.

## Two-stream networks and optical flow

A parallel line of thinking asks: if motion is the hard part, why not hand it to the network directly? **Optical flow** is a per-pixel field of two numbers — how far each pixel moved horizontally and vertically between two frames — a pre-computed motion signal. Classically it was solved by hand-built energy minimization; today it is usually a learned network (RAFT and its descendants) that is itself a small CNN. Either way it is a separate pass over the video before the classifier even starts.

A **two-stream** network runs two CNNs on top of it: a *spatial* stream over RGB frames (appearance: what is in the scene) and a *temporal* stream over stacked optical-flow fields (motion: how it moves), then fuses their predictions — typically by averaging the two streams' logits, sometimes with a learned fusion layer. It works well because it explicitly separates the two things video carries, and the temporal stream can be trained even when the spatial stream is frozen. The catch is that computing flow is expensive and usually done offline, which makes two-stream models awkward for real-time use — a recurring theme: in video, you pay for motion one way or another. This cost is precisely what SlowFast and 3D-conv models try to avoid by learning motion from raw RGB instead.

## SlowFast: two speeds instead of two modalities

**SlowFast** keeps the two-pathway idea but drops optical flow. It runs a *slow* pathway at a low frame rate with many channels (rich appearance, few frames) and a *fast* pathway at a high frame rate with few channels (coarse appearance, fine motion), with lateral connections feeding fast into slow. The insight is that spatial semantics change slowly (a "car" stays a car) while motion needs high temporal resolution, so you should spend frames and channels asymmetrically rather than treating every frame equally. It captures motion from raw RGB alone, sidestepping flow computation.

## Video transformers

After [vision transformers](07-vision-transformers.md) took over images, the same shift came to video. The naive move — treat every space-time patch as a token and run full self-attention over all of them — is quadratic in the number of tokens, and a clip has *thousands* of patches. Full space-time attention is unaffordable.

To feel the problem: a 32-frame clip cut into 16×16 patches at 224×224 is 32 × 196 ≈ 6,300 tokens, and full self-attention is quadratic — about 40 million pairwise scores per layer. That is why joint space-time attention does not scale.

The fix is **factorized attention**. **TimeSformer** and **ViViT** split attention into a spatial step (each frame attends within itself) and a temporal step (each patch attends across time at its own location), instead of one giant joint attention. This drops cost from quadratic-in-(space × time) to roughly the sum of the two, for a small accuracy price. It is the same factorization instinct as depthwise-separable convolutions: don't mix everything at once when two cheaper stages compose to nearly the same thing. **MViT** takes a related route, pooling tokens to shrink resolution as depth increases — the channels-grow-space-shrinks pattern from CNNs, ported to attention.

```python
# tokens x: (batch, T, S, dim)  -- S spatial patches per frame
def factorized_attention(x, spatial_attn, temporal_attn):
    B, T, S, D = x.shape
    x = spatial_attn(x.reshape(B * T, S, D)).reshape(B, T, S, D)   # within frame
    x = x.permute(0, 2, 1, 3).reshape(B * S, T, D)                 # across time
    x = temporal_attn(x).reshape(B, S, T, D).permute(0, 2, 1, 3)
    return x
```

## Adding sound

Video usually arrives with audio, and audio is often the cheapest route to the answer — a smashing sound, speech, music. The common recipe encodes the audio (typically as a log-mel spectrogram, which is just an image you can run a CNN or transformer over) into its own embedding and fuses it with the video embedding, either late (concatenate the two summary vectors before the classifier) or with cross-attention between the streams. Multimodal fusion helps most exactly where vision is ambiguous — a phone in hand looks the same whether or not it is ringing, but the audio settles it. Crucially, audio and video are *naturally paired* in every clip: the fact that a sound co-occurs with the frames it came from is a free training signal, no labels required. That audio-visual correspondence is one of the cleanest examples of a broader idea — supervision hiding inside unlabeled data — which is exactly where the next lesson begins.

## Which architecture to reach for

With so many families, the practical choice collapses to a few rules:

- **Appearance-dominated task** (scene, object, or setting gives away the label): start with the **frame-pool baseline** on a strong 2D backbone. Often you can stop here.
- **Need real motion, have a good 2D backbone and moderate data**: **inflate** it to a 3D CNN (I3D). This is the reliable default for action recognition.
- **Latency-sensitive, motion matters, no offline flow allowed**: **SlowFast** or an efficient 3D variant that reads motion from raw RGB.
- **Large dataset and compute, want the accuracy ceiling**: a **video transformer** (TimeSformer / ViViT / MViT) with factorized attention.
- **Two-stream with optical flow**: strong accuracy, but reserve it for offline batch jobs where the flow-computation cost is acceptable.
- **Audio is available and informative**: add an audio stream before you add a heavier video model — it is often cheaper and more decisive than more frames.

The through-line: pick the cheapest family that clears your accuracy bar, and let the frame-pool baseline tell you how much temporal machinery the task actually needs.

## Datasets and honest evaluation

Modern video models are trained on large action datasets — **Kinetics** (hundreds of thousands of clips across ~400–700 human actions) is the de facto ImageNet of video, with Something-Something (fine-grained motions like "pushing something left to right") stressing genuine temporal reasoning. The choice reveals a subtlety: a model can top Kinetics largely on appearance yet fail Something-Something, because the latter is designed so that a single frame gives nothing away.

Evaluation has a video-specific trap. A model sees a short clip, but you want to label a whole video, so inference samples *several* clips (and often multiple spatial crops per clip) and averages their scores — **multi-view** or multi-crop testing. Reported accuracy usually assumes this. If you benchmark a single center clip and compare it to a paper's multi-view number, you will conclude your model is broken when it is only being tested more cheaply.

```python
def video_predict(model, video, clips=10, sampler=sample_clip):
    views = [sampler(video) for _ in range(clips)]      # several sampled clips
    logits = torch.stack([model(v.unsqueeze(0))[0] for v in views])
    return logits.softmax(-1).mean(0)                   # average over views
```

## The compute reality of video

It is worth saying plainly: video is the most compute-hungry corner of applied vision, and every production system is organized around coping with that. A one-hour upload at 30 fps is over 100,000 frames; you obviously cannot run a 3D CNN on all of them. Real systems survive by attacking the frame budget from several sides at once:

- **Sample hard.** Most pipelines classify from 8–32 sampled frames per clip, not the raw stream — the sampling logic from the top of this lesson is doing the heavy lifting.
- **Decode is a bottleneck, not just the model.** Reading and resizing frames off disk often dominates GPU time, so pipelines pre-decode, pre-resize, and stream shards rather than decoding on the fly.
- **Cascade cheap-to-expensive.** A fast lightweight model (even a per-frame 2D CNN) filters the boring 99%, and the heavy spatio-temporal model runs only on the promising clips.
- **Distill.** A large accurate teacher trains a small student that meets the latency budget, trading a little accuracy for the order-of-magnitude speedup that makes real-time feasible.

The lesson from the field is that the exotic architecture is rarely the constraint; the frame budget and the data pipeline are. A team that samples well and decodes efficiently will beat a team with a fancier model and a naive pipeline.

## Why this matters for ML

Video understanding is behind content moderation and search at every large platform, action recognition in sports and security, quality inspection on production lines, and the perception stacks of robots and cars. Sora-class generative video models grab headlines, but the everyday production job is overwhelmingly *understanding* existing video under a tight latency and cost budget. If you interview for such a role, you will be asked less about the newest backbone and more about how you would sample frames, where the pipeline bottlenecks, and how you would hit a latency target — which is exactly why the compute section above matters more than the architecture zoo.

## Key takeaways

- Video adds a time axis: tensors are `(N, C, T, H, W)`, and **frame sampling** — usually sparse, 8–32 frames — is the first and most consequential design choice.
- **3D convolutions** model motion directly but can't reuse 2D pretraining; the **I3D inflation trick** copies pretrained 2D kernels across time (scaled by `1/t`) so a 3D net starts from ImageNet knowledge.
- **Two-stream** nets split appearance (RGB) from motion (optical flow); **SlowFast** gets motion from RGB alone via a slow high-channel and a fast high-frame-rate pathway.
- **Video transformers** (TimeSformer, ViViT) use **factorized** spatial-then-temporal attention to escape the quadratic cost of joint space-time attention.
- Video is compute-bound; production systems cope with aggressive sampling, efficient decoding, cheap-to-expensive cascades, and **distillation** — the pipeline usually matters more than the model.

## Try it

Take any short video, decode it to a `(3, T, 224, 224)` tensor, and implement the `sample_clip` function above for `num_frames` of 8, 16, and 32; print the resulting shapes and the total element counts to feel the memory scaling. Then load a pretrained video classifier from `torchvision.models.video` (e.g. `r3d_18` or an S3D/MViT variant), run it on your sampled clips, and check whether the predicted action changes as you vary the number of sampled frames. Write two sentences on how sampling density affected both the prediction and the memory footprint.

Next, [self-supervised learning](11-self-supervised-learning.md) picks up the thread this lesson left dangling — that the correspondence *within* data (a sound and its frames, an image and its own crops) is a free supervision signal — and turns it into the pretraining recipe behind every modern backbone.
