# 14 — 3D Vision

Everything so far has lived on the flat image plane — a grid of pixels, `(C, H, W)`, exactly as the [first module](01-images-as-tensors.md) set up. But the world is three-dimensional, and a photo is a projection that throws the depth axis away. Robots, AR headsets, self-driving cars, and film pipelines all need to recover or represent that lost dimension. The [diffusion module](13-diffusion-models.md) closed on a lesson that carries straight into 3D: the *representation you optimize decides everything*. Nowhere is that truer than here. There is no single obvious way to put a 3D scene into a tensor, and the entire subject is organized around that choice.

## Why 3D is hard: the representation problem

A 2D image has one canonical layout — a dense grid, every cell filled, neighbors well-defined, and convolution slides across it cleanly. 3D has no such luxury. Do you store a solid grid of little cubes? A cloud of surface points with no fixed order? A mesh of connected triangles? A function you query at any coordinate? Each choice makes some operations trivial and others miserable, and unlike pixels, most of a 3D volume is empty space you do not want to pay for. Getting the representation right *is* the problem; the network is almost secondary. It helps to sort the options into **explicit** representations, which store geometry directly, and **implicit** ones, which store a function that describes it.

## Explicit representations: voxels, point clouds, meshes

**Voxels** are the direct extension of pixels: a 3D grid where each cell is occupied or empty (or holds a value). They are conceptually simple and let you reuse 3D convolutions, but cost is brutal — memory grows with resolution *cubed*, so a modest 256³ grid is already 16 million cells, most of them empty air. Voxels teach the idea cleanly but rarely scale.

**Point clouds** store only a list of 3D points on surfaces — exactly what a LiDAR sensor or depth camera returns. They are compact because they skip empty space, but they are unordered and irregular, so a plain CNN cannot process them. **PointNet** (Qi et al., 2017) solved this with a simple, durable insight: apply the *same* small network to every point independently, then combine the results with a symmetric pooling operation (like a max over all points) that does not care about ordering. That permutation-invariant pooling is the whole trick, and it made deep learning on raw point sets possible.

```python
import torch, torch.nn as nn

class TinyPointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                 nn.Linear(64, 256), nn.ReLU())
        self.head = nn.Linear(256, num_classes)

    def forward(self, pts):            # pts: (batch, num_points, 3)
        feats = self.mlp(pts)          # same MLP applied to every point
        global_feat = feats.max(dim=1).values   # symmetric pool -> order-invariant
        return self.head(global_feat)

torch.manual_seed(0)
print(TinyPointNet()(torch.randn(2, 1024, 3)).shape)   # (2, 10)
```

Shuffle the 1024 points and the output is unchanged — that invariance is exactly what an unordered set demands, and no amount of convolution gives it to you for free.

**Meshes** — vertices connected into triangles — are what graphics and film actually use, because they represent surfaces efficiently and render fast on GPUs. But their connectivity is awkward for a neural network to predict directly; you cannot just regress a variable, connected graph the way you regress a grid. The tradeoff across all three is the recurring theme:

- **Voxels** — regular grid, reuse 3D convolution, but memory scales as resolution cubed; most cells are empty.
- **Point clouds** — compact, sensor-native, but unordered and give you no surface or connectivity directly.
- **Meshes** — compact surfaces that render fast, the graphics standard, but hard for a network to generate because of their irregular connectivity.

Keep this table in mind: almost every 3D method is a bet on one of these tradeoffs, or an attempt to sidestep them with an implicit function.

## Implicit representations: occupancy and SDF

Instead of storing geometry, store a *function* that answers questions about it. An **occupancy network** learns a function that takes any 3D coordinate and returns the probability that the point is inside the object. A **signed distance function** (SDF) returns the distance to the nearest surface, negative inside and positive outside; the surface is wherever the function crosses zero.

```python
class SDF(nn.Module):                  # scene as a coordinate -> distance function
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 1))   # signed distance

    def forward(self, xyz):            # xyz: (N, 3) query points
        return self.net(xyz)           # < 0 inside, > 0 outside, 0 on surface
```

Both occupancy and SDF are typically just a small MLP, so resolution is no longer baked into memory — you can query the function as finely as you like and extract a mesh afterward with an algorithm like marching cubes (which walks a grid and stitches triangles wherever the function crosses its threshold). The cost moves from storage to compute: a megabyte of weights can describe a surface you could sample at any resolution. This shift from storing geometry to storing a queryable function is what set up the breakthrough that reshaped the field.

## NeRF: the core idea

**NeRF** (Neural Radiance Fields, Mildenhall et al., 2020) took the implicit idea and aimed it at photorealistic rendering. The core idea is to represent an entire scene as a single MLP that maps a 3D position and a viewing direction to a color and a density — a **radiance field**. To render a pixel, you shoot a ray from the camera through that pixel into the scene, sample many points along the ray, query the MLP at each, and composite the colors weighted by density. This compositing step is **volume rendering**, and crucially it is differentiable: you can compare the rendered pixel to the real photo and backpropagate all the way into the MLP's weights.

```python
# volume rendering along one ray (the heart of NeRF), schematic
def render_ray(mlp, points, dirs, deltas):
    rgb, sigma = mlp(points, dirs)                 # color + density per sample
    alpha = 1.0 - torch.exp(-sigma * deltas)       # opacity of each segment
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=0)  # transmittance so far
    T = torch.roll(T, 1, dims=0); T[0] = 1.0
    weights = T * alpha
    return (weights[..., None] * rgb).sum(dim=0)   # composited pixel color
```

One subtlety made NeRF actually work: a raw coordinate MLP produces oversmooth, blurry output because small networks struggle to fit high-frequency detail from low-dimensional inputs. NeRF fixes this by first mapping each coordinate through a **positional encoding** — a bank of sines and cosines at many frequencies — before the MLP sees it, which lets the network represent sharp edges and fine texture. It is the same frequency-encoding idea that appears with transformers and, as you just saw, with diffusion timesteps.

The results were startling — free-viewpoint photorealism from a few dozen photos (with known camera poses, usually recovered beforehand by classical structure-from-motion). The cost, though, is steep. A NeRF is optimized **per scene**: you train a fresh MLP on each new set of photos, which historically took hours, and rendering a single frame meant millions of MLP queries, so it was far from real-time. Speedups followed quickly (hash grids and other tricks cut training to minutes), but the fundamental shape — expensive optimization, slow rendering — left an opening.

## 3D Gaussian splatting: why it took over real-time

**3D Gaussian splatting** (Kerbl et al., 2023) filled that opening and, by 2026, is the default for real-time novel-view rendering. It swings back to an *explicit* representation: the scene is a few million little 3D Gaussian blobs, each with a position, shape, color, and opacity. Rendering is not ray-marching an MLP — it is **rasterizing** these blobs, projecting and blending them onto the image plane, which GPUs do extremely fast. The key is that this rasterizer is differentiable, so you still optimize the blobs by comparing renders to real photos, exactly as NeRF optimizes its MLP. You keep NeRF's differentiable-rendering training loop but swap the slow implicit field for a fast explicit primitive. The payoff is rendering at hundreds of frames per second where NeRF crawled — which is why anything interactive (AR, telepresence, live scene capture) moved to splatting almost immediately.

The contrast is the clearest illustration of this module's whole theme — same objective, opposite representation:

- **NeRF** — *implicit* MLP, rendered by ray-marching and volume integration; compact in memory, slow to render, blurs less-observed regions gracefully.
- **Gaussian splatting** — *explicit* blobs, rendered by rasterization; larger in memory, real-time to render, easy to edit and move.

Neither is universally better; you pick by whether you need real-time interaction (splatting) or a compact, easily-regularized field (NeRF).

## Single-image 3D and depth as the workhorse

Reconstructing from many photos is one thing; recovering 3D from a *single* image is far harder and inherently ambiguous, but it is where the practical value concentrates. **Large reconstruction models** now regress a full 3D representation from one or a few images in a single forward pass, trained across huge object datasets — no per-scene optimization at all — and this is an active, fast-moving frontier.

But the genuine workhorse of applied 3D vision is **monocular depth estimation**: predict a depth value for every pixel from a single ordinary image. It does not give you a full scene you can fly through, but a per-pixel depth map is enough for an enormous range of products — background blur, occlusion in AR, obstacle sensing, 3D photo effects. **Depth Anything** and similar models, trained on massive mixed datasets, produce robust depth from arbitrary images and run cheaply, which is exactly why depth estimation ships in far more products than NeRF or splatting ever will. If you touch 3D in a real job, this is probably the piece you will use first.

One honest caveat: most monocular models predict *relative* depth — which pixels are nearer than others — not *metric* depth in meters, because scale is fundamentally ambiguous from a single image (a toy car and a real car can project to the same pixels). If you need true distances you either use a model trained for metric depth, calibrate against a known object, or fall back on stereo or a depth sensor. Knowing whether your task needs relative or metric depth is the first question to ask before wiring one of these models in.

```python
# monocular depth in a few lines with a pretrained transformer
from transformers import pipeline
depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
out = depth("photo.jpg")           # returns a per-pixel depth map
out["depth"].save("depth.png")

import numpy as np
d = np.asarray(out["depth"], dtype=np.float32)
d = (d - d.min()) / (d.max() - d.min())   # normalize to [0,1] for display
print(d.shape)                             # (H, W) - one depth per pixel
```

## Generating 3D, not just reconstructing it

The [diffusion models](13-diffusion-models.md) from the previous module reappear here as a generative engine for 3D. The obstacle is data: there is no internet-scale corpus of 3D scenes the way there is of 2D images, so training a 3D diffusion model directly is starved for examples. The influential workaround is **score distillation**: use a frozen, powerful 2D image diffusion model as a critic, and optimize a 3D representation (a NeRF or a set of Gaussians) so that its renders, from many random viewpoints, look like plausible samples to the 2D model. In effect you borrow the 2D model's vast knowledge of what things look like and lift it into three dimensions one rendered view at a time.

```python
# text-to-3D by score distillation, schematic
def sds_step(scene_3d, diffusion_2d, prompt, cam):
    view = render(scene_3d, cam)              # differentiable render from a random pose
    noisy, eps = add_noise(view)              # noise it as in the diffusion forward process
    eps_pred = diffusion_2d(noisy, prompt)    # what the frozen 2D model expects
    grad = (eps_pred - eps)                   # push the render toward "realistic"
    view.backward(grad)                       # update the 3D representation, not the 2D model
```

The output is still rough compared to native 2D generation, and speed and consistency across views remain open problems — but text-to-3D and image-to-3D are moving fast, and score distillation is the idea to know.

## Where 3D vision ships

The applications are concrete and well-funded. **Robotics** needs depth and point clouds to grasp objects and avoid collisions. **AR and VR** need real-time scene reconstruction and depth to place virtual objects convincingly — splatting and monocular depth both feature heavily. **Autonomous driving** fuses LiDAR point clouds with camera images for 3D object detection and mapping. **Film and games** use meshes, and increasingly captured splats and NeRFs, for virtual sets and visual effects. Across all of them the representation question from the top of this module is the first design decision, not an afterthought.

## Why this matters for ML

3D vision is a specialization, but a lucrative and growing one, and it rewards knowing the tradeoffs rather than memorizing one architecture. In a real role you will most often reach for monocular depth (a pretrained model, a few lines of code) or consume point clouds from a sensor; less often you will capture a scene with Gaussian splatting or fine-tune a reconstruction model. The durable skill is diagnosing which representation a problem wants — Is the geometry dense or sparse? Do I need real-time rendering? One image or many? Watertight surfaces or just depth? — because that choice, far more than the network, decides whether the project is feasible. The same "representation is everything" instinct you built on flat images carries directly into three dimensions.

## Key takeaways

- 3D has **no single canonical tensor layout**; the representation choice dominates every 3D problem.
- **Explicit** representations store geometry directly: **voxels** (regular but cost grows cubically), **point clouds** (compact but unordered — **PointNet**'s symmetric pooling handles them), **meshes** (efficient to render, hard to generate).
- **Implicit** representations store a queryable function — **occupancy** or **SDF** — decoupling detail from memory.
- **NeRF** is an MLP radiance field rendered by differentiable **volume rendering**, with **positional encoding** for sharpness; photorealistic but optimized **per scene** and slow to render.
- **3D Gaussian splatting** keeps the differentiable-rendering loop but uses explicit blobs and fast **rasterization**, making real-time rendering practical — the 2026 default.
- **Monocular depth estimation** (e.g. Depth Anything) is the cheap, robust practical workhorse that ships in the most products — but usually gives *relative*, not *metric*, depth.
- 3D vision powers **robotics, AR/VR, autonomous driving, and film**; the durable skill is choosing the representation the problem wants.

## Try it

Run a pretrained monocular depth model (Depth Anything V2 via `transformers`, as above) on several of your own photos and visualize the depth maps — note where it succeeds (clear foreground/background) and where it is ambiguous (reflections, flat textures). Then reason on paper about memory: a 512³ voxel grid versus the same scene as a 2-million-point cloud versus a small MLP — write down the storage each needs and which operations (rendering, editing, neural processing) each makes easy or hard. Finally, watch any short Gaussian-splatting capture demo and identify, from this module, why it renders in real time where a NeRF would not.

Next, in [vision and language](15-vision-and-language.md), we reconnect vision to text — how models like CLIP learn a shared image-text space, how that space powers search, zero-shot classification, and the conditioning behind the diffusion models you just met, and how it forms the bridge from pure vision into the [VLMs](../vlm-guide/04_vlms.md) that combine seeing and reasoning.
