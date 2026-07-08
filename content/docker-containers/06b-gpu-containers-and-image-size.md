# 06b — GPU Containers and Image Size

The last lesson packaged a small scikit-learn model that runs happily on a CPU. But the reason most ML people reach for Docker in the first place is different: they have a PyTorch or TensorFlow model that needs a GPU, and they want it to run the same way on their laptop, on a rented cloud box, and in production. That turns out to be the trickiest part of containerizing ML work, and it comes with a second headache — GPU-enabled images are enormous. This lesson covers both: how to give a container access to the GPU, and how to stop your image from ballooning to five gigabytes.

## Why a normal container can't see the GPU

Here is the thing that surprises everyone. You have a working GPU on your machine, you build a container, you run your PyTorch code inside it, and `torch.cuda.is_available()` returns `False`. The container is completely blind to the hardware.

This is by design. A container is isolated from the host, and that isolation includes hardware devices. The GPU is driven by the NVIDIA driver installed on the *host*, and nothing inside the container can reach it unless you explicitly bridge the gap.

The bridge is the **NVIDIA Container Toolkit**. It is a piece of software you install on the host that hooks into Docker and, at run time, exposes the host's GPU driver and devices into the container. Crucially, you do *not* install the driver inside the image — the toolkit injects the host's driver at launch. Your image only needs the CUDA *libraries*, not the driver itself.

On an Ubuntu host, installation looks like this:

```bash
# Add NVIDIA's package repository and its signing key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Tell Docker to use it, then restart Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

The `nvidia-ctk runtime configure` step edits Docker's config so it knows how to wire up GPUs. Once Docker restarts, you get the GPU hello-world:

```bash
# Run a CUDA base image and ask it what GPUs it can see
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this prints the familiar `nvidia-smi` table listing your GPU, the plumbing works. The magic flag is `--gpus all`, which tells Docker to expose every GPU to the container. To hand over just one specific card, use `--gpus '"device=0"'` (the quoting is fussy but required), or limit the count with `--gpus 2`.

You may see older tutorials use `--runtime=nvidia`. That is the deprecated way — it was replaced by `--gpus` back in Docker 19.03 and you should not reach for it in new work. `--gpus` is the current, supported flag.

## CUDA base images: runtime vs devel

Your image needs the CUDA libraries so your framework can talk to the GPU. NVIDIA publishes official base images at `nvidia/cuda` for exactly this. The tags follow a pattern like `12.8.0-runtime-ubuntu22.04`, and the middle word matters a lot:

- **`base`** — bare minimum, just enough to run `nvidia-smi`. Rarely what you want directly.
- **`runtime`** — the CUDA runtime libraries needed to *execute* GPU code. This is what a finished inference service ships with.
- **`devel`** — everything in `runtime` plus the CUDA compilers, headers, and static libraries needed to *build* GPU code. Larger, and only needed at build time.

There are also `cudnn` variants (e.g. `12.8.0-cudnn-runtime-ubuntu22.04`) that bundle the deep-learning primitives most frameworks rely on.

Two rules save you pain here. First, **the CUDA version in the tag must match what your framework was built against.** A PyTorch wheel compiled for CUDA 12.8 wants CUDA 12.8 libraries. Mismatch and you get cryptic runtime errors. Second, **avoid the `latest` tag** — NVIDIA deprecated it, and pinning an explicit version keeps your builds reproducible.

If all this version-matching sounds fragile, there is a shortcut: start from the **official PyTorch image**, `pytorch/pytorch`. It ships CUDA, cuDNN, and a matching PyTorch already installed and tested together. Tags look like `2.8.0-cuda12.8-cudnn9-runtime`. For beginners this is often the path of least resistance — you skip the version-matching dance entirely. The tradeoff is size and less control, which brings us to the real problem.

## The image-size problem

Build a naive PyTorch GPU image and check its size:

```bash
docker images
# REPOSITORY   TAG   SIZE
# my-model     ...   6.8GB
```

Seven gigabytes is normal for a naive build, and it hurts everywhere: slow to push to a registry, slow to pull onto each production machine, slow to start. The bulk comes from CUDA libraries, cuDNN, and the framework itself — much of which is only needed to *build*, not to *run*.

The first, cheapest win costs nothing but ordering your Dockerfile correctly. Docker caches each layer and rebuilds from the first line that changed. If you copy your code *before* installing dependencies, every code edit re-runs the multi-minute dependency install. Copy `requirements.txt` first, install, *then* copy code:

```dockerfile
# Dependencies change rarely — install them in their own cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code changes constantly — copy it after, so edits don't bust the deps cache
COPY app.py .
```

Now editing `app.py` reuses the cached dependency layer and rebuilds in seconds. Note `--no-cache-dir`, which stops pip from stashing its download cache inside the image.

## Multi-stage builds

The dependency-ordering trick speeds up builds but does not shrink the final image. For that, use a **multi-stage build**: one stage with all the heavy build tools, and a second, slim stage that copies over only the finished artifacts and throws the build tools away.

The idea is simple. A `devel` CUDA image (with compilers) or a `builder` stage installs everything. Then a fresh `runtime` stage — which has no compilers, no build caches, no headers — copies just the installed Python packages. The final image never contains the multi-gigabyte toolchain.

A modern, fast way to do the install stage is [uv](https://docs.astral.sh/uv/guides/integration/docker/), a drop-in replacement for pip that is dramatically faster and pairs well with multi-stage builds. You copy the `uv` binary from its official image, install into a self-contained virtual environment, and then copy only that environment into the runtime stage. Multi-stage builds routinely cut PyTorch images by 45–60%.

## Keep model weights out of the image

Baking a small scikit-learn model into the image (as we did last lesson) is fine. Baking a multi-gigabyte set of model weights is not — it bloats the image, forces a full rebuild every time you retrain, and means one image can only ever serve one model version.

Instead, keep weights *out* of the image and supply them at run time:

- **Mount them** from the host with `-v /host/weights:/models`, exactly like the volume trick from the ports-and-volumes lesson.
- **Pull them at startup** from object storage (S3, GCS) or a model registry when the container boots.

Either way, the same lean image can serve any model version, and retraining never means rebuilding.

A `.dockerignore` file backstops this. It stops large or irrelevant files from being sent into the build at all:

```
# .dockerignore
*.pt
*.pth
*.ckpt
data/
notebooks/
.git/
__pycache__/
.venv/
```

## A short note on good practice

Two habits mark a production-grade image. First, **don't run as root.** By default containers run as the root user; a `USER` directive drops to an unprivileged account so a compromised container has less power. Second, **add a `HEALTHCHECK`** so Docker itself knows whether your service is actually alive, not just running.

## A complete worked Dockerfile

Here is a full multi-stage Dockerfile for a GPU inference service. The build stage installs dependencies with `uv`; the runtime stage starts from a lean CUDA `runtime` image and copies only the finished environment.

```dockerfile
# ---- Stage 1: build dependencies ----
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

# Grab the uv binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Faster, more predictable installs in containers
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Install into a self-contained virtual environment (deps layer, cached)
COPY requirements.txt .
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r requirements.txt

# ---- Stage 2: lean runtime ----
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Copy ONLY the finished virtual environment — no compilers, no caches
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY app.py .

# Run as a non-root user for safety
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Let Docker verify the service is actually serving
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Weights are NOT copied in — mount or pull them at run time
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build it, then run it with the GPU exposed and the weights mounted:

```bash
docker build -t gpu-model-server .

docker run -d --name server \
  --gpus all \
  -p 8000:8000 \
  -v /host/weights:/models \
  gpu-model-server
```

The `--gpus all` flag gives the container the GPU, `-v` supplies the weights without baking them in, and the image itself stays lean because the compilers never made it past the build stage.

## Key takeaways

- A normal container cannot see the GPU — install the **NVIDIA Container Toolkit** on the host, then run with `docker run --gpus all` (not the deprecated `--runtime=nvidia`).
- Verify the plumbing with `docker run --rm --gpus all nvidia/cuda:...-base... nvidia-smi`.
- Pick a CUDA base image whose version **matches your framework**; use a `runtime` tag to ship and a `devel` tag to build. Or start from the official `pytorch/pytorch` image to skip version-matching.
- Copy `requirements.txt` before your code so dependency layers stay cached across edits.
- Use a **multi-stage build** — heavy build stage, lean runtime stage — to cut PyTorch images by half or more.
- Keep large model weights **out of the image**; mount or pull them at run time, and use a `.dockerignore`.
- Run as a non-root `USER` and add a `HEALTHCHECK` in production images.

## Try it

1. On a machine with an NVIDIA GPU, confirm the toolkit works: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`. You should see your GPU listed.
2. Write a tiny `app.py` (reuse the FastAPI service from lesson 06) whose `/health` endpoint also returns `torch.cuda.is_available()`.
3. Put PyTorch in `requirements.txt`, then build the naive single-stage way and record the image size with `docker images`.
4. Rewrite it as the multi-stage Dockerfile above, rebuild, and compare sizes. Note how much you saved.
5. Run it with `--gpus all` and a mounted weights directory, then `curl localhost:8000/health` and confirm the service reports the GPU is visible.
6. Bonus: add a `.dockerignore` that excludes `*.pt` and `data/`, rebuild, and confirm your weights never enter the image.
