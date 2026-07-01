# 03 — Dockerfile Basics

So far you have run images that already existed. Now you will build your own. The recipe for an image is a plain text file called a `Dockerfile`. This lesson covers the essential instructions and walks you through building a real image from scratch.

## What a Dockerfile is

A `Dockerfile` is a list of instructions, read top to bottom, that describes how to assemble an image. Each instruction adds a layer (remember the layered images from the last lesson). Docker executes the file in order and produces a finished, reusable image at the end.

The file is named exactly `Dockerfile`, with a capital D and no extension. It lives in your project folder alongside your code.

## The core instructions

There are many instructions, but five carry most of the weight. Learn these and you can build almost anything.

### FROM — the starting point

Every image is built on top of another image, called the *base image*. `FROM` picks it. Rather than starting from an empty machine, you start from something useful, like an image that already has Python installed.

```dockerfile
# Start from an official Python 3.11 base image (slim = smaller)
FROM python:3.11-slim
```

The `slim` variant is a trimmed-down version with fewer system packages, which keeps your final image smaller. `FROM` must be the first real instruction in the file.

### WORKDIR — set the working directory

`WORKDIR` sets the directory inside the container where following commands run and where your app will live. If it does not exist, Docker creates it.

```dockerfile
# All following commands run inside /app
WORKDIR /app
```

### COPY — bring files into the image

`COPY` copies files from your project (the *build context*) into the image. The first argument is the source on your machine; the second is the destination inside the image.

```dockerfile
# Copy the requirements file from your project into /app
COPY requirements.txt .
```

The `.` means "into the current working directory," which is `/app` because of the `WORKDIR` above.

### RUN — execute a command at build time

`RUN` executes a command *while the image is being built*, and the result becomes part of the image. This is how you install libraries and system packages.

```dockerfile
# Install the Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
```

`--no-cache-dir` tells pip not to keep its download cache, which shaves size off the final image.

### CMD — what runs when the container starts

`CMD` sets the default command that runs when a container starts from this image. Unlike `RUN`, it executes at *run time*, not build time. There is usually one `CMD` per Dockerfile.

```dockerfile
# Run this when the container starts
CMD ["python", "app.py"]
```

The bracket form (a JSON array of strings) is the recommended way to write it. It avoids a shell wrapping your process, which makes stopping the container behave more predictably.

## Putting it together

Here is a complete, ordered `Dockerfile` for a small Python app. Read it top to bottom and you can narrate exactly what happens.

```dockerfile
# Start from a slim Python base image
FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Copy the dependency list first, then install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Default command when the container starts
CMD ["python", "app.py"]
```

Notice the ordering trick: we copy `requirements.txt` and install dependencies *before* copying the rest of the code. This is deliberate and it is about the layer cache. Dependencies change rarely; your code changes constantly. By installing dependencies in their own layer, Docker can reuse that cached layer every time you edit code, so rebuilds only redo the cheap final `COPY`. Flip the order and every code change would reinstall every library.

## Building the image

With the `Dockerfile` and code in a folder, build the image with `docker build`. The `-t` flag tags it with a name so you can refer to it later. The `.` at the end tells Docker to use the current directory as the build context.

```bash
# Build an image named "myapp" from the current directory
docker build -t myapp .
```

Docker prints each instruction as it runs, and you will see layers being created or pulled from cache. When it finishes, `docker images` will list your new `myapp` image.

Then run it exactly like any other image:

```bash
# Start a container from your freshly built image
docker run myapp
```

## The .dockerignore file

Just as Git has `.gitignore`, Docker has `.dockerignore`. When you `COPY . .`, you do not want to drag in giant folders like local virtual environments, caches, or datasets. Listing them in a `.dockerignore` file keeps them out of the build, making it faster and the image smaller.

```
# .dockerignore
__pycache__
*.pyc
.git
venv/
data/
```

## Key takeaways

- A `Dockerfile` is a top-to-bottom recipe for building an image; each instruction adds a layer.
- The five core instructions: `FROM` (base image), `WORKDIR` (working directory), `COPY` (bring in files), `RUN` (execute at build time), and `CMD` (default command at run time).
- Copy and install dependencies *before* copying your code, so the dependency layer stays cached across code changes.
- Build with `docker build -t name .` and run the result with `docker run name`.
- Use `.dockerignore` to keep unneeded files out of the build.

## Try it

1. Make a new folder. Inside it, create a file `app.py` containing a single line: `print("Hello from inside a container")`.
2. Create an empty `requirements.txt` (your app has no dependencies yet, but the file lets you practice the pattern).
3. Create a `Dockerfile` using the complete example above.
4. Build it: `docker build -t hello-app .`. Watch the layers get created.
5. Run it: `docker run hello-app`. You should see your message printed from inside the container.
6. Now change the message in `app.py` and rebuild. Notice that the dependency-install layer is reused from cache and the build is nearly instant.
