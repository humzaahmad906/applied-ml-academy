# 06 — Packaging an ML Model

This is where everything comes together. You have learned images, containers, Dockerfiles, run flags, and Compose. Now you will use all of it to do the thing this course is really about: take a trained ML model and wrap it in a container that serves predictions over an API. The result is a portable artifact you can run anywhere.

## What "packaging a model" means

A trained model is usually a file of learned weights sitting on disk. On its own it does nothing useful for other people or systems. To make it usable, you wrap it in an *inference service*: a small program that loads the model, exposes an endpoint, accepts input, runs a prediction, and returns the result.

Packaging means putting that service, its dependencies, and the model file into one container. Once packaged, anyone can run your model with a single command, on any machine, without installing Python or your libraries or matching your environment. The container carries all of it.

## The inference service

We will build a tiny web service that serves predictions. The framework here is FastAPI, a lightweight Python web framework, but the pattern is identical regardless of what you use. Here is `app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the model once, when the service starts
model = joblib.load("model.joblib")

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
```

Two things are worth calling out. First, the model is loaded *once* at startup, not on every request. Loading is slow, so you do it a single time and reuse the loaded model for every prediction. Second, there is a `/health` endpoint that just returns "ok." This is a convention: deployment systems ping it to confirm the service is alive before sending real traffic.

The dependencies go in `requirements.txt`:

```
fastapi
uvicorn
scikit-learn
joblib
```

`uvicorn` is the server that actually runs the FastAPI app.

## The Dockerfile

Now wrap it. This `Dockerfile` follows the pattern from earlier lessons, with the dependency-caching ordering trick intact:

```dockerfile
# Start from a slim Python base
FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Install dependencies first (cached across code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model file
COPY app.py .
COPY model.joblib .

# Document the port the service listens on
EXPOSE 8000

# Start the server, listening on all interfaces so it is reachable
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

A couple of details specific to serving a model:

- `COPY model.joblib .` bakes the trained model into the image. For a small model this is the simplest approach: the image is fully self-contained. For large models (hundreds of megabytes or gigabytes), you would instead mount the model as a volume at run time to keep the image lean, using the `-v` flag from the ports-and-volumes lesson.
- `EXPOSE 8000` documents which port the service uses. It is informational and does not publish the port; you still map it with `-p` when you run.
- `--host 0.0.0.0` is critical and catches everyone once. By default a server binds to `localhost` *inside the container*, which is unreachable from outside. Binding to `0.0.0.0` makes it listen on all interfaces so your `-p` mapping can reach it.

## Build and run

Build the image, then run it with the port published so you can reach the API:

```bash
# Build the inference image
docker build -t model-server .

# Run it, mapping container port 8000 to host port 8000
docker run -d --name server -p 8000:8000 model-server
```

Now the service is live on `localhost:8000`. Confirm it is healthy, then send it a prediction request:

```bash
# Check the health endpoint
curl localhost:8000/health

# Send features and get a prediction back
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

The first returns `{"status": "ok"}`. The second sends a list of input features and gets back the model's prediction as JSON. You now have a running model served from a container.

## Keeping the image lean

ML images have a habit of ballooning to multiple gigabytes because ML libraries are heavy. A few habits keep them manageable:

- Start from a `slim` base image, not the full one.
- Use `pip install --no-cache-dir` so pip's download cache is not stored in the image.
- Add a `.dockerignore` that excludes datasets, notebooks, virtual environments, and caches so they never enter the build.
- For genuinely large models, mount them as a volume instead of copying them in, so the image stays small and the same image can serve different model versions.

Smaller images build faster, push and pull faster, and start faster. In production, where images are pulled onto many machines, this adds up.

## Key takeaways

- Packaging a model means wrapping an inference service, its dependencies, and the model into one portable container.
- Load the model once at startup, not per request, and expose a `/health` endpoint by convention.
- Bake small models into the image with `COPY`; mount large models as volumes to keep the image lean.
- Bind the server to `0.0.0.0`, not `localhost`, or your published port will not reach it.
- Keep ML images small with slim bases, no pip cache, and a `.dockerignore`.

## Try it

1. Train or grab any tiny scikit-learn model (an iris classifier is perfect) and save it with `joblib.dump(model, "model.joblib")`.
2. Create the `app.py`, `requirements.txt`, and `Dockerfile` shown above in the same folder as your saved model.
3. Build the image: `docker build -t model-server .`.
4. Run it: `docker run -d --name server -p 8000:8000 model-server`.
5. Hit the health check with `curl localhost:8000/health`, then send a real prediction with the POST request above. Confirm you get a prediction back.
6. Bonus: check your image size with `docker images`. Then add a `.dockerignore` and rebuild, and see whether you can shrink it.
