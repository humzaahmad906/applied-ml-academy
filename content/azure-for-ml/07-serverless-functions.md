# 07 — Serverless: Azure Functions

Not every piece of an ML system deserves an always-on server. A lot of the glue — reacting when a new file lands in the lake, transforming a message off a stream, calling a model when a webhook fires, running a nightly aggregation — is bursty, event-driven, and idle most of the time. **Azure Functions** is the serverless compute service built for exactly this: you write a small function, bind it to an event, and Azure runs it on demand, scaling from zero to many instances and back, billing only for execution. In the end-to-end solution, Functions is the reactive connective tissue that wires services together and triggers ML actions in response to events.

## The programming model: triggers and bindings

A Function is defined by one **trigger** and any number of **bindings**. The trigger is the event that causes the function to run; bindings are declarative connections to other services for input and output, so you get data in and push results out without writing SDK boilerplate.

Triggers you will use in ML:

- **HTTP trigger** — a REST endpoint. Wraps a lightweight model or a call to a hosted model behind a URL; good for low-traffic inference or webhooks.
- **Blob trigger (via Event Grid)** — fires when a blob is created or updated. The canonical "new data landed → kick off processing/scoring" hook. On modern hosting the blob trigger uses the Event Grid source for reliable, low-latency delivery.
- **Queue / Service Bus trigger** — fires per message on a queue. The backbone of decoupled, retryable async processing — enqueue scoring requests, process them one at a time with automatic retries.
- **Event Hubs trigger** — fires on batches of streaming events. For real-time feature computation or scoring on a high-throughput event stream.
- **Timer trigger** — cron-scheduled. Nightly retraining kickoffs, periodic drift checks, cache warms.

Bindings then let the same function, say, read a blob as input and write a row to Cosmos DB as output, purely by configuration.

```python
import azure.functions as func
import logging

app = func.FunctionApp()

# Fires when a blob lands under scoring/inputs/, scores it, writes to outputs/
@app.blob_trigger(arg_name="incoming", path="scoring/inputs/{name}",
                  connection="DataLakeConnection")
@app.blob_output(arg_name="result", path="scoring/outputs/{name}.json",
                 connection="DataLakeConnection")
def score_on_arrival(incoming: func.InputStream, result: func.Out[str]):
    logging.info("Scoring blob %s (%d bytes)", incoming.name, incoming.length)
    payload = incoming.read()
    prediction = run_model(payload)          # your model call
    result.set(prediction)                   # written to outputs/ by the binding
```

```python
# HTTP-triggered thin inference proxy calling a managed online endpoint
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    body = req.get_json()
    score = call_online_endpoint(body)       # forwards to the AML endpoint
    return func.HttpResponse(score, mimetype="application/json")
```

## Hosting plans: Flex Consumption and the others

Where a Function runs is set by its **hosting plan**, and the choice affects scaling, cold starts, and networking:

- **Flex Consumption** — the current recommended serverless plan. It scales per-function to zero and up quickly, supports VNet integration (so functions can reach your private endpoints), lets you choose instance memory size, and gives more deterministic scaling. This is the default to reach for in new ML solutions.
- **Consumption (classic)** — the original pay-per-execution plan; still around but the Linux variant is being superseded by Flex Consumption.
- **Premium** — pre-warmed instances (no cold start), more CPU/memory, full VNet integration; for latency-sensitive or heavier functions.
- **Dedicated (App Service) / Container Apps** — run functions on always-on infrastructure or as containers when you need custom dependencies, GPU-adjacent workloads, or predictable capacity.

Flex Consumption's **per-function scaling** is worth understanding: it groups functions by trigger type and scales each group independently — all HTTP functions scale together into their own instances, all Durable functions together, all Event-Grid-sourced blob functions together — so a flood of blob events does not starve your HTTP endpoints. New-instance allocation is rate-limited (roughly once per second for HTTP, once per ~30 seconds for non-HTTP triggers), which sets a ceiling on how fast the cold fleet grows.

```bash
# Create a storage account, then a Flex Consumption function app on Python
az functionapp create \
  --resource-group rg-mlx-dev --name func-mlx-scoring \
  --storage-account stmlxfunc \
  --flexconsumption-location eastus2 \
  --runtime python --runtime-version 3.12 \
  --instance-memory 2048 --maximum-instance-count 100 \
  --assign-identity id-mlplatform
```

The presence of `--flexconsumption-location` is what selects the Flex Consumption plan — you do not pre-create a plan object as you do for Premium or Dedicated. `--instance-memory` (512, 2048, or 4096 MB) sets the per-instance size that Flex scales out, and `--maximum-instance-count` caps the blast radius so a runaway trigger cannot scale into a surprise bill. After creation, the everyday operations are list/show/delete plus configuring app settings and identity:

```bash
az functionapp list   -g rg-mlx-dev -o table
az functionapp show   -g rg-mlx-dev -n func-mlx-scoring --query "{state:state, plan:sku}" -o table
az functionapp delete -g rg-mlx-dev -n func-mlx-scoring

# App settings are your environment variables — point the function at its downstream endpoint
az functionapp config appsettings set -g rg-mlx-dev -n func-mlx-scoring \
  --settings SCORING_ENDPOINT="https://mlw-mlx-dev.eastus2.inference.ml.azure.com/score"
az functionapp config appsettings list -g rg-mlx-dev -n func-mlx-scoring -o table

# Attach the shared managed identity so the function reaches storage / Key Vault / the ML endpoint keyless
az functionapp identity assign -g rg-mlx-dev -n func-mlx-scoring --identities id-mlplatform

# VNet integration (Flex Consumption / Premium) so the function reaches private endpoints
az functionapp vnet-integration add -g rg-mlx-dev -n func-mlx-scoring \
  --vnet vnet-mlx --subnet snet-functions
az functionapp vnet-integration list -g rg-mlx-dev -n func-mlx-scoring -o table
```

Premium and Dedicated plans, unlike Flex, are created explicitly and then a function app is bound to the plan. That extra step is how you get pre-warmed (always-ready) instances that eliminate cold starts:

```bash
# Premium plan (EP1/EP2/EP3) with two always-ready warm instances, then an app on it
az functionapp plan create -g rg-mlx-dev -n plan-mlx-premium \
  --sku EP1 --min-instances 2 --max-burst 20 --location eastus2
az functionapp create -g rg-mlx-dev -n func-mlx-lowlatency \
  --plan plan-mlx-premium --storage-account stmlxfunc \
  --runtime python --runtime-version 3.12 --assign-identity id-mlplatform
az functionapp plan list -g rg-mlx-dev -o table
```

The **local development** loop uses the Azure Functions Core Tools (`func`), separate from `az`: `func init` scaffolds a project, `func new` adds a trigger, `func start` runs it locally against the same bindings, and `func azure functionapp publish` deploys the code to the app you created with `az`:

```bash
func init func-mlx-scoring --python
func new --name score_on_arrival --template "Azure Blob Storage trigger"
func start                                            # run locally with the local.settings.json bindings
func azure functionapp publish func-mlx-scoring       # deploy code to the Azure app
```

For safe rollouts, non-Consumption plans support **deployment slots** (a staging slot you publish to, warm up, then swap into production with zero downtime) and Flex/Premium let you tune **scaling** limits after the fact. And because a Function's HTTP endpoints are protected by keys, retrieving and rotating those keys is a first-class `az` operation — put the host or function key behind API Management or a managed-identity call rather than embedding it:

```bash
# Deployment slots (Premium/Dedicated) — publish to staging, then swap
az functionapp deployment slot create -g rg-mlx-dev -n func-mlx-lowlatency --slot staging
az functionapp deployment slot swap   -g rg-mlx-dev -n func-mlx-lowlatency --slot staging --target-slot production

# Scaling controls
az functionapp scale config set -g rg-mlx-dev -n func-mlx-scoring --maximum-instance-count 40   # Flex
az functionapp plan update -g rg-mlx-dev -n plan-mlx-premium --min-instances 3                  # Premium warm floor

# Keys — list, then rotate the host key rather than shipping it in code
az functionapp keys list -g rg-mlx-dev -n func-mlx-scoring
az functionapp keys set  -g rg-mlx-dev -n func-mlx-scoring --key-type functionKeys --key-name mykey --key-value <new>
```

## Durable Functions: stateful orchestration

Plain functions are stateless and short. **Durable Functions** is an extension for stateful workflows — orchestrating a sequence of function calls, fanning out and fanning in, waiting on external events, and managing retries with checkpointed state that survives restarts. For ML this is how you express a multi-step reactive pipeline in code: "new data arrives → validate → extract features → call scoring → aggregate results → notify," with each step a function and Durable managing the flow. On Flex Consumption, Durable Functions is supported with Azure Storage or the Durable Task Scheduler as the state backend, and the Durable functions scale as their own group.

Durable is the lightweight orchestrator for **event-driven** ML glue. For scheduled, DAG-heavy data and training pipelines you will reach for Azure Machine Learning pipelines or Data Factory (later topics) — Durable Functions and those orchestrators are complementary, not competing.

## Limits and when *not* to use Functions

Functions is glue, not a training rig. Respect its boundaries:

- **Execution timeout.** Serverless plans cap how long a single execution can run (minutes, not hours on Consumption/Flex; longer on Premium/Dedicated). Long training or large batch scoring belongs on Azure ML compute, not in a function.
- **No serverless GPU.** Functions runs CPU workloads. For GPU inference, have the function *call* a GPU-backed endpoint rather than trying to run the model itself.
- **Cold starts.** On scale-from-zero plans, the first request after idle pays a startup penalty. For strict low-latency SLAs, use Premium (pre-warmed) or keep a managed online endpoint warm and put a thin function in front.
- **Package size and memory** are bounded per plan; heavyweight model weights are better loaded from blob or served by a dedicated endpoint.

The right pattern is: **Functions orchestrates and reacts; heavy compute lives elsewhere.** A blob-triggered function detects new data and enqueues a batch job; an HTTP function validates a request and forwards it to a managed online endpoint; a timer function kicks off a retraining pipeline. The function is small, fast, and cheap; the ML muscle is on the right compute.

## How Functions fits the whole solution

In the reference architecture, Functions is the event layer that keeps the system reactive without standing servers. A **blob trigger** fires when curated data lands, enqueuing work or starting a batch scoring run. An **Event Hubs trigger** computes lightweight features on streaming events in real time. An **HTTP trigger** is the thin, cheap public front door that authenticates a request and forwards it to the model endpoint (or to Azure AI Foundry for a GenAI call), adding auth, rate limiting, and logging without the model itself paying for that overhead. A **timer trigger** runs nightly drift checks and kicks off retraining. **Durable Functions** stitches multi-step reactions together. All of it authenticates with the shared managed identity and, on Flex Consumption or Premium, integrates into the VNet so it reaches private endpoints — so the reactive layer is as secure as the rest of the platform.

## Key takeaways

- **Azure Functions** is event-driven serverless compute: one **trigger** plus declarative **bindings**, scaling from zero, billed per execution — ideal ML *glue*.
- Key ML triggers: **HTTP** (thin inference/webhooks), **Blob via Event Grid** (new-data reactions), **Queue/Service Bus** (retryable async), **Event Hubs** (streaming), **Timer** (scheduled jobs).
- Prefer the **Flex Consumption** plan for new work — fast scale-to-zero, VNet integration, per-function scaling; use **Premium** to avoid cold starts.
- **Durable Functions** expresses stateful, multi-step event-driven workflows; for scheduled DAG pipelines use Azure ML pipelines or Data Factory instead.
- Functions has **execution-time, memory, and no-GPU** limits: let it *orchestrate and react*, and delegate heavy training/inference to Azure ML compute and endpoints.

## CLI cheat-sheet

```bash
# --- Local development (Azure Functions Core Tools) ---
func init func-mlx-scoring --python
func new --name score_on_arrival --template "Azure Blob Storage trigger"
func start                                          # run + debug locally
func azure functionapp publish func-mlx-scoring     # deploy code to the app

# --- Create the app (Flex Consumption = the modern default) ---
az functionapp create -g rg-mlx-dev -n func-mlx-scoring --storage-account stmlxfunc \
  --flexconsumption-location eastus2 --runtime python --runtime-version 3.12 \
  --instance-memory 2048 --maximum-instance-count 100 --assign-identity id-mlplatform

# --- Premium plan (pre-warmed, no cold start) ---
az functionapp plan create -g rg-mlx-dev -n plan-mlx-premium --sku EP1 --min-instances 2 --max-burst 20 --location eastus2
az functionapp create -g rg-mlx-dev -n func-mlx-lowlatency --plan plan-mlx-premium \
  --storage-account stmlxfunc --runtime python --runtime-version 3.12 --assign-identity id-mlplatform
az functionapp plan list   -g rg-mlx-dev -o table
az functionapp plan update -g rg-mlx-dev -n plan-mlx-premium --min-instances 3

# --- Lifecycle ---
az functionapp list   -g rg-mlx-dev -o table
az functionapp show   -g rg-mlx-dev -n func-mlx-scoring
az functionapp stop   -g rg-mlx-dev -n func-mlx-scoring
az functionapp start  -g rg-mlx-dev -n func-mlx-scoring
az functionapp delete -g rg-mlx-dev -n func-mlx-scoring

# --- Config, identity, networking ---
az functionapp config appsettings set  -g rg-mlx-dev -n func-mlx-scoring --settings SCORING_ENDPOINT="https://..."
az functionapp config appsettings list -g rg-mlx-dev -n func-mlx-scoring -o table
az functionapp identity assign -g rg-mlx-dev -n func-mlx-scoring --identities id-mlplatform
az functionapp vnet-integration add  -g rg-mlx-dev -n func-mlx-scoring --vnet vnet-mlx --subnet snet-functions
az functionapp vnet-integration list -g rg-mlx-dev -n func-mlx-scoring -o table

# --- Scaling, slots, keys ---
az functionapp scale config set -g rg-mlx-dev -n func-mlx-scoring --maximum-instance-count 40
az functionapp deployment slot create -g rg-mlx-dev -n func-mlx-lowlatency --slot staging
az functionapp deployment slot swap   -g rg-mlx-dev -n func-mlx-lowlatency --slot staging --target-slot production
az functionapp keys list -g rg-mlx-dev -n func-mlx-scoring
az functionapp keys set  -g rg-mlx-dev -n func-mlx-scoring --key-type functionKeys --key-name mykey --key-value <new>
```

## Try it

Create a Flex Consumption Function app on Python with the platform managed identity attached. Write two functions: a blob-triggered one that logs the name and size of any file dropped under `scoring/inputs/` and writes a stub result to `scoring/outputs/`, and an HTTP-triggered one that accepts a JSON body and returns a canned prediction. Upload a test blob and confirm the trigger fires end to end. Then reason about which parts of your eventual ML system should be Functions (reactive glue) versus Azure ML compute (heavy lifting) — and write that boundary down.
