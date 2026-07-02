# 17 — Azure ML Feature Store, Registry, and Data Assets

The training module produced models; the deployment module served them. But two of the hardest problems in production ML sit *between* raw data and a trained model: how do you define a feature once and use the *identical* computation in both training and serving (so you never get train/serve skew), and how do you version data and models so any prediction is traceable back to exactly what produced it. Azure ML answers these with three managed asset systems: **datastores and data assets** (versioned, governed pointers to data), the **managed feature store** (features defined once, materialized to offline and online stores), and the **model registry** (versioned models with lineage and dev→prod promotion). This module goes deep on all three using the `az ml` (v2) CLI and the `azure-ai-ml` SDK. In the end-to-end solution, these are the asset-management backbone that makes the ML lifecycle reproducible and auditable rather than a pile of one-off files.

## Datastores and data assets: governed, versioned data

A **datastore** is a named, credential-managed reference to a storage location (a Blob container, an ADLS Gen2 filesystem) registered in the workspace, so jobs address data by a short `azureml://` URI instead of an account URL plus secret. The workspace ships with default datastores; you add your own for the medallion containers from the storage module. Prefer **identity-based** access (the workspace/compute managed identity) over stored credentials so no key lives in the datastore definition.

```bash
# Register the curated container as an identity-based datastore (no key stored)
az ml datastore create --file datastore-gold.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml datastore list -g rg-mlx-dev -w mlw-mlx-dev -o table
```

A **data asset** is a *versioned* pointer to data within a datastore — a `uri_file`, `uri_folder`, or `mltable`. Versioning is the point: a training job records the exact data-asset version it consumed, so you can always answer "what data produced this model." Registering a new version never mutates the old one, so historical runs stay reproducible.

```bash
# Version a curated training folder as a data asset
az ml data create --name fraud-train --version 3 --type uri_folder \
  --path azureml://datastores/gold/paths/datasets/fraud/train/ \
  -g rg-mlx-dev -w mlw-mlx-dev
az ml data show --name fraud-train --version 3 -g rg-mlx-dev -w mlw-mlx-dev
az ml data list --name fraud-train -g rg-mlx-dev -w mlw-mlx-dev -o table   # all versions
```

An **MLTable** asset adds a materialization spec (schema, column types, transformations, delimiter/format), so tabular data loads consistently regardless of who reads it — the recommended type for structured training data because it removes "it parsed differently on my machine" from the equation.

## The managed feature store: define once, use everywhere

The classic production-ML bug is **train/serve skew**: the feature pipeline that built the training set and the code that computes features at request time drift apart, so the model sees subtly different inputs in production than it learned on, and quality silently degrades. The **Azure ML managed feature store** fixes this structurally by making the feature *definition* the single source of truth for both paths.

Its top-level entities:

- **Feature store** — a specialized workspace that hosts feature assets. Creating one provisions its own storage, container registry, Key Vault, and Application Insights (like a regular workspace).
- **Entity** — the business object features attach to (a `user`, an `account`, a `transaction`), with an index/join key.
- **Feature set** — a versioned definition that computes a group of related features from a source, expressed as a transformation (a Spark/SQL spec) plus **materialization settings** for where computed values are stored.

```bash
# Create the feature store, then register an entity and a feature set from YAML specs
az ml feature-store create --file feature-store.yml -g rg-mlx-dev
az ml feature-store-entity create --file entity-user.yml \
  --feature-store-name fs-mlx -g rg-mlx-dev
az ml feature-set create --file featureset-user-txn.yml \
  --feature-store-name fs-mlx -g rg-mlx-dev
az ml feature-set list --feature-store-name fs-mlx -g rg-mlx-dev -o table
```

**Materialization** is what makes one definition serve both worlds. The feature store computes the feature values and writes them to:

- an **offline store** (ADLS Gen2, in Delta) — the historical feature values used to build **point-in-time-correct** training sets, so a training row only ever sees feature values that existed *before* its label timestamp (this is how the feature store prevents the other classic bug, **label leakage** from future data).
- an **online store** (a low-latency cache, e.g. Azure Cache for Redis / Cosmos-style store) — the latest feature values, read in single-digit milliseconds during a scoring request.

Because both stores are populated from the *same* feature-set definition, the value a model trains on and the value it scores on are computed identically — skew eliminated by construction. You trigger materialization as a backfill or on a recurring schedule:

```bash
# Backfill the offline store, and enable scheduled online+offline materialization
az ml feature-set backfill --name user-txn --version 1 --feature-store-name fs-mlx -g rg-mlx-dev \
  --feature-window-start-time "2026-01-01T00:00:00" --feature-window-end-time "2026-07-01T00:00:00"
```

At training time you build a dataset by joining feature sets to an observation (label) DataFrame with `get_offline_features`, which enforces point-in-time correctness; at serving time you fetch the same features from the online store with `get_online_features`. The training module's `command` job then references the resulting feature-retrieval spec as a data input, and the exact feature-set versions are recorded in lineage.

## The model registry: versioned models and lineage

A trained model becomes a first-class, **versioned** asset in the **model registry**, with lineage back to the job, environment, and data-asset versions that produced it — the handoff from training to deployment.

```bash
# Register a model from a completed job's output (auto-increments the version)
az ml model create --name fraud-detector --version 1 --type mlflow_model \
  --path azureml://jobs/$JOB_NAME/outputs/model \
  -g rg-mlx-dev -w mlw-mlx-dev
az ml model list --name fraud-detector -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml model show --name fraud-detector --version 1 -g rg-mlx-dev -w mlw-mlx-dev -o jsonc
```

Model **types** matter for deployment: `mlflow_model` carries its signature and dependencies so an online endpoint auto-generates the scoring wrapper; `custom_model` and `triton_model` are the alternatives when you provide your own scorer or serve on Triton. Tag models with stage/candidate metadata to track which version is production.

The key distinction is **workspace registry** versus **shared registry**. A workspace registry is local to one workspace; a **registry** (a separate resource) is shared across workspaces and regions, so you can register a model in a dev workspace and *promote the exact artifact* into a prod workspace — the backbone of promotion-based MLOps. The same registry can hold shared **components** and **environments**, so a whole team reuses one vetted training component and one curated environment.

```bash
# Create a shared registry, then promote a model artifact dev -> prod by copying to it
az ml registry create --file registry.yml -g rg-mlx-dev
az ml model create --name fraud-detector --version 1 --type mlflow_model \
  --path azureml://jobs/$JOB_NAME/outputs/model --registry-name reg-mlx-shared
# a prod workspace then references registry:reg-mlx-shared/models/fraud-detector:1
az ml model share --name fraud-detector --version 1 -w mlw-mlx-dev -g rg-mlx-dev \
  --registry-name reg-mlx-shared --share-with-name fraud-detector --share-with-version 1
```

## How the asset systems fit the whole solution

These three systems are the reproducibility spine. Curated **gold** data in the lake is exposed as identity-based **datastores** and pinned as versioned **data assets** / **MLTables**. The **feature store** defines features once and materializes them to an **offline** store (point-in-time-correct training sets) and an **online** store (millisecond serving lookups) — eliminating train/serve skew and label leakage by construction, and superseding the hand-rolled "Cosmos as online store" pattern for teams that adopt it. A training **pipeline** consumes specific feature-set and data-asset versions, and registers its output to the **model registry** with full lineage. A **shared registry** promotes the winning model artifact from the dev to the prod workspace, where the deployment module serves it. Because every input (data version, feature-set version, environment) and every output (model version) is tracked, any prediction traces back to exactly the data and code that produced it — the property that makes the system auditable, not folklore.

## Key takeaways

- **Datastores** are credential-managed (prefer identity-based) references to storage; **data assets** are *versioned* pointers (`uri_file`/`uri_folder`/`mltable`) so a job records exactly which data version it used — reproducibility by version.
- The **managed feature store** defines a feature *once* and materializes it to an **offline** store (point-in-time-correct training sets, preventing label leakage) and an **online** store (ms serving lookups), eliminating **train/serve skew** by construction.
- Feature-store entities: **feature store** (a specialized workspace) → **entity** (join key) → **feature set** (versioned definition + materialization); trigger with **backfill** and scheduled materialization.
- The **model registry** versions models with **lineage** to job/data/environment; model **type** (`mlflow_model` vs custom/triton) drives how it deploys.
- A **shared registry** (distinct from the workspace registry) promotes the exact model artifact **dev → prod** across workspaces/regions — the backbone of promotion-based MLOps.

## CLI cheat-sheet

```bash
# --- datastores ---
az ml datastore create --file datastore-gold.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml datastore list   -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml datastore show   --name gold  -g rg-mlx-dev -w mlw-mlx-dev

# --- data assets (versioned) ---
az ml data create --name fraud-train --version 3 --type uri_folder \
  --path azureml://datastores/gold/paths/datasets/fraud/train/ -g rg-mlx-dev -w mlw-mlx-dev
az ml data list  --name fraud-train -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml data show  --name fraud-train --version 3 -g rg-mlx-dev -w mlw-mlx-dev
az ml data archive --name fraud-train --version 2 -g rg-mlx-dev -w mlw-mlx-dev   # soft-hide old version

# --- feature store ---
az ml feature-store create --file feature-store.yml -g rg-mlx-dev
az ml feature-store list   -g rg-mlx-dev -o table
az ml feature-store-entity create --file entity-user.yml --feature-store-name fs-mlx -g rg-mlx-dev
az ml feature-set create   --file featureset-user-txn.yml --feature-store-name fs-mlx -g rg-mlx-dev
az ml feature-set list     --feature-store-name fs-mlx -g rg-mlx-dev -o table
az ml feature-set show     --name user-txn --version 1 --feature-store-name fs-mlx -g rg-mlx-dev
az ml feature-set backfill --name user-txn --version 1 --feature-store-name fs-mlx -g rg-mlx-dev \
  --feature-window-start-time "2026-01-01T00:00:00" --feature-window-end-time "2026-07-01T00:00:00"

# --- model registry (workspace) ---
az ml model create --name fraud-detector --version 1 --type mlflow_model \
  --path azureml://jobs/$JOB_NAME/outputs/model -g rg-mlx-dev -w mlw-mlx-dev
az ml model list --name fraud-detector -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml model show --name fraud-detector --version 1 -g rg-mlx-dev -w mlw-mlx-dev -o jsonc
az ml model archive --name fraud-detector --version 1 -g rg-mlx-dev -w mlw-mlx-dev

# --- shared registry (dev -> prod promotion) ---
az ml registry create --file registry.yml -g rg-mlx-dev
az ml model create --name fraud-detector --version 1 --type mlflow_model \
  --path azureml://jobs/$JOB_NAME/outputs/model --registry-name reg-mlx-shared
az ml model share --name fraud-detector --version 1 -w mlw-mlx-dev -g rg-mlx-dev \
  --registry-name reg-mlx-shared --share-with-name fraud-detector --share-with-version 1
az ml environment create --file env.yml --registry-name reg-mlx-shared   # share envs too
az ml component  create --file train-component.yml --registry-name reg-mlx-shared
```

## Try it

Register the curated container as an identity-based datastore, then create a versioned `uri_folder` data asset over a training folder and confirm registering a new version leaves the old one intact. Stand up a feature store, define a `user` entity and a `user-txn` feature set, and backfill its offline store over a date window. Register a model from a prior job's output into the workspace registry, then create a shared registry and promote that exact artifact into it — and confirm a (second) workspace can reference `registry:reg-mlx-shared/models/fraud-detector:1`. Finally, trace one registered model back through its lineage to the feature-set versions and data-asset versions that produced it — that trace is what "reproducible ML" actually means.
