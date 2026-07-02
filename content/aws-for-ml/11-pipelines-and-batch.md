# 11 — Pipelines, Batch, and Orchestration

A model you trained by hand in a notebook is a liability: no one can reproduce it, retrain it on fresh data, or trust that what is in production matches what was evaluated. MLOps turns those manual steps into automated, versioned, gated pipelines. This module covers the orchestration options on AWS — SageMaker Pipelines, Step Functions, and managed Airflow — plus batch transform and the registry-and-monitoring machinery that make a pipeline trustworthy.

## Why orchestration matters

An ML workflow is a directed graph of steps: pull data, validate it, engineer features, train, evaluate, register the model if it beats a threshold, then deploy. Orchestration makes that graph declarative and repeatable — it runs on a schedule or a trigger, passes artifacts between steps, retries failures, and records what happened. The payoff is reproducibility and safe automation: retraining on Monday's data produces a traceable model that only reaches production if it passes the same gates every time.

## SageMaker Pipelines

**SageMaker Pipelines** is the native ML workflow tool, purpose-built for these graphs. You define steps — processing, training, evaluation, conditional, register-model — in Python, and Pipelines executes them with artifact lineage tracked automatically. A **condition step** encodes the quality gate: only register and deploy if the evaluation metric clears a bar.

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

train = TrainingStep(name="Train", estimator=estimator, inputs=train_inputs)
evaluate = ProcessingStep(name="Evaluate", processor=eval_processor, ...)
gate = ConditionStep(
    name="AccuracyGate",
    conditions=[ConditionGreaterThanOrEqualTo(left=accuracy, right=0.90)],
    if_steps=[register_step],   # only register if accuracy >= 0.90
    else_steps=[],
)
pipeline = Pipeline(name="fraud-retrain", steps=[train, evaluate, gate])
pipeline.upsert(role_arn=role)
pipeline.start()
```

Because Pipelines understands SageMaker's own concepts (estimators, processors, model artifacts, lineage), it is the lowest-friction choice when the whole workflow is ML training and deployment.

You author the graph in the Python SDK, but everything after `upsert` is operable from the CLI — which is what CI/CD and on-call runbooks actually use. `upsert` compiles the DAG to a JSON definition and registers it; `start-pipeline-execution` kicks off a run and can override any pipeline parameter (data date, instance type, threshold) without touching code, so the same definition retrains on Monday's data by passing a different `--pipeline-parameters`. Watch a run with `describe-pipeline-execution` and `list-pipeline-execution-steps` — the latter is the fastest way to see which step failed and why.

```bash
# List registered pipelines and start a parameterized run
aws sagemaker list-pipelines
aws sagemaker start-pipeline-execution \
  --pipeline-name fraud-retrain \
  --pipeline-parameters Name=DataDate,Value=2026-07-01 Name=AccuracyThreshold,Value=0.90

# Follow the execution and drill into the step that failed
aws sagemaker describe-pipeline-execution \
  --pipeline-execution-arn <exec-arn>
aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn <exec-arn>

# Stop a runaway execution
aws sagemaker stop-pipeline-execution --pipeline-execution-arn <exec-arn>
```

A common gotcha: `start-pipeline-execution` fails if a parameter name is not declared on the pipeline, and passing a wrong type silently coerces — declare parameters with explicit `ParameterInteger`/`ParameterFloat` types in the SDK so the gate compares numbers, not strings. When registering the winning model, the raw `update-model-package --model-approval-status Approved` flips a version from `PendingManualApproval` to `Approved`, which is exactly what a manual approval gate or an approver Lambda calls — the registry and Model Monitor get their own module, so we keep them at this level here.

```bash
# Create the model group once, then register + approve versions (registry has its own module)
aws sagemaker create-model-package-group --model-package-group-name fraud-models
aws sagemaker list-model-packages --model-package-group-name fraud-models
aws sagemaker update-model-package \
  --model-package-arn <version-arn> \
  --model-approval-status Approved
```

## Model Registry, Model Monitor, and Clarify

Three SageMaker components make the pipeline trustworthy:

- **Model Registry** is the versioned catalog of trained models with approval status. The pipeline registers a candidate as `PendingManualApproval` or `Approved`; deployment reads from the registry, so what ships is always a tracked, approved version. This is the hand-off point between training and inference.
- **Model Monitor** watches a live endpoint for **data quality** and **model quality** drift, comparing production traffic against a baseline captured at training time and raising alerts (via CloudWatch) when inputs or predictions drift. Drift is the signal to retrain.
- **Clarify** measures bias in data and models and explains predictions (feature attributions), which belongs in the evaluation step and in ongoing monitoring.

Together these close the loop: train → gate on quality and bias → register → deploy approved version → monitor for drift → trigger retraining.

## Batch transform in the pipeline

For workflows that score data in bulk rather than serve a live endpoint, **batch transform** is a first-class pipeline step: it runs the model over an S3 dataset and writes predictions back to S3, then shuts the instances down. Nightly scoring of an entire customer base, backfilling predictions after a model update, or generating features for the next model all fit here — and slot directly into a scheduled pipeline.

## Step Functions for broader orchestration

**AWS Step Functions** is a general-purpose workflow engine (state machines) that orchestrates *any* AWS services, not just SageMaker. It has direct integrations to start training jobs, batch transforms, and endpoints, plus Lambda, Glue, ECS tasks, and more. Choose Step Functions when your workflow spans beyond ML — for example, a pipeline that runs a Glue ETL job, calls a Lambda for validation, kicks off SageMaker training, and then updates a DynamoDB table and notifies a queue. **Standard** workflows suit long-running orchestration; **Express** workflows suit high-volume, short-duration event processing.

```json
{
  "StartAt": "GlueETL",
  "States": {
    "GlueETL":   {"Type": "Task", "Resource": "arn:aws:states:::glue:startJobRun.sync",
                  "Parameters": {"JobName": "curate-features"}, "Next": "Train"},
    "Train":     {"Type": "Task", "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
                  "Parameters": {"TrainingJobName.$": "$.jobName", "...": "..."}, "Next": "Done"},
    "Done":      {"Type": "Succeed"}
  }
}
```

A common division of labor: SageMaker Pipelines owns the ML-internal graph, and Step Functions wraps it into the larger business workflow.

Operationally, the state machine is a resource you create once and then execute repeatedly. `create-state-machine` registers the JSON definition and the execution role; `start-execution` runs it with an `--input` JSON blob that becomes the initial state (`$`), so you pass the run's data date or job name here; `describe-execution` returns status and the final output; `list-executions --status-filter FAILED` is the on-call's first query after a page. The `.sync` suffix on a `Resource` ARN (as in `sagemaker:createTrainingJob.sync`) tells Step Functions to block until the job finishes and surface its result — omit it and the state completes the instant the job is *submitted*, which is a classic race that "succeeds" while training is still running.

```bash
# Register the state machine, then run it with per-execution input
aws stepfunctions create-state-machine \
  --name ml-retrain --definition file://statemachine.json \
  --role-arn arn:aws:iam::<acct>:role/StepFunctionsMLRole
aws stepfunctions start-execution \
  --state-machine-arn <sm-arn> \
  --input '{"jobName":"fraud-2026-07-01","dataDate":"2026-07-01"}'

# Triage: what ran, what failed
aws stepfunctions describe-execution --execution-arn <exec-arn>
aws stepfunctions list-executions --state-machine-arn <sm-arn> --status-filter FAILED
```

## Managed Airflow (MWAA)

**Amazon Managed Workflows for Apache Airflow** runs Airflow without operating the servers. Choose MWAA when your team already lives in Airflow, needs its rich operator ecosystem, or orchestrates a mixed data-and-ML platform where ML is one of many DAGs. It coexists with SageMaker via operators that trigger training and deployment.

The environment is heavyweight infrastructure (a sized Airflow cluster in your VPC), so `create-environment` takes an instance class, a DAGs-S3 bucket, and networking — you provision it once, then deploy DAGs by syncing Python files to that bucket. Because the Airflow web UI sits behind IAM, scripting a DAG trigger from CI means minting a short-lived token with `create-cli-token` and calling the Airflow REST/CLI with it — that token, not your AWS creds, is what authenticates the DAG operation.

```bash
# Provision the environment once (sized, VPC-attached, points at a DAGs bucket)
aws mwaa create-environment \
  --name ml-airflow --airflow-version 2.10.1 \
  --environment-class mw1.medium \
  --dag-s3-path dags \
  --source-bucket-arn arn:aws:s3:::my-mwaa-bucket \
  --execution-role-arn arn:aws:iam::<acct>:role/MwaaExecutionRole \
  --network-configuration SubnetIds=subnet-a,subnet-b,SecurityGroupIds=sg-123

# Deploy a DAG = sync it to the bucket; trigger it via a short-lived CLI token
aws s3 cp retrain_dag.py s3://my-mwaa-bucket/dags/
aws mwaa create-cli-token --name ml-airflow
```

## CI/CD for the pipeline itself

The pipeline definition is code, so it belongs in a CI/CD flow. **AWS CodePipeline / CodeBuild**, or **GitHub Actions**, build the training container, run tests, and deploy the pipeline definition; **EventBridge** schedules retraining or triggers it on new data landing in S3. This is what makes the whole thing hands-off: a commit updates the pipeline, and a schedule or data event runs it.

CodePipeline defines the stages (source → build → deploy) from a JSON document and can be kicked manually with `start-pipeline-execution`; CodeBuild runs the actual build via `start-build`, overriding env vars per run so the same project builds a dev or prod image. For the trigger side there are two distinct EventBridge tools that people conflate: classic **EventBridge rules** (`put-rule` + `put-targets`) fire a target on an *event* — the canonical one being an S3 `Object Created` event that starts a pipeline the moment fresh data lands — while **EventBridge Scheduler** (`create-schedule`) is the modern, higher-limit replacement for cron-style *time-based* triggers, with a `--flexible-time-window` so thousands of schedules don't all stampede at midnight. A drift alarm from Model Monitor lands as an event on the same bus, closing the retraining loop without any human in the path.

```bash
# CI/CD: define/run the deploy pipeline and its build
aws codepipeline create-pipeline --cli-input-json file://pipeline.json
aws codepipeline start-pipeline-execution --name ml-deploy-pipeline
aws codebuild start-build --project-name train-image \
  --environment-variables-override name=IMAGE_TAG,value=prod

# Event trigger: start retraining when new data lands in S3
aws events put-rule --name new-training-data \
  --event-pattern '{"source":["aws.s3"],"detail-type":["Object Created"],"detail":{"bucket":{"name":["ml-landing"]}}}'
aws events put-targets --rule new-training-data \
  --targets 'Id=1,Arn=<state-machine-arn>,RoleArn=<events-role-arn>'

# Time trigger: nightly retrain via EventBridge Scheduler (cron, with a flex window)
aws scheduler create-schedule --name nightly-retrain \
  --schedule-expression 'cron(0 3 * * ? *)' \
  --flexible-time-window '{"Mode":"FLEXIBLE","MaximumWindowInMinutes":15}' \
  --target '{"Arn":"<state-machine-arn>","RoleArn":"<scheduler-role-arn>"}'
```

## How this fits the whole ML solution

Orchestration is the automation layer that turns the individual services — data, training, registry, endpoints, monitoring — into a self-running system. SageMaker Pipelines chains the ML steps and gates quality; the registry gates what deploys; Model Monitor detects drift and, via EventBridge, triggers a retrain; Step Functions or MWAA stitches ML into the wider data and business flow; CI/CD ships changes to the pipeline itself. Without this layer you have a pile of capable services; with it you have a production ML system that retrains, re-evaluates, and redeploys on its own.

## Key takeaways

- Orchestration makes ML workflows declarative, repeatable, and gated — the core of MLOps.
- SageMaker Pipelines is the native ML workflow tool with built-in lineage and condition steps for quality gates.
- Model Registry versions and approves models (the train→deploy hand-off); Model Monitor detects drift; Clarify measures bias and explains predictions.
- Batch transform handles bulk offline scoring as a pipeline step.
- Step Functions orchestrates across all AWS services; MWAA suits Airflow-centric teams; CI/CD plus EventBridge automate and schedule the whole loop.

## CLI cheat-sheet

```bash
# ── SageMaker Pipelines (authored in Python SDK; run/observe from CLI) ──
aws sagemaker list-pipelines
aws sagemaker start-pipeline-execution --pipeline-name NAME \
  --pipeline-parameters Name=DataDate,Value=2026-07-01
aws sagemaker describe-pipeline-execution --pipeline-execution-arn ARN
aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn ARN
aws sagemaker stop-pipeline-execution --pipeline-execution-arn ARN

# ── Model Registry (raw API; has its own module for depth) ──
aws sagemaker create-model-package-group --model-package-group-name GROUP
aws sagemaker create-model-package --model-package-group-name GROUP \
  --inference-specification file://inference-spec.json
aws sagemaker list-model-packages --model-package-group-name GROUP
aws sagemaker update-model-package --model-package-arn ARN \
  --model-approval-status Approved        # PendingManualApproval -> Approved

# ── Batch transform (bulk offline scoring step) ──
aws sagemaker create-transform-job --transform-job-name JOB \
  --model-name MODEL \
  --transform-input '{"DataSource":{"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":"s3://in/"}}}' \
  --transform-output '{"S3OutputPath":"s3://out/"}' \
  --transform-resources '{"InstanceType":"ml.m5.xlarge","InstanceCount":2}'
aws sagemaker describe-transform-job --transform-job-name JOB

# ── Step Functions (cross-service orchestration; use .sync ARNs to block) ──
aws stepfunctions create-state-machine --name SM \
  --definition file://statemachine.json --role-arn ROLE_ARN
aws stepfunctions start-execution --state-machine-arn ARN \
  --input '{"jobName":"run-2026-07-01"}'
aws stepfunctions describe-execution --execution-arn ARN
aws stepfunctions list-executions --state-machine-arn ARN --status-filter FAILED

# ── MWAA (managed Airflow; DAGs = files in the S3 bucket) ──
aws mwaa create-environment --name ENV --airflow-version 2.10.1 \
  --environment-class mw1.medium --dag-s3-path dags \
  --source-bucket-arn arn:aws:s3:::BUCKET \
  --execution-role-arn ROLE_ARN \
  --network-configuration SubnetIds=subnet-a,subnet-b,SecurityGroupIds=sg-1
aws s3 cp dag.py s3://BUCKET/dags/
aws mwaa create-cli-token --name ENV         # short-lived token for the Airflow API

# ── EventBridge: event-driven rules vs. time-based Scheduler ──
aws events put-rule --name new-data \
  --event-pattern '{"source":["aws.s3"],"detail-type":["Object Created"]}'
aws events put-targets --rule new-data \
  --targets 'Id=1,Arn=TARGET_ARN,RoleArn=EVENTS_ROLE_ARN'
aws scheduler create-schedule --name nightly \
  --schedule-expression 'cron(0 3 * * ? *)' \
  --flexible-time-window '{"Mode":"FLEXIBLE","MaximumWindowInMinutes":15}' \
  --target '{"Arn":"TARGET_ARN","RoleArn":"SCHEDULER_ROLE_ARN"}'

# ── CI/CD for the pipeline itself ──
aws codepipeline create-pipeline --cli-input-json file://pipeline.json
aws codepipeline start-pipeline-execution --name PIPELINE
aws codebuild start-build --project-name PROJECT \
  --environment-variables-override name=IMAGE_TAG,value=prod
```

## Try it

Build a SageMaker Pipeline with train → evaluate → condition → register steps, where the model is only registered if validation accuracy clears a threshold. Run it and confirm a below-threshold model is *not* registered. Then wrap that pipeline in a Step Functions state machine that first runs a Glue job to prepare data, and schedule the state machine nightly with EventBridge. Finally, enable Model Monitor on a deployed endpoint, feed it drifted data, and confirm it raises a CloudWatch alarm — the trigger that would start the retraining loop.
