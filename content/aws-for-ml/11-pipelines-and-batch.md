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

## Managed Airflow (MWAA)

**Amazon Managed Workflows for Apache Airflow** runs Airflow without operating the servers. Choose MWAA when your team already lives in Airflow, needs its rich operator ecosystem, or orchestrates a mixed data-and-ML platform where ML is one of many DAGs. It coexists with SageMaker via operators that trigger training and deployment.

## CI/CD for the pipeline itself

The pipeline definition is code, so it belongs in a CI/CD flow. **AWS CodePipeline / CodeBuild**, or **GitHub Actions**, build the training container, run tests, and deploy the pipeline definition; **EventBridge** schedules retraining or triggers it on new data landing in S3. This is what makes the whole thing hands-off: a commit updates the pipeline, and a schedule or data event runs it.

## How this fits the whole ML solution

Orchestration is the automation layer that turns the individual services — data, training, registry, endpoints, monitoring — into a self-running system. SageMaker Pipelines chains the ML steps and gates quality; the registry gates what deploys; Model Monitor detects drift and, via EventBridge, triggers a retrain; Step Functions or MWAA stitches ML into the wider data and business flow; CI/CD ships changes to the pipeline itself. Without this layer you have a pile of capable services; with it you have a production ML system that retrains, re-evaluates, and redeploys on its own.

## Key takeaways

- Orchestration makes ML workflows declarative, repeatable, and gated — the core of MLOps.
- SageMaker Pipelines is the native ML workflow tool with built-in lineage and condition steps for quality gates.
- Model Registry versions and approves models (the train→deploy hand-off); Model Monitor detects drift; Clarify measures bias and explains predictions.
- Batch transform handles bulk offline scoring as a pipeline step.
- Step Functions orchestrates across all AWS services; MWAA suits Airflow-centric teams; CI/CD plus EventBridge automate and schedule the whole loop.

## Try it

Build a SageMaker Pipeline with train → evaluate → condition → register steps, where the model is only registered if validation accuracy clears a threshold. Run it and confirm a below-threshold model is *not* registered. Then wrap that pipeline in a Step Functions state machine that first runs a Glue job to prepare data, and schedule the state machine nightly with EventBridge. Finally, enable Model Monitor on a deployed endpoint, feed it drifted data, and confirm it raises a CloudWatch alarm — the trigger that would start the retraining loop.
