# Module 14 — Domain Variations: Healthcare, Finance, Autonomous Systems & Manufacturing

## Why this module matters

The core patterns you have built through this course — multi-stage funnels, eval-driven development, feedback loops, training-serving skew monitoring, cascade architectures — are domain-invariant. The same skeleton that runs a recommender runs a radiology triage system. What changes is the physics: the regulatory overhead, the acceptable failure modes, the latency budget, the cost of being wrong. In an interview, the question "design a system for healthcare" or "design this for low-latency trading" is not testing a different skill — it is testing whether you know *which levers bend* and by how much. This module is a concise map of those bends.

A note on cross-references: each section below points back to the core chapters where the invariant technique lives. This module covers only the domain-specific deltas.

---

## 1. Healthcare

### What changes

**Data types.** Medical imaging is DICOM — a standard that bundles pixel data with dozens of required metadata tags (patient ID, modality, institution, series UIDs). The pixel data is 12–16-bit grayscale for CT/MRI (not the 8-bit RGB most CV pipelines expect), and studies are volumetric: a chest CT is 300–500 axial slices, not a single image. Clinical text is HL7/FHIR-structured or free-text EHR notes — dense with domain jargon, abbreviations, and implicit context ("pt s/p CABG x3" is not NLP-friendly out of the box). Structured EHR data (labs, vitals, medications, diagnoses) is available in bulk but riddled with missing-at-random patterns (a lab wasn't ordered, not because it was normal — because the patient wasn't seen).

**Regulation.** HIPAA governs data handling: any individually identifiable health information is PHI, carries strict access controls, audit logging, and breach-notification requirements. AI/ML that influences clinical decisions is additionally governed as Software as a Medical Device (SaMD) — FDA De Novo or 510(k) clearance in the US, EU MDR in Europe. These pathways require prospective clinical validation, locked algorithm specifications (an "algorithm change protocol" must be submitted and approved before you retrain on new data), and documentation of intended use, indications, and contraindications. This fundamentally changes your iteration speed: the eval-driven development loop from the evaluation chapter does not disappear, but each loop that crosses a regulatory decision boundary is measured in months, not days.

**In-hospital SLAs.** A chest X-ray AI that flags pneumothorax runs in a radiologist's worklist queue — reads should complete in under an hour; critical findings ("stat" pathology) may need surfacing within minutes. An ICU sepsis-prediction model fires continuously; acting on a prediction takes seconds. These are not web-scale latency budgets, but they are hard: an alert that fires ten minutes after it should — or that fires at all during a system outage — has patient-safety implications.

**Failure modes.** The cost asymmetry flips by task. Screening (e.g., cancer detection) tolerates more false positives — a recall is cheap; a missed cancer is not. Alarm systems in the ICU face the opposite problem: alert fatigue from too many false positives means clinicians stop responding — so a false positive is not free. You must state the cost matrix explicitly before choosing a threshold, and this is a senior signal the interviewer is listening for.

### What stays the same

The multi-stage funnel from the recommendations chapter applies directly: a radiology worklist triager runs candidate generation (all new studies arriving in the last hour), a fast classifier scores urgency, and a post-processing layer applies business rules (suppress duplicates, suppress modalities outside the model's indications). Eval-driven development — golden set, LLM-or-clinician-as-judge, CI gate — is identical in structure; only the annotation source (radiologists instead of crowd workers) and cost change. The feedback loop is the same data flywheel: production predictions and eventual clinical outcomes flow back to training; you just need IRB approval and a formal data-use agreement to close it. Training-serving skew monitoring (the log-and-wait pattern from the data-engineering chapter) is equally important — hospital scanners, software versions, and acquisition protocols change, and a CT model trained on one scanner brand can degrade on another.

### Unique challenges and how to address them

**De-identification.** DICOM metadata de-identification is straightforward with standard tooling (pydicom + a tag-whitelisting approach), but the hard problem is **burned-in pixel PHI**: some modalities (ultrasound, screenshots, scanned documents embedded as DICOM) have patient name and MRN rendered directly into the pixel data. A metadata scrub that misses pixel-level PHI leaves you in HIPAA violation. The mitigation: run an OCR-based pixel-PHI detector before any data leaves the facility boundary; treat false negatives here as a critical failure mode.

**Annotation cost and active learning.** Radiologist or pathologist annotation runs roughly an order of magnitude more expensive than crowd annotation. You cannot afford to label the full long tail. The standard response is a carefully designed active learning loop: train on an initial seed set, run inference on unlabeled data, surface the highest-uncertainty and highest-diversity cases to annotators, retrain — the same uncertainty-based querying you'd use anywhere, but the oracle is scarce and expensive. Semi-supervised methods (pseudo-labeling with calibrated confidence thresholds) are also standard in medical imaging for exactly this reason.

**Calibration and uncertainty.** A model that outputs "0.92 probability of malignancy" must mean roughly 92 out of 100 such cases are malignant — otherwise probability-weighted clinical decision trees downstream are broken. Calibrate every medical ML model (temperature scaling, Platt, isotonic as covered in the evaluation chapter) and report calibration curves (reliability diagrams) as a first-class output. Report expected calibration error (ECE) on test slices, not just AUC.

**Concept extraction and de-identification in clinical NLP.** Named entity recognition over clinical text targets medical concepts: disease mentions, medications + dosages, lab values, procedures. The standard architecture is a fine-tuned biomedical language model (the BioBERT/PubMedBERT lineage, or a more recent clinical variant) on annotated corpora, followed by a rule-based normalization step that maps extracted mentions to standard ontologies (SNOMED CT, ICD-10, RxNorm). De-identification for clinical text is its own NLP task: the PHI types are defined by HIPAA Safe Harbor (18 identifier categories), and the standard approach is a fine-tuned NER model on the i2b2 de-identification challenge corpora. Both tasks require domain-specific evaluation — a general NER model's accuracy on clinical text is substantially lower.

**Auditability under FDA SaMD.** Every prediction that contributed to a clinical decision should be traceable: what model version, what input data, what output, at what time. This is the tracing requirement from the evaluation chapter applied with regulatory force. Store inference logs with the model artifact hash, the input study UIDs, and the prediction with confidence — at minimum. Your deployment pipeline must treat a model version bump as a regulated change, not a config push.

---

## 2. Finance

### What changes

**Latency and adversarial dynamics.** The latency spectrum in finance is wider than almost any other domain: algorithmic trading in equity markets operates at sub-millisecond latency (co-location with the exchange, FPGA-accelerated execution, kernel-bypass networking); fraud scoring runs at 100–200 ms inside an authorization flow; credit scoring can take seconds. The adversarial dimension is unique: in equity markets, **alpha decays** — as a profitable signal becomes known or crowded, it is arbitraged away. The model you trained six months ago on a signal that correlates with next-day returns may be picking up a transient pattern that no longer exists. This is a form of distribution shift that is adversarially driven, not randomly occurring, and it demands more aggressive retraining cadences and novelty detection than most other domains.

**Regulation.** In the US, SR 11-7 (Federal Reserve and OCC guidance on model risk management) requires formal model validation by an independent team before any model goes into production for a regulated use. The Equal Credit Opportunity Act (ECOA) and Fair Housing Act create disparate impact liability for credit models: if a model has statistically significant adverse impact on a protected class (race, sex, national origin) even without discriminatory intent, the lender faces regulatory and legal exposure. An adverse action notice must accompany credit denials, listing the primary reasons — which means SHAP values (or equivalent) are legally significant outputs, not optional explainability garnish.

**Point-in-time correctness.** The data-engineering chapter names this as the most common source of leakage; in finance it is catastrophically easy to get wrong. Macroeconomic indicators are **revised** — GDP, unemployment, and CPI are released as preliminary estimates, then revised multiple times over months. A model that trains on the current (revised) value of an indicator for a point in history is learning from data that did not exist at that point. Feature engineering for finance must use point-in-time snapshots of all revise-able data. The same trap applies to fundamental corporate data (earnings restatements). Survivorship bias in backtests is the sibling problem: historical index constituent lists differ from today's list because companies were removed due to bankruptcy or acquisition — training on "today's S&P 500, historically" evaluates on survivors, inflating backtested returns.

**Asymmetric costs.** A chargeback in payments fraud costs the lender the full transaction amount plus fees. A false positive (blocking a legitimate transaction) costs a customer relationship and a service call. These are not symmetric — state the cost ratio explicitly and let it drive your threshold and cost-sensitive weighting strategy.

### What stays the same

Temporal train/test splits are mandatory — the data-engineering chapter's warning about random splits leaking the future applies here with full force. The fraud funnel is the same multi-stage architecture from the classic-ML-systems chapter: features at three freshness tiers (batch profile, near-real-time velocity counters, live transaction context), a calibrated model, and a decision layer with step-up actions as a middle tier between approve and block. The evaluation chapter's emphasis on calibration applies directly — credit models feeding into pricing or risk capital calculations require calibrated probabilities.

### Unique challenges and how to address them

**Fairness and disparate impact.** The standard approach: train on the full feature set, then run a disparate impact analysis (4/5ths rule, or statistical tests on approval rates by protected class) before production. If disparate impact exists, options include: removing or regularizing the offending features, re-weighting the training distribution, using an adversarial debiasing objective, or — critically — auditing whether the protected attribute is correlated with a legitimate risk factor through a third variable (which is legally distinguishable from direct discrimination but requires documentation). Counterfactual fairness analysis (would the outcome have changed if only the protected attribute changed?) is a richer frame. Whatever approach you take, the audit trail must be documented — model risk management requires it.

**Explainability as a product requirement.** For adverse action notices, SHAP values or a LIME-based explanation at inference time must produce the top-N reasons for a denial in human-readable form. This is not optional on the credit side. The practical implication: you cannot deploy a model where attribution is prohibitively expensive to compute. Tree-based models (GBDT) have fast exact Shapley via TreeSHAP and are favored for exactly this reason. Deep models require approximate methods and carry more explainability overhead.

**Model risk management workflow.** SR 11-7 requires: a conceptual soundness review (does the model's assumptions hold?), a data quality review, an independent validation by a team separate from the development team, sensitivity analysis and stress testing, ongoing performance monitoring with escalation triggers, and periodic revalidation (typically annual). Build this into your deployment plan from the start; retrofitting it onto a running production model is painful. The champion/challenger framework from the classic-ML-systems chapter maps cleanly onto MRM's ongoing monitoring requirement.

**Low-latency serving for algo trading.** Below 1 ms, the model inference itself is often not the bottleneck — it is data transport, serialization, and memory allocation. The typical approach: pre-load all features into a shared-memory store, use a statically allocated model (no dynamic dispatch, no Python runtime, ONNX on a fixed-size input compiled to native code), and pin the process to a dedicated core. The ML problem is often simple by model-complexity standards (linear models, shallow trees) because the latency budget leaves no room for depth.

---

## 3. Autonomous Systems

### What changes

**Latency and safety criticality.** The perception-to-actuation latency budget in an autonomous vehicle is roughly 50–100 ms end-to-end — from sensor capture through detection, tracking, prediction, planning, and control signal. Within that budget, perception alone (object detection + semantic segmentation + depth estimation over camera/LiDAR fusion) needs to complete in under 50 ms. Robotics control loops run at 100–1000 Hz, demanding sub-10 ms cycle times for low-level controllers. These are not soft SLOs — exceeding them means the vehicle or robot acts on stale world state, which at highway speed can mean meters of error. Unlike a recommendation miss, a perception miss can cause injury or death. This changes the engineering culture around failure: you design for graceful degradation into a safe state (pull over, freeze, hand off to human) rather than best-effort service.

**Sensor fusion.** No single sensor is reliable alone. Camera provides rich semantic information but fails in low light and adverse weather. LiDAR provides precise 3D geometry but is expensive and degrades in fog and rain. RADAR is weather-robust and provides velocity but has low resolution. The standard stack fuses all three at the feature or object level (early, mid, or late fusion depending on latency and accuracy tradeoffs). Sensor calibration (extrinsic transforms between sensors) drifts over time and must be monitored — a miscalibrated camera–LiDAR pair produces inconsistent 3D projections that cascade into detection errors silently.

**Edge deployment.** The inference hardware in a vehicle is a fixed-power embedded SoC — NVIDIA Drive, Qualcomm Snapdragon Ride, or custom silicon. The inference optimization techniques from the inference-optimization chapter are not optional; they are the only path to meeting latency budgets on constrained hardware. INT8 quantization, structured pruning, and hardware-specific kernel fusion are standard practice. The model must be profiled on the *target chip*, not on a datacenter GPU, because throughput and memory bandwidth characteristics differ substantially.

**Operational design domain (ODD).** A deployed autonomous system is not certified for all conditions — it has a defined ODD: geographic region, weather conditions, road types, time of day, speed range. Predictions outside the ODD must trigger a safe-state handoff rather than a best-effort inference. This is the flip of a general-purpose model: you accept narrower coverage in exchange for reliable performance within the envelope.

### What stays the same

The inference optimization chapter's full toolkit applies: quantization, distillation, and hardware-specific compilation are exactly the techniques used to fit a perception stack on an embedded SoC. The eval-driven development pattern is the same — golden sets of labeled sensor captures (including edge-case scenarios: pedestrians in occluding shadows, cut-in vehicles, construction zones) gated in CI, with regression tests on every model update. The training-serving skew monitor applies, but here it is called the **sim-to-real gap**.

### Unique challenges and how to address them

**Sim-to-real gap.** Training a perception model entirely on real-world labeled data is prohibitively expensive for rare and safety-critical scenarios (a child running into the road from between parked cars). Simulation fills the gap — rendered synthetic scenes with perfect pixel-level labels. The problem: simulation is imperfect, and models trained on synthetic data degrade on real data. Mitigations: domain randomization (randomize lighting, textures, weather in the simulator, making the model invariant to appearance details); photorealistic rendering pipelines (NVIDIA Omniverse, NeRF-based scene reconstruction from real captures fed into the simulator); and fine-tuning on real-world data as the final adaptation step. The mix ratio between synthetic and real data is a hyperparameter to tune carefully; report real-world validation metrics, not synthetic ones.

**Long-tail safety.** The distribution of sensor captures is dominated by nominal highway and urban driving. The failure modes that matter — an overturned truck blocking the road, a pedestrian in a wheelchair, a construction worker gesturing — are rare in training data and are exactly where the model is most likely to fail. Active learning and scenario mining over the production fleet (find cases where the model had high uncertainty or made a correction that was later overridden) are the standard retrieval mechanisms for this long tail. Safety-critical edge cases must have explicit coverage in the golden test set with hard recall requirements (not just average precision).

**OTA (over-the-air) model updates.** Updating software on a deployed fleet of vehicles is a regulated activity in automotive (UNECE WP.29 regulations, ISO 24089). A model update is not a config push — it requires validation that the update does not regress safety-critical scenarios, may require regulator notification, and needs a staged rollout with automatic rollback on safety-metric degradation. This is the deployment chapter's canary + rollback workflow applied with regulatory force and much higher stakes per incident.

**Sim-to-real for robotics: the manipulation gap.** For robot manipulation (grasping, assembly), the gap is compounded by contact dynamics — real-world friction, deformability, and sensor noise are hard to simulate faithfully. The 2026 approach is **diffusion policy** or **action-chunking transformer** models trained on real demonstration data (teleoperation), with simulation used only for initial policy seeding and curriculum structuring. Minimizing real-world demonstration data needs is the central research and engineering problem.

---

## 4. Manufacturing and Industrial

### What changes

**Sensor data characteristics.** Industrial telemetry is IoT time-series: vibration (accelerometer, acoustic emission), temperature, electrical current, pressure, and flow rates, sampled at anywhere from 1 Hz to tens of kHz depending on the sensor and failure mode. Unlike web event logs, industrial sensor data is frequently: irregularly sampled (sensor polling failures), gappy (machines go offline, planned maintenance windows), multi-rate (a vibration sensor at 10 kHz alongside a temperature sensor at 1 Hz), and drifting (sensor degradation is itself a signal, but it also corrupts other signals). Your data pipeline must handle these gracefully — forward-fill and interpolation need to be applied with an understanding that they can mask the very anomalies you are trying to detect.

**Cost asymmetry for predictive maintenance.** A false negative (the model misses an impending bearing failure) results in unplanned downtime, cascading equipment damage, and potentially safety incidents — costs that are one to two orders of magnitude higher than a false positive (an unnecessary maintenance stop). You must state this asymmetry when choosing a threshold and when framing the evaluation: optimize recall at a fixed acceptable false-positive rate, not F1. This mirrors the fraud-detection framing from the classic-ML-systems chapter but with even more extreme asymmetry.

**Quality control: inverse cost asymmetry.** For inline product inspection (visual defect detection), the cost direction reverses by product: in safety-critical manufacturing (aerospace parts, automotive brakes), a false negative (shipping a defective part) is catastrophic; in consumer goods, the cost of false positives (scrapping good product) is the binding constraint. The model must be calibrated and the threshold set by explicit cost modeling, not by default argmax.

**Edge deployment at the factory floor.** Many factories have limited or air-gapped connectivity. Inference runs on an industrial PC, a PLC (programmable logic controller), or an embedded vision system at the line. Model size and inference latency matter; the inference optimization chapter's techniques apply, with the additional constraint that the serving environment may not support modern ML runtimes — ONNX Runtime is widely deployed for exactly this reason.

**Domain expertise and labeling.** Manufacturing engineers often know what failure signatures look like in the sensor data — they can label them in retrospect. But run-to-failure data is scarce and expensive to generate deliberately (you are destroying equipment). The standard dataset for predictive maintenance research is limited in real-world fidelity because most real failure data is proprietary. This means your models often train on a limited labeled set and must rely heavily on unsupervised or self-supervised pretraining over unlabeled normal-operation data.

### What stays the same

The feedback loop is identical: production inference results, joined with maintenance logs and inspection outcomes, flow back as training data — the flywheel from the foundations chapter. Drift monitoring is essential: equipment ages, production recipes change, new product variants appear on the same line. The data-engineering chapter's feature-drift monitors (PSI, KL on feature distributions) apply; additionally monitor the sensor signal statistics themselves (mean, variance, autocorrelation structure) against a baseline window. The classic-ML-systems chapter's asymmetric-cost framing for fraud applies directly to both predictive maintenance and quality control.

### Unique challenges and how to address them

**Label scarcity and anomaly detection.** With limited run-to-failure examples, a pure supervised approach is fragile. The standard two-stage approach: (1) train an autoencoder or a normalizing-flow model on normal-operation data (abundant and unlabeled) to learn a compact reconstruction; (2) flag high-reconstruction-error windows as anomalies; (3) use the small labeled failure set to calibrate the anomaly threshold and assign fault categories to flagged windows. The reconstruction error is not a calibrated probability — it is a ranking score. Convert it to a probability only if you have enough failure examples for proper calibration.

**Hierarchical supply-chain forecasting.** Demand forecasting in manufacturing must be coherent across levels: global demand → regional → product family → individual SKU. A model that forecasts SKUs independently will produce aggregate-level totals that are inconsistent with the global forecast, leading to inventory misallocation. The standard framework is **reconciliation** (bottom-up, top-down, or optimal MinTrace reconciliation) applied on top of base forecasters at each level. Exogenous variables (economic indicators, promotions, competitor actions, supplier capacity signals) matter here more than in consumer-facing forecasting — models that ignore them fail hard at demand inflection points.

**Industrial protocols.** Factory data arrives via industrial protocols — OPC-UA (the standard for modern industrial equipment; hierarchical, browseable address space, authentication support), MQTT (lightweight pub/sub common in IIoT), and legacy protocols (Modbus, PROFIBUS). Your data pipeline must speak these, and they are not HTTP. OPC-UA clients in Python (`asyncua` is a common library; confirm current version) work well for lab setups; at production scale, a dedicated SCADA historian or industrial IoT gateway (Siemens MindSphere, PTC ThingWorx, AWS IoT Greengrass) is the entry point for the ML pipeline. Knowing this architecture — SCADA historian as the source-of-truth time-series store, with a CDC or export feed into your ML feature platform — is a senior signal for manufacturing ML interviews.

---

## Mapping domains back to the core patterns

A quick reference for the interview:

| Domain | Dominant constraint | Key delta from core |
|---|---|---|
| Healthcare | Regulation + annotation cost | Algorithm change protocol locks iteration; calibration is a legal output; PHI controls |
| Finance | Adversarial drift + explainability mandate | Point-in-time correctness critical; SHAP is a regulatory artifact; MRM gating |
| Autonomous | Safety-criticality + edge compute | Sim-to-real gap; ODD-aware failure; OTA update regulation; long-tail coverage |
| Manufacturing | Label scarcity + cost asymmetry | Anomaly detection over supervised learning; industrial protocols; hierarchical forecasting |

---

## Interview Q&A

**Q1. Design a medical imaging AI that triages chest X-rays for emergency radiology.**

**Requirements (state aloud):** ~500 studies/hour at a large hospital; findings to triage: pneumothorax, pulmonary edema, pneumonia, normal; SLO: critical finding (pneumothorax) surfaced to radiologist within 5 minutes of scan arrival; regulatory context: deployed as SaMD, so FDA clearance pathway applies; HIPAA data handling; radiologist remains the clinical decision-maker — this is a worklist prioritizer, not an autonomous diagnoser.

**Architecture:**

- **Ingest:** DICOM listener (DIMSE C-STORE or DICOMweb STOW-RS) on the hospital PACS; metadata de-identification + pixel-PHI scan before the image leaves the hospital network or enters the ML pipeline; convert 12-bit grayscale DICOM to normalized numpy arrays; reject or flag studies with equipment-type outside the model's validated ODD (lateral views, pediatric cases if not trained on).
- **Model:** fine-tuned classification/detection model — a ViT or EfficientNet-class backbone pretrained on a large chest-X-ray corpus (CheXpert, MIMIC-CXR lineage), fine-tuned on labeled clinical data with radiologist annotations. Multi-label output (the pathologies are not mutually exclusive). Output includes per-finding probability and an optional localization heatmap (GradCAM or attention rollout) for the radiologist's UI.
- **Calibration:** mandatory — temperature scaling on the held-out validation set, verified with reliability diagrams by finding type. The output probability feeds the triage priority queue; an uncalibrated model produces miscalibrated queue ordering.
- **Triage layer:** rule-based priority escalation — any study with pneumothorax probability above a calibrated threshold fires a STAT alert (pager/EHR notification) within 2 minutes of arrival; all studies are enqueued with a priority score for the worklist; normal-probability-high studies can be batched lower.
- **Monitoring:** per-finding AUC and calibration ECE on a rolling held-out set with radiologist-confirmed outcomes (feedback loop — the radiologist's final report is the label); alert on AUC drop > X% over a 30-day rolling window; monitor input distribution shift (scanner acquisition parameters logged and trended).
- **Regulatory notes:** this is the answer the interviewer is listening for — FDA clearance means the model version is locked; a performance-improving retrain requires a submission under the algorithm change protocol before deployment; the eval and monitoring infrastructure is also the regulatory evidence package.
- **Failure modes to volunteer:** pixel-level PHI missed by metadata-only scrubbing; scanner brand/model distribution shift (a model trained on one manufacturer's images degrades on another's acquisition profile); annotation label noise (different radiologists grade borderline findings differently — inter-annotator agreement should be measured and model uncertainty should correlate with it).

---

**Q2. Design a real-time credit decisioning system for a digital lender.**

**Requirements:** consumer unsecured installment loans; decision at application time, < 3 seconds end-to-end; regulatory: ECOA adverse action notice required on denials; SR 11-7 model validation before production; fairness constraints on protected classes; ~50k applications/day, growing.

**Architecture:**

- **Feature engineering:** point-in-time correct by construction — all feature values are timestamped at application receipt, never joined to revised data; features span credit bureau tradeline history, income/employment verification, bank account transaction aggregates (velocity features from the data-engineering chapter), and behavioral signals from the application session. Every feature has a documented source-of-truth and a staleness policy.
- **Model:** calibrated GBDT (LightGBM/XGBoost) as the primary — fast inference, exact TreeSHAP, interpretable for adverse action reasons. Potentially a stacked ensemble with a shallow neural network for interaction capture, but the GBDT component must remain the primary explanation vehicle. All models calibrated (Platt/isotonic) on a holdout set; calibration curves reported by applicant cohort.
- **Fairness pipeline:** post-training disparate impact analysis (4/5ths rule and chi-squared test on protected class approval rates); SHAP-based feature importance audit for protected attributes and their proxies; if disparate impact is detected, adversarial debiasing or reweighting applied and re-audited; the audit trail is a regulatory deliverable, not a nice-to-have.
- **Adverse action:** at inference time, SHAP values are computed (TreeSHAP is fast — milliseconds on a GBDT), and the top 4 negative factors are mapped to a standardized adverse-action code table (as required by FCRA) and included in the denial notice.
- **Model risk management gate:** independent validation team reviews conceptual soundness, data quality, performance on stress scenarios (recession-period data), sensitivity analysis (how much does the score change if income is perturbed by 10%?), and fairness audit before production. This is a hard gate — no deployment without sign-off.
- **Monitoring:** PSI on input feature distributions monthly; score distribution by vintage; rolling calibration error; protected-class approval rates on a 30-day window with alert on threshold violation; champion/challenger for the next version.
- **Failure modes to volunteer:** survivorship bias in the training set (only approved applicants have observed outcomes — the reject inference problem); label delay (defaults occur months or years post-origination; early-vintage monitoring uses proxy labels like 30-day delinquency, with known limitations); model staleness in a credit cycle shift (a model trained in a low-default environment underestimates risk in a rising-default cycle).

---

**Q3. Design the object detection and tracking stack for a mobile delivery robot operating on a university campus.**

**Requirements:** wheeled ground robot, outdoor campus environment; must detect pedestrians, bicycles, vehicles, curbs, and construction; max inference latency 50 ms per cycle; inference on embedded SoC (roughly NVIDIA Jetson-class); safety-critical — must not collide with people; ODD: paved paths, daylight and dusk, dry and light-wet conditions only; no highway-speed dynamics (max 5 km/h); human operator can remote-takeover.

**Architecture:**

- **Sensor configuration:** 360° camera ring (3–6 cameras) for semantic richness + a 16-beam LiDAR for 3D geometry + ultrasonic sensors for close-range obstacle detection below camera/LiDAR resolution. This is a deliberate redundancy design: if the LiDAR fails, camera-based depth estimation plus ultrasonic provides a degraded-but-functional safety layer. Sensor fusion at the feature level (project LiDAR point cloud into camera frustums, fuse point-pillar features with image features) rather than late fusion, for better performance on occluded objects.
- **Model:** a multi-task backbone (shared ResNet or EfficientDet-derived) with heads for 2D detection, monocular depth estimation, and semantic segmentation; INT8 quantized and compiled to the target SoC's DLA (deep learning accelerator) with ONNX Runtime or TensorRT (representative as of 2026 — confirm current SDK version). Profile on the target chip, not a datacenter GPU — the SoC's memory bandwidth is the bottleneck, not FLOPs.
- **Tracking:** a Kalman filter + Hungarian algorithm baseline (SORT-style) extended with appearance re-identification embeddings for long-track consistency (DeepSORT lineage). Tracking maintains object state between frames; the planner consumes tracks, not raw detections.
- **Sim-to-real mitigation:** initial training on synthetic campus-environment data (domain-randomized lighting, pedestrian textures, weather conditions) plus fine-tuning on real campus captures; a sim-to-real gap metric (detection mAP on real validation captures vs synthetic) tracked per training run.
- **ODD enforcement:** a lightweight ODD monitor runs in parallel — detects rain intensity from camera image statistics, estimates ambient light level, and flags ODD boundary conditions; on ODD exit, the robot decelerates to a stop and requests human takeover rather than running inference outside its validated envelope.
- **Failure modes to volunteer:** camera lens contamination (mud, rain droplets) degrades image quality silently — monitor reconstruction error or image statistics for early detection; LiDAR point-density drop under heavy rain; tracking ID-switch events when pedestrians cross (requires appearance re-ID); sim-to-real gap for construction zones (novel scene types not in training distribution — monitor detection confidence distribution for new semantic classes).

## You can now

- identify which levers bend in an unfamiliar domain — regulation, latency budget, cost asymmetry, failure tolerance — while reusing the domain-invariant core (funnels, eval-driven development, feedback loops, skew monitoring).
- design a HIPAA/SaMD-compliant medical-imaging triager, including pixel-PHI de-identification, calibration as a first-class output, and the algorithm-change-protocol constraint that turns iteration into a months-long regulated loop.
- design an ECOA/SR 11-7-compliant credit system with point-in-time-correct features, TreeSHAP adverse-action reasons, disparate-impact auditing, and an independent model-risk-management gate.
- design a safety-critical edge perception stack with redundant sensor fusion, ODD enforcement with safe-state handoff, and sim-to-real mitigation via domain randomization plus real fine-tuning.
- choose anomaly-detection-over-supervised approaches and hierarchical (reconciled) forecasting for label-scarce industrial settings, and state the cost asymmetry that sets the decision threshold.

## Try it

Pick one domain from this module and rewrite the real-time fraud mock (Mock 5 in the interview-playbook chapter) for that domain's constraints. In one page: name the specific regulation that gates deployment, the cost asymmetry that sets your decision threshold, the one feature-freshness or latency change the domain forces relative to the fraud baseline, and the single failure mode an interviewer is most likely to probe. Then defend your threshold and cascade choice against a curveball — "what changes if scale grows 10×?" or "what breaks when the regulator locks your model version?" — and write the two-sentence answer you'd give out loud.
