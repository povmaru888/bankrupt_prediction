# Corporate Bankruptcy Risk Prediction

**An explainable machine learning system for corporate financial-distress screening, risk prioritization, and model governance.**

Python · XGBoost · scikit-learn · SHAP · FastAPI · Streamlit

> **Project status:** Local MVP implementation with documented model experiments and automated tests. A trained XGBoost candidate passed the project's offline evaluation criteria; production activation and final release sign-off should be independently verified. This is a decision-support research prototype, not an automated credit decision engine.

## Overview

Corporate bankruptcy is a rare but consequential event. This project explores whether financial indicators can help analysts **rank firms by bankruptcy-related risk** and identify a manageable shortlist for further review.

The system goes beyond model training: it includes data contracts, leakage-conscious evaluation, probability calibration, explainability, a local scoring API, an interactive dashboard, and versioned model artifacts.

**Key capabilities**
- Compare a transparent **Logistic Regression baseline** with **XGBoost** under the same validation protocol.
- Rank records by model confidence and measure how many positive cases appear in the **top 10%**.
- Inspect influential financial indicators through **SHAP**, with contributions explicitly described as associations within the model—not causal drivers.
- Submit JSON or CSV scoring requests, examine individual cases, and export batch results.
- Separate offline training and model activation from online inference, with artifact integrity checks and rollback support.

## Results at a glance

Reported results from the October 22, 2026 local training run:

| Metric | Selected XGBoost candidate |
| --- | ---: |
| Final-test Average Precision (AP) | **0.5204** |
| Final-test positive prevalence | 0.0323 |
| AP / prevalence | **16.13×** |
| Precision at validation-selected threshold | 0.3333 |
| Recall at validation-selected threshold | 0.6591 |
| F1 score | 0.4427 |
| Brier score | **0.02215** |
| Recall@Top10% | **0.7500** |
| Final-test records | 1,364 |
| Positive records retrieved in top 10% | 33 of 44 |

The top-decile result means that, **within this held-out random split**, reviewing the 10% highest-scoring records would identify 75% of positive cases. It does **not** establish performance on future companies or a particular forecast horizon.

### Baseline comparison

Both candidate families used 30 randomized hyperparameter settings, five-fold stratified cross-validation, and sigmoid calibration based on the training partition. Model selection used validation AP; only the selected model was evaluated on the final test.

| Model | Training-CV AP | Validation AP |
| --- | ---: | ---: |
| Logistic Regression | 0.4075 | 0.4235 |
| **XGBoost** | **0.4448** | **0.5366** |

XGBoost improved validation AP by approximately **26.7% relative to Logistic Regression**. The selection rule prefers Logistic Regression when the validation AP difference is below 0.01; XGBoost exceeded that margin.

**Interpretation:** AP evaluates ranking quality, while Brier score summarizes squared error between predicted scores and binary outcomes. Neither metric alone proves that individual predictions are calibrated for a future time window.

For full evidence, consult `reports/logistic_vs_xgboost_comparison.md` and `reports/local_formal_training.log`.

## Dataset and feature engineering

| Property | Value |
| --- | --- |
| Source file | `Database/bankrupt_data.csv` |
| Records | 6,819 |
| Original numerical features | 95 |
| Positive (bankrupt) records | 220 (3.23%) |
| Negative records | 6,599 (96.77%) |
| Features retained for model training | **92** |
| Target | `Bankrupt?` |

The feature selector removes one constant field (`Net Income Flag`) and two verified duplicates (`Current Liability to Liability` and `Current Liability to Equity`). Highly correlated features are audited but **not automatically removed**. The CSV scoring contract still expects the original 95 feature columns, with an optional `record_id`.

Missing values are handled by training-fold-fitted median imputation. Row-level checks reject non-numeric/infinite values and records with ten or more missing features; file-level checks enforce the input schema.

**Important dataset limitation:** The source does not contain usable company identifiers, fiscal years, or bankruptcy event dates. Therefore, this project cannot claim a chronological backtest, an externally validated deployment population, or a defined “bankruptcy within the next 12 months” forecast.

## Modeling and evaluation

The main training workflow is:

1. Audit and load the input data; preserve schema and source-file provenance.
2. Create a reproducible stratified **64% / 16% / 20%** train / validation / test split using seed **42**.
3. Fit preprocessing inside training/CV folds to limit leakage.
4. Run **30 configurations × 5 CV folds** for Logistic Regression and XGBoost, optimizing average precision.
5. Calibrate candidates using training-only, five-fold sigmoid calibration.
6. Select the model on validation AP, with an explicit simpler-model tie-break rule.
7. Select the operating threshold on validation data by maximizing **F2** (favoring recall).
8. Evaluate the frozen winner once on the held-out test set.
9. Package the candidate with model metadata, schema, checksums, metrics, visualizations, and an HTML report.

Additional experiments compare XGBoost `binary:logistic`, `rank:pairwise`, and `reg:pseudohubererror` objectives. The ranking objective achieved higher validation AP in the recorded experiment, but slightly lower Recall@Top10%; it was **not automatically promoted**. See `reports/xgboost_objective_comparison.md`.

## Architecture

```text
                    OFFLINE / CONTROL PLANE
 Financial indicators CSV
          |
     Data validation
          |
  Feature selection + imputation
          |
  Stratified split / model search
          |
 Calibration / validation selection / F2 threshold
          |
   Frozen final-test evaluation
          |
 Versioned candidate model package
          |
 Manual review -> activate / rollback
                         |
                         v
                    ONLINE / DATA PLANE
  Streamlit UI (localhost:8501)
          |
      FastAPI (localhost:8000)
          |
 Schema + row validation -> active model -> scoring / SHAP
          |
  Ranked results / case explanations / CSV export
```

The Streamlit application calls FastAPI rather than directly loading the model. Training, activation, and rollback are CLI-only operations; web requests cannot retrain or switch the active model.

### Core modules

| Location | Responsibility |
| --- | --- |
| `src/bankruptcy_prediction/data/` | Training loader and strict CSV schema/row validation |
| `features/` | Deterministic feature cleanup, imputation, range monitoring |
| `training/` | Search pipelines, calibration, selection, thresholding, orchestration |
| `evaluation/` | AP, precision/recall, F1, Brier, top-decile metrics |
| `explanation/` | Model-specific SHAP explanations |
| `model_package/` and `model_registry/` | Artifact verification, activation, rollback, read-only retrieval |
| `api/` | FastAPI application and scoring/model/health routes |
| `ui/` | Streamlit overview, batch scoring, individual explanations, performance |
| `tests/` | Unit, integration, UI, security, leakage and performance checks |

## Getting started

### Requirements

- Windows 11 (the documented local development environment)
- Python **3.11+** (Python 3.12 was used during local testing)
- A CPU-capable installation of the pinned dependencies
- PowerShell for the supplied launch scripts

```powershell
git clone https://github.com/povmaru888/bankrupt_prediction.git
cd bankrupt_prediction
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

> These commands require the corresponding source files to be committed to this repository. At the time this README was prepared, the local project and the remote repository had not yet been fully synchronized.

### Train a candidate model

```powershell
.\.venv\Scripts\python.exe -m bankruptcy_prediction.cli train --help
```

Review `config/training.yaml`, `Docs/RUNBOOK.md`, and the CLI help for the exact command arguments before launching a full training run. Full search is CPU-intensive and creates a **candidate**, not an automatically activated production model.

### Start the local services

After a trusted candidate has been reviewed and activated via the documented CLI workflow:

```powershell
.\scripts\start_api.ps1
# In a separate PowerShell terminal:
.\scripts\start_ui.ps1
```

- API health endpoint: `http://127.0.0.1:8000/health`
- Streamlit interface: `http://127.0.0.1:8501`

Both applications are configured for **localhost-only** use. If no valid active model exists, scoring is unavailable until activation.

### Streamlit Community Cloud

The repository root contains the Community Cloud entry point `streamlit_app.py`.
When the local launcher has not supplied `BANKRUPTCY_API_BASE_URL`, that entry
point starts the same FastAPI application on loopback inside the managed
Streamlit container. The API remains inaccessible from the public network;
browser traffic reaches only Streamlit.

Deployment settings:

- Repository: `povmaru888/bankrupt_prediction`
- Branch: `main`
- Main file: `streamlit_app.py`
- Python: 3.12

The checked-in model package is intentionally limited to the verified active
candidate required for inference. Uploaded CSV content is request-scoped and
must not contain confidential data because the hosted dashboard is a public
research demonstration.

### Run checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format --check src tests
```

A historical validation report records 223 passing tests; a subsequent objective-experiment report records 226. These are **reported historical results**, not fresh CI results for the current GitHub repository.

## Model interpretability

The explanation layer uses SHAP implementations appropriate to each model family (Linear SHAP for Logistic Regression and Tree SHAP for XGBoost). Results identify features contributing positively or negatively to a model score and support case-level inspection. **SHAP contributions are model explanations, not evidence that changing a financial ratio would cause or prevent bankruptcy.**

## Safety, reproducibility, and governance

- Strict input schema, record identifiers, row-level validation, and configurable batch limits (10 MB / 10,000 rows).
- Separation of training, validation, and final-test responsibilities.
- Model candidate manifests, artifact checksums, and explicit manual activation/rollback.
- Local-only serving with temporary-upload cleanup and sanitized audit logging.
- No automated lending, investment, or credit decisions.
- Model binaries, temporary data, and machine-specific activation state should not be committed without deliberate review.

**Release status:** The project checklist includes outstanding steps for production candidate verification/activation, hardware-specific performance evidence, browser-based end-to-end validation, and sign-off. See `Docs/RELEASE_CHECKLIST.md`. Some earlier acceptance documents predate later successful training; check artifact timestamps and manifests rather than treating all documents as current.

## Repository layout

```text
config/             Training and service configuration
Database/           Input CSV and dataset documentation
Docs/               Design documents, roadmap, runbook, release checklist
experiments/        XGBoost objective comparison artifacts
reports/            Recorded training, model, and evaluation evidence
scripts/            Local launch, benchmark, and packaging utilities
src/                Python package: ML, API, UI, explanations, governance
tests/              Automated validation suites
artifacts/colab/    Optional Colab training bundle
models/             Local versioned candidates (binary artifacts may be ignored)
```

## Limitations and future work

The primary limitation is **external validity**: random stratified splits of a dataset without dates cannot establish forward-looking performance. Meaningful next steps include obtaining dated, entity-linked financial statements and bankruptcy outcomes; defining a prediction horizon; performing time-based and out-of-entity validation; monitoring calibration and class prevalence drift; and independently verifying deployment readiness.

## Disclaimer

This repository is an educational and research-oriented financial-risk screening prototype. Scores and rankings require human interpretation and domain review. They are **not** verified forward-looking bankruptcy probabilities and should not be used as the sole basis for lending, investment, regulatory, or other consequential decisions.

## Project documentation

- [Technical roadmap](Docs/TASK_ROADMAP.md)
- [Operational runbook](Docs/RUNBOOK.md)
- [Release checklist](Docs/RELEASE_CHECKLIST.md)
- [Logistic Regression vs. XGBoost](reports/logistic_vs_xgboost_comparison.md)
- [XGBoost objective experiments](reports/xgboost_objective_comparison.md)
