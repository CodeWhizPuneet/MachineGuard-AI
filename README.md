# MachineGuard AI: Explainable Predictive Maintenance & Machine Failure Prediction

![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)
![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)
![Model: Gradient Boosting](https://img.shields.io/badge/Selected%20Model-Gradient%20Boosting-emerald.svg)
![Decision Threshold](https://img.shields.io/badge/Operating%20Threshold-0.475-amber.svg)
![Test AP](https://img.shields.io/badge/Test%20AP-0.8004-blue.svg)
![Test ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.9635-purple.svg)
![Tests](https://img.shields.io/badge/Unit%20Tests-15%20Passed-success.svg)

**MachineGuard AI** is a complete, production-grade academic Machine Learning and MLOps system that estimates the probability of industrial machinery failure using continuous operating conditions. Designed specifically to solve the severe class imbalance (~3.39% historical failure rate) found in real-world predictive maintenance, the project includes zero-leakage data pipelines, multi-algorithm benchmarking, validation-based decision-threshold optimization, local/global explainability, automated test suites, and an interactive Streamlit web dashboard ready for cloud deployment.

---

## Architecture Flowchart

```mermaid
flowchart TD
    A[UCI AI4I 2020 Dataset\n10,000 Observations] --> B[Data Validation & Schema Audit\nExclusion of UDI, Product ID, & Failure Subtypes]
    B --> C[Stratified Splitting\n64% Train | 16% Val | 20% Final Test]
    C --> D[Leakage-Free Preprocessing Pipeline\nOneHotEncoder + StandardScaler inside folds]
    D --> E[Multi-Algorithm Benchmark & 5-Fold Stratified CV\nDummy, LogReg, RF, Extra Trees, Gradient Boosting, HistGB]
    E --> F[Model Selection on Validation Set\nPrimary Ranking Metric: Average Precision]
    F --> G[Threshold Optimization on Validation Set\nOptimal Operating Threshold: 0.475]
    G --> H[Final Independent Test Set Evaluation\nEvaluated Once: AP 0.8004 | ROC-AUC 0.9635 | Recall 73.5%]
    H --> I[Artifact Serialization\nartifacts/model.joblib, metadata.json, metrics.json]
    I --> J[Streamlit Multi-Page Dashboard & Web UI\nOverview, Explorer, Benchmarks, Single & Batch Diagnosis]
    J --> K[Cloud Deployment\nStreamlit Community Cloud / Hugging Face Spaces]
```

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Dataset Source & Schema](#3-dataset-source--schema)
4. [Target Leakage Prevention & Feature Selection](#4-target-leakage-prevention--feature-selection)
5. [Preprocessing Methodology](#5-preprocessing-methodology)
6. [Train / Validation / Test Strategy](#6-train--validation--test-strategy)
7. [Algorithms Evaluated](#7-algorithms-evaluated)
8. [Hyperparameter Tuning & Cross-Validation](#8-hyperparameter-tuning--cross-validation)
9. [Model Selection Criteria](#9-model-selection-criteria)
10. [Overfitting Prevention & Generalization Analysis](#10-overfitting-prevention--generalization-analysis)
11. [Evaluation Metrics & Meaning](#11-evaluation-metrics--meaning)
12. [Decision-Threshold Optimization](#12-decision-threshold-optimization)
13. [Explainable AI (Global & Local Attribution)](#13-explainable-ai-global--local-attribution)
14. [Project Directory Structure](#14-project-directory-structure)
15. [Windows Setup Guide (PowerShell / VS Code / Anaconda)](#15-windows-setup-guide)
16. [Commands to Download Data & Train Models](#16-commands-to-download-data--train-models)
17. [Commands to Run Automated Tests](#17-commands-to-run-automated-tests)
18. [Commands to Launch Streamlit Application](#18-commands-to-launch-streamlit-application)
19. [GitHub Setup & Cloud Deployment Guide](#19-github-setup--cloud-deployment-guide)
20. [Known Limitations & Future Work](#20-known-limitations--future-work)

---

## 1. Project Overview
Industrial milling machines, CNC equipment, and lathe spindles experience severe mechanical and thermodynamic degradation over continuous duty cycles. MachineGuard AI continuously monitors:
- Product Quality Variant (`Type`: L, M, H)
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

The system transforms raw sensor signals through an isolated scikit-learn pipeline, calculates calibrated failure probabilities, classifies machines into actionable risk tiers, and isolates which operating variables elevated failure risk.

---

## 2. Problem Statement & Motivation
In factory operations, unexpected machine breakdowns cause catastrophic downtime costs, ruined tooling, and production bottlenecks. However:
1. **Extreme Class Imbalance**: In healthy plants, machines run normally >96% of the time (~3.39% failure incidence). Traditional classifiers tuned for raw accuracy default to predicting "nominal" 100% of the time, achieving 96.6% accuracy while catching zero failures.
2. **Asymmetric Error Costs**: A false negative (missed machine breakdown) causes thousands of dollars in mechanical destruction. A false positive (routine inspection alert) costs only 15 minutes of technician inspection time.
3. **Black-Box Skepticism**: Plant reliability engineers will not trust predictions without understanding why the system flagged a machine (e.g., torque overload vs thermal dissipation failure).

---

## 3. Dataset Source & Schema
- **Official Repository**: UCI Machine Learning Repository
- **Dataset**: AI4I 2020 Predictive Maintenance Dataset
- **URL**: `https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset`
- **Citation**: Stephan Matzka, School of Engineering - Technology and Life, HTW Berlin, Germany.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Observations**: 10,000 synthetic industrial records modeled on real machinery behavior.

### Column Specification
| Column | Type | Role | Engineering Rationale |
|:---|:---|:---|:---|
| `UDI` | Integer (1 - 10000) | **Excluded** | Arbitrary database row ID. Causes spurious memorization. |
| `Product ID` | String (e.g. L47181) | **Excluded** | Hardware serial number. Causes overfitting. |
| `Type` | Categorical ('L', 'M', 'H') | **Predictor** | Quality variant: Low (50%), Medium (30%), High (20%). |
| `Air temperature [K]` | Float (295.3 - 304.5 K) | **Predictor** | Ambient factory environment temperature. |
| `Process temperature [K]` | Float (305.7 - 313.8 K) | **Predictor** | Working process temperature. |
| `Rotational speed [rpm]` | Integer (1168 - 2886 rpm) | **Predictor** | Spindle angular velocity. |
| `Torque [Nm]` | Float (3.8 - 76.6 Nm) | **Predictor** | Spindle load torque (normal mean: 40 Nm). |
| `Tool wear [min]` | Integer (0 - 253 min) | **Predictor** | Cumulative cutting tool operational minutes. |
| `Machine failure` | Binary (0 / 1) | **Target** | 1 = Failure, 0 = Nominal (339 positive cases, 3.39%). |
| `TWF` | Binary (0 / 1) | **Excluded** | Tool Wear Failure mode (post-failure indicator). |
| `HDF` | Binary (0 / 1) | **Excluded** | Heat Dissipation Failure mode (post-failure indicator). |
| `PWF` | Binary (0 / 1) | **Excluded** | Power Failure mode (post-failure indicator). |
| `OSF` | Binary (0 / 1) | **Excluded** | Overstrain Failure mode (post-failure indicator). |
| `RNF` | Binary (0 / 1) | **Excluded** | Random Failure mode (post-failure indicator). |

---

## 4. Target Leakage Prevention & Feature Selection
Academic predictive maintenance papers frequently suffer from silent target leakage. MachineGuard AI strictly enforces:
1. **Identifier Elimination**: `UDI` and `Product ID` are dropped immediately upon data loading.
2. **Subtype Exclusion**: Columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` describe post-event failure modes. In live industrial operations, these flags do not exist until after the machine breaks. Feeding them into the model would yield an artificial ~99.9% score that collapses in production.
3. **Temporal Sanity**: Only features measurable *prior* to a failure event are utilized.

---

## 5. Preprocessing Methodology
All data-dependent transformations are encapsulated inside a scikit-learn `ColumnTransformer`:
- **Categorical Encoder**: `OneHotEncoder(categories=[['L', 'M', 'H']], handle_unknown='ignore', sparse_output=False)`
- **Numerical Scaler**: `StandardScaler()` applied to continuous operating features.
- **Strict Leakage Prevention**: Encoders and scalers are fitted **only on training folds** inside cross-validation loops, never on the full dataset or validation/test sets prior to splitting.

---

## 6. Train / Validation / Test Strategy
To ensure honest, unbiased performance estimates:
- **64% Training Set** (6,400 samples, 217 failures, 3.39%): Model parameter fitting & 5-fold Stratified Cross-Validation.
- **16% Validation Set** (1,600 samples, 54 failures, 3.38%): Algorithm comparison, hyperparameter selection, generalization-gap auditing, and decision-threshold tuning.
- **20% Independent Test Set** (2,000 samples, 68 failures, 3.40%): Evaluated **exactly once** after all modeling and threshold decisions were locked.
- **Random Seed**: Fixed globally to `42`.

---

## 7. Algorithms Evaluated
Six candidate algorithms spanning diverse learning paradigms were benchmarked:
1. **Dummy Classifier (Baseline)**: Stratified random guessing reflecting class prior.
2. **Logistic Regression**: Linear classifier with L2 regularization and class weighting.
3. **Random Forest Classifier**: Ensemble of 150 bagged decision trees with depth and leaf constraints.
4. **Extra Trees Classifier**: Extremely Randomized Trees testing random split thresholds for variance reduction.
5. **Gradient Boosting Classifier**: Sequential boosting optimizing binomial deviance with shallow trees.
6. **HistGradientBoostingClassifier**: Histogram-binned gradient boosting with native fast discretization.

---

## 8. Hyperparameter Tuning & Cross-Validation
- **Strategy**: `RandomizedSearchCV` confined strictly to the 64% training split.
- **Cross-Validation**: 5-fold Stratified K-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Scoring Objective**: `average_precision` (optimizes the precision-recall trade-off on the minority failure class).
- **Tuned Parameter Spaces**: Bounded tree depth (3-5 for boosting, 6-15 for random forest), learning rate (0.05-0.15), subsampling (0.8-1.0), and minimum samples per leaf (1-4).

---

## 9. Model Selection Criteria
The winning algorithm was selected based on four documented operational criteria:
1. **Validation Average Precision (Primary)**: Precision-recall performance on the minority class.
2. **Overfitting Gap**: Difference between Training AP and Validation AP ($Gap = AP_{train} - AP_{val}$).
3. **Failure Recall at Acceptable False-Alarm Rates**: High detection rate without flooding maintenance staff with false alarms.
4. **Deployment Stability**: Simplicity, deterministic inference speed, and zero external binary dependencies.

### Actual Benchmark Comparison Results
| Algorithm | CV Train AP | Val AP | Val ROC-AUC | Val F1 (0.50) | Val Recall (0.50) | Gen Gap (AP) | Overfit Risk |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting (WINNER)** | 0.8124 | **0.7687** | **0.9753** | **0.7368** | **0.6481** | 0.2117 | Moderate |
| Extra Trees | 0.7812 | 0.7397 | 0.9540 | 0.5333 | 0.3704 | 0.2602 | Moderate |
| Hist Gradient Boosting | 0.7540 | 0.6892 | 0.9741 | 0.6458 | 0.5741 | 0.3108 | High |
| Random Forest | 0.8410 | 0.5932 | 0.9594 | 0.5606 | 0.6852 | 0.4022 | High (Overfit) |
| Logistic Regression | 0.4420 | 0.4551 | 0.8926 | 0.2985 | 0.1852 | 0.0038 | Low |
| Dummy Baseline | 0.0339 | 0.0348 | 0.5103 | 0.0541 | 0.0556 | -0.0011 | None |

---

## 10. Overfitting Prevention & Generalization Analysis
1. **Detection**: Random Forest achieved near-perfect Training AP (~0.995) but plummeted to 0.5932 on Validation data ($Gap = 0.4022$). This heavy memorization disqualified it.
2. **Remediation**: Gradient Boosting uses shallow trees (`max_depth=3`), conservative learning rates (`learning_rate=0.1`), and shrinkage to bound tree complexity.
3. **Generalization Proof**: When evaluated on the completely unseen 2,000-sample final test set, Gradient Boosting maintained an exceptional **0.8004 AP** and **0.9635 ROC-AUC**.

---

## 11. Evaluation Metrics & Meaning
- **Average Precision (AP)**: Summarizes the Precision-Recall curve as the weighted mean of precisions achieved at each threshold. This is the primary metric for imbalanced data.
- **ROC-AUC**: Measures discrimination capacity across all possible thresholds.
- **Precision**: Proportion of flagged machines that actually failed ($\frac{TP}{TP + FP}$).
- **Recall**: Proportion of actual failures caught by the model ($\frac{TP}{TP + FN}$).
- **False Alarm Rate (FPR)**: False alarms per healthy machine ($\frac{FP}{FP + TN}$).
- **Balanced Accuracy**: Arithmetic mean of sensitivity and specificity ($\frac{Recall + Specificity}{2}$).

---

## 12. Decision-Threshold Optimization
The default probability threshold of $0.50$ is arbitrary and suboptimal for asymmetric maintenance costs.
- **Optimization Data**: Evaluated 35 candidate thresholds from $0.05$ to $0.90$ using **held-out validation data only**.
- **Optimal Threshold Selected**: **0.475**
- **Operational Rationale**: Maximizes validation F1-score (0.7423) while achieving 66.67% recall with minimal false alarms (6 false alarms out of 1,600 machines).

### Independent Final Test Set Performance (2,000 Machines • 68 Failures)
| Metric | At Default Threshold (0.500) | At Optimized Threshold (0.475) | Operational Impact |
|:---|:---:|:---:|:---|
| **Average Precision** | 0.8004 | **0.8004** | Precision-Recall curve summary |
| **ROC-AUC** | 0.9635 | **0.9635** | High rank discrimination |
| **F1-Score** | 0.8000 | **0.8000** | Harmonized precision/recall balance |
| **Failure Recall** | 73.53% | **73.53%** | **Caught 50 of 68 actual failures** |
| **False Positives (FP)** | 7 | **7** | Only 7 false alarms out of 1,932 healthy machines |
| **False Alarm Rate (FPR)** | 0.36% | **0.36%** | Extremely low maintenance distraction |
| **False Negatives (FN)** | 18 | **18** | 18 missed failures |

---

## 13. Explainable AI (Global & Local Attribution)
### Global Permutation Importance (Validation Set)
1. **Torque [Nm]** ($\Delta AP = 0.4285 \pm 0.0214$): Primary load predictor. Exceeding 60 Nm creates extreme motor strain.
2. **Rotational speed [rpm]** ($\Delta AP = 0.2641 \pm 0.0182$): Inversely coupled with torque via mechanical power. Low speed under load indicates stalling.
3. **Tool wear [min]** ($\Delta AP = 0.1984 \pm 0.0150$): Physical tool degradation. Failure hazard accelerates nonlinearly above 200 min.
4. **Process temperature [K]** ($\Delta AP = 0.0712 \pm 0.0095$): Process heat dissipation tracking.
5. **Air temperature [K]** ($\Delta AP = 0.0435 \pm 0.0078$): Ambient environmental baseline.
6. **Machine Type** ($\Delta AP = 0.0120 \pm 0.0032$): Baseline variant quality differential.

### Local Sensitivity Attribution
For single-machine diagnosis, the system calculates counterfactual delta impacts:
$$\Delta P = P(\text{current telemetry}) - P(\text{feature} \leftarrow \text{baseline})$$
Classifying features as **INCREASED RISK** ($\Delta P > +1\%$), **REDUCED RISK** ($\Delta P < -1\%$), or **NEUTRAL**.

---

## 14. Project Directory Structure
```
MachineGuard-AI/
├── app.py                         # Streamlit multi-page application entry point
├── requirements.txt               # Pinned production Python dependencies
├── README.md                      # Complete system documentation & setup guide
├── MachineGuard-AI.zip            # Self-contained project zip archive
├── .streamlit/
│   └── config.toml                # Industrial dark theme styling configuration
├── data/
│   ├── ai4i2020.csv               # Official verified UCI dataset (10,000 rows)
│   └── README.md                  # Dataset lineage, schema, and license details
├── artifacts/
│   ├── machineguard_pipeline.joblib  # Trained scikit-learn pipeline
│   ├── model_metadata.json        # Hyperparameters, threshold (0.475), timestamp
│   ├── model_comparison.json      # Benchmark results across all 6 models
│   ├── threshold_analysis.json    # 35-point validation threshold optimization
│   ├── final_test_metrics.json    # Final held-out test metrics
│   └── feature_importance.json    # Validation permutation & tree importances
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, seeds, and schema definitions
│   ├── validation.py              # Schema validation & leakage column exclusion
│   ├── data_loader.py             # 64/16/20 stratified split & EDA statistics
│   ├── preprocessing.py           # Preprocessing ColumnTransformer factory
│   ├── model_selection.py         # Multi-model factory & metrics evaluation
│   ├── train.py                   # Full training, CV tuning, & threshold pipeline
│   ├── evaluate.py                # Final evaluation reporting & model card generator
│   ├── explainability.py          # Global permutation & local attribution engine
│   └── predict.py                 # Single and batch inference engine
├── pages/
│   ├── 1_Data_Explorer.py         # Streamlit EDA & distribution visualizer
│   ├── 2_Model_Comparison.py      # Streamlit algorithm comparison benchmarks
│   ├── 3_Predict_Failure.py       # Streamlit single diagnosis & batch CSV upload
│   ├── 4_Explainability.py        # Streamlit permutation & attribution breakdowns
│   └── 5_Model_Information.py     # Streamlit model card & technical specs
├── reports/
│   ├── final_evaluation_report.md # Formal markdown evaluation report
│   └── model_card.md              # Complete industrial model card
├── scripts/
│   ├── download_data.py           # Automated dataset downloader with checksum validation
│   ├── train_model.py             # CLI pipeline to retrain models and regenerate artifacts
│   └── evaluate_model.py          # CLI report generator
└── tests/
    ├── test_data_validation.py    # Schema integrity and leakage tests
    ├── test_preprocessing.py      # Encoding and 64/16/20 split isolation tests
    ├── test_prediction.py         # Single/batch inference and threshold tests
    └── test_artifacts.py          # Serialized pipeline and metadata integrity tests
```

---

## 15. Windows Setup Guide
Follow these step-by-step commands using **Windows PowerShell** and **VS Code**.

### Option A: Standard Python Virtual Environment (Recommended)
```powershell
# 1. Create project workspace directory
mkdir -Force D:\Workspace\MachineGuard-AI
cd D:\Workspace\MachineGuard-AI

# 2. Extract MachineGuard-AI.zip or clone repository here
# Expand-Archive -Path MachineGuard-AI.zip -DestinationPath .

# 3. Create clean virtual environment
python -m venv .venv

# 4. Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 5. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Using Existing Anaconda Environment (`jupyter_env`)
```powershell
# 1. Open Anaconda Prompt or PowerShell with conda initialized
conda activate jupyter_env

# 2. Navigate to workspace
cd D:\Workspace\MachineGuard-AI

# 3. Install required libraries
pip install -r requirements.txt
```

---

## 16. Commands to Download Data & Train Models
```powershell
# 1. Download and verify official UCI AI4I 2020 dataset
python scripts\download_data.py

# 2. Verify dataset checksum and required columns
python scripts\download_data.py --verify-only

# 3. Execute the full training, tuning, and threshold selection pipeline
python scripts\train_model.py

# 4. Generate evaluation reports and model cards
python scripts\evaluate_model.py
```

---

## 17. Commands to Run Automated Tests
```powershell
# Run the complete test suite with verbose output
pytest tests\ -v

# Run individual test modules
pytest tests\test_data_validation.py -v
pytest tests\test_prediction.py -v
pytest tests\test_preprocessing.py -v
pytest tests\test_artifacts.py -v
```
*(All 15 tests pass out of the box).*

---

## 18. Commands to Launch Streamlit Application
```powershell
# Launch the interactive Streamlit dashboard
streamlit run app.py
```
Streamlit will automatically open your default browser at: `http://localhost:8501`

---

## 19. GitHub Setup & Cloud Deployment Guide
To deploy this project to **Streamlit Community Cloud**:

### 1. Initialize Git Repository
```powershell
cd D:\Workspace\MachineGuard-AI
git init
git add .
git commit -m "Initial commit: MachineGuard AI end-to-end predictive maintenance system"
```

### 2. Push to GitHub
```powershell
# Create a new public repository on GitHub named 'MachineGuard-AI'
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/MachineGuard-AI.git
git push -u origin main
```

### 3. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/MachineGuard-AI`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Deploy!**

The cloud platform will automatically install `requirements.txt`, read the cached artifacts in `artifacts/`, and launch the web application with zero manual retraining required.

---

## 20. Known Limitations & Future Work
1. **Synthetic Nature of AI4I 2020**: While physically grounded in HTW Berlin milling equipment models, real-world machining telemetry experiences vibration sensor noise, thermal drift, and variable workpiece metallurgy.
2. **Snapshot vs Sequential Telemetry**: The dataset provides aggregated minutes of tool wear rather than raw 100 Hz tri-axial accelerometer vibration waveforms. Future versions could integrate LSTM/Transformer temporal encoders for continuous vibration spectrograms.
3. **Sensor Drift Recalibration**: Operational thresholds must be recalibrated periodically if physical sensors experience systematic temperature or tachometer drift.

---

*MachineGuard AI was developed by a CSE AI/ML student as a defensible, production-grade demonstration of end-to-end machine learning engineering.*
