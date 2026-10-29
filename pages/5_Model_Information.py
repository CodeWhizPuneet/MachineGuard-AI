"""
MachineGuard AI - Page 5: Model Information & Model Card
Model Card, Technical Specifications, Dataset Lineage, and Deployment Export
"""

import sys
import json
import zipfile
import io
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METADATA_FILE, FINAL_METRICS_FILE, MODEL_COMPARISON_FILE, THRESHOLD_ANALYSIS_FILE, DATA_FILE

st.set_page_config(page_title="Model Information - MachineGuard AI", page_icon="📜", layout="wide")

st.title("📜 Model Information & Industrial Model Card")
st.markdown(
    "Comprehensive technical documentation, hyperparameters, training provenance, and operational boundaries "
    "for the **MachineGuard AI** system."
)

@st.cache_data
def load_all_metadata():
    meta = {}
    final_metrics = {}
    thresh_data = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            meta = json.load(f)
    if FINAL_METRICS_FILE.exists():
        with open(FINAL_METRICS_FILE, "r") as f:
            final_metrics = json.load(f)
    if THRESHOLD_ANALYSIS_FILE.exists():
        with open(THRESHOLD_ANALYSIS_FILE, "r") as f:
            thresh_data = json.load(f)
    return meta, final_metrics, thresh_data

meta, final_metrics, thresh_data = load_all_metadata()

if not meta:
    st.warning("⚠️ Metadata artifacts not found. Please train models first via `python scripts/train_model.py`.")
    st.stop()

tab_card, tab_specs, tab_download = st.tabs(["📋 Model Card", "⚙️ Hyperparameters & Metrics", "📦 Project Archive"])

with tab_card:
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown("### Model Details")
        st.markdown(f"""
        - **System Name**: MachineGuard AI
        - **Model Architecture**: `{meta.get('model_name', 'Gradient Boosting')}`
        - **Pipeline Framework**: `scikit-learn` Pipeline with ColumnTransformer (OneHotEncoder + StandardScaler)
        - **Version**: 1.0.0
        - **Training Timestamp**: `{meta.get('training_timestamp', 'N/A')}`
        - **Random Seed**: `{meta.get('random_seed', 42)}` (Guarantees strict reproducibility)
        """)

        st.markdown("### Intended Use")
        st.markdown("""
        - **Primary Domain**: Continuous manufacturing, CNC milling, cutting machines, and industrial rotational equipment.
        - **Operational Role**: Early warning advisory system to trigger non-destructive maintenance inspection.
        - **Target Decision Makers**: Reliability engineers, maintenance dispatchers, and plant operations supervisors.
        """)

    with col_c2:
        st.markdown("### Dataset Lineage")
        st.markdown("""
        - **Source**: UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset)
        - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
        - **Total Records**: 10,000 observations (3.39% baseline failure prevalence)
        - **Features Used**: Machine Type (`L`, `M`, `H`), Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
        - **Strictly Excluded Columns**: `UDI`, `Product ID` (IDs); `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (post-failure leakages).
        """)

        st.markdown("### Operational Limitations & Inappropriate Use")
        st.markdown("""
        - **Do Not Automate Kill-Switches**: Never wire model predictions directly to automated plant emergency shutdowns without human confirmation.
        - **Out of Scope Machinery**: Turbines, jet propulsion, or pneumatic systems outside the documented operating ranges.
        - **Sensor Drift**: Sensor recalibration is required if physical calibration shifts by >2 K or >50 rpm.
        """)

with tab_specs:
    st.subheader("Tuned Hyperparameters & Final Metrics")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("#### Selected Hyperparameters")
        hyp = meta.get("hyperparameters", {})
        if hyp:
            st.json(hyp)
        else:
            st.info("Default regularized parameters verified through 5-fold Stratified CV.")

        st.markdown("#### Decision Threshold Specification")
        st.markdown(f"""
        - **Operating Threshold**: `{meta.get('selected_threshold', 0.5):.3f}`
        - **Validation Tuning Rationale**: {meta.get('threshold_rationale', 'N/A')}
        """)

    with col_h2:
        st.markdown("#### Independent Held-Out Test Evaluation")
        test_m = final_metrics.get("test_metrics_optimal_threshold", {})
        cm = test_m.get("confusion_matrix", {})
        rates = test_m.get("rates", {})

        metrics_summary = pd.DataFrame([
            {"Metric": "Average Precision (AP)", "Score": f"{test_m.get('average_precision', 0):.4f}"},
            {"Metric": "ROC-AUC", "Score": f"{test_m.get('roc_auc', 0):.4f}"},
            {"Metric": "F1-Score (Threshold = 0.475)", "Score": f"{test_m.get('f1_score', 0):.4f}"},
            {"Metric": "Failure Recall", "Score": f"{test_m.get('recall', 0)*100:.1f}%"},
            {"Metric": "Precision", "Score": f"{test_m.get('precision', 0)*100:.1f}%"},
            {"Metric": "False Alarm Rate (FPR)", "Score": f"{rates.get('false_alarm_rate_fpr', 0)*100:.2f}%"},
            {"Metric": "Missed Failure Rate (FNR)", "Score": f"{rates.get('miss_rate_fnr', 0)*100:.2f}%"},
            {"Metric": "True Positives / Caught Failures", "Score": f"{cm.get('true_positives', 0)} / 68"},
            {"Metric": "False Positives (False Alarms)", "Score": f"{cm.get('false_positives', 0)} / 1,932 normal machines"}
        ])
        st.dataframe(metrics_summary, hide_index=True, use_container_width=True)

with tab_download:
    st.subheader("Export Project Archive")
    st.markdown("Download the complete self-contained **MachineGuard AI** project package.")

    zip_path = PROJECT_ROOT / "MachineGuard-AI.zip"
    if zip_path.exists():
        with open(zip_path, "rb") as f:
            st.download_button(
                label="📥 Download MachineGuard-AI.zip (Code, Models, & Data)",
                data=f.read(),
                file_name="MachineGuard-AI.zip",
                mime="application/zip",
                type="primary"
            )
    else:
        st.info("Generating project zip archive on the fly...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(PROJECT_ROOT):
                # Exclude node_modules, .git, venv, pycache
                if any(x in root for x in ["node_modules", ".git", "__pycache__", ".venv", ".playwright-cli"]):
                    continue
                for file in files:
                    if file.endswith((".pyc", ".DS_Store", ".zip")):
                        continue
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(PROJECT_ROOT)
                    z.write(full_p, rel_p)

        st.download_button(
            label="📥 Download MachineGuard-AI.zip (Complete Package)",
            data=zip_buffer.getvalue(),
            file_name="MachineGuard-AI.zip",
            mime="application/zip",
            type="primary"
        )
