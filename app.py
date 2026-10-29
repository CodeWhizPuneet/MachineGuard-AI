"""
MachineGuard AI: Industrial Explainable Predictive Maintenance System
Main Streamlit Application Entry Point - Overview Dashboard
"""

import os
import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_FILE,
    METADATA_FILE,
    FINAL_METRICS_FILE,
    MODEL_COMPARISON_FILE,
    THRESHOLD_ANALYSIS_FILE,
)
from src.predict import get_inference_engine

st.set_page_config(
    page_title="MachineGuard AI - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Styling
st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .callout-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

@st.cache_resource
def load_cached_artifacts():
    engine = get_inference_engine()
    metadata = {}
    final_metrics = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    if FINAL_METRICS_FILE.exists():
        with open(FINAL_METRICS_FILE, "r") as f:
            final_metrics = json.load(f)
    return engine, metadata, final_metrics

engine, metadata, final_metrics = load_cached_artifacts()

# Sidebar Navigation Summary
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/engineering.png", width=64)
    st.title("MachineGuard AI")
    st.caption("Predictive Maintenance & Failure Risk System")
    st.markdown("---")
    
    st.markdown("### System Status")
    if engine.pipeline is not None and metadata:
        st.markdown(f"🟢 **Model Active**: `{metadata.get('model_name', 'Trained Model')}`")
        st.markdown(f"🎯 **Decision Threshold**: `{metadata.get('selected_threshold', 0.5):.3f}`")
        st.markdown(f"⏱️ **Trained**: `{metadata.get('training_timestamp', 'N/A')[:10]}`")
    else:
        st.markdown("🟡 **Model Status**: Artifacts loading or training required.")
    
    st.markdown("---")
    st.markdown(f"📊 **Session Predictions**: `{st.session_state.prediction_count}`")
    st.markdown("---")
    st.markdown("**Dataset**: UCI AI4I 2020 Predictive Maintenance (10,000 observations)")

# Main Header
st.title("⚙️ MachineGuard AI: Overview Dashboard")
st.markdown(
    "**Production-grade Explainable Predictive Maintenance System** estimating mechanical failure probability "
    "from continuous operating measurements (temperatures, spindle speed, torque, tool wear)."
)

if not metadata or not final_metrics:
    st.warning("⚠️ Model training artifacts are currently generating or not yet present. Run `python scripts/train_model.py` to produce full results.")
    st.stop()

# Key Performance Indicators
st.markdown("### 📈 Core System Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

test_opt = final_metrics.get("test_metrics_optimal_threshold", {})
val_opt = final_metrics.get("validation_metrics_optimal_threshold", {})
threshold_val = metadata.get("selected_threshold", 0.5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Selected Model</div>
        <div class="metric-value" style="font-size:1.2rem; color:#f8fafc;">{}</div>
        <div style="font-size:0.75rem; color:#10b981;">Best Validation AP</div>
    </div>
    """.format(metadata.get("model_name", "N/A")), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Operating Threshold</div>
        <div class="metric-value">{:.3f}</div>
        <div style="font-size:0.75rem; color:#94a3b8;">Tuned on Validation Set</div>
    </div>
    """.format(threshold_val), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Test Average Precision</div>
        <div class="metric-value">{:.4f}</div>
        <div style="font-size:0.75rem; color:#38bdf8;">Independent Final Test</div>
    </div>
    """.format(test_opt.get("average_precision", 0.0)), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Test Failure Recall</div>
        <div class="metric-value">{:.1f}%</div>
        <div style="font-size:0.75rem; color:#10b981;">Failures Caught</div>
    </div>
    """.format(test_opt.get("recall", 0.0) * 100), unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Test ROC-AUC</div>
        <div class="metric-value">{:.4f}</div>
        <div style="font-size:0.75rem; color:#a855f7;">Discrimination Ability</div>
    </div>
    """.format(test_opt.get("roc_auc", 0.0)), unsafe_allow_html=True)

# Data Distribution & Honest Boundary Callout
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### 📊 Dataset & Split Stratification")
    split_info = metadata.get("dataset_split", {})
    split_df = pd.DataFrame([
        {"Split": "Training Set (64%)", "Total Samples": split_info.get("train_samples", 6400), "Failure Count": split_info.get("train_failures", 217), "Failure Rate": f"{(split_info.get('train_failures', 217)/split_info.get('train_samples', 6400))*100:.2f}%", "Role": "Model fitting & 5-Fold Stratified CV"},
        {"Split": "Validation Set (16%)", "Total Samples": split_info.get("val_samples", 1600), "Failure Count": split_info.get("val_failures", 54), "Failure Rate": f"{(split_info.get('val_failures', 54)/split_info.get('val_samples', 1600))*100:.2f}%", "Role": "Model comparison & Threshold tuning"},
        {"Split": "Independent Test Set (20%)", "Total Samples": split_info.get("test_samples", 2000), "Failure Count": split_info.get("test_failures", 68), "Failure Rate": f"{(split_info.get('test_failures', 68)/split_info.get('test_samples', 2000))*100:.2f}%", "Role": "Single unbiased final evaluation"}
    ])
    st.dataframe(split_df, hide_index=True, use_container_width=True)

    st.markdown("""
    **Evaluation Hygiene Guarantee**:
    - Validation metrics and Test metrics are reported strictly separately.
    - Post-failure subtypes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and IDs (`UDI`, `Product ID`) are 100% excluded from model inputs.
    """)

with col_right:
    st.markdown("### 🛡️ System Scope & Capabilities")
    st.markdown("""
    <div class="callout-box">
        <h4 style="color:#38bdf8; margin-top:0;">What MachineGuard AI Can Do:</h4>
        <ul style="color:#cbd5e1; font-size:0.9rem;">
            <li>Quantifies calibrated failure risk probability from operational telemetry.</li>
            <li>Classifies risk using a defensible threshold tuned specifically to balance missed failures against false alarms.</li>
            <li>Identifies top feature contributions (Torque, Spindle Speed, Tool Wear) for each machine.</li>
            <li>Executes high-throughput batch CSV diagnosis with row-level validation.</li>
        </ul>
        <h4 style="color:#f59e0b; margin-top:1rem;">What The Model Cannot Do:</h4>
        <ul style="color:#cbd5e1; font-size:0.9rem;">
            <li>Does not guarantee deterministic failure timing or replace physical inspection.</li>
            <li>Cannot diagnose physical root causes (e.g. bearing fatigue vs motor burnout) beyond statistical feature sensitivity.</li>
            <li>Should not automate emergency plant shutdowns without human-in-the-loop review.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("💡 *Use the left sidebar navigation to explore the dataset, inspect model comparison benchmarks, run live predictions, or review the explainability breakdowns.*")
