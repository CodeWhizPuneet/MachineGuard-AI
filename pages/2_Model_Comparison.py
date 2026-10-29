"""
MachineGuard AI - Page 2: Model Comparison
Benchmarking candidate algorithms, cross-validation, and generalization gap analysis.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_COMPARISON_FILE, METADATA_FILE, THRESHOLD_ANALYSIS_FILE, FINAL_METRICS_FILE

st.set_page_config(page_title="Model Comparison - MachineGuard AI", page_icon="⚖️", layout="wide")

st.title("⚖️ Model Benchmark & Algorithm Comparison")
st.markdown(
    "Rigorous evaluation across 6 distinct algorithms trained on the 64% training split with 5-fold Stratified CV. "
    "Models are ranked by **Average Precision (AP)** on the independent 16% validation split to address extreme class imbalance."
)

@st.cache_data
def load_comparison_data():
    if not MODEL_COMPARISON_FILE.exists():
        return None, None, None
    with open(MODEL_COMPARISON_FILE, "r") as f:
        comp = json.load(f)
    with open(METADATA_FILE, "r") as f:
        meta = json.load(f)
    thresh_data = {}
    if THRESHOLD_ANALYSIS_FILE.exists():
        with open(THRESHOLD_ANALYSIS_FILE, "r") as f:
            thresh_data = json.load(f)
    return comp, meta, thresh_data

comp, meta, thresh_data = load_comparison_data()

if comp is None:
    st.warning("⚠️ Model comparison artifacts not found. Please train models first via `python scripts/train_model.py`.")
    st.stop()

# Build comparison DataFrame
rows = []
for name, data in comp.items():
    v = data["val_metrics_default_0_5"]
    t = data["train_metrics"]
    rows.append({
        "Algorithm": name,
        "CV Train AP": data.get("cv_train_average_precision", 0.0),
        "Val AP": v.get("average_precision", 0.0),
        "Val ROC-AUC": v.get("roc_auc", 0.0),
        "Val F1 (0.50)": v.get("f1_score", 0.0),
        "Val Recall (0.50)": v.get("recall", 0.0),
        "Val Precision (0.50)": v.get("precision", 0.0),
        "Balanced Acc": v.get("balanced_accuracy", 0.0),
        "Train AP": t.get("average_precision", 0.0),
        "Generalization Gap (AP)": data.get("generalization_gap_ap", 0.0),
        "Overfitting Risk": data.get("overfitting_risk", "Unknown")
    })

comp_df = pd.DataFrame(rows).sort_values(by="Val AP", ascending=False).reset_index(drop=True)

# Highlight Winner
winner = meta.get("model_name", "Gradient Boosting")
st.success(
    f"🏆 **Selected Model Candidate**: `{winner}` | "
    f"Primary Criterion: Highest Validation Average Precision ({comp[winner]['val_metrics_default_0_5']['average_precision']:.4f}) "
    f"with balanced generalization gap ({comp[winner]['generalization_gap_ap']:.4f})."
)

# Comparison Table
st.markdown("### 📊 Benchmark Comparison Matrix (Threshold = 0.50)")
st.dataframe(
    comp_df[[
        "Algorithm", "Val AP", "Val ROC-AUC", "Val F1 (0.50)", 
        "Val Recall (0.50)", "Val Precision (0.50)", "Generalization Gap (AP)", "Overfitting Risk"
    ]],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Visual Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_ap = px.bar(
        comp_df,
        x="Algorithm",
        y="Val AP",
        color="Algorithm",
        text="Val AP",
        title="Validation Average Precision (Primary Ranking Metric)",
    )
    fig_ap.update_layout(template="plotly_dark", showlegend=False)
    fig_ap.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    st.plotly_chart(fig_ap, use_container_width=True)

with col_chart2:
    fig_roc = px.bar(
        comp_df,
        x="Algorithm",
        y="Val ROC-AUC",
        color="Algorithm",
        text="Val ROC-AUC",
        title="Validation ROC-AUC",
    )
    fig_roc.update_layout(template="plotly_dark", showlegend=False)
    fig_roc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("### 🔍 Overfitting & Generalization Gap Analysis")
st.markdown(
    "To detect subtle overfitting, we compare performance on the **Training set** vs the **held-out Validation set**. "
    "A massive drop in Average Precision (Generalization Gap > 0.35) flags severe memorization of the majority negative class."
)

fig_gap = go.Figure()
fig_gap.add_trace(go.Bar(
    name="Train AP",
    x=comp_df["Algorithm"],
    y=comp_df["Train AP"],
    marker_color="#38bdf8"
))
fig_gap.add_trace(go.Bar(
    name="Validation AP",
    x=comp_df["Algorithm"],
    y=comp_df["Val AP"],
    marker_color="#10b981"
))
fig_gap.update_layout(
    barmode="group",
    template="plotly_dark",
    title="Train AP vs Validation AP (Generalization Stability)",
    yaxis_title="Average Precision"
)
st.plotly_chart(fig_gap, use_container_width=True)

# Deep Dive into Decision Rationale
st.markdown("### 📝 Model Selection Rationale")
st.markdown(f"""
1. **Why Dummy Baseline Failed**: With a ~3.4% failure prevalence, the stratified baseline achieves AP of 0.035, confirming the severe penalty of random guessing under high imbalance.
2. **Why Logistic Regression Was Insufficient**: While computationally simple with low overfitting gap (+0.0038), it achieved only 0.4551 AP and 18.5% recall at default threshold. Failure mechanics in CNC machinery are governed by compound nonlinear interactions (e.g. high torque coupled with high spindle speeds), which a linear hyperplane cannot adequately separate.
3. **Why Random Forest Underperformed Gradient Boosting**: Unconstrained Random Forest suffered from an excessive generalization gap (Train AP ~0.995 vs Val AP ~0.593), memorizing majority training patterns.
4. **Why Gradient Boosting Won**: Gradient Boosting achieved the highest Validation Average Precision (**{comp[winner]['val_metrics_default_0_5']['average_precision']:.4f}**) and Validation ROC-AUC (**{comp[winner]['val_metrics_default_0_5']['roc_auc']:.4f}**), sequential residual minimization on the minority class with shallow trees effectively bounded variance.
""")
