"""
MachineGuard AI - Page 4: Explainable AI & Feature Attribution
Global Permutation Importance, Tree Importance, and Local Sensitivity Analysis
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

from src.config import FEATURE_IMPORTANCE_FILE, METADATA_FILE
from src.explainability import load_global_feature_importance

st.set_page_config(page_title="Explainability - MachineGuard AI", page_icon="💡", layout="wide")

st.title("💡 Explainable AI & Feature Attribution")
st.markdown(
    "Predictive maintenance models in industrial plants must not be opaque black boxes. "
    "MachineGuard AI provides **global feature importances** on held-out validation data and **local directional sensitivity** for individual machines."
)

feat_imp = load_global_feature_importance()
metadata = {}
if METADATA_FILE.exists():
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

if not feat_imp:
    st.warning("⚠️ Feature importance artifacts not found. Please train models first via `python scripts/train_model.py`.")
    st.stop()

tab_global, tab_local, tab_limitations = st.tabs([
    "🌍 Global Feature Importance", 
    "🔬 Local Attribution Methodology", 
    "⚠️ Engineering Limitations"
])

with tab_global:
    st.subheader("Global Importance on Held-Out Validation Data")
    st.markdown(
        "Permutation importance measures the drop in **Average Precision (AP)** when a feature's values are randomly shuffled in the validation set. "
        "Unlike training impurity-based importance, permutation importance on held-out data cannot be inflated by feature cardinality or memorization."
    )

    perm_data = feat_imp.get("permutation_importance_validation", {})
    if perm_data:
        perm_rows = []
        for feat, vals in perm_data.items():
            perm_rows.append({
                "Feature": feat,
                "Importance Mean (Δ AP)": vals["importance_mean"],
                "Std Dev": vals["importance_std"]
            })
        perm_df = pd.DataFrame(perm_rows).sort_values(by="Importance Mean (Δ AP)", ascending=True)

        col_g1, col_g2 = st.columns([1.3, 1])

        with col_g1:
            fig_perm = px.bar(
                perm_df,
                x="Importance Mean (Δ AP)",
                y="Feature",
                orientation="h",
                error_x="Std Dev",
                title="Validation Permutation Importance (Drop in AP)",
                color="Importance Mean (Δ AP)",
                color_continuous_scale="Viridis"
            )
            fig_perm.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig_perm, use_container_width=True)

        with col_g2:
            st.markdown("### Top Drivers Summary")
            st.markdown("""
            1. **Torque [Nm]**: Consistently the single strongest mechanical predictor. Severe torque overloads directly trigger power and overstrain mechanical failures.
            2. **Rotational Speed [rpm]**: Interacts inversely with torque. Abnormally low speeds under high load indicate spindle stall.
            3. **Tool Wear [min]**: Represents continuous cumulative physical degradation of the cutting edge. Exceeding 200 minutes drastically elevates failure probability.
            4. **Process & Air Temperature [K]**: Thermal differential drives heat dissipation failures (HDF).
            5. **Machine Type (L/M/H)**: Subtle baseline risk difference across product quality grades (L variants suffer slightly higher failure rates).
            """)

    # Native Tree Importance comparison
    native_data = feat_imp.get("native_tree_importance", {})
    if native_data:
        with st.expander("🌲 Inspect Model-Specific Gini Impurity Importance"):
            native_df = pd.DataFrame(list(native_data.items()), columns=["Transformed Feature", "Impurity Importance"]).sort_values(by="Impurity Importance", ascending=False)
            st.dataframe(native_df, use_container_width=True, hide_index=True)

with tab_local:
    st.subheader("Instance-Level Prediction Explanations")
    st.markdown("""
    When evaluating an individual machine in the **Predict Failure** tab, MachineGuard AI computes local directional contributions:
    - **Reference Baseline**: Nominal operating conditions (300K Air Temp, 310K Process Temp, 1500 RPM, 40 Nm Torque, 60 min Wear, Type M).
    - **Counterfactual Perturbation**: Computes $\Delta P = P(\text{current}) - P(\text{feature} \leftarrow \text{baseline})$.
    - **Directional Classification**:
        - **INCREASED RISK** ($\Delta P > +1\%$): This specific parameter is actively pushing the machine toward the warning threshold.
        - **REDUCED RISK** ($\Delta P < -1\%$): This parameter is benign and mitigating risk relative to baseline.
        - **NEUTRAL** ($|\Delta P| \le 1\%$): Operating within safe nominal bounds.
    """)

    st.info("💡 To view live local explanations for any custom parameter combination, navigate to the **Predict Failure** page.")

with tab_limitations:
    st.subheader("Critical Engineering & Scientific Limitations")
    st.markdown("""
    ### 1. Statistical Association vs Mechanical Causation
    - Machine learning models identify **predictive correlation** within the training distribution.
    - An elevated torque reading strongly correlates with failure, but the root physical cause could be motor electrical phase imbalance, bad bearings, or defective workpiece material.
    - Feature importance **never proves physical root-cause causation**.

    ### 2. Sensor Collinearity
    - Ambient air temperature and internal process temperature share an 88% correlation.
    - Shuffling or perturbing one feature in isolation can temporarily create out-of-distribution pairs (e.g. cold process temp inside hot ambient air).

    ### 3. Degradation History
    - The AI4I 2020 dataset provides snapshots of tool wear in minutes, but does not provide complete 100 Hz high-frequency vibrational accelerometer waveforms or acoustic emission time-series.
    """)
