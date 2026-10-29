"""
MachineGuard AI - Page 1: Data Explorer
Exploratory Data Analysis, Distribution Analysis, and Leakage Inspection
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, LEAKAGE_COLUMNS, ID_COLUMNS
from src.data_loader import load_raw_dataset, get_eda_summary

st.set_page_config(page_title="Data Explorer - MachineGuard AI", page_icon="🔍", layout="wide")

st.title("🔍 Exploratory Data Analysis & Schema Inspection")
st.markdown(
    "Deep dive into the **AI4I 2020 Predictive Maintenance Dataset**, verifying data types, distributions, "
    "class imbalance, and strictly isolating target leakage columns."
)

@st.cache_data
def get_data_and_eda():
    if not DATA_FILE.exists():
        return None, None
    df = load_raw_dataset(str(DATA_FILE))
    eda = get_eda_summary(df)
    return df, eda

df, eda = get_data_and_eda()

if df is None:
    st.error("⚠️ Dataset not found. Please run `python scripts/download_data.py` to acquire the official dataset.")
    st.stop()

# Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview & Schema", "📊 Feature Distributions", "🔗 Correlation Heatmap", "🛡️ Leakage Audit"])

with tab1:
    st.subheader("Dataset Dimensions & Class Balance")
    c1, c2, c3, c4 = st.columns(4)
    total_samples = len(df)
    failures = int(df[TARGET_COLUMN].sum())
    fail_rate = (failures / total_samples) * 100

    c1.metric("Total Records", f"{total_samples:,}")
    c2.metric("Observed Failures", f"{failures} ({fail_rate:.2f}%)")
    c3.metric("Missing Values", f"{df.isnull().sum().sum()}")
    c4.metric("Duplicate Rows", f"{df.duplicated().sum()}")

    st.markdown("### Interactive Dataset Preview")
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        type_filter = st.multiselect("Filter by Product Type", ["L", "M", "H"], default=["L", "M", "H"])
    with col_f2:
        failure_filter = st.selectbox("Filter by Failure State", ["All", "Failures Only (1)", "Normal Operation Only (0)"])

    filtered_df = df[df["Type"].isin(type_filter)]
    if failure_filter == "Failures Only (1)":
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == 1]
    elif failure_filter == "Normal Operation Only (0)":
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == 0]

    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} records matching filters (capped at first 100 rows for preview).")

    st.markdown("### Failure Rate Across Machine Types")
    type_stats = df.groupby("Type")[TARGET_COLUMN].agg(
        Total="count",
        Failures="sum",
        Failure_Rate="mean"
    ).reset_index()
    type_stats["Failure_Rate_%"] = (type_stats["Failure_Rate"] * 100).round(2)

    fig_type = px.bar(
        type_stats,
        x="Type",
        y="Failure_Rate_%",
        text="Failure_Rate_%",
        title="Failure Rate by Machine Variant (L: Low 50%, M: Medium 30%, H: High 20%)",
        color="Type",
        color_discrete_map={"L": "#38bdf8", "M": "#3b82f6", "H": "#1d4ed8"}
    )
    fig_type.update_layout(template="plotly_dark", yaxis_title="Failure Rate (%)", showlegend=False)
    st.plotly_chart(fig_type, use_container_width=True)

with tab2:
    st.subheader("Continuous Feature Distributions by Target Class")
    selected_feature = st.selectbox("Select Operating Parameter to Inspect", NUMERICAL_FEATURES)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_hist = px.histogram(
            df,
            x=selected_feature,
            color=TARGET_COLUMN,
            marginal="box",
            barmode="overlay",
            title=f"Histogram & Boxplot: {selected_feature}",
            color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
            labels={TARGET_COLUMN: "Machine Failure"}
        )
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        fig_scatter = px.scatter(
            df,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color=TARGET_COLUMN,
            title="Spindle Speed vs Torque (Power Hyperbolic Envelope)",
            color_discrete_map={0: "#38bdf8", 1: "#f43f5e"},
            opacity=0.7,
            labels={TARGET_COLUMN: "Failure"}
        )
        fig_scatter.update_layout(template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Note how mechanical failures cluster around extreme torque spikes and power boundaries.")

with tab3:
    st.subheader("Feature Correlation Matrix")
    corr_cols = NUMERICAL_FEATURES + [TARGET_COLUMN]
    corr_df = df[corr_cols].corr().round(3)

    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        title="Pearson Correlation Heatmap (Operating Variables + Machine Failure)",
        color_continuous_scale="Blues"
    )
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Key Correlation Findings**:
    - **Air temperature** and **Process temperature** exhibit high collinearity (r ~ 0.88), consistent with thermodynamic heating.
    - **Torque** and **Rotational speed** exhibit an inverse relationship (r ~ -0.87) governed by motor mechanical power: $P = \\tau \\cdot \\omega$.
    - Individual linear correlations with `Machine failure` are small (all |r| < 0.20), showing that failures depend on **nonlinear compound thresholds** rather than a single linear driver.
    """)

with tab4:
    st.subheader("Strict Leakage Prevention & Audit")
    st.markdown("""
    Predictive maintenance models often suffer from catastrophic subtle target leakage in academic benchmarks. 
    MachineGuard AI implements the following strict exclusions:
    """)

    leak_table = pd.DataFrame([
        {"Column": "UDI", "Type": "Identifier", "Action": "STRICTLY EXCLUDED", "Rationale": "Monotonic index sequence; model would learn arbitrary row positions."},
        {"Column": "Product ID", "Type": "Identifier", "Action": "STRICTLY EXCLUDED", "Rationale": "Serial numbers encourage spurious memorization instead of physics."},
        {"Column": "TWF", "Type": "Failure Subtype", "Action": "STRICTLY EXCLUDED", "Rationale": "Tool Wear Failure indicator is a post-event diagnostic flag."},
        {"Column": "HDF", "Type": "Failure Subtype", "Action": "STRICTLY EXCLUDED", "Rationale": "Heat Dissipation Failure directly leaks machine failure target."},
        {"Column": "PWF", "Type": "Failure Subtype", "Action": "STRICTLY EXCLUDED", "Rationale": "Power Failure indicator directly implies target = 1."},
        {"Column": "OSF", "Type": "Failure Subtype", "Action": "STRICTLY EXCLUDED", "Rationale": "Overstrain Failure directly leaks target = 1."},
        {"Column": "RNF", "Type": "Failure Subtype", "Action": "STRICTLY EXCLUDED", "Rationale": "Random Failure indicator leaks post-mortem diagnosis."}
    ])
    st.dataframe(leak_table, hide_index=True, use_container_width=True)

    st.warning(
        "⚠️ **Operational Caveat**: Historical batch data contains retrospective observations. In live factory deployments, "
        "sensor data arrives sequentially as streaming telemetry subject to noise, calibration drift, and transient communication dropouts."
    )
