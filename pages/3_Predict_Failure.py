"""
MachineGuard AI - Page 3: Machine Failure Prediction
Interactive Single-Machine Diagnosis and High-Throughput Batch CSV Prediction
"""

import sys
import io
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METADATA_FILE, FEATURE_RANGES
from src.predict import get_inference_engine
from src.validation import ValidationError

st.set_page_config(page_title="Predict Failure - MachineGuard AI", page_icon="⚡", layout="wide")

st.title("⚡ Machine Failure Risk Prediction")
st.markdown(
    "Estimate machine failure probability using operational conditions. "
    "Classify risk against the tuned decision threshold and explore parameter contributions."
)

@st.cache_resource
def get_engine():
    return get_inference_engine()

engine = get_engine()

if engine.pipeline is None:
    st.error("⚠️ Model pipeline is not loaded. Please train the model via `python scripts/train_model.py` first.")
    st.stop()

# Load default threshold
default_threshold = float(engine.metadata.get("selected_threshold", 0.475))

tab_single, tab_batch = st.tabs(["🎯 Single Machine Diagnosis", "📁 Batch Prediction (CSV Upload)"])

# ==============================================================================
# TAB 1: SINGLE MACHINE DIAGNOSIS
# ==============================================================================
with tab_single:
    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.subheader("Operational Input Parameters")
        
        # Presets for quick student demonstration
        preset = st.selectbox(
            "Load Operating Scenario Preset",
            ["Custom Input", "Nominal Healthy Machine", "High Tool Wear Warning", "High Torque / Power Stress", "Thermal Dissipation Risk"]
        )

        preset_vals = {
            "Custom Input": {"type": "M", "air": 300.0, "proc": 310.0, "rpm": 1500, "torque": 40.0, "wear": 60},
            "Nominal Healthy Machine": {"type": "M", "air": 298.1, "proc": 308.6, "rpm": 1551, "torque": 42.8, "wear": 15},
            "High Tool Wear Warning": {"type": "L", "air": 300.5, "proc": 310.2, "rpm": 1420, "torque": 48.0, "wear": 215},
            "High Torque / Power Stress": {"type": "H", "air": 301.2, "proc": 310.8, "rpm": 1280, "torque": 68.5, "wear": 120},
            "Thermal Dissipation Risk": {"type": "L", "air": 304.5, "proc": 313.2, "rpm": 1380, "torque": 58.0, "wear": 190}
        }
        active_vals = preset_vals[preset]

        input_type = st.selectbox("Machine Type (Quality Variant)", ["L", "M", "H"], index=["L", "M", "H"].index(active_vals["type"]))
        air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=float(active_vals["air"]), step=0.1)
        proc_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=float(active_vals["proc"]), step=0.1)
        rot_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=int(active_vals["rpm"]), step=10)
        torque = st.number_input("Torque [Nm]", min_value=1.0, max_value=100.0, value=float(active_vals["torque"]), step=0.5)
        tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=int(active_vals["wear"]), step=5)

        st.markdown("---")
        with st.expander("⚙️ Decision Threshold Setting"):
            threshold_slider = st.slider(
                "Operating Probability Threshold",
                min_value=0.05,
                max_value=0.90,
                value=default_threshold,
                step=0.025,
                help="Adjust threshold to balance false alarms vs failure detection sensitivity."
            )
            st.caption(f"Tuned default: {default_threshold:.3f}")

        predict_btn = st.button("🚀 Analyze Machine Health", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Diagnostic Assessment")
        
        # Auto-run or button-run
        rec = {
            "Type": input_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }

        try:
            res = engine.predict_single(rec, threshold_override=threshold_slider)
            st.session_state.prediction_count += 1
            proba = res["predicted_probability"]
            proba_pct = res["predicted_probability_percent"]
            t_used = res["decision_threshold"]

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Predicted Failure Probability (%)", 'font': {'size': 20, 'color': '#f8fafc'}},
                number={'suffix': "%", 'font': {'size': 36, 'color': res['risk_color']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': res['risk_color']},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, t_used * 70], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [t_used * 70, t_used * 100], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [t_used * 100, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 4},
                        'thickness': 0.75,
                        'value': t_used * 100
                    }
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk Classification Badge
            st.markdown(f"""
            <div style="background:#1e293b; border-left:5px solid {res['risk_color']}; padding:1rem; border-radius:6px; margin-bottom:1rem;">
                <h4 style="margin:0; color:{res['risk_color']};">{res['risk_band']}</h4>
                <p style="margin:0.25rem 0 0 0; color:#cbd5e1; font-size:0.9rem;">
                    Decision Threshold: <b>{t_used:.3f}</b> | Predicted Class: <b>{res['predicted_class']}</b> ({res['prediction_label']})
                </p>
                <p style="margin:0.5rem 0 0 0; color:#94a3b8; font-size:0.85rem;">
                    <b>Recommended Maintenance Action:</b> {res['recommended_action']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Feature Contributions
            st.markdown("#### 🔬 Local Feature Risk Contributions")
            contribs = res["explanation"]["contributions"]
            contrib_df = pd.DataFrame(contribs)

            fig_contrib = px.bar(
                contrib_df,
                x="impact_percentage",
                y="feature",
                orientation="h",
                color="direction",
                text="impact_percentage",
                color_discrete_map={
                    "INCREASED RISK": "#ef4444",
                    "REDUCED RISK": "#10b981",
                    "NEUTRAL": "#64748b"
                },
                title="Directional Probability Impact vs Reference Baseline (%)"
            )
            fig_contrib.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=30, b=10))
            fig_contrib.update_traces(texttemplate='%{text:+.1f}%', textposition='outside')
            st.plotly_chart(fig_contrib, use_container_width=True)

            st.caption(f"ℹ️ {res['explanation']['summary']}")

        except ValidationError as e:
            st.error(f"Input Validation Error: {e}")

# ==============================================================================
# TAB 2: BATCH PREDICTION (CSV UPLOAD)
# ==============================================================================
with tab_batch:
    st.subheader("Batch Machine Health Assessment")
    st.markdown(
        "Upload a CSV file containing machine operating conditions. "
        "The system validates schemas, isolates any leakage columns, and generates probabilities."
    )

    col_u1, col_u2 = st.columns([2, 1])

    with col_u1:
        uploaded_file = st.file_uploader("Upload Machine Telemetry CSV", type=["csv"])

    with col_u2:
        st.markdown("**Sample Batch Template**")
        sample_batch = pd.DataFrame([
            {"UDI": 10001, "Product ID": "L50001", "Type": "L", "Air temperature [K]": 298.5, "Process temperature [K]": 308.5, "Rotational speed [rpm]": 1500, "Torque [Nm]": 42.0, "Tool wear [min]": 25},
            {"UDI": 10002, "Product ID": "M50002", "Type": "M", "Air temperature [K]": 301.2, "Process temperature [K]": 310.5, "Rotational speed [rpm]": 1380, "Torque [Nm]": 58.4, "Tool wear [min]": 195},
            {"UDI": 10003, "Product ID": "H50003", "Type": "H", "Air temperature [K]": 299.0, "Process temperature [K]": 309.0, "Rotational speed [rpm]": 1620, "Torque [Nm]": 35.0, "Tool wear [min]": 45}
        ])
        csv_buffer = io.StringIO()
        sample_batch.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=csv_buffer.getvalue(),
            file_name="machineguard_batch_template.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            input_batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(input_batch_df)} records from uploaded file.**")

            results_df, invalid_df, summary = engine.predict_batch(
                input_batch_df,
                threshold_override=default_threshold
            )

            # Summary Cards
            c_tot, c_val, c_fail, c_rate = st.columns(4)
            c_tot.metric("Total Submitted", f"{summary['total_input_rows']}")
            c_val.metric("Valid Evaluated", f"{summary['valid_evaluated_rows']}")
            c_fail.metric("Predicted Failures", f"{summary['predicted_failures_count']}")
            c_rate.metric("Failure Rate", f"{summary['predicted_failure_rate_%']}%")

            if summary.get("warnings"):
                for w in summary["warnings"]:
                    st.info(f"ℹ️ {w}")

            st.markdown("### Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            # Download Predictions CSV
            out_csv = io.StringIO()
            results_df.to_csv(out_csv, index=False)
            st.download_button(
                label="📥 Download Diagnosed Results CSV",
                data=out_csv.getvalue(),
                file_name="machineguard_predictions.csv",
                mime="text/csv",
                type="primary"
            )

            if len(invalid_df) > 0:
                st.warning(f"⚠️ {len(invalid_df)} records were skipped due to schema/range validation errors:")
                st.dataframe(invalid_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing uploaded batch: {e}")
