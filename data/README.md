# AI4I 2020 Predictive Maintenance Dataset

## Dataset Provenance & Overview
- **Name**: AI4I 2020 Predictive Maintenance Dataset
- **Repository**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Authors**: Stephan Matzka, School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin, Germany
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Observations**: 10,000 synthetic instances reflecting real predictive maintenance scenarios
- **Features**: 14 attributes (identifiers, operating conditions, binary failure target, failure modes)

## Column Schema
| Column Name | Type | Role | Description |
|:---|:---|:---|:---|
| `UDI` | Integer (1 - 10000) | **Excluded Identifier** | Unique row index. Excluded from models to prevent spurious memorization. |
| `Product ID` | String (L/M/H + serial) | **Excluded Identifier** | Variant identifier (e.g. L47181). Excluded from default models to prevent target leakage. |
| `Type` | Categorical ('L', 'M', 'H') | **Predictor Feature** | Product quality variant: Low (50%), Medium (30%), High (20%). Encoded via One-Hot encoding. |
| `Air temperature [K]` | Float | **Predictor Feature** | Ambient air temperature in Kelvin (generated via random walk around 300 K). |
| `Process temperature [K]` | Float | **Predictor Feature** | Temperature inside the manufacturing process in Kelvin (~Air temp + 10 K). |
| `Rotational speed [rpm]` | Integer | **Predictor Feature** | Spindle speed calculated from 2860 W motor power with overlaid noise. |
| `Torque [Nm]` | Float | **Predictor Feature** | Torque values normally distributed around 40 Nm (std: 10 Nm). |
| `Tool wear [min]` | Integer | **Predictor Feature** | Tool usage time in minutes; wears down during machining (0 - 250+ min). |
| `Machine failure` | Binary (0 / 1) | **Target Variable** | 1 if machine failed, 0 if normal operation. Observed rate ~3.39% (339 / 10000). |
| `TWF` | Binary (0 / 1) | **Target Subtype (Excluded)** | Tool Wear Failure mode. Excluded from predictors (causes direct target leakage). |
| `HDF` | Binary (0 / 1) | **Target Subtype (Excluded)** | Heat Dissipation Failure mode. Excluded from predictors (causes target leakage). |
| `PWF` | Binary (0 / 1) | **Target Subtype (Excluded)** | Power Failure mode. Excluded from predictors (causes target leakage). |
| `OSF` | Binary (0 / 1) | **Target Subtype (Excluded)** | Overstrain Failure mode. Excluded from predictors (causes target leakage). |
| `RNF` | Binary (0 / 1) | **Target Subtype (Excluded)** | Random Failure mode. Excluded from predictors (causes target leakage). |

## Leakage Prevention Rules
1. `UDI` and `Product ID` are strictly removed before model training.
2. `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` describe post-failure categorization and must **never** be used as inputs to predict `Machine failure`.
3. Preprocessing (scalers, encoders) is fit **strictly on training folds** within stratified cross-validation.
