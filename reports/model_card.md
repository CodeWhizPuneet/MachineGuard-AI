# MachineGuard AI - Model Card

## Model Details
- **System**: MachineGuard AI (Explainable Predictive Maintenance System)
- **Model Architecture**: `Gradient Boosting` with scikit-learn preprocessing pipeline
- **Version**: 1.0.0
- **Input Features**: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- **Excluded Features**: `UDI`, `Product ID` (identifiers); `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (post-failure subtypes)
- **Target**: `Machine failure` (0 = Nominal Operation, 1 = Mechanical Failure)
- **Decision Threshold**: `0.475`

## Intended Use
- **Primary Domain**: Continuous manufacturing, CNC milling, lathe equipment, and rotational machinery monitoring.
- **Intended Users**: Maintenance reliability engineers, plant supervisors, and ML practitioners.
- **Use Case**: Early warning system flagging machines with elevated failure risk for preventive inspection.

## Inappropriate Use
- **Safety-Critical Shutdown Automation**: This model is an early advisory warning tool. It should not be connected directly to emergency automated safety kill-switches without human-in-the-loop review.
- **Out-of-Distribution Hardware**: Turbines, jet engines, or machinery operating outside documented ranges should not use this model without domain fine-tuning.

## Performance & Fairness
- **Stratified Test AP**: `0.8004`
- **Stratified Test ROC-AUC**: `0.9635`
- **Test Recall**: `0.7353`
- **False Alarm Rate**: `0.36%`

## Ethical & Operational Limitations
The model reflects patterns in the AI4I 2020 dataset. In real factory environments, sensor drift, unexpected thermal surges, and mechanical anomalies outside historical distributions require ongoing recalibration and human diagnostic confirmation.
