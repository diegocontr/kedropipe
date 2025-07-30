"""
Global Analysis Features Summary
===============================

The following global analysis plots have been added to the model validation pipeline:

1. **Prediction Distribution Plots**
   - Shows histogram distributions of predictions from new model (and old model if available)
   - Helps understand the range and shape of model predictions
   - Useful for detecting prediction drift between models

2. **Calibration Plots** 
   - Bins predictions by percentiles and shows mean predicted vs mean observed values
   - Separate plots for new model and old model (if available)
   - Includes identity line for perfect calibration reference
   - Critical for understanding if model probabilities are well-calibrated

3. **ROC Curves**
   - Shows Receiver Operating Characteristic curves for model discrimination
   - Combined plot showing both new and old model performance
   - Includes AUC (Area Under Curve) scores for quantitative comparison
   - Automatically handles non-binary targets by median split

All plots are:
- Saved as MLflow artifacts under "global_analysis_plots/" directory
- Styled consistently with the existing model monitoring plots
- Generated automatically as part of the validation pipeline
- Include proper error handling and logging

Integration:
- Added to `nodes_global.py` for modular organization
- Integrated into `generate_validation_reports()` function
- No changes needed to pipeline structure or parameters
- Maintains backward compatibility with existing functionality

The plots provide comprehensive model evaluation beyond segmented analysis,
giving a holistic view of model performance and calibration.
"""
