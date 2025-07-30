"""
Code Refactoring Summary: generate_validation_reports Function
============================================================

## Problem
The `generate_validation_reports` function was too long (~150+ lines) and had multiple responsibilities:
- Setting up MLflow configuration
- Generating segmented validation plots
- Generating global analysis plots  
- Saving plots as artifacts
- Creating summary files

## Solution
Split the monolithic function into smaller, focused helper functions:

### New Helper Functions:

1. **`_save_plot_as_artifact()`**
   - Purpose: Handle the common pattern of saving plots as MLflow artifacts or locally
   - Responsibility: File I/O and artifact management
   - Reused by both segment and global plot generation

2. **`_generate_segment_reports()`**
   - Purpose: Generate feature-segmented validation plots
   - Responsibility: Model monitoring plots for age, income, credit_score segments
   - Uses existing model_monitoring plotting framework

3. **`_generate_global_reports()`** 
   - Purpose: Generate global analysis plots (distribution, calibration, ROC)
   - Responsibility: Non-segmented model performance analysis
   - Uses new global analysis framework

4. **`_generate_summary_file()`**
   - Purpose: Create text summary of validation results
   - Responsibility: Report generation and metrics summarization

### Refactored Main Function:
The `generate_validation_reports()` function is now much cleaner (~50 lines):
- Handles setup and configuration
- Orchestrates the different report types
- Maintains the same public interface
- Returns the same results

## Benefits:

### Code Quality:
- **Single Responsibility**: Each function has one clear purpose
- **Readability**: Much easier to understand and navigate
- **Maintainability**: Changes to specific report types are isolated
- **Testability**: Individual functions can be tested separately

### Performance:
- **Memory Management**: Better figure cleanup in helper functions
- **Error Isolation**: Errors in one report type don't affect others
- **Logging**: More granular logging for debugging

### Extensibility:
- **New Plot Types**: Easy to add new global or segment plot types
- **Different Backends**: Easy to modify artifact storage logic
- **Custom Reports**: Simple to add new report formats

## File Structure:
```
nodes.py (main orchestration functions)
├── generate_validation_reports()     # Main entry point
├── _save_plot_as_artifact()          # Common utility
├── _generate_segment_reports()       # Feature-based plots  
├── _generate_global_reports()        # Global analysis plots
└── _generate_summary_file()          # Text summaries

nodes_by_feature.py (feature analysis)
├── create_feature_segments()
└── calculate_validation_metrics()

nodes_global.py (global analysis)  
├── create_prediction_distribution_plots()
├── create_calibration_plots()
├── create_roc_curves()
└── generate_global_analysis_plots()
```

## Backward Compatibility:
- ✅ Same function signature
- ✅ Same return format
- ✅ Same MLflow artifacts
- ✅ Same file outputs
- ✅ No pipeline changes needed

The refactoring improves code organization without breaking any existing functionality.
"""
