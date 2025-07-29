# Model Validation Pipeline

This pipeline performs comprehensive validation and analysis of trained models, similar to the analysis implemented in the `01_model_validation.ipynb` notebook.

## Pipeline Overview

The model validation pipeline consists of the following nodes:

1. **Generate Model Predictions** - Creates predictions from the trained model on test data
2. **Prepare Validation Data** - Combines features, targets, predictions, and optional old model data
3. **Calculate Validation Metrics** - Computes statistics using the model monitoring framework
4. **Compare with Old Model** - Compares new model performance with an optional old model
5. **Generate Validation Reports** - Creates visualizations and summary reports

## Features

### Statistical Analysis
- **S/PP Ratios**: Observed vs Predicted ratios across segments
- **Gini Coefficients**: Model discriminatory power measurement
- **Exposure Analysis**: Volume analysis across segments
- **Segmented Analysis**: Performance breakdown by age groups, income levels, credit scores, etc.

### Model Comparison
- Compares new model with optional old model predictions
- Calculates improvement metrics for S/PP and Gini coefficients
- Provides segment-wise comparison analysis

### Reporting
- Generates publication-ready visualizations saved to `data/reporting/`
- Creates summary statistics files
- Produces multi-panel reports with customizable themes

## Configuration

Configure the pipeline through `conf/base/parameters.yml`:

```yaml
model_validation:
  # Column mappings
  target_column: target_B                    # Target column name
  prediction_column: prediction_new          # New model predictions column
  weight_column: weight                      # Weight column for aggregations
  old_model_column: prediction_A             # Optional: old model column
  
  # Analysis settings
  bootstrap: false                           # Bootstrap confidence intervals
  old_model_noise_factor: 0.1               # Noise for synthetic old model
  
  # Plotting theme
  plot_theme:
    annotation_fontsize: 14
    style: "ggplot"
    target_color: "#1E1D25"
    h_line_style: ":"
```

## Usage

### Run the full pipeline:
```bash
kedro run --pipeline model_validation
```

### Run specific nodes:
```bash
# Generate predictions only
kedro run --nodes generate_predictions_node

# Generate reports only  
kedro run --nodes generate_reports_node
```

### Run with different target columns:
```bash
# For target_A
kedro run --pipeline model_validation --params "model_validation.target_column:target_A"

# For target_C
kedro run --pipeline model_validation --params "model_validation.target_column:target_C"
```

## Outputs

The pipeline generates the following outputs:

### Data Assets
- `model_predictions`: Model predictions on test data
- `validation_dataset`: Combined validation dataset
- `validation_metrics`: Overall performance metrics
- `segmented_metrics`: Performance metrics by segments
- `model_comparison_results`: Comparison with old model (if available)

### Reports (saved to `data/reporting/`)
- `validation_report_[segment].png`: Visualization reports for each segment
- `validation_summary.txt`: Text summary of all metrics
- `validation_report_paths.json`: Paths to generated report files

## Integration with Notebook Analysis

This pipeline implements the same analysis methodology as the `01_model_validation.ipynb` notebook:

- Uses the same `model_monitoring` framework
- Applies identical segmentation strategies (age groups, income levels, etc.)
- Calculates the same metrics (S/PP, Gini, exposure)
- Generates similar visualizations with consistent theming

The key advantage is that this pipeline can be run automatically as part of your MLOps workflow, while the notebook remains useful for interactive exploration and development.

## Dependencies

- `model_monitoring`: Custom monitoring framework
- `pandas`: Data manipulation
- `numpy`: Numerical operations  
- `matplotlib/plotly`: Visualization (via model_monitoring.plotting)
- `kedro`: Pipeline orchestration
