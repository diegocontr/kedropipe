# Model Monitoring

A Python package for monitoring and analyzing the performance of predictive models. It helps in understanding model behavior by calculating Key Performance Indicators (KPIs) across various data segments and provides tools for comparing different models.

## Core Components

- `AnalysisDataBuilder`: The central class that manages data loading, preprocessing (treatments), and segmentation.
- `SegmentCategorical` & `SegmentCustom`: Classes to define how to slice your data into segments for analysis. For example, by region, or by binning a continuous variable like age.
- `calculate_statistics`: A function that computes aggregate statistics and KPIs on the segmented data.
- `plot_segment_statistics`: A powerful plotting function to visualize the calculated KPIs across different segments, allowing for multi-panel reports.
- **Treatments**: A concept for applying data transformations, like scaling predictions or creating aggregate features.

## Usage

Here is a typical workflow for using the `predlab` package to analyze model KPIs.

### 1. Prepare your data

First, you need a dataset, typically in a pandas DataFrame, containing your features, predictions, and target variables. 

### 2. Configure and Run Analysis

Use `AnalysisDataBuilder` to set up the monitoring pipeline.

```python
from predlab.monitoring import AnalysisDataBuilder
from predlab.segmentation import SegmentCategorical, SegmentCustom

# Initialize AnalysisDataBuilder
lr_analysis = AnalysisDataBuilder()

# Define segmentation strategies
segments = [
    SegmentCustom(seg_col="age", seg_name="age_group", bins=[18, 30, 45, 60, 75]),
    SegmentCategorical(seg_col="region", seg_name="region_segment"),
]

for s in segments:
    lr_analysis.add_segment(s)

# Load data and apply segments
lr_analysis.load_data(data=df)
lr_analysis.apply_segments()
```

### 3. Calculate KPIs

Define the statistics you want to compute and use `calculate_statistics`.

```python
from predlab.agg_stats import calculate_statistics

# Define statistics to calculate
func_dict = {
    "aggregations": {
        "prediction_A": ("weighted_mean", ["prediction_A", "weight", "weight"]),
        "target_A": ("observed_charge", ["target_A", "weight", "weight"]),
        "LR": ("lr", ["target_A", "weight", "weight", "market_premium"]),
        "exposure": (lambda df, e: df[e].sum(), ["weight"]),
    },
    "post_aggregations": {
        "S/PP": ("division", ["target_A", "prediction_A"]),
    },
}

# Calculate statistics
dict_stats, agg_stats = calculate_statistics(lr_analysis, func_dict)
```

### 4. Visualize the Results

Use `plot_segment_statistics` to create a visual report of your KPIs for a specific segment.

```python
from predlab.plotting import plot_segment_statistics, set_plot_theme

# Set a plot theme
set_plot_theme()

# Define the panels for the plot
report_panels = [
    {
        "title": "Prediction vs. Target",
        "type": "pred_vs_target",
        "pred_col": ["prediction_A"],
        "target_col": "target_A",
        "plot_type": "line",
    },
    {
        "title": "S/PP (Observed/Predicted Ratio)",
        "type": "spp",
        "spp_col": ["S/PP"],
        "plot_type": "line",
    },
    {
        "title": "Loss Ratio",
        "type": "loss_ratios",
        "lr_col": "LR",
        "plot_type": "line",
    },
    {
        "title": "Exposure",
        "type": "exposure",
        "metric_col": "exposure",
        "plot_type": "bar",
    },
]

# Generate plot for the 'age_group' segment
stats_df = dict_stats["age_group"]
plot_segment_statistics(stats_df, panel_configs=report_panels, agg_stats=agg_stats)
```
This will generate a multi-panel plot showing how the predictions, targets, loss ratios, and exposure are distributed across different age groups.
