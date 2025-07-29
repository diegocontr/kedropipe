# Data Preparation Pipeline

This pipeline prepares data for model training with any target column from the segmentation dataset.

## Overview

The data preparation pipeline:
1. Loads raw segmentation data from `data/raw/segmentation_data.parquet`
2. Selects specified feature columns and target column
3. Performs basic data quality checks
4. Removes rows with missing values
5. Outputs prepared data and feature column information

## Configuration

Configure the pipeline in `conf/base/parameters.yml`:

```yaml
data_preparation:
  feature_columns:
    - age
    - income 
    - credit_score
  target_column: target_C  # Change this to any target column
```

## Usage

### Train on different targets:

**For target_A:**
```bash
# Update parameters.yml to set target_column: target_A
kedro run --pipeline data_preparation
```

**For target_B:**
```bash  
# Update parameters.yml to set target_column: target_B
kedro run --pipeline data_preparation
```

**For target_C:**
```bash
# Update parameters.yml to set target_column: target_C  
kedro run --pipeline data_preparation
```

### Using different feature sets:

You can also modify the `feature_columns` list to include different features:

```yaml
data_preparation:
  feature_columns:
    - age
    - income
    - credit_score
    - market_premium  # Add additional features
  target_column: target_A
```

## Outputs

The pipeline produces:
- `data/processed/prepared_model_data.parquet`: Cleaned dataset with selected features and target
- `data/processed/feature_columns.txt`: Comma-separated list of feature column names

## Extending the Pipeline

To add more sophisticated data preparation steps:
1. Add new functions to `nodes.py`
2. Update the pipeline in `pipeline.py` 
3. Add any new parameters to `conf/base/parameters.yml`

Example extensions:
- Feature scaling/normalization
- Feature engineering
- Train/validation/test splits
- Outlier detection and handling
