# Generic Feature Analysis Configuration

The model validation pipeline now automatically analyzes **all numeric features** in your dataset and supports flexible binning configurations.

## How It Works

The pipeline automatically:
1. **Detects all numeric features** (int64, float64) in your dataset
2. **Excludes non-feature columns** (target, predictions, weights)
3. **Creates segments** for each feature using configurable binning strategies
4. **Generates validation reports** for each feature segment

## Binning Configuration

Configure feature binning in `conf/base/parameters.yml` under `model_validation.feature_binning`:

### Default Binning
```yaml
model_validation:
  feature_binning:
    default_bins: 5  # Applied to all features unless specified
```

### Custom Binning Examples

```yaml
model_validation:
  feature_binning:
    default_bins: 5  # Fallback for unspecified features
    
    # Custom bin edges with labels
    age:
      bins: [18, 30, 45, 60, 75]
      labels: ["18-29", "30-44", "45-59", "60+"]
    
    # Custom bin edges (auto-generated labels)
    credit_score:
      bins: [300, 550, 650, 750, 850]
    
    # Number of quantile bins
    income:
      bins: 4  # Creates 4 quantile-based bins
    
    # Large number of bins for detailed analysis
    transaction_amount:
      bins: 10  # Creates 10 quantile-based bins
```

## Supported Binning Types

### 1. Quantile Bins (Integer)
```yaml
feature_name:
  bins: 5  # Creates 5 equal-frequency bins
```
- **Use case**: Balanced sample sizes across segments
- **Naming**: `{feature}_level` (e.g., `income_level`)

### 2. Custom Edges (List)
```yaml
feature_name:
  bins: [0, 25, 50, 75, 100]  # Custom breakpoints
```
- **Use case**: Business-meaningful thresholds
- **Naming**: `{feature}_group` (e.g., `age_group`)

### 3. Custom Edges with Labels
```yaml
feature_name:
  bins: [300, 550, 650, 750, 850]
  labels: ["Poor", "Fair", "Good", "Excellent"]
```
- **Use case**: Interpretable business categories
- **Naming**: `{feature}_group` with custom labels

## Output

For each feature, the pipeline generates:
- **Segmented metrics**: S/PP ratios, Gini coefficients, exposure by segment
- **Validation plots**: Combined curves showing model performance across segments
- **MLflow artifacts**: Plots saved as artifacts for tracking and comparison

## Example: Adding New Features

If you add new features to your model (e.g., `customer_tenure`, `monthly_spend`):

1. **No configuration needed** - they'll use `default_bins: 5`
2. **Custom binning** (optional):
```yaml
customer_tenure:
  bins: [0, 6, 12, 24, 36]  # months
  labels: ["New", "Short", "Medium", "Long"]

monthly_spend:
  bins: 8  # 8 quantile bins for detailed analysis
```

3. **Pipeline runs automatically** - creates segments and reports for all features

## Benefits

- **Scalable**: Works with any number of features
- **Flexible**: Mix different binning strategies per feature
- **Consistent**: Same analysis framework across all features
- **Automated**: No code changes needed for new features
- **Traceable**: All results tracked in MLflow with artifacts
