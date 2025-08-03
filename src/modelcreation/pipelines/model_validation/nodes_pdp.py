"""Partial Dependency Plot (PDP) analysis for model validation pipeline."""

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence


logger = logging.getLogger(__name__)


def calculate_partial_dependence(
    model: Any,
    X_data: pd.DataFrame,
    feature_names: List[str],
    n_points: int = 50
) -> Dict[str, Dict[str, np.ndarray]]:
    """Calculate partial dependence for each feature.
    
    Args:
        model: Trained model with predict method
        X_data: Feature data
        feature_names: List of feature names to analyze
        n_points: Number of points for PDP grid
        
    Returns:
        Dictionary with PDP values for each feature
    """
    logger.info(f"Calculating partial dependence for features: {feature_names}")
    
    pdp_results = {}
    
    for feature in feature_names:
        if feature not in X_data.columns:
            logger.warning(f"Feature {feature} not found in data, skipping")
            continue
            
        try:
            # Calculate partial dependence
            feature_idx = X_data.columns.get_loc(feature)
            
            # Convert to float to avoid sklearn warnings
            X_float = X_data.astype(float)
            
            # Use sklearn's partial_dependence function
            pd_result = partial_dependence(
                model, 
                X_float, 
                features=[feature_idx],
                grid_resolution=n_points,
                kind='average'
            )
            
            pdp_results[feature] = {
                'values': pd_result['grid_values'][0],  # Grid values (first element in list)
                'average': pd_result['average'][0],     # PDP values (first row)
                'feature_name': feature
            }
            
            logger.info(f"Calculated PDP for {feature}: {len(pd_result['grid_values'][0])} points")
            
        except Exception as e:
            logger.error(f"Error calculating PDP for {feature}: {e}")
            continue
    
    logger.info(f"Successfully calculated PDP for {len(pdp_results)} features")
    return pdp_results


def create_segment_pdp_plots(
    model: Any,
    validation_dataset: pd.DataFrame,
    segmented_metrics: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any]
) -> Dict[str, plt.Figure]:
    """Create PDP plots for each segment using the same features as the segmented analysis.
    
    Args:
        model: Trained model
        validation_dataset: Full validation dataset
        segmented_metrics: Segmented metrics (to get segment definitions)
        parameters: Validation parameters
        
    Returns:
        Dictionary of figures for each segment
    """
    logger.info("Creating segment-based PDP plots")
    
    # Get feature binning configuration to recreate segments
    feature_binning = parameters.get('feature_binning', {})
    default_bins = feature_binning.get('default_bins', 5)
    
    # Get feature columns (same logic as in nodes_by_feature.py)
    exclude_cols = {
        parameters.get('target_column', 'target_B'),
        parameters.get('prediction_column', 'prediction_new'), 
        parameters.get('weight_column', 'weight'),
        parameters.get('old_model_column', 'prediction_A')
    }
    
    # Add any prediction columns that might exist
    prediction_cols = [col for col in validation_dataset.columns if col.startswith('prediction')]
    exclude_cols.update(prediction_cols)
    
    # Get feature columns
    feature_cols = [col for col in validation_dataset.columns 
                   if col not in exclude_cols and validation_dataset[col].dtype in ['int64', 'float64']]
    
    logger.info(f"Creating PDP plots for features: {feature_cols}")
    
    pdp_figures = {}
    
    for feature in feature_cols:
        try:
            # Get feature configuration
            feature_config = feature_binning.get(feature, {})
            
            # Create segments for this feature
            if 'bins' in feature_config and 'labels' in feature_config:
                bins = feature_config['bins']
                labels = feature_config['labels']
                segment_name = f"{feature}_group"
            elif 'bins' in feature_config and isinstance(feature_config['bins'], list):
                bins = feature_config['bins']
                labels = None
                segment_name = f"{feature}_group"
            elif 'bins' in feature_config and isinstance(feature_config['bins'], int):
                bins = feature_config['bins']
                labels = None
                segment_name = f"{feature}_level"
            else:
                bins = default_bins
                labels = None
                segment_name = f"{feature}_level"
            
            # Create segments
            if labels:
                validation_dataset[segment_name] = pd.cut(
                    validation_dataset[feature], 
                    bins=bins, 
                    labels=labels, 
                    include_lowest=True
                )
            else:
                validation_dataset[segment_name] = pd.qcut(
                    validation_dataset[feature], 
                    q=bins if isinstance(bins, int) else len(bins)-1, 
                    duplicates='drop'
                )
            
            # Create PDP plot for this feature across segments
            fig = create_feature_pdp_plot(
                model=model,
                data=validation_dataset,
                feature_name=feature,
                segment_column=segment_name,
                parameters=parameters,
                n_points=50
            )
            
            pdp_figures[f"pdp_{segment_name}"] = fig
            
        except Exception as e:
            logger.error(f"Error creating PDP plot for {feature}: {e}")
            continue
    
    logger.info(f"Created {len(pdp_figures)} PDP plots")
    return pdp_figures


def create_feature_pdp_plot(
    model: Any,
    data: pd.DataFrame,
    feature_name: str,
    segment_column: str,
    parameters: Dict[str, Any],
    n_points: int = 50
) -> plt.Figure:
    """Create a PDP plot for a single feature across different segments.
    
    Args:
        model: Trained model
        data: Validation dataset with segments
        feature_name: Name of the feature to analyze
        segment_column: Name of the segment column
        parameters: Validation parameters
        n_points: Number of points for PDP grid
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique segments
    segments = data[segment_column].dropna().unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    
    # Feature range for PDP grid
    feature_min = data[feature_name].min()
    feature_max = data[feature_name].max()
    feature_range = np.linspace(feature_min, feature_max, n_points)
    
    for i, segment in enumerate(segments):
        try:
            # Get data for this segment
            segment_data = data[data[segment_column] == segment]
            
            if len(segment_data) < 10:  # Skip segments with too few samples
                logger.warning(f"Skipping segment {segment} - too few samples: {len(segment_data)}")
                continue
            
            # Get features for PDP calculation (exclude target and prediction columns)
            feature_cols = [col for col in segment_data.columns 
                           if col not in [segment_column, parameters.get('target_column', 'target_B')] 
                           and not col.startswith('prediction')
                           and segment_data[col].dtype in ['int64', 'float64']]
            
            X_segment = segment_data[feature_cols]
            
            # Calculate PDP for this segment
            pdp_results = calculate_partial_dependence(
                model=model,
                X_data=X_segment,
                feature_names=[feature_name],
                n_points=n_points
            )
            
            if feature_name in pdp_results:
                pdp_data = pdp_results[feature_name]
                ax.plot(
                    pdp_data['values'], 
                    pdp_data['average'],
                    label=f"{segment} (n={len(segment_data)})",
                    color=colors[i],
                    linewidth=2,
                    marker='o',
                    markersize=3
                )
            
        except Exception as e:
            logger.error(f"Error calculating PDP for segment {segment}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel(f"{feature_name}")
    ax.set_ylabel("Partial Dependence")
    ax.set_title(f"Partial Dependence Plot: {feature_name} by Segments")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add overall PDP line for comparison
    try:
        feature_cols = [col for col in data.columns 
                       if col not in [segment_column, parameters.get('target_column', 'target_B')] 
                       and not col.startswith('prediction')
                       and data[col].dtype in ['int64', 'float64']]
        
        X_all = data[feature_cols].dropna()
        overall_pdp = calculate_partial_dependence(
            model=model,
            X_data=X_all,
            feature_names=[feature_name],
            n_points=n_points
        )
        
        if feature_name in overall_pdp:
            pdp_data = overall_pdp[feature_name]
            ax.plot(
                pdp_data['values'], 
                pdp_data['average'],
                label="Overall",
                color='black',
                linewidth=3,
                linestyle='--',
                alpha=0.7
            )
    except Exception as e:
        logger.warning(f"Could not add overall PDP line: {e}")
    
    plt.tight_layout()
    return fig


def generate_pdp_analysis(
    trained_model: Any,
    model_metrics: str,
    validation_dataset: pd.DataFrame,
    segmented_metrics: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any]
) -> Dict[str, plt.Figure]:
    """Generate PDP analysis using the same segments as feature analysis.
    
    Args:
        trained_model: Trained model object
        model_metrics: Model metrics (for MLflow model loading if needed)
        validation_dataset: Validation dataset
        segmented_metrics: Segmented metrics (to get segment definitions)
        parameters: Validation parameters
        
    Returns:
        Dictionary of PDP figures
    """
    logger.info("Generating PDP analysis for segmented features")
    
    # Use the trained model directly for PDP analysis
    # (PDP works better with the original model object than MLflow-loaded model)
    model = trained_model
    
    # Create PDP plots for each feature
    pdp_figures = create_segment_pdp_plots(
        model=model,
        validation_dataset=validation_dataset,
        segmented_metrics=segmented_metrics,
        parameters=parameters
    )
    
    logger.info(f"Generated {len(pdp_figures)} PDP analysis plots")
    return pdp_figures
