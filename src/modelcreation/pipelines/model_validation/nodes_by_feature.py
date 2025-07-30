"""Model monitoring and feature analysis functions for model validation pipeline."""

import logging
import os
import tempfile
from typing import Any, Dict, Tuple

import pandas as pd

from model_monitoring import (
    AnalysisDataBuilder,
    SegmentCustom,
    calculate_statistics,
)


logger = logging.getLogger(__name__)


def create_feature_segments(validation_dataset: pd.DataFrame, parameters: Dict[str, Any]) -> list:
    """Create segments for all feature columns based on configuration.
    
    Args:
        validation_dataset: Validation dataset with features
        parameters: Model validation parameters including feature_binning config
        
    Returns:
        List of SegmentCustom objects for all features
    """
    segments = []
    
    # Get binning configuration
    feature_binning = parameters.get('feature_binning', {})
    default_bins = feature_binning.get('default_bins', 5)
    
    # Get feature columns (exclude target, prediction, weight columns)
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
    
    logger.info(f"Creating segments for features: {feature_cols}")
    
    for feature in feature_cols:
        # Get custom binning for this feature, or use default
        feature_config = feature_binning.get(feature, {})
        
        # Check if custom bins and labels are provided
        if 'bins' in feature_config and 'labels' in feature_config:
            # Custom bins with labels
            bins = feature_config['bins']
            labels = feature_config['labels']
            segment_name = f"{feature}_group"
            
            logger.info(f"Using custom binning for {feature}: bins={bins}, labels={labels}")
            
        elif 'bins' in feature_config and isinstance(feature_config['bins'], list):
            # Custom bin edges without labels
            bins = feature_config['bins']
            labels = None  # Will be auto-generated
            segment_name = f"{feature}_group"
            
            logger.info(f"Using custom bin edges for {feature}: bins={bins}")
            
        elif 'bins' in feature_config and isinstance(feature_config['bins'], int):
            # Number of quantile bins
            bins = feature_config['bins']
            labels = None
            segment_name = f"{feature}_level"
            
            logger.info(f"Using {bins} quantile bins for {feature}")
            
        else:
            # Use default binning
            bins = default_bins
            labels = None
            segment_name = f"{feature}_level"
            
            logger.info(f"Using default {bins} quantile bins for {feature}")
        
        # Create segment
        if labels:
            segment = SegmentCustom(
                seg_col=feature,
                seg_name=segment_name,
                bins=bins,
                bin_labels=labels
            )
        else:
            segment = SegmentCustom(
                seg_col=feature,
                seg_name=segment_name, 
                bins=bins
            )
            
        segments.append(segment)
    
    logger.info(f"Created {len(segments)} feature segments")
    return segments


def calculate_validation_metrics(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Calculate validation metrics using the model monitoring framework.
    
    Args:
        validation_dataset: Combined validation dataset
        parameters: Validation parameters
        
    Returns:
        Tuple of (overall_metrics, segmented_metrics)
    """
    logger.info("Calculating validation metrics")
    
    # Get column names from parameters
    target_col = parameters.get('target_column', 'target_B')
    pred_col = parameters.get('prediction_column', 'prediction_new')
    weight_col = parameters.get('weight_column', 'weight')
    
    # Ensure columns exist
    if target_col not in validation_dataset.columns:
        available_targets = [col for col in validation_dataset.columns if col.startswith('target_')]
        if available_targets:
            target_col = available_targets[0]
            logger.info(f"Using detected target column: {target_col}")
        else:
            raise ValueError(f"Target column not found. Available columns: {list(validation_dataset.columns)}")
    
    # Create temporary parquet file for AnalysisDataBuilder
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
        temp_path = temp_file.name
        validation_dataset.to_parquet(temp_path)
    
    try:
        # Initialize AnalysisDataBuilder
        extra_cols = [weight_col, pred_col, target_col]
        old_model_col = parameters.get('old_model_column')
        if old_model_col and old_model_col in validation_dataset.columns:
            extra_cols.append(old_model_col)
            
        analysis = AnalysisDataBuilder(
            data=temp_path,
            extra_cols=extra_cols
        )
        
        # Define segments using the generic function
        segments = create_feature_segments(validation_dataset, parameters)
        
        # Add segments to analysis
        for segment in segments:
            analysis.add_segment(segment)
        
        # Load and process data
        analysis.load_data()
        analysis.apply_treatments()
        analysis.apply_segments()
        
        # Define statistics to calculate
        func_dict = {
            "aggregations": {
                pred_col: ("weighted_mean", [pred_col, weight_col]),
                target_col: ("observed_charge", [target_col, weight_col]),
                "gini": ("gini", [target_col, pred_col, weight_col]),
                "exposure(k)": (lambda df, e: df[e].sum() / 1000, [weight_col]),
            },
            "post_aggregations": {
                "S/PP": ("division", [target_col, pred_col]),
            },
        }
        
        # Add old model metrics if available
        if old_model_col and old_model_col in validation_dataset.columns:
            func_dict["aggregations"][old_model_col] = ("weighted_mean", [old_model_col, weight_col])
            func_dict["post_aggregations"]["S/PP_old"] = ("division", [target_col, old_model_col])
            func_dict["aggregations"]["gini_old"] = ("gini", [target_col, old_model_col, weight_col])
            logger.info(f"Added old model metrics for column: {old_model_col}")
        
        # Calculate statistics
        dict_stats, agg_stats = calculate_statistics(
            analysis, 
            func_dict, 
            bootstrap=parameters.get('bootstrap', False)
        )
        
        # Create comparison flag for plotting
        has_old_model = old_model_col is not None and old_model_col in validation_dataset.columns
        
        logger.info("Validation metrics calculated successfully")
        logger.info(f"Overall metrics: {agg_stats.to_dict()}")
        logger.info(f"Segmented metrics for {len(dict_stats)} segments")
        logger.info(f"Old model comparison available: {has_old_model}")
        
        # Convert Series to DataFrame for saving
        agg_stats_df = agg_stats.to_frame(name='value')
        
        # Add comparison flag to the metrics for use in plotting
        agg_stats_df.loc['has_old_model'] = has_old_model
        
        return agg_stats_df, dict_stats
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
