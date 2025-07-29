"""Data preparation nodes for flexible model training.

This module provides data preparation functionality that can work with any target column.
Simply change the target_column parameter in conf/base/parameters.yml to train on 
different targets (e.g., target_A, target_B, target_C, or any custom column).
"""

import logging
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_model_data(
    raw_data: pd.DataFrame, 
    params: Dict
) -> Tuple[pd.DataFrame, str]:
    """Prepare data for model training with configurable target.
    
    Args:
        raw_data: Raw segmentation data from parquet file
        params: Parameters from conf/base/parameters.yml containing:
            - feature_columns: List of feature column names to use
            - target_column: Name of the target column to predict
        
    Returns:
        Tuple of:
        - Prepared dataframe with features and target
        - Feature column names as comma-separated string
    """
    logger.info("Starting data preparation for model training")
    
    # Get parameters
    feature_columns = params.get("feature_columns", ["age", "income", "credit_score"])
    target_column = params.get("target_column")
    
    if not target_column:
        raise ValueError("target_column must be specified in parameters")
    
    logger.info(f"Using features: {feature_columns}")
    logger.info(f"Using target: {target_column}")
    
    # Validate that all required columns exist in the data
    missing_features = [col for col in feature_columns if col not in raw_data.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in data: {missing_features}")
    
    if target_column not in raw_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Select features and target
    columns_to_keep = [*feature_columns, target_column]
    prepared_data = raw_data[columns_to_keep].copy()
    
    # Basic data quality checks
    logger.info(f"Original data shape: {raw_data.shape}")
    logger.info(f"Prepared data shape: {prepared_data.shape}")
    
    # Check for missing values
    missing_counts = prepared_data.isna().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
    
    # Remove rows with missing values if any
    initial_rows = len(prepared_data)
    prepared_data = prepared_data.dropna()
    final_rows = len(prepared_data)
    
    if initial_rows != final_rows:
        logger.info(f"Removed {initial_rows - final_rows} rows with missing values")
    
    # Basic statistics
    logger.info("Feature statistics:")
    for col in feature_columns:
        logger.info(f"  {col}: mean={prepared_data[col].mean():.2f}, "
                   f"std={prepared_data[col].std():.2f}, "
                   f"min={prepared_data[col].min():.2f}, "
                   f"max={prepared_data[col].max():.2f}")
    
    logger.info("Target statistics:")
    logger.info(f"  {target_column}: mean={prepared_data[target_column].mean():.2f}, "
               f"std={prepared_data[target_column].std():.2f}, "
               f"min={prepared_data[target_column].min()}, "
               f"max={prepared_data[target_column].max()}")
    
    logger.info("Data preparation completed successfully")
    
    # Convert feature columns list to comma-separated string for text storage
    feature_columns_str = ",".join(feature_columns)
    
    return prepared_data, feature_columns_str
