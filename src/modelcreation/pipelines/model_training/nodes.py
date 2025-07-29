"""Model training nodes for CatBoost models.

This module provides model training functionality using CatBoost with Poisson loss,
specifically designed for count/regression tasks.
"""

import logging
from typing import Dict, Tuple

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(
    prepared_data: pd.DataFrame, 
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Split prepared data into train and test sets.
    
    Args:
        prepared_data: Prepared dataframe with features and target
        params: Parameters from conf/base/parameters.yml containing:
            - test_size: Fraction of data to use for testing
            - random_state: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - X_train: Training features
        - X_test: Test features  
        - y_train: Training target as CSV string
        - y_test: Test target as CSV string
    """
    logger.info("Starting data splitting for model training")
    
    # Get parameters
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    
    logger.info(f"Test size: {test_size}")
    logger.info(f"Random state: {random_state}")
    
    # Get target column name (last column should be target)
    target_col = prepared_data.columns[-1]
    feature_cols = prepared_data.columns[:-1].tolist()
    
    logger.info(f"Target column: {target_col}")
    logger.info(f"Feature columns: {feature_cols}")
    
    # Split features and target
    X = prepared_data[feature_cols]
    y = prepared_data[target_col]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Training target distribution: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    logger.info(f"Test target distribution: mean={y_test.mean():.3f}, std={y_test.std():.3f}")
    
    # Convert target series to CSV string for text storage
    y_train_str = y_train.to_csv(index=False)
    y_test_str = y_test.to_csv(index=False)
    
    return X_train, X_test, y_train_str, y_test_str


def train_catboost_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train_str: str,
    y_test_str: str,
    feature_columns_str: str,
    params: Dict
) -> Tuple[str, str]:
    """Train a CatBoost model with Poisson loss.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train_str: Training target as CSV string
        y_test_str: Test target as CSV string
        feature_columns_str: Comma-separated feature column names
        params: Parameters from conf/base/parameters.yml containing:
            - loss_function: Loss function to use (default: Poisson)
            - iterations: Number of boosting iterations
            - learning_rate: Learning rate
            - depth: Tree depth
            - early_stopping_rounds: Early stopping patience
            - verbose: Verbosity level
        
    Returns:
        Tuple of:
        - Trained CatBoost model as string (serialized)
        - Dictionary of model metrics as string
    """
    logger.info("Starting CatBoost model training")
    
    # Convert text inputs back to pandas Series
    import io
    y_train = pd.read_csv(io.StringIO(y_train_str)).iloc[:, 0]
    y_test = pd.read_csv(io.StringIO(y_test_str)).iloc[:, 0]
    
    # Parse feature columns
    feature_columns = feature_columns_str.split(',')
    logger.info(f"Feature columns: {feature_columns}")
    
    # Get model parameters
    loss_function = params.get("loss_function", "Poisson")
    iterations = params.get("iterations", 1000)
    learning_rate = params.get("learning_rate", 0.1)
    depth = params.get("depth", 6)
    early_stopping_rounds = params.get("early_stopping_rounds", 100)
    verbose = params.get("verbose", 100)
    random_state = params.get("random_state", 42)
    
    logger.info("Model parameters:")
    logger.info(f"  Loss function: {loss_function}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Depth: {depth}")
    logger.info(f"  Early stopping rounds: {early_stopping_rounds}")
    
    # Initialize CatBoost model
    model = CatBoostRegressor(
        loss_function=loss_function,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        random_state=random_state,
        train_dir=None  # Disable training directory to avoid clutter
    )
    
    # Train the model
    logger.info("Training CatBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        plot=False
    )
    
    # Make predictions
    logger.info("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = (mean_squared_error(y_train, y_train_pred)) ** 0.5
    test_rmse = (mean_squared_error(y_test, y_test_pred)) ** 0.5
    
    # Get feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    
    # Compile metrics
    metrics = {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "best_iteration": model.best_iteration_,
        "feature_importance": feature_importance,
        "model_params": {
            "loss_function": loss_function,
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "best_iteration": model.best_iteration_
        }
    }
    
    logger.info("Model training completed!")
    logger.info(f"Best iteration: {model.best_iteration_}")
    logger.info(f"Training MAE: {train_mae:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Training RMSE: {train_rmse:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    
    logger.info("Feature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Serialize model and metrics as text
    import base64
    import pickle
    
    model_bytes = pickle.dumps(model)
    model_str = base64.b64encode(model_bytes).decode('utf-8')
    
    import json
    metrics_str = json.dumps(metrics, indent=2)
    
    return model_str, metrics_str
