"""Model training nodes for CatBoost models.

This module provides model training functionality using CatBoost with Poisson loss,
specifically designed for count/regression tasks.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.catboost

logger = logging.getLogger(__name__)


def setup_mlflow_experiment(params: Dict[str, Any]) -> str:
    """Setup MLflow experiment based on parameters.
    
    Args:
        params: Model training parameters containing MLflow configuration
        
    Returns:
        Experiment ID as string
    """
    experiment_id = params.get("mlflow_experiment_id")
    experiment_name = params.get("mlflow_experiment_name")
    
    if experiment_id is not None:
        # Use provided experiment ID
        try:
            experiment = mlflow.get_experiment(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment with ID {experiment_id} does not exist")
            logger.info(f"Using existing experiment: ID={experiment_id}, Name={experiment.name}")
            mlflow.set_experiment(experiment_id=experiment_id)
            return str(experiment_id)
        except Exception as e:
            logger.warning(f"Failed to use experiment ID {experiment_id}: {e}")
            # Fall through to create new experiment
    
    # Create new experiment with time-based name if no valid experiment_id
    if experiment_name is None:
        # Generate time-based experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"kedropipe_experiment_{timestamp}"
    
    try:
        # Try to get existing experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: ID={experiment_id}, Name={experiment_name}")
        else:
            # Create new experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: ID={experiment_id}, Name={experiment_name}")
        
        mlflow.set_experiment(experiment_id=experiment_id)
        return str(experiment_id)
        
    except Exception as e:
        logger.error(f"Failed to setup MLflow experiment: {e}")
        # Use default experiment as fallback
        logger.warning("Falling back to default experiment (ID=0)")
        mlflow.set_experiment(experiment_id="0")
        return "0"


def split_data(
    prepared_data: pd.DataFrame,
    params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split prepared full DataFrame into train/test full DataFrames.

    Returns train_dataset, test_dataset (each including features + target column).
    Target is assumed to be the last column (consistent with previous logic).
    """
    logger.info("Starting data splitting (full DataFrames) for model training")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    train_df, test_df = train_test_split(
        prepared_data, test_size=test_size, random_state=random_state
    )
    logger.info("Full train rows=%d, test rows=%d", len(train_df), len(test_df))
    return train_df, test_df


def separate_features_target(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    target_column: str,
    feature_columns_str: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Split full train/test DataFrames into feature matrices and target CSV strings.

    Only the originally declared feature columns (feature_columns_str) are used for X;
    any passthrough/reference columns (e.g. old model predictions) remain in train/test
    datasets for later analyses but are excluded from model training.
    """
    if target_column not in train_dataset.columns:
        raise ValueError(f"target_column '{target_column}' not in train_dataset")
    if target_column not in test_dataset.columns:
        raise ValueError(f"target_column '{target_column}' not in test_dataset")

    declared_features = [c for c in feature_columns_str.split(',') if c]
    feature_cols = [c for c in declared_features if c in train_dataset.columns]
    X_train = train_dataset[feature_cols].copy()
    X_test = test_dataset[feature_cols].copy()

    y_train = train_dataset[target_column]
    y_test = test_dataset[target_column]

    y_train_str = y_train.to_csv(index=False)
    y_test_str = y_test.to_csv(index=False)
    logger.info(
        "Separated features/target: features=%d target=%s", len(feature_cols), target_column
    )
    return X_train, X_test, y_train_str, y_test_str


def train_catboost_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train_str: str,
    y_test_str: str,
    feature_columns_str: str,
    params: Dict
) -> Tuple[str, str]:
    """Train a CatBoost model with Poisson loss and log to MLflow.
    
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
    
    # Setup MLflow experiment
    experiment_id = setup_mlflow_experiment(params)
    logger.info(f"Using MLflow experiment ID: {experiment_id}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="catboost_training"):
        # Log parameters
        mlflow.log_param("loss_function", loss_function)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("depth", depth)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
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
        
        # Log metrics to MLflow
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("best_iteration", model.best_iteration_)
        
        # Get feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Log feature importance as metrics
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log the model to MLflow
        mlflow.catboost.log_model(
            model, 
            "catboost_model"
        )
        
        # Store the run ID for later retrieval
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Compile metrics
        metrics = {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "best_iteration": model.best_iteration_,
            "feature_importance": feature_importance,
            "mlflow_run_id": run_id,
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
        logger.info(f"MLflow run ID: {run_id}")
        
        logger.info("Feature importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feature}: {importance:.4f}")
        
        # Return the actual model object and metrics as JSON string
        import json
        metrics_str = json.dumps(metrics, indent=2)
        
        return model, metrics_str


