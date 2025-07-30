"""Global analysis functions for model validation (non-segmented plots)."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


logger = logging.getLogger(__name__)


def create_prediction_distribution_plots(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any]
) -> List[plt.Figure]:
    """Create distribution plots for model predictions.
    
    Args:
        validation_dataset: Validation dataset with predictions
        parameters: Validation parameters
        
    Returns:
        List of matplotlib figures
    """
    logger.info("Creating prediction distribution plots")
    
    pred_col = parameters.get('prediction_column', 'prediction_new')
    old_model_col = parameters.get('old_model_column')
    has_old_model = old_model_col and old_model_col in validation_dataset.columns
    
    figures = []
    
    # Set style
    plt.style.use('ggplot')
    
    # Create combined distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot new model distribution
    ax.hist(validation_dataset[pred_col], bins=50, alpha=0.7, 
            label='New Model', color='#3D27B9', density=True)
    
    # Plot old model distribution if available
    if has_old_model:
        ax.hist(validation_dataset[old_model_col], bins=50, alpha=0.7,
                label='Old Model', color='#FF6B6B', density=True)
    
    ax.set_xlabel('Prediction Value')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    figures.append(fig)
    
    return figures


def create_calibration_plots(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any],
    n_bins: int = 10
) -> List[plt.Figure]:
    """Create calibration plots for model predictions.
    
    Args:
        validation_dataset: Validation dataset with predictions and targets
        parameters: Validation parameters
        n_bins: Number of bins for calibration analysis
        
    Returns:
        List of matplotlib figures
    """
    logger.info("Creating calibration plots")
    
    pred_col = parameters.get('prediction_column', 'prediction_new')
    target_col = parameters.get('target_column', 'target_B')
    old_model_col = parameters.get('old_model_column')
    has_old_model = old_model_col and old_model_col in validation_dataset.columns
    
    figures = []
    
    # Set style
    plt.style.use('ggplot')
    
    def create_calibration_data(predictions: pd.Series, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create calibration data by binning predictions into percentiles."""
        # Create bins based on prediction percentiles
        bin_edges = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        # If we have fewer unique values than bins, adjust
        if len(bin_edges) <= 2:
            logger.warning(f"Too few unique prediction values for {n_bins} bins. Using available values.")
            bin_centers = [predictions.mean()]
            bin_targets = [targets.mean()]
        else:
            # Bin the predictions
            binned = pd.cut(predictions, bins=bin_edges, include_lowest=True, duplicates='drop')
            
            # Calculate mean prediction and target for each bin
            bin_stats = pd.DataFrame({
                'prediction': predictions,
                'target': targets,
                'bin': binned
            }).groupby('bin', observed=True).agg({
                'prediction': 'mean',
                'target': 'mean'
            })
            
            bin_centers = bin_stats['prediction'].to_numpy()
            bin_targets = bin_stats['target'].to_numpy()
        
        return bin_centers, bin_targets
    
    # New model calibration plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    pred_means, target_means = create_calibration_data(
        validation_dataset[pred_col], 
        validation_dataset[target_col]
    )
    
    # Plot calibration curve
    ax.scatter(pred_means, target_means, color='#3D27B9', s=100, 
               alpha=0.7, label='New Model', edgecolors='black', linewidth=1)
    ax.plot(pred_means, target_means, color='#3D27B9', alpha=0.5, linewidth=2)
    
    # Add identity line
    min_val = min(np.min(pred_means), np.min(target_means))
    max_val = max(np.max(pred_means), np.max(target_means))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, 
            linewidth=2, label='Perfect Calibration')
    
    ax.set_xlabel('Mean Predicted Value')
    ax.set_ylabel('Mean Observed Value')
    ax.set_title('Calibration Plot - New Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make it square
    ax.set_aspect('equal', adjustable='box')
    
    figures.append(fig)
    
    # Old model calibration plot if available
    if has_old_model:
        fig_old, ax_old = plt.subplots(figsize=(8, 8))
        
        pred_means_old, target_means_old = create_calibration_data(
            validation_dataset[old_model_col], 
            validation_dataset[target_col]
        )
        
        # Plot calibration curve
        ax_old.scatter(pred_means_old, target_means_old, color='#FF6B6B', s=100,
                      alpha=0.7, label='Old Model', edgecolors='black', linewidth=1)
        ax_old.plot(pred_means_old, target_means_old, color='#FF6B6B', alpha=0.5, linewidth=2)
        
        # Add identity line
        min_val_old = min(np.min(pred_means_old), np.min(target_means_old))
        max_val_old = max(np.max(pred_means_old), np.max(target_means_old))
        ax_old.plot([min_val_old, max_val_old], [min_val_old, max_val_old], 'k--', 
                   alpha=0.7, linewidth=2, label='Perfect Calibration')
        
        ax_old.set_xlabel('Mean Predicted Value')
        ax_old.set_ylabel('Mean Observed Value')
        ax_old.set_title('Calibration Plot - Old Model')
        ax_old.legend()
        ax_old.grid(True, alpha=0.3)
        
        # Make it square
        ax_old.set_aspect('equal', adjustable='box')
        
        figures.append(fig_old)
    
    return figures


def create_roc_curves(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any]
) -> List[plt.Figure]:
    """Create ROC curves for model predictions.
    
    Args:
        validation_dataset: Validation dataset with predictions and targets
        parameters: Validation parameters
        
    Returns:
        List of matplotlib figures
    """
    logger.info("Creating ROC curves")
    
    pred_col = parameters.get('prediction_column', 'prediction_new')
    target_col = parameters.get('target_column', 'target_B')
    old_model_col = parameters.get('old_model_column')
    has_old_model = old_model_col and old_model_col in validation_dataset.columns
    
    figures = []
    
    # Set style
    plt.style.use('ggplot')
    
    # Check if target is binary for ROC analysis
    unique_targets = validation_dataset[target_col].unique()
    if len(unique_targets) > 2:
        logger.warning("ROC curves are typically for binary classification. "
                      f"Target has {len(unique_targets)} unique values. "
                      "Converting to binary using median split.")
        # Convert to binary using median
        target_binary = (validation_dataset[target_col] > validation_dataset[target_col].median()).astype(int)
    else:
        target_binary = validation_dataset[target_col].astype(int)
    
    # Create combined ROC plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # New model ROC
    fpr_new, tpr_new, _ = roc_curve(target_binary, validation_dataset[pred_col])
    auc_new = auc(fpr_new, tpr_new)
    
    ax.plot(fpr_new, tpr_new, color='#3D27B9', linewidth=2, 
            label=f'New Model (AUC = {auc_new:.3f})')
    
    # Old model ROC if available
    if has_old_model:
        fpr_old, tpr_old, _ = roc_curve(target_binary, validation_dataset[old_model_col])
        auc_old = auc(fpr_old, tpr_old)
        
        ax.plot(fpr_old, tpr_old, color='#FF6B6B', linewidth=2,
                label=f'Old Model (AUC = {auc_old:.3f})')
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make it square
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    figures.append(fig)
    
    return figures


def generate_global_analysis_plots(
    validation_dataset: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, List[plt.Figure]]:
    """Generate all global analysis plots.
    
    Args:
        validation_dataset: Validation dataset
        parameters: Validation parameters
        
    Returns:
        Dictionary with plot categories and their figures
    """
    logger.info("Generating global analysis plots")
    
    plot_results = {}
    
    try:
        # 1. Distribution plots
        dist_plots = create_prediction_distribution_plots(validation_dataset, parameters)
        plot_results['distribution'] = dist_plots
        logger.info(f"Created {len(dist_plots)} distribution plots")
        
        # 2. Calibration plots
        cal_plots = create_calibration_plots(validation_dataset, parameters)
        plot_results['calibration'] = cal_plots
        logger.info(f"Created {len(cal_plots)} calibration plots")
        
        # 3. ROC curves
        roc_plots = create_roc_curves(validation_dataset, parameters)
        plot_results['roc'] = roc_plots
        logger.info(f"Created {len(roc_plots)} ROC plots")
        
    except Exception as e:
        logger.error(f"Error generating global plots: {e}")
        raise
    
    total_plots = sum(len(plots) for plots in plot_results.values())
    logger.info(f"Generated {total_plots} global analysis plots")
    
    return plot_results
