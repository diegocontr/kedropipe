import numpy as np
import pandas as pd
from scipy import stats


def _get_agg_function(func_ref):
    """Get aggregation function from AGG_FUNCTIONS dict or return callable directly."""
    # Import here to avoid circular imports
    from .agg_stats import AGG_FUNCTIONS

    if isinstance(func_ref, str):
        return AGG_FUNCTIONS[func_ref]
    return func_ref


def _get_post_agg_function(func_ref):
    """Get post-aggregation function from POST_AGG_FUNCTIONS dict or return callable directly."""
    # Import here to avoid circular imports
    from .agg_stats import POST_AGG_FUNCTIONS

    if isinstance(func_ref, str):
        return POST_AGG_FUNCTIONS[func_ref]
    return func_ref


def apply_functions(db, segment, func_dict):
    """Inner helper function to apply a single function to a group."""
    # Import here to avoid circular imports
    from .agg_stats import apply_functions as _apply_functions

    return _apply_functions(db, segment, func_dict)


def _calculate_statistics_from_db(db, segments_names, func_dict):
    """Calculates various statistics for different segments within a DataFrame
    based on a provided dictionary of functions.

    Args:
        db (pd.DataFrame): The DataFrame containing the data.
        segments_names (list): A list of column names to be used for segmentation.
        func_dict (dict): A dictionary with 'aggregations' and optional 'post_aggregations'.
                          'aggregations' contains functions to apply group-wise.
                          'post_aggregations' contains string expressions to evaluate on the aggregated results.

    Returns:
        tuple: A tuple containing:
            - dict_dbg (dict): A dictionary where keys are segment names and values are
                               DataFrames with calculated statistics for each segment.
            - agg_stat (pd.Series): A Series with aggregated statistics across all data.
    """
    agg_funcs = func_dict.get("aggregations", {})
    post_agg_calcs = func_dict.get("post_aggregations", {})

    dict_dbg = {}

    for group in segments_names:
        df = apply_functions(db, group, agg_funcs)

        for new_col, (func, cols) in post_agg_calcs.items():
            resolved_func = _get_post_agg_function(func)
            df[new_col] = resolved_func(df, *cols)

        df["segment"] = group
        df = df.rename(columns={group: "segment_values"})
        dict_dbg[group] = df

    # Calculate aggregated statistics for the entire dataset
    agg_stat = pd.Series(
        {
            col: _get_agg_function(func)(db, *cols)
            for col, (func, cols) in agg_funcs.items()
        }
    )

    for new_col, (func, cols) in post_agg_calcs.items():
        resolved_func = _get_post_agg_function(func)
        agg_stat[new_col] = resolved_func(agg_stat, *cols)

    return dict_dbg, agg_stat


def _perform_bootstrap_resampling(
    db, segments_names, func_dict, n_resamples, dict_stats
):
    bootstrap_results_segmented = {seg: [] for seg in dict_stats.keys()}
    bootstrap_results_agg = []

    for _ in range(n_resamples):
        resampled_db = db.sample(frac=1, replace=True)
        dict_stats_resampled, agg_stats_resampled = _calculate_statistics_from_db(
            resampled_db, segments_names, func_dict
        )
        for seg_name, df_resampled in dict_stats_resampled.items():
            bootstrap_results_segmented[seg_name].append(df_resampled)
        bootstrap_results_agg.append(agg_stats_resampled)

    return bootstrap_results_segmented, bootstrap_results_agg


def _calculate_segmented_ci(
    dict_stats, bootstrap_results_segmented, lower_quantile, upper_quantile
):
    dict_stats_with_ci = {}
    for seg_name, stat_df in dict_stats.items():
        bootstrap_concat = pd.concat(bootstrap_results_segmented[seg_name])
        numeric_df = bootstrap_concat.select_dtypes(include=[np.number])
        ci_df = (
            numeric_df.groupby(numeric_df.index, observed=False)
            .quantile([lower_quantile, upper_quantile])
            .unstack()
        )

        new_cols = [
            f"{metric}_{'low' if q == lower_quantile else 'up'}"
            for metric, q in ci_df.columns
        ]
        ci_df.columns = new_cols

        joined_df = stat_df.join(ci_df)
        metric_cols = stat_df.select_dtypes(include=[np.number]).columns

        new_col_order = []
        for metric in metric_cols:
            if f"{metric}_low" in joined_df.columns:
                new_col_order.append(f"{metric}_low")
            new_col_order.append(metric)
            if f"{metric}_up" in joined_df.columns:
                new_col_order.append(f"{metric}_up")

        other_cols = [c for c in joined_df.columns if c not in new_col_order]
        dict_stats_with_ci[seg_name] = joined_df[other_cols + new_col_order]

    return dict_stats_with_ci


def _calculate_aggregated_ci(
    agg_stats, bootstrap_results_agg, lower_quantile, upper_quantile
):
    agg_bootstrap_df = pd.DataFrame(bootstrap_results_agg)
    numeric_agg_df = agg_bootstrap_df.select_dtypes(include=[np.number])
    agg_ci = numeric_agg_df.quantile([lower_quantile, upper_quantile])

    agg_stats_with_ci = agg_stats.copy()
    for col in numeric_agg_df.columns:
        agg_stats_with_ci[f"{col}_low"] = agg_ci.loc[lower_quantile, col]
        agg_stats_with_ci[f"{col}_up"] = agg_ci.loc[upper_quantile, col]

    metric_cols = agg_stats.index
    new_col_order = []
    for metric in metric_cols:
        if f"{metric}_low" in agg_stats_with_ci.index:
            new_col_order.append(f"{metric}_low")
        new_col_order.append(metric)
        if f"{metric}_up" in agg_stats_with_ci.index:
            new_col_order.append(f"{metric}_up")

    other_cols = [c for c in agg_stats_with_ci.index if c not in new_col_order]
    return agg_stats_with_ci.reindex(new_col_order + other_cols)


def calculate_bootstrap_statistics_from_db(
    db, segments_names, func_dict, n_resamples=100, ci_level=0.95, method="empirical"
):
    """Calculates statistics with bootstrap confidence intervals from a DataFrame.

    Args:
        db (pd.DataFrame): The DataFrame containing the data.
        segments_names (list): A list of column names for segmentation.
        func_dict (dict): Dictionary defining aggregations and post-aggregations.
        n_resamples (int): The number of bootstrap resamples.
        ci_level (float): The confidence level for the intervals.
        method (str): Bootstrap method, either "empirical" or "bca" (bias-corrected and accelerated).

    Returns:
        tuple: (dict_stats_with_ci, agg_stats_with_ci)
    """
    if method not in ["empirical", "bca"]:
        raise ValueError("Method must be either 'empirical' or 'bca'")

    # 1. Calculate statistics on the original database
    dict_stats, agg_stats = _calculate_statistics_from_db(db, segments_names, func_dict)

    # 2. Perform N resamples
    bootstrap_results_segmented, bootstrap_results_agg = _perform_bootstrap_resampling(
        db, segments_names, func_dict, n_resamples, dict_stats
    )

    # 3. Calculate confidence intervals
    if method == "empirical":
        lower_quantile = (1 - ci_level) / 2
        upper_quantile = 1 - lower_quantile

        dict_stats_with_ci = _calculate_segmented_ci(
            dict_stats, bootstrap_results_segmented, lower_quantile, upper_quantile
        )

        agg_stats_with_ci = _calculate_aggregated_ci(
            agg_stats, bootstrap_results_agg, lower_quantile, upper_quantile
        )
    else:  # method == "bca"
        dict_stats_with_ci = _calculate_segmented_bca_ci(
            db,
            segments_names,
            func_dict,
            dict_stats,
            bootstrap_results_segmented,
            ci_level,
        )

        agg_stats_with_ci = _calculate_aggregated_bca_ci(
            db, segments_names, func_dict, agg_stats, bootstrap_results_agg, ci_level
        )

    return dict_stats_with_ci, agg_stats_with_ci


def _calculate_bias_correction(original_stat, bootstrap_stats):
    """Calculate bias correction for BCa bootstrap."""
    prop_less = np.mean(bootstrap_stats <= original_stat)
    # Avoid edge cases where prop_less is 0 or 1
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)
    return z0


def _calculate_acceleration(
    db, segments_names, func_dict, statistic_func, *statistic_args
):
    """Calculate acceleration parameter for BCa bootstrap using jackknife."""
    n = len(db)
    jackknife_stats = []

    for i in range(n):
        # Create jackknife sample (remove observation i)
        jack_db = db.drop(index=db.index[i])
        jack_stat = statistic_func(jack_db, segments_names, func_dict, *statistic_args)
        jackknife_stats.append(jack_stat)

    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = np.mean(jackknife_stats, axis=0)

    # Calculate acceleration
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3, axis=0)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2, axis=0)) ** 1.5

    # Avoid division by zero
    denominator = np.where(denominator == 0, np.inf, denominator)
    acceleration = numerator / denominator

    return acceleration


def _calculate_bca_bounds(original_stat, bootstrap_stats, ci_level):
    """Calculate BCa confidence interval bounds."""
    alpha = 1 - ci_level
    z_alpha_2 = stats.norm.ppf(alpha / 2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

    # Calculate bias correction
    z0 = _calculate_bias_correction(original_stat, bootstrap_stats)

    # For simplicity, set acceleration to 0 (this becomes bias-corrected bootstrap)
    # Full BCa would require jackknife estimation which is computationally expensive
    a = 0

    # Calculate adjusted quantiles
    alpha_1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2)))
    alpha_2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2)))

    # Clip to valid range
    alpha_1 = np.clip(alpha_1, 0.001, 0.999)
    alpha_2 = np.clip(alpha_2, 0.001, 0.999)

    # Calculate quantiles
    lower_bound = np.quantile(bootstrap_stats, alpha_1)
    upper_bound = np.quantile(bootstrap_stats, alpha_2)

    return lower_bound, upper_bound


def _calculate_segmented_bca_ci(
    db, segments_names, func_dict, dict_stats, bootstrap_results_segmented, ci_level
):
    """Calculate BCa confidence intervals for segmented statistics."""
    dict_stats_with_ci = {}

    for seg_name, stat_df in dict_stats.items():
        bootstrap_concat = pd.concat(bootstrap_results_segmented[seg_name])
        numeric_df = bootstrap_concat.select_dtypes(include=[np.number])

        ci_data = {}

        # Calculate BCa CI for each metric and segment value
        for metric in stat_df.select_dtypes(include=[np.number]).columns:
            ci_data[f"{metric}_low"] = []
            ci_data[f"{metric}_up"] = []

            for segment_value in stat_df.index:
                original_stat = stat_df.loc[segment_value, metric]
                bootstrap_stats = numeric_df[numeric_df.index == segment_value][
                    metric
                ].to_numpy()

                if len(bootstrap_stats) > 0:
                    lower_bound, upper_bound = _calculate_bca_bounds(
                        original_stat, bootstrap_stats, ci_level
                    )
                    ci_data[f"{metric}_low"].append(lower_bound)
                    ci_data[f"{metric}_up"].append(upper_bound)
                else:
                    ci_data[f"{metric}_low"].append(np.nan)
                    ci_data[f"{metric}_up"].append(np.nan)

        # Create CI DataFrame
        ci_df = pd.DataFrame(ci_data, index=stat_df.index)

        # Join with original statistics
        joined_df = stat_df.join(ci_df)
        metric_cols = stat_df.select_dtypes(include=[np.number]).columns

        # Reorder columns
        new_col_order = []
        for metric in metric_cols:
            if f"{metric}_low" in joined_df.columns:
                new_col_order.append(f"{metric}_low")
            new_col_order.append(metric)
            if f"{metric}_up" in joined_df.columns:
                new_col_order.append(f"{metric}_up")

        other_cols = [c for c in joined_df.columns if c not in new_col_order]
        dict_stats_with_ci[seg_name] = joined_df[other_cols + new_col_order]

    return dict_stats_with_ci


def _calculate_aggregated_bca_ci(
    db, segments_names, func_dict, agg_stats, bootstrap_results_agg, ci_level
):
    """Calculate BCa confidence intervals for aggregated statistics."""
    agg_bootstrap_df = pd.DataFrame(bootstrap_results_agg)
    numeric_agg_df = agg_bootstrap_df.select_dtypes(include=[np.number])

    agg_stats_with_ci = agg_stats.copy()

    for col in numeric_agg_df.columns:
        original_stat = agg_stats[col]
        bootstrap_stats = numeric_agg_df[col].to_numpy()

        lower_bound, upper_bound = _calculate_bca_bounds(
            original_stat, bootstrap_stats, ci_level
        )

        agg_stats_with_ci[f"{col}_low"] = lower_bound
        agg_stats_with_ci[f"{col}_up"] = upper_bound

    # Reorder columns
    metric_cols = agg_stats.index
    new_col_order = []
    for metric in metric_cols:
        if f"{metric}_low" in agg_stats_with_ci.index:
            new_col_order.append(f"{metric}_low")
        new_col_order.append(metric)
        if f"{metric}_up" in agg_stats_with_ci.index:
            new_col_order.append(f"{metric}_up")

    other_cols = [c for c in agg_stats_with_ci.index if c not in new_col_order]
    return agg_stats_with_ci.reindex(new_col_order + other_cols)
