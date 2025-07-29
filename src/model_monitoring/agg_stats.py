import pandas as pd

from .bootstrap_ci import calculate_bootstrap_statistics_from_db
from .metrics import gini


def safe_division(numerator, denominator):
    """Helper function to safely divide, returning 0 if the denominator is 0."""
    return numerator / denominator if denominator != 0 else 0


def df_weighted_mean(df, x, w, s):
    """This function calculates a weighted mean of 'x' using 'w' as weights and 's' as a special factor."""
    m = (df[x] < 9999).astype(int)
    numerator = (m * df[x] * df[w] * df[s]).sum()
    denominator = (m * df[w] * df[s]).sum()
    return safe_division(numerator, denominator)


def df_observed_charge(df, x, w, s):
    """This function calculates the sum of 'x' divided by the sum of 'w' multiplied by 's'."""
    numerator = df[x].sum()
    denominator = (df[w] * df[s]).sum()
    return safe_division(numerator, denominator)


def df_mean_of_ratio(df, v1, v2, w, s):
    """Calculates the weighted mean of the ratio of two columns (v1 / v2)."""
    m1 = (df[v1] < 9999).astype(int)
    m2 = (df[v2] < 9999).astype(int)
    ratio = df[v1] / df[v2]
    numerator = (m1 * m2 * ratio * df[w] * df[s]).sum()
    denominator = (m1 * m2 * df[w] * df[s]).sum()
    return safe_division(numerator, denominator)


def df_mean_of_variation(df, v1, v2, w, s):
    """Calculates the weighted mean of the variation of two columns (v1 - v2)."""
    m1 = (df[v1] < 9999).astype(int)
    m2 = (df[v2] < 9999).astype(int)
    ratio = df[v1] / df[v2]
    numerator = (m1 * m2 * (ratio - 1) * df[w] * df[s]).sum()
    denominator = (m1 * m2 * df[w] * df[s]).sum()
    return 100 * safe_division(numerator, denominator)


def df_mean_of_difference(df, v1, v2, w, s):
    """Calculates the weighted mean of the difference of two columns (v1 - v2)."""
    m1 = (df[v1] < 9999).astype(int)
    m2 = (df[v2] < 9999).astype(int)
    difference = df[v1] - df[v2]
    numerator = (m1 * m2 * difference * df[w] * df[s]).sum()
    denominator = (m1 * m2 * df[w] * df[s]).sum()
    return safe_division(numerator, denominator)


def df_e_lr(df, x, w, s, pc):
    """This function seems to be calculating an expected loss ratio."""
    m = (df[x] < 9999).astype(int)
    numerator = (m * df[x] * df[w]).sum()
    denominator = (m * df[w] * df[pc]).sum()
    return safe_division(numerator, denominator)


def df_lr(df, o, w, s, pc):
    """This function calculates a loss ratio."""
    numerator = df[o].sum()
    denominator = (df[w] * df[pc]).sum()
    return safe_division(numerator, denominator)  # / df_weighted_mean(df, pc, w, s)


def df_gini(df, o, p, w, s):
    """This function calculates a Gini coefficient."""
    return gini(df[o], df[p], df[w] * df[s])


AGG_FUNCTIONS = {
    "weighted_mean": df_weighted_mean,
    "observed_charge": df_observed_charge,
    "mean_of_ratio": df_mean_of_ratio,
    "mean_of_variation_perc": df_mean_of_variation,
    "mean_of_difference": df_mean_of_difference,
    "gini": df_gini,
    "e_lr": df_e_lr,
    "lr": df_lr,
    "sum": lambda df, col: df[col].sum(),
}

POST_AGG_FUNCTIONS = {
    "division": lambda d, c1, c2: d[c1] / d[c2],
    "subtraction": lambda d, c1, c2: d[c1] - d[c2],
    "multiplication": lambda d, c1, c2: d[c1] * d[c2],
    "addition": lambda d, c1, c2: d[c1] + d[c2],
    "variation_perc": lambda d, c1, c2: 100 * (d[c1] - d[c2]) / d[c2],
    "/": lambda d, c1, c2: d[c1] / d[c2],
    "-": lambda d, c1, c2: d[c1] - d[c2],
    "*": lambda d, c1, c2: d[c1] * d[c2],
    "+": lambda d, c1, c2: d[c1] + d[c2],
}


def _get_agg_function(func_ref):
    """Get aggregation function from string reference or return the function directly.

    Args:
        func_ref: Either a string key to look up in AGG_FUNCTIONS or a callable function.

    Returns:
        callable: The resolved function.
    """
    if isinstance(func_ref, str):
        if func_ref not in AGG_FUNCTIONS:
            raise ValueError(
                f"Unknown aggregation function '{func_ref}'. Available functions: {list(AGG_FUNCTIONS.keys())}"
            )
        return AGG_FUNCTIONS[func_ref]
    elif callable(func_ref):
        return func_ref
    else:
        raise TypeError(
            f"func_ref must be either a string or callable, got {type(func_ref)}"
        )


def _get_post_agg_function(func_ref):
    """Get post-aggregation function from string reference or return the function directly.

    Args:
        func_ref: Either a string key to look up in POST_AGG_FUNCTIONS or a callable function.

    Returns:
        callable: The resolved function.
    """
    if isinstance(func_ref, str):
        if func_ref not in POST_AGG_FUNCTIONS:
            raise ValueError(
                f"Unknown post-aggregation function '{func_ref}'. Available functions: {list(POST_AGG_FUNCTIONS.keys())}"
            )
        return POST_AGG_FUNCTIONS[func_ref]
    elif callable(func_ref):
        return func_ref
    else:
        raise TypeError(
            f"func_ref must be either a string or callable, got {type(func_ref)}"
        )


def apply_functions(db, segment, func_dict):
    """Inner helper function to apply a single function to a group."""

    def apply_func(group, func, cols):
        resolved_func = _get_agg_function(func)
        return resolved_func(group, *cols)

    # Group the DataFrame by 'segment' and apply functions from 'func_dict'.
    # 'func_dict' is expected to have structure {col_name: (function, [list_of_cols_for_func])}
    result = db.groupby(segment, observed=False).apply(
        lambda group: pd.Series(
            {
                col: apply_func(group, func, cols)
                for col, (func, cols) in func_dict.items()
            }
        ),
        include_groups=False,
    )
    return result


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


def calculate_statistics_from_data(
    db,
    segments_names,
    func_dict,
    bootstrap=False,
    n_resamples=100,
    ci_level=0.95,
    method="empirical",
):
    """Calculates various statistics for different segments within a DataFrame.
    Optionally performs bootstrapping to calculate confidence intervals.

    Args:
        db (pd.DataFrame): The DataFrame containing the data.
        segments_names (list): A list of column names to be used for segmentation.
        func_dict (dict): A dictionary defining the statistics to calculate.
        bootstrap (bool): If True, performs bootstrap resampling to calculate CIs.
        n_resamples (int): Number of bootstrap samples to generate if bootstrap is True.
        ci_level (float): Confidence level for intervals if bootstrap is True.
        method (str): Bootstrap method, either "empirical" or "bca".

    Returns:
        tuple: A tuple containing:
            - dict_dbg (dict): A dictionary with calculated statistics for each segment.
            - agg_stat (pd.Series): A Series with aggregated statistics across all data.
    """
    if bootstrap:
        return calculate_bootstrap_statistics_from_db(
            db, segments_names, func_dict, n_resamples, ci_level, method
        )
    else:
        return _calculate_statistics_from_db(db, segments_names, func_dict)


def calculate_statistics(
    analysis_obj,
    func_dict,
    bootstrap=False,
    n_resamples=100,
    ci_level=0.95,
    method="empirical",
    query=None,
):
    """Calculates various statistics for different segments within the analysis object.
    Optionally performs bootstrapping to calculate confidence intervals.

    Args:
        analysis_obj: An object containing the database (analysis_obj.db) and segment names
                      (analysis_obj.segments_names).
        func_dict (dict): A dictionary defining the statistics to calculate.
        bootstrap (bool): If True, performs bootstrap resampling to calculate CIs.
        n_resamples (int): Number of bootstrap samples to generate if bootstrap is True.
        ci_level (float): Confidence level for intervals if bootstrap is True.
        method (str): Bootstrap method, either "empirical" or "bca".
        query (str): Optional query string to filter the DataFrame.

    Returns:
        tuple: A tuple containing:
            - dict_dbg (dict): A dictionary with calculated statistics for each segment.
            - agg_stat (pd.Series): A Series with aggregated statistics across all data.
    """
    db = analysis_obj.db
    if query is not None:
        db = db.query(query)
    segments_names = analysis_obj.segments_names

    return calculate_statistics_from_data(
        db, segments_names, func_dict, bootstrap, n_resamples, ci_level, method
    )
