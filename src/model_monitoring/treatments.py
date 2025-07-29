from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Treatment(ABC):
    """Base class for all data treatments."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the treatment to the DataFrame."""
        pass

    def get_cols(self):
        """Get all columns used in the treatment."""
        return []

    def get_created_cols(self):
        """Get the names of columns created by the treatment."""
        return []


class TreatmentIsoResourceScaling(Treatment):
    """Applies isotonic regression scaling to predictions."""

    def __init__(self, pred_dict):
        """Initialize the TreatmentIsoResourceScaling.

        Args:
            pred_dict: A dictionary mapping coverage names to their column names.
        """
        self.pred_dict = pred_dict

    def get_cols(self):
        """Get all columns used in the treatment."""
        cols = []
        for __, info in self.pred_dict.items():
            cols.append(info["pred_col"])
            cols.append(info["target_col"])
            cols.append(info["sel_col"])
        return list(set(cols))

    def get_created_cols(self):
        """Get the names of columns created by the treatment."""
        return [info["pred_col"] + "_iso" for info in self.pred_dict.values()]

    def apply(self, df):
        """For each coverage in pred_dict, create a new column with the name pred_col + '_iso'
        such that the weighted mean of the new prediction column matches the mean_observed of the target.
        """
        df_copy = df.copy()
        metrics = {}
        for cover, info in self.pred_dict.items():
            pred_col = info["pred_col"]
            target_col = info["target_col"]
            weight_col = info["sel_col"]
            # Calculate scaling factor
            mean_obs = mean_observed(df, target_col, weight_col)
            weighted_pred_mean = weighted_mean(df, pred_col, weight_col)
            scale = mean_obs / weighted_pred_mean if weighted_pred_mean != 0 else 1.0
            metrics[f"{cover}_iso_scale"] = scale
            # Create new column
            iso_col = pred_col + "_iso"
            df[iso_col] = df[pred_col] * scale
        return {"db": df, "metrics": metrics}


class TreatmentAddTotalCoverage(Treatment):
    """Adds total coverage columns by summing up individual coverages."""

    def __init__(
        self,
        pred_dict,
        suffix="",
        target_total_col="target_total",
        prediction_total_col=None,
    ):
        """Initialize the TreatmentAddTotalCoverage.

        Args:
            pred_dict: A dictionary mapping coverage names to their column names.
            suffix: A suffix to add to the created prediction column name.
            target_total_col: Name for the total target column (default 'target_total').
            prediction_total_col: Name for the total prediction column (default 'prediction_total{suffix}').
        """
        self.pred_dict = pred_dict
        self.suffix = suffix
        self.target_total_col = target_total_col
        if prediction_total_col is None:
            self.prediction_total_col = f"prediction_total{self.suffix}"
        else:
            self.prediction_total_col = prediction_total_col

    def get_cols(self):
        """Get all columns used in the treatment."""
        cols = []
        for info in self.pred_dict.values():
            cols.append(info["target_col"])
            if self.suffix:
                cols.append(info["pred_col"] + self.suffix)
            else:
                cols.append(info["pred_col"])
        return list(set(cols))

    def get_created_cols(self):
        """Get the names of columns created by the treatment."""
        return [self.prediction_total_col, self.target_total_col]

    def apply(self, df):
        """Adds total coverage columns to the dataframe with configurable names."""
        target_cols = [info["target_col"] for info in self.pred_dict.values()]
        pred_cols = [info["pred_col"] + self.suffix for info in self.pred_dict.values()]
        df[self.target_total_col] = df[target_cols].sum(axis=1)
        df[self.prediction_total_col] = df[pred_cols].sum(axis=1)
        return {"db": df, "metrics": {}}


class TreatmentColumnScaling(Treatment):
    """Applies scaling factors to specified columns, creating new columns with a suffix."""

    def __init__(self, column_factor_dict, suffix="_scaled"):
        """Initialize the TreatmentColumnScaling.

        Args:
            column_factor_dict: A dictionary mapping column names to their scaling factors.
                               e.g. {"column1": 1.5, "column2": 0.8}
            suffix: Suffix to add to the original column name for the new scaled column.
        """
        self.column_factor_dict = column_factor_dict
        self.suffix = suffix

    def get_cols(self):
        """Get all columns used in the treatment."""
        return list(self.column_factor_dict.keys())

    def get_created_cols(self):
        """Get the names of columns created by the treatment."""
        return [col + self.suffix for col in self.column_factor_dict.keys()]

    def apply(self, df):
        """Create new scaled columns by multiplying original columns with their factors."""
        for column, factor in self.column_factor_dict.items():
            new_col_name = column + self.suffix
            df[new_col_name] = df[column] * factor
        return {"db": df, "metrics": {}}


class TreatmentLeftJoin(Treatment):
    """Performs a left join with another dataframe using specified merge key and columns."""

    def __init__(self, data_source, merge_key, columns_to_merge):
        """Initialize the TreatmentLeftJoin.

        Args:
            data_source: Either a pandas DataFrame or a string path to a parquet file.
            merge_key: The column name to use as the merge key for the left join.
            columns_to_merge: List of column names to retrieve from the right dataframe
                            (excluding the merge key which is automatically included).
        """
        from .data_loader import DataFrameDataLoader, ParquetDataLoader

        self.merge_key = merge_key
        self.columns_to_merge = columns_to_merge

        # Set up the appropriate data loader
        if isinstance(data_source, str):
            self.data_loader = ParquetDataLoader(data_source)
        elif hasattr(data_source, "columns"):  # Check if it's a DataFrame-like object
            self.data_loader = DataFrameDataLoader(data_source)
        else:
            raise TypeError(
                "data_source must be either a pandas DataFrame or a string path to a parquet file."
            )

    def get_cols(self):
        """Get all columns used in the treatment (just the merge key from the main dataframe)."""
        return [self.merge_key]

    def get_created_cols(self):
        """Get the names of columns created by the treatment."""
        return self.columns_to_merge

    def apply(self, df):
        """Perform a left join with the external dataframe."""
        # Load the external data with only the columns we need (merge key + columns to merge)
        columns_to_load = [self.merge_key, *self.columns_to_merge]
        external_df = self.data_loader.load(columns=columns_to_load)

        # Perform the left join
        merged_df = df.merge(external_df, on=self.merge_key, how="left")

        return {"db": merged_df, "metrics": {}}


class TreatmentPairwiseIsoResourceScaling(Treatment):
    """Applies Iso resource scaling between pairs of prediction columns."""

    def __init__(self, pairwise_pred_dict):
        """Initialize the TreatmentPairwiseIsoResourceScaling.

        Args:
            pairwise_pred_dict: A dictionary mapping pair names to their column names.
                                e.g. {
                                    "pair_name": {
                                        "pred_to_scale": "pred_col_1",
                                        "pred_target": "pred_col_2",
                                        "weight_col": "weight",
                                        "mask_col": "selection_mask",
                                        "new_col_name": "pred_col_1_scaled"
                                    }
                                }
        """
        self.pairwise_pred_dict = pairwise_pred_dict

    def get_cols(self):
        """Get all columns used in the treatment."""
        cols = []
        for __, info in self.pairwise_pred_dict.items():
            cols.append(info["pred_to_scale"])
            cols.append(info["pred_target"])
            cols.append(info["weight_col"])
            cols.append(info["mask_col"])
        return list(set(cols))

    def get_created_cols(self):
        """Get the names of columns created by the treatment."""
        return [info["new_col_name"] for info in self.pairwise_pred_dict.values()]

    def apply(self, df):
        """For each pair in pairwise_pred_dict, create a new scaled column.

        The scaling ensures that the weighted mean of the 'pred_to_scale' column
        matches the weighted mean of the 'pred_target' column, using an effective
        weight calculated from 'weight_col' and 'mask_col'.
        """
        metrics = {}
        for pair_name, info in self.pairwise_pred_dict.items():
            pred_to_scale = info["pred_to_scale"]
            pred_target = info["pred_target"]
            weight_col = info["weight_col"]
            mask_col = info["mask_col"]
            new_col_name = info["new_col_name"]

            effective_weight = df[weight_col] * df[mask_col]

            weighted_mean_target = np.sum(df[pred_target] * effective_weight) / np.sum(
                effective_weight
            )
            weighted_mean_to_scale = np.sum(
                df[pred_to_scale] * effective_weight
            ) / np.sum(effective_weight)

            scale = (
                weighted_mean_target / weighted_mean_to_scale
                if weighted_mean_to_scale != 0
                else 1.0
            )
            metrics[f"{pair_name}_pairwise_iso_scale"] = scale

            df[new_col_name] = df[pred_to_scale] * scale

        return {"db": df, "metrics": metrics}


def weighted_mean(df, col, weight_col):
    """Calculate weighted mean of a column."""
    return np.sum(df[col] * df[weight_col]) / np.sum(df[weight_col])


def mean_observed(df, col, weight_col):
    """Calculate mean of a column."""
    return df[col].sum() / df[weight_col].sum()


def apply_iso_resource_scaling(df, pred_dict):
    """For each coverage in pred_dict, create a new column with the name pred_col + '_iso'
    such that the weighted mean of the new prediction column matches the mean_observed of the target.
    """
    for _cover, info in pred_dict.items():
        pred_col = info["pred_col"]
        target_col = info["target_col"]
        weight_col = info["sel_col"]
        # Calculate scaling factor
        mean_obs = mean_observed(df, target_col, weight_col)
        weighted_pred_mean = weighted_mean(df, pred_col, weight_col)
        scale = mean_obs / weighted_pred_mean if weighted_pred_mean != 0 else 1.0
        # Create new column
        iso_col = pred_col + "_iso"
        df[iso_col] = df[pred_col] * scale
    return {"db": df}


def add_total_coverage_columns(df, pred_dict, suffix=""):
    """Adds 'target_total{suffix}' and 'prediction_total{suffix}' columns to the dataframe,
    which are the sum of all target and prediction columns across coverages, with optional suffix.
    Only generates columns with the given suffix.
    """
    target_cols = [info["target_col"] for info in pred_dict.values()]
    pred_cols = [info["pred_col"] + suffix for info in pred_dict.values()]
    df["target_total"] = df[target_cols].sum(axis=1)
    df[f"prediction_total{suffix}"] = df[pred_cols].sum(axis=1)
    return {"db": df}
