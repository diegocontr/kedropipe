from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, columns=None):
        """Load data with specified columns."""
        pass


class ParquetDataLoader(DataLoader):
    """Loads data from a parquet file."""

    def __init__(self, path):
        """Initialize the ParquetDataLoader with a file path."""
        if not isinstance(path, str):
            raise TypeError("path must be a string representing the file path.")
        self.path = path

    def load(self, columns=None):
        """Load data from a parquet file."""
        return pd.read_parquet(self.path, columns=columns)


class DataFrameDataLoader(DataLoader):
    """Loads data from a pandas DataFrame."""

    def __init__(self, df):
        """Initialize the DataFrameDataLoader with a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        self.df = df.copy()

    def load(self, columns=None):
        """Load data from a pandas DataFrame."""
        if columns:
            # Return a copy to avoid side effects
            return self.df[columns].copy()
        return self.df.copy()


class TripleParquetDataLoader(DataLoader):
    """Loads data from three parquet files and merges them."""

    def __init__(
        self,
        features_path,
        claims_path,
        predictions_path,
        claims_cols=None,
        predictions_cols=None,
        merge_keys=None,
    ):
        """Initialize the TripleParquetDataLoader."""
        self.features_path = features_path
        self.claims_path = claims_path
        self.predictions_path = predictions_path
        self.claims_cols = claims_cols if claims_cols is not None else []
        self.predictions_cols = predictions_cols if predictions_cols is not None else []
        self.merge_keys = merge_keys

    def load(self, columns=None):
        """Load data from three parquet files and merge them.
        Only loads necessary columns if a list is provided.
        """
        cols_to_load_claims = None
        cols_to_load_preds = None
        cols_to_load_features = None

        if columns:
            all_known_cols = set(self.claims_cols + self.predictions_cols)
            feature_cols_in_request = [c for c in columns if c not in all_known_cols]
            claim_cols_in_request = [c for c in columns if c in self.claims_cols]
            prediction_cols_in_request = [
                c for c in columns if c in self.predictions_cols
            ]

            # Ensure merge keys are loaded if they exist
            if self.merge_keys:
                cols_to_load_features = list(
                    set(feature_cols_in_request + self.merge_keys)
                )
                cols_to_load_claims = list(set(claim_cols_in_request + self.merge_keys))
                cols_to_load_preds = list(
                    set(prediction_cols_in_request + self.merge_keys)
                )
            else:
                cols_to_load_features = feature_cols_in_request
                cols_to_load_claims = claim_cols_in_request
                cols_to_load_preds = prediction_cols_in_request

        features_df = pd.read_parquet(self.features_path, columns=cols_to_load_features)
        claims_df = pd.read_parquet(self.claims_path, columns=cols_to_load_claims)
        predictions_df = pd.read_parquet(
            self.predictions_path, columns=cols_to_load_preds
        )

        if self.merge_keys:
            # Merge using specified keys, with features as the left table
            merged_df = features_df.merge(claims_df, on=self.merge_keys, how="left")
            merged_df = merged_df.merge(predictions_df, on=self.merge_keys, how="left")
        else:
            # Join on index if no merge keys are provided
            merged_df = features_df.join(claims_df).join(predictions_df)

        # The final filtering should be done on the merged dataframe if columns were specified
        if columns:
            return merged_df[columns]

        return merged_df
