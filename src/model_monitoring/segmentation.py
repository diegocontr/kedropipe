import numpy as np
import pandas as pd


def _round_to_significant(x, sig_digits=3):
    """Round a number to the given number of significant digits.

    Returns NaN unchanged. For 0, returns 0.
    """
    if pd.isna(x):
        return x
    if x == 0:
        return 0
    try:
        return float(f"{float(x):.{int(sig_digits)}g}")
    except Exception:
        # Fallback to original value if formatting fails for any reason
        return x


def _format_edge(x, sig_digits=3):
    """Format a bin edge value using significant digits for display.

    Attempts to avoid scientific notation for readability (e.g., 14500 instead of 1.45e+04)
    when feasible after rounding to the requested significant digits.
    """
    if pd.isna(x):
        return "nan"
    x_rounded = _round_to_significant(x, sig_digits)

    # If it's an integer value, show as integer without casting warnings
    if isinstance(x_rounded, (int, np.integer)) or (
        isinstance(x_rounded, float) and float(x_rounded).is_integer()
    ):
        return f"{float(x_rounded):.0f}"

    # Prefer non-scientific formatting when within a reasonable magnitude
    absx = abs(float(x_rounded))
    if 1e-3 <= absx < 1e6:
        s = f"{x_rounded:.10f}"  # overspecify then trim
        # Trim trailing zeros and optional decimal point
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    # Fallback to general format
    return f"{x_rounded:.{int(sig_digits)}g}"


def compute_percentile_bins(series, n_bins, sig_digits=3, round_bounds=False):
    """Compute percentile-based bin edges and optionally labels.

    Args:
        series (pd.Series): Continuous data to segment.
        n_bins (int): Number of quantile bins to compute.
        sig_digits (int): Significant digits for rounding/formatting.
        round_bounds (bool): If True, round the numeric bin edges to significant digits.
            If False, keep precise edges but return human-friendly labels with reduced precision.

    Returns:
        tuple[list[float], list[str] | None]: (edges, labels)
            - edges: strictly increasing list of bin boundaries (length k+1 for k bins)
            - labels: None if round_bounds is True (let pandas label with Interval using rounded edges),
              otherwise a list of k strings formatted with reduced precision.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Cannot compute percentile bins on empty or non-numeric data.")

    # Compute percentile edges
    quantiles = np.linspace(0, 100, int(n_bins) + 1)
    edges = np.nanpercentile(s.values, quantiles)

    # Ensure strictly increasing edges by dropping duplicates (can happen with ties)
    unique_edges = np.unique(edges)
    if unique_edges.size < 2:
        raise ValueError("Not enough distinct values to create bins.")

    # If duplicates reduced the number of intervals, use the unique edges
    edges = unique_edges.astype(float).tolist()

    if round_bounds:
        # Round numeric edges to requested significant digits
        edges = [_round_to_significant(e, sig_digits) for e in edges]
        # After rounding, ensure strict monotonicity by collapsing any potential duplicates
        collapsed = []
        for e in edges:
            if not collapsed or e > collapsed[-1]:
                collapsed.append(e)
        if len(collapsed) < 2:
            raise ValueError(
                "Rounding collapsed all bin edges; increase significant digits or disable rounding."
            )
        edges = collapsed
        # Also build friendly labels from the (rounded) edges
        labels = [
            f"[{_format_edge(edges[i], sig_digits)}, {_format_edge(edges[i + 1], sig_digits)})"
            for i in range(len(edges) - 1)
        ]
    else:
        # Keep precise edges, but prepare readable labels with formatted bounds
        labels = [
            f"[{_format_edge(edges[i], sig_digits)}, {_format_edge(edges[i + 1], sig_digits)})"
            for i in range(len(edges) - 1)
        ]

    return edges, labels


class SegmentCustom:
    """A class to represent a segmentation of a column in a DataFrame."""

    def __init__(
        self,
        seg_col,
        seg_name,
        bins,
        bin_labels=None,
        seg_label=None,
        added_to_db=False,
        sig_digits=3,
        round_bounds=False,
    ):
        """Initializes the SegmentCustom object.

        Args:
            seg_col (str): The name of the column to be segmented.
            seg_name (str): The name of the new column that will contain the segments.
            bins (int or sequence of scalars): The criteria for binning.
                - If an integer, it defines the number of percentile-based bins (quantiles).
                - If a sequence of scalars, it defines the bin edges.
            bin_labels (list of str, optional): The labels for the bins. If None, labels will be
                generated automatically. Defaults to None.
            seg_label (str, optional): A display label for the segmentation. If None, seg_name is used.
                Defaults to None.
            added_to_db (bool, optional): A flag indicating if the segmentation column has been
                added to the database. Defaults to False.
            sig_digits (int, optional): Significant digits to use when rounding bounds or labels.
                Defaults to 3.
            round_bounds (bool, optional): If True, round the computed percentile bin edges to
                the specified significant digits. If False, keep precise edges but generate
                human-friendly labels with reduced precision. Defaults to False.
        """
        if bin_labels and isinstance(bins, (list, tuple)):
            if len(bins) != len(bin_labels) + 1:
                raise ValueError(
                    "The number of bin edges (`bins`) must be one more than the number of `bin_labels`."
                )

        self.seg_col = seg_col
        self.seg_name = seg_name
        self.bins = bins
        self.bin_labels = bin_labels
        self.seg_label = seg_label if seg_label is not None else seg_name
        self.added_to_db = added_to_db
        self.sig_digits = sig_digits
        self.round_bounds = round_bounds

    def get_cols(self):
        """Returns the columns required for this segmentation."""
        if not self.added_to_db:
            return [self.seg_col]
        return []

    def apply(self, db):
        """Applies the segmentation to a DataFrame.

        Args:
            db (pd.DataFrame): The DataFrame to apply the segmentation to.
        """
        labels = self.bin_labels
        bins_to_use = self.bins

        # If bins is an integer, compute percentile-based bins automatically
        if isinstance(self.bins, int):
            edges, auto_labels = compute_percentile_bins(
                db[self.seg_col],
                self.bins,
                sig_digits=self.sig_digits,
                round_bounds=self.round_bounds,
            )
            bins_to_use = edges
            # Only use auto-generated labels if user didn't provide custom labels
            if labels is None:
                labels = auto_labels

            # Validate label count if user provided labels
            if labels is not None and (len(labels) != (len(bins_to_use) - 1)):
                raise ValueError(
                    "The number of `bin_labels` must be exactly one less than the number of computed bin edges."
                )

        db[self.seg_name] = pd.cut(
            db[self.seg_col],
            bins=bins_to_use,
            labels=labels,
            right=False,
            include_lowest=True,
        )


class SegmentCategorical:
    """A class to represent a segmentation of a categorical column in a DataFrame."""

    def __init__(
        self, seg_col, seg_name, mapping=None, seg_label=None, added_to_db=False
    ):
        """Initializes the SegmentCategorical object.

        Args:
            seg_col (str): The name of the categorical column to be segmented.
            seg_name (str): The name of the new column that will contain the segments.
            mapping (dict, optional): A dictionary to map original category values to new segment labels.
                If None, each unique category will be its own segment. Defaults to None.
            seg_label (str, optional): A display label for the segmentation. If None, seg_name is used.
                Defaults to None.
            added_to_db (bool, optional): A flag indicating if the segmentation column has been
                added to the database. Defaults to False.
        """
        self.seg_col = seg_col
        self.seg_name = seg_name
        self.mapping = mapping
        self.seg_label = seg_label if seg_label is not None else seg_name
        self.added_to_db = added_to_db

    def get_cols(self):
        """Returns the columns required for this segmentation."""
        if not self.added_to_db:
            return [self.seg_col]
        return []

    def apply(self, db):
        """Applies the segmentation to a DataFrame.

        Args:
            db (pd.DataFrame): The DataFrame to apply the segmentation to.
        """
        if self.mapping:
            # Invert the mapping to map from category to segment
            reverse_mapping = {
                value: key for key, values in self.mapping.items() for value in values
            }
            db[self.seg_name] = db[self.seg_col].map(reverse_mapping)
        else:
            # If no mapping is provided, use the original categorical column
            db[self.seg_name] = db[self.seg_col]
