import pandas as pd


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
    ):
        """Initializes the SegmentCustom object.

        Args:
            seg_col (str): The name of the column to be segmented.
            seg_name (str): The name of the new column that will contain the segments.
            bins (int or sequence of scalars): The criteria for binning.
                - If an integer, it defines the number of equal-width bins.
                - If a sequence of scalars, it defines the bin edges.
            bin_labels (list of str, optional): The labels for the bins. If None, labels will be
                generated automatically. Defaults to None.
            seg_label (str, optional): A display label for the segmentation. If None, seg_name is used.
                Defaults to None.
            added_to_db (bool, optional): A flag indicating if the segmentation column has been
                added to the database. Defaults to False.
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
        if isinstance(self.bins, int) and self.bin_labels is None:
            # If number of bins is provided without labels, we can let pandas create them
            # or create simple integer-based labels. Let's let pd.cut handle it.
            labels = None

        db[self.seg_name] = pd.cut(
            db[self.seg_col],
            bins=self.bins,
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
