from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


def generate_roc_curve(*, y_true, y_pred_proba) -> Tuple[Dict[str, plt.Figure], Dict[str, Any]]:
	"""Generate ROC curve and metrics (binary y_true; any monotonic score)."""
	fig, ax = plt.subplots(figsize=(6, 5))
	metrics: Dict[str, Optional[float]] = {"auc": None, "gini": None}
	try:
		fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
		roc_auc = auc(fpr, tpr)
		gini = 2 * roc_auc - 1
		ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
		metrics = {"auc": float(roc_auc), "gini": float(gini)}
	except ValueError as e:  # single-class or invalid labels
		ax.text(0.5, 0.5, f"ROC not available: {e}", ha="center", va="center")
	ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC Curve")
	ax.legend(loc="lower right")
	fig.tight_layout()
	return {"roc": fig}, metrics


def generate_calibration_plot(
	*, y_true, y_pred, n_bins: int = 10
) -> Tuple[Dict[str, plt.Figure], Dict[str, Any]]:
	"""Calibration plot: ratio mean(y_true)/mean(y_pred) per percentile bin of predictions."""
	df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
	try:
		df["bin"] = pd.qcut(df["y_pred"], q=n_bins, duplicates="drop")
	except ValueError:
		uniq = df["y_pred"].nunique()
		q = max(2, min(n_bins, uniq))
		df["bin"] = pd.qcut(df["y_pred"], q=q, duplicates="drop")
	grouped = df.groupby("bin", observed=True).agg(
		mean_pred=("y_pred", "mean"),
		mean_true=("y_true", "mean"),
		count=("y_true", "size"),
	)
	grouped["ratio"] = grouped.apply(
		lambda r: (r["mean_true"] / r["mean_pred"]) if r["mean_pred"] != 0 else float("nan"),
		axis=1,
	)
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.plot(grouped["mean_pred"], grouped["ratio"], marker="o", lw=2)
	ax.axhline(1.0, color="grey", linestyle=":", lw=1.5)
	ax.set_xlabel("Mean prediction (per percentile bin)")
	ax.set_ylabel("Mean observed / Mean prediction")
	ax.set_title("Calibration (Observed / Predicted)")
	fig.tight_layout()
	table_records = [
		{
			"bin": str(idx),
			"mean_pred": float(row["mean_pred"]),
			"mean_true": float(row["mean_true"]),
			"count": int(row["count"]),
			"ratio": float(row["ratio"]) if pd.notna(row["ratio"]) else None,
		}
		for idx, row in grouped.iterrows()
	]
	return {"calibration": fig}, {"bins": table_records}


__all__ = ["generate_calibration_plot", "generate_roc_curve"]
