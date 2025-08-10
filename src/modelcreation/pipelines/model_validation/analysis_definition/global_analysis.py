from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_calibration_plot(
    *, y_true, y_pred, y_pred_old: Optional[pd.Series] = None, n_bins: int = 10
) -> Tuple[Dict[str, plt.Figure], Dict[str, Any]]:
    """Calibration plot: ratio mean(y_true)/mean(y_pred) per percentile bin of predictions."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

    def _make_bins(work_df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, list[dict]]:
        tmp = work_df[["y_true", col]].rename(columns={col: "y_pred"}).dropna()
        if tmp.empty:
            return pd.DataFrame(), []
        try:
            tmp["bin"] = pd.qcut(tmp["y_pred"], q=n_bins, duplicates="drop")
        except ValueError:
            uniq = tmp["y_pred"].nunique()
            q = max(2, min(n_bins, uniq))
            tmp["bin"] = pd.qcut(tmp["y_pred"], q=q, duplicates="drop")
        grouped = tmp.groupby("bin", observed=True).agg(
            mean_pred=("y_pred", "mean"),
            mean_true=("y_true", "mean"),
            count=("y_true", "size"),
        )
        grouped["ratio"] = grouped.apply(
            lambda r: (r["mean_true"] / r["mean_pred"]) if r["mean_pred"] != 0 else float("nan"),
            axis=1,
        )
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
        return grouped, table_records

    grouped_curr, table_curr = _make_bins(df, "y_pred")

    have_old = y_pred_old is not None
    if have_old:
        df_old = pd.DataFrame({"y_true": y_true, "y_pred_old": y_pred_old})
        grouped_old, table_old = _make_bins(df_old, "y_pred_old")
    else:
        grouped_old, table_old = pd.DataFrame(), []

    fig, ax = plt.subplots(figsize=(7, 5))
    if not grouped_curr.empty:
        ax.plot(grouped_curr["mean_pred"], grouped_curr["ratio"], marker="o", lw=2, label="Current")
    if have_old and not grouped_old.empty:
        ax.plot(
            grouped_old["mean_pred"],
            grouped_old["ratio"],
            marker="s",
            lw=2,
            label="Old",
            color="tab:orange",
        )
    ax.axhline(1.0, color="grey", linestyle=":", lw=1.5)
    ax.set_xlabel("Mean prediction (per percentile bin)")
    ax.set_ylabel("Mean observed / Mean prediction")
    ax.set_title("Calibration (Observed / Predicted)")
    ax.legend(loc="best")
    fig.tight_layout()

    metrics: Dict[str, Any] = {"bins": table_curr}
    if table_old:
        metrics["bins_old"] = table_old

    return {"calibration": fig}, metrics


def generate_lorenz_curve(
    *, y_true, y_pred, y_pred_old: Optional[pd.Series] = None
) -> Tuple[Dict[str, plt.Figure], Dict[str, Any]]:
    """Generate Lorenz curve for regression predictions.

    Procedure:
      1. Order samples by predicted value ascending (classic Lorenz ordering).
      2. Compute cumulative share of population vs cumulative share of actual target.
      3. Plot Lorenz curve against line of equality.

    Metrics:
      - gini: 1 - 2 * area_under_lorenz (area computed via trapezoidal rule)
      - area_under_lorenz
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No data for Lorenz", ha="center", va="center")
        ax.set_axis_off()
        return {"lorenz": fig}, {"gini": None, "area_under_lorenz": None}

    def _prepare(df_work: pd.DataFrame, pred_col: str):
        if (df_work["y_true"] < 0).any():
            shift = -df_work["y_true"].min()
            target_col_local = "y_true_shifted"
            df_work = df_work.copy()
            df_work[target_col_local] = df_work["y_true"] + shift
        else:
            target_col_local = "y_true"
        total_local = df_work[target_col_local].sum()
        if total_local == 0:
            return None
        srt = df_work.sort_values(pred_col, ascending=True).reset_index(drop=True)
        srt["cum_target"] = srt[target_col_local].cumsum()
        srt["cum_pop"] = (np.arange(1, len(srt) + 1)) / len(srt)
        srt["lorenz_y"] = srt["cum_target"] / total_local
        lx = np.concatenate([[0.0], srt["cum_pop"].to_numpy()])
        ly = np.concatenate([[0.0], srt["lorenz_y"].to_numpy()])
        area = np.trapz(ly, lx)
        gini_local = 1 - 2 * area
        return lx, ly, area, gini_local

    current = _prepare(df, "y_pred")
    old = None
    if y_pred_old is not None:
        df_old = pd.DataFrame({"y_true": y_true, "y_pred_old": y_pred_old}).dropna()
        if not df_old.empty:
            old = _prepare(df_old, "y_pred_old")

    fig, ax = plt.subplots(figsize=(6, 5))
    metrics: Dict[str, Any] = {}

    if current is None:
        ax.text(0.5, 0.5, "Lorenz undefined (zero total)", ha="center", va="center")
    else:
        lx, ly, area, gini_val = current
        ax.plot(lx, ly, label="Current", color="tab:blue")
        ax.fill_between(lx, ly, lx, color="tab:blue", alpha=0.15)
        metrics.update({"gini": float(gini_val), "area_under_lorenz": float(area)})

    if old is not None:
        lx_o, ly_o, area_o, gini_o = old
        ax.plot(lx_o, ly_o, label="Old", color="tab:orange")
        ax.fill_between(lx_o, ly_o, lx_o, color="tab:orange", alpha=0.10)
        metrics.update({"gini_old": float(gini_o), "area_under_lorenz_old": float(area_o)})

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Equality")
    ax.set_xlabel("Cumulative share of population")
    ax.set_ylabel("Cumulative share of actual target")
    ax.set_title("Lorenz Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    return {"lorenz": fig}, metrics


__all__ = ["generate_calibration_plot", "generate_lorenz_curve"]
