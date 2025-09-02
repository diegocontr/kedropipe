from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analyses import GlobalAnalysis, _ColumnResolver


class CalibrationCurveAnalysis(GlobalAnalysis):
    """Compute calibration curves for one or multiple models.

    Data returned by get_data() is a wide DataFrame with columns:
    - bin: 1..n_bins (quantile bins by prediction)
    - x_{name}: weighted mean predicted value for model 'name'
    - y_{name}: weighted mean observed value for model 'name' (target per weight)
    where 'name' comes from model spec 'name' or the model key.
    """

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        return _ColumnResolver.required_columns(config)

    def run(self, df: pd.DataFrame) -> None:
        r = _ColumnResolver.resolve(self.config)
        n_bins = int(self.config.get("n_bins", 20))

        result = pd.DataFrame({"bin": np.arange(1, n_bins + 1, dtype=int)})
        result = result.set_index("bin")

        for _, pred_col, weight_col, target_col, display_name in r.model_columns:
            pred = df[pred_col].astype(float)
            target = df[target_col].astype(float)
            if weight_col is None:
                weights = pd.Series(1.0, index=df.index)
            else:
                weights = df[weight_col].astype(float)

            order = pred.sort_values(ascending=True).index
            pred_sorted = pred.loc[order]
            weights_sorted = weights.loc[order]
            target_sorted = target.loc[order]

            try:
                bin_ids = pd.qcut(
                    np.arange(len(order)),
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                bin_ids = np.zeros(len(order), dtype=int)

            tmp = pd.DataFrame(
                {
                    "bin": bin_ids,
                    "w": weights_sorted.to_numpy(),
                    "pred_w": (pred_sorted.to_numpy() * weights_sorted.to_numpy()),
                    "tgt": target_sorted.to_numpy(),
                }
            )

            by = (
                tmp.groupby("bin", sort=True, observed=True)
                .agg(
                    exposure=("w", "sum"),
                    pred_sum=("pred_w", "sum"),
                    tgt_sum=("tgt", "sum"),
                )
                .reset_index()
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                by["x"] = np.where(
                    by["exposure"] > 0, by["pred_sum"] / by["exposure"], 0.0
                )
                by["y"] = np.where(
                    by["exposure"] > 0, by["tgt_sum"] / by["exposure"], 0.0
                )

            series_x = by.set_index(by["bin"].astype(int) + 1)["x"].reindex(
                result.index
            )
            series_y = by.set_index(by["bin"].astype(int) + 1)["y"].reindex(
                result.index
            )

            result[f"x_{display_name}"] = series_x
            result[f"y_{display_name}"] = series_y

        self._data = result.reset_index()

    def get_data_and_metadata(self) -> Dict[str, Dict[str, object]]:
        data = self.get_data()
        names = sorted({col[2:] for col in data.columns if col.startswith("x_")})
        meta = {
            "names": names,
            "title": self.config.get("title", "Calibration Curve"),
        }
        return {"calibration_curve": {"data": data, "metadata": meta}}

    def plot(self, ax, theme: Dict, color_provider, panel_cfg: Optional[Dict] = None):
        data = self.get_data()
        names = sorted({col[2:] for col in data.columns if col.startswith("x_")})

        min_val = float(
            min(
                data[[f"x_{n}" for n in names]].min().min(),
                data[[f"y_{n}" for n in names]].min().min(),
            )
        )
        max_val = float(
            max(
                data[[f"x_{n}" for n in names]].max().max(),
                data[[f"y_{n}" for n in names]].max().max(),
            )
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="black",
            linestyle=theme.get("h_line_style", "--"),
            linewidth=1,
            label="perfect",
        )

        colors = color_provider(len(names))
        for name, color in zip(names, colors, strict=False):
            ax.plot(
                data[f"x_{name}"],
                data[f"y_{name}"],
                marker="o",
                label=name,
                color=color,
            )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=False)
