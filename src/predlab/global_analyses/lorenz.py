from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analyses import GlobalAnalysis, _ColumnResolver


class LorenzCurveAnalysis(GlobalAnalysis):
    """Compute Lorenz curves for one or multiple models vs. an observed target.

    Data returned by get_data() is a wide DataFrame with columns:
    - bin: 1..n_bins (ordered by prediction per model)
    - x_{name}: cumulative share of exposure (weights) for model 'name'
    - y_{name}: cumulative share of target (weighted) for model 'name'
    where 'name' comes from model spec 'name' or the model key.
    """

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        return _ColumnResolver.required_columns(config)

    def run(self, df: pd.DataFrame) -> None:
        r = _ColumnResolver.resolve(self.config)
        n_bins = int(self.config.get("n_bins", 20))
        ascending = bool(self.config.get("ascending", True))

        result = pd.DataFrame({"bin": np.arange(1, n_bins + 1, dtype=int)})
        result = result.set_index("bin")

        for _, pred_col, weight_col, target_col, display_name in r.model_columns:
            pred = df[pred_col]
            target = df[target_col]
            if weight_col is None:
                weights = pd.Series(1.0, index=df.index)
            else:
                weights = df[weight_col].astype(float)

            order = pred.sort_values(ascending=ascending).index
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
                    "weight": weights_sorted.to_numpy(),
                    "target_w": (target_sorted.to_numpy() * weights_sorted.to_numpy()),
                }
            )

            by = (
                tmp.groupby("bin", sort=True, observed=True)
                .agg(
                    exposure=("weight", "sum"),
                    target=("target_w", "sum"),
                )
                .reset_index()
            )

            total_exp = by["exposure"].sum()
            total_tgt = by["target"].sum()
            by["exp_share"] = 0.0 if total_exp == 0 else by["exposure"] / total_exp
            by["tgt_share"] = 0.0 if total_tgt == 0 else by["target"] / total_tgt

            by = by.sort_values("bin")
            by["cum_exposure_share"] = by["exp_share"].cumsum().clip(upper=1.0)
            by["cum_target_share"] = by["tgt_share"].cumsum().clip(upper=1.0)

            series_x = by.set_index(by["bin"].astype(int) + 1)["cum_exposure_share"]
            series_y = by.set_index(by["bin"].astype(int) + 1)["cum_target_share"]
            series_x = series_x.reindex(result.index)
            series_y = series_y.reindex(result.index)

            result[f"x_{display_name}"] = series_x
            result[f"y_{display_name}"] = series_y

        self._data = result.reset_index()

    def get_data_and_metadata(self) -> Dict[str, Dict[str, object]]:
        data = self.get_data()
        names = sorted({col[2:] for col in data.columns if col.startswith("x_")})
        meta = {
            "names": names,
            "title": self.config.get("title", "Lorenz Curve"),
        }
        return {"lorenz_curve": {"data": data, "metadata": meta}}

    def plot(self, ax, theme: Dict, color_provider, panel_cfg: Optional[Dict] = None):
        data = self.get_data()
        names = sorted({col[2:] for col in data.columns if col.startswith("x_")})

        ax.plot(
            [0, 1],
            [0, 1],
            color="black",
            linestyle=theme.get("h_line_style", "--"),
            linewidth=1,
            label="baseline",
        )

        colors = color_provider(len(names))
        for name, color in zip(names, colors, strict=False):
            x = data[f"x_{name}"]
            y = data[f"y_{name}"]
            ax.plot(x, y, marker="o", label=name, color=color)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Cumulative exposure share")
        ax.set_ylabel("Cumulative target share")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=False)
