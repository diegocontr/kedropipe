from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analyses import GlobalAnalysis


class PredictionAnalysis(GlobalAnalysis):
    """Compute histograms of model predictions.

    get_data() returns a dict with:
      - 'describe': DataFrame where each row corresponds to a model name and columns are describe() stats
      - 'histogram': DataFrame with columns [bin_left, bin_right, bin_center, h_{name} ...]
    """

    @staticmethod
    def required_columns(config: Dict) -> List[str]:
        if (
            "models" not in config
            or not isinstance(config["models"], dict)
            or len(config["models"]) == 0
        ):
            raise ValueError("config must contain a non-empty 'models' dictionary")
        cols = set()
        default_weight = config.get("weight_col")
        for model_key, spec in config["models"].items():
            if not isinstance(spec, dict) or "pred_col" not in spec:
                raise ValueError(
                    f"Each model spec must be a dict with 'pred_col'. Problem with model '{model_key}'."
                )
            cols.add(spec["pred_col"])
            w = spec.get("weight_col", default_weight)
            if w:
                cols.add(w)
        return list(cols)

    def run(self, df: pd.DataFrame) -> None:
        models = self.config["models"]
        default_weight = self.config.get("weight_col")
        n_bins = int(self.config.get("n_bins", 20))
        density = bool(self.config.get("density", False))
        xscale = self.config.get("xscale")

        # Collect predictions to set common bin edges
        all_preds = []
        names = []
        pred_series = {}
        weight_series = {}
        for mkey, spec in models.items():
            name = spec.get("name", mkey)
            pred_col = spec["pred_col"]
            w_col = spec.get("weight_col", default_weight)
            p = df[pred_col].astype(float)
            w = df[w_col].astype(float) if w_col else pd.Series(1.0, index=df.index)
            pred_series[name] = p
            weight_series[name] = w
            all_preds.append(p)
            names.append(name)

        if len(all_preds) == 0:
            raise ValueError("No models provided for prediction_analysis")
        p_min = min(s.min() for s in all_preds)
        p_max = max(s.max() for s in all_preds)
        if not np.isfinite(p_min) or not np.isfinite(p_max) or p_min == p_max:
            # Fallback to simple range
            p_min, p_max = float(p_min), float(p_min + 1.0)

        # Determine bins: log-spaced if requested and feasible
        if isinstance(xscale, str) and xscale.lower() == "log":
            # Use only positive values for log bins
            positive_mins = [float(s[s > 0].min()) for s in all_preds if (s > 0).any()]
            if positive_mins:
                p_min_pos = max(min(positive_mins), 1e-12)
                p_max_pos = max(p_max, p_min_pos * (1.0 + 1e-9))
                bins = np.logspace(np.log10(p_min_pos), np.log10(p_max_pos), n_bins + 1)
            else:
                # No positive values: fallback to linear bins
                bins = np.linspace(p_min, p_max, n_bins + 1)
        else:
            bins = np.linspace(p_min, p_max, n_bins + 1)

        # Build histogram dataframe
        hist_df = pd.DataFrame(
            {
                "bin_left": bins[:-1],
                "bin_right": bins[1:],
            }
        )
        hist_df["bin_center"] = (hist_df["bin_left"] + hist_df["bin_right"]) / 2.0

        for name in names:
            counts, _ = np.histogram(
                pred_series[name].to_numpy(),
                bins=bins,
                weights=weight_series[name].to_numpy(),
            )
            if density:
                total = counts.sum()
                counts = counts / total if total > 0 else counts
            hist_df[f"h_{name}"] = counts

        # Describe table per model (unweighted describe)
        desc_rows = []
        for name in names:
            s = pred_series[name]
            d = s.describe()
            d["model"] = name
            desc_rows.append(d)
        desc_df = pd.DataFrame(desc_rows).set_index("model")

        self._data = {
            "describe": desc_df,
            "histogram": hist_df,
        }

    def get_data(self):  # override for clarity in types
        if self._data is None:
            raise RuntimeError(
                "Analysis has not been executed yet. Call run(df) first."
            )
        return self._data

    def get_data_and_metadata(self) -> Dict[str, Dict[str, object]]:
        data = self.get_data()
        names = sorted([c[2:] for c in data["histogram"].columns if c.startswith("h_")])
        meta = {
            "names": names,
            "density": bool(self.config.get("density", False)),
            "title": self.config.get("title", "Prediction Histogram"),
            "xscale": self.config.get("xscale"),
            "yscale": self.config.get("yscale"),
        }
        return {"prediction_analysis": {"data": data, "metadata": meta}}

    def plot(self, ax, theme: Dict, color_provider, panel_cfg: Optional[Dict] = None):
        payload_data = self.get_data()
        hist_df = payload_data.get("histogram")
        desc_df = payload_data.get("describe")
        if hist_df is None or desc_df is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        names = sorted([c[2:] for c in hist_df.columns if c.startswith("h_")])
        colors = color_provider(len(names))

        # Build bin edges from left edges and the last right edge
        bin_left = hist_df["bin_left"].to_numpy()
        bin_right = hist_df["bin_right"].to_numpy()
        if bin_right.size == 0:
            ax.text(0.5, 0.5, "Empty histogram", ha="center", va="center")
            return
        edges = np.concatenate([bin_left, bin_right[-1:]])

        # Plot histogram as step lines
        for name, color in zip(names, colors, strict=False):
            counts = hist_df[f"h_{name}"].to_numpy()
            if counts.size == 0:
                continue
            y_step = np.concatenate([counts, counts[-1:]])
            ax.step(edges, y_step, where="post", color=color, label=name)

        ax.set_xlabel("Prediction")
        ylabel = "Frequency" + (
            " (density)" if bool(self.config.get("density", False)) else ""
        )
        ax.set_ylabel(ylabel)

        # Axis scaling: panel overrides config
        xscale = panel_cfg.get("xscale") if isinstance(panel_cfg, dict) else None
        if not xscale:
            xscale = self.config.get("xscale")
        if isinstance(xscale, str) and xscale:
            ax.set_xscale(xscale)

        yscale = panel_cfg.get("yscale") if isinstance(panel_cfg, dict) else None
        if not yscale:
            yscale = self.config.get("yscale")
        if isinstance(yscale, str) and yscale:
            ax.set_yscale(yscale)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=False)
