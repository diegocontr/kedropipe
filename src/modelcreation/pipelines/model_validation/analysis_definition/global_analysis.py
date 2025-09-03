from __future__ import annotations

from typing import Any, Dict, Optional

from ..analysis_class import ModelAnalysis  # type: ignore


class GlobalAnalysesRunner:
    """Notebook-style builder & executor for global analyses.

    Mirrors the structure of the example notebook: defines three configs
    (Lorenz, Calibration, Prediction) then runs them for train & test datasets,
    logging artifacts to MLflow via the shared ModelAnalysis utility.
    """

    def __init__(
        self,
        *,
        train_df,
        test_df,
        target_column: str,
        prediction_column: str,
        old_model_column: Optional[str],
        params: Optional[dict],
        run_id: Optional[str],
        resolved_run_extractor,
        model_metrics: Optional[Any],
    ) -> None:
        """Initialize the global analyses runner.

        Keeps structure close to the reference notebook while encapsulating logic.
        """
        from model_monitoring.plotting.core import set_plot_theme

        self.train_df = train_df
        self.test_df = test_df
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.old_model_column = old_model_column
        self.params = params or {}
        self.run_id = run_id
        self.model_metrics = model_metrics
        self._extract_run_id = resolved_run_extractor

        # Theme
        theme = self.params.get("plot_theme") or {}
        if theme:
            set_plot_theme(theme)

        # Weight column handling
        weight_col = self.params.get("weight_column")
        if weight_col and weight_col not in self.train_df.columns:
            print(
                f"Warning: weight_column '{weight_col}' not found. Proceeding without weights."
            )
            weight_col = None
        self.weight_col = weight_col

        self.calibration_bins = int(self.params.get("calibration_bins", 20))

        self.resolved_run = self._extract_run_id(self.run_id, self.model_metrics)

        # Build model spec (optional old model)
        models_cfg: Dict[str, Dict[str, Any]] = {
            "New model": {"pred_col": prediction_column, "name": "new"}
        }
        if (
            old_model_column
            and old_model_column in self.train_df.columns
            and old_model_column in self.test_df.columns
        ):
            models_cfg["Old model"] = {"pred_col": old_model_column, "name": "old"}
        self.models_cfg = models_cfg

        # Observation spec
        obs = {"target_col": target_column}
        if self.weight_col:
            obs["weight_col"] = self.weight_col
        self.observation_spec = obs

        # Build configs (mirroring notebook variable names)
        self.func_dict_lorenz = {
            "models": self.models_cfg,
            "observation": self.observation_spec,
            "weight_col": self.weight_col,
            "n_bins": 20,
            "ascending": True,
            "title": "Lorenz Curve",
        }
        self.func_dict_calibration = {
            "models": self.models_cfg,
            "observation": self.observation_spec,
            "weight_col": self.weight_col,
            "n_bins": self.calibration_bins,
            "title": "Calibration Curve",
        }
        self.func_dict_prediction = {
            "models": self.models_cfg,
            "weight_col": self.weight_col,
            "n_bins": 40,
            "xscale": "log",
            "yscale": "log",
            "title": "Prediction Histogram",
        }

    # ---- internal helpers -------------------------------------------------
    def _make_callable(self, analysis_key: str, cfg: dict):
        """Wrap a global analysis into a simple callable returning (fig, tables)."""
        from model_monitoring.global_analyses.analyses import ANALYSIS_REGISTRY
        from model_monitoring.plotting import plot_global_statistics

        AnalysisCls = ANALYSIS_REGISTRY[analysis_key]

        def _run(df):
            analysis_obj = AnalysisCls(cfg)
            analysis_obj.run(df)
            payload = analysis_obj.get_data_and_metadata()
            tables: Dict[str, Any] = {}
            for k, v in payload.items():
                entry = dict(v)
                data = entry.get("data")
                if hasattr(data, "to_dict") and not isinstance(data, dict):
                    entry["data"] = data.to_dict(orient="records")
                elif isinstance(data, dict):  # prediction_analysis returns dict of DFs
                    converted = {}
                    for dk, dv in data.items():
                        if hasattr(dv, "to_dict"):
                            converted[dk] = dv.to_dict(orient="records")
                        else:
                            converted[dk] = dv
                    entry["data"] = converted
                tables[k] = entry
            panel_cfg = [
                {"type": analysis_key, "title": cfg.get("title", analysis_key)}
            ]
            fig_axes = plot_global_statistics(
                {analysis_key: analysis_obj}, panel_configs=panel_cfg, show=False
            )
            fig = None
            if isinstance(fig_axes, tuple) and len(fig_axes) >= 1:
                fig = fig_axes[0]
            return fig, tables

        return _run

    # ---- public API -------------------------------------------------------
    def run(self) -> None:
        analyses_sequence = [
            ("lorenz_curve", self.func_dict_lorenz),
            ("calibration_curve", self.func_dict_calibration),
            ("prediction_analysis", self.func_dict_prediction),
        ]
        for subset_label, df in ("train", self.train_df), ("test", self.test_df):
            for key, cfg in analyses_sequence:
                title = f"{cfg.get('title', key)} ({subset_label})"
                callable_func = self._make_callable(key, cfg)
                ma = ModelAnalysis(callable_func, title, config=cfg)
                ma.run_analysis(key, df=df)
                ma.save_to_mlflow(identifier_run=self.resolved_run)


def build_and_run_global_analyses(**kwargs) -> None:
    runner = GlobalAnalysesRunner(**kwargs)
    runner.run()
