from __future__ import annotations

from typing import Any, Dict, Optional

from ..analysis_class import BaseAnalysis  # type: ignore


class GlobalAnalysesRunner(BaseAnalysis):
    """Global analyses orchestrator with minimal subclass surface.

    Responsibilities:
        1. __init__ builds configuration
        2. run_analysis() performs heavy computations and stores raw objects
        3. create_artifacts() transforms raw objects into figures + JSON tables via add_artifact

    save_to_mlflow + get_artifacts come from the base so a node does:
        runner.run_analysis(); runner.create_artifacts(); runner.save_to_mlflow(...)
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
        """Initialize the global analyses runner."""
        from model_monitoring.plotting.core import set_plot_theme

        super().__init__(artifact_root="global_analyses")
        self.analysis_name = "Global Analyses"
        self.train_df = train_df
        self.test_df = test_df
        self.target_column = target_column
        self.prediction_column = prediction_column
    # Use old_model_column exactly as provided (no normalization)
        self.old_model_column = old_model_column
        self.params = params or {}
        self.run_id = run_id
        self.model_metrics = model_metrics
        self._extract_run_id = resolved_run_extractor

        theme = self.params.get("plot_theme") or {}
        if theme:
            set_plot_theme(theme)

        weight_col = self.params.get("weight_column")
        if weight_col and weight_col not in self.train_df.columns:
            # Silently drop invalid weight column
            weight_col = None
            weight_col = None
        self.weight_col = weight_col

        self.calibration_bins = int(self.params.get("calibration_bins", 20))
        self.resolved_run = self._extract_run_id(self.run_id, self.model_metrics)

        # Build model spec. Always include new model. Optionally add old model if provided.
        models_cfg: Dict[str, Dict[str, Any]] = {
            "New model": {"pred_col": prediction_column, "name": "new"}
        }
        if old_model_column:
            if (
                old_model_column in self.train_df.columns
                and old_model_column in self.test_df.columns
            ):
                models_cfg["Old model"] = {
                    "pred_col": old_model_column,
                    "name": "old",
                }
            # If missing we just proceed with new model only (no noisy prints)
        self.models_cfg = models_cfg

        obs = {"target_col": target_column}
        if self.weight_col:
            obs["weight_col"] = self.weight_col
        self.observation_spec = obs

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

    # ---- analysis execution (no artifact logging here) --------------------
    def run_analysis(self) -> None:
        """Replicate notebook-style workflow using GlobalAnalysisDataBuilder for train/test.

        Builds separate builders for train and test datasets, registers analyses,
        loads data (if needed), calculates, and stores analysis objects for later
        artifact creation.
        """
        from model_monitoring.global_analyses import GlobalAnalysisDataBuilder

        self._subset_results = {}
        analyses_cfg = [
            ("lorenz_curve", self.func_dict_lorenz),
            ("calibration_curve", self.func_dict_calibration),
            ("prediction_analysis", self.func_dict_prediction),
        ]

        for subset_label, df in ("train", self.train_df), ("test", self.test_df):
            builder = GlobalAnalysisDataBuilder(data=df, extra_cols=[])
            for key, cfg in analyses_cfg:
                builder.add_analysis(key, cfg)
            builder.load_data()
            builder.calculate()
            analyses_objs = builder.get_analyses_objects()
            self._subset_results[subset_label] = {
                "builder": builder,
                "analyses_objs": analyses_objs,
            }

    # ---- artifact materialization ----------------------------------------
    def create_artifacts(self) -> None:
        """Create combined panel figures per subset and store artifacts.

        Returns analyses objects & figure per subset similarly to notebook logic.
        """
        from model_monitoring.plotting import plot_global_statistics

        report_panels = [
            {"title": "Lorenz Curve", "type": "lorenz_curve"},
            {"title": "Calibration Curve", "type": "calibration_curve"},
            {
                "title": "Prediction histogram",
                "type": "prediction_analysis",
                "xscale": "log",
                "yscale": "log",
            },
        ]

        self._figures_by_subset = {}

        for subset_label, payload in self._subset_results.items():
            analyses_objs = payload["analyses_objs"]
            fig_axes = plot_global_statistics(
                analyses_objs, panel_configs=report_panels, show=False
            )
            fig = None
            if isinstance(fig_axes, tuple) and len(fig_axes) >= 1:
                fig = fig_axes[0]
            self._figures_by_subset[subset_label] = {
                "analyses_objs": analyses_objs,
                "figure": fig,
            }

            # Build tables extraction combining all analyses metadata
            tables: Dict[str, Any] = {}
            for key, analysis_obj in analyses_objs.items():
                meta = analysis_obj.get_data_and_metadata()
                # Flatten meta into serializable dict
                ser_meta: Dict[str, Any] = {}
                for mk, mv in meta.items():
                    entry = dict(mv)
                    data = entry.get("data")
                    if hasattr(data, "to_dict") and not isinstance(data, dict):
                        entry["data"] = data.to_dict(orient="records")
                    elif isinstance(data, dict):
                        converted = {}
                        for dk, dv in data.items():
                            if hasattr(dv, "to_dict"):
                                converted[dk] = dv.to_dict(orient="records")
                            else:
                                converted[dk] = dv
                        entry["data"] = converted
                    ser_meta[mk] = entry
                tables[key] = ser_meta

            self.add_artifact(
                f"global_panels_{subset_label}",
                figures=fig,
                tables=tables,
                include_config=True,
                config={"panels": report_panels},
            )

        return self._figures_by_subset

    def get_artifacts(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of stored artifacts (figures objects & tables).

        Shape: {artifact_name: {"figures": {...}, "tables": {...}}}
        Useful if caller wants to inspect or test without relying on MLflow side-effects.
        """
        return {
            k: {"figures": v.figures, "tables": v.tables}
            for k, v in self.artifacts.items()
        }


def build_and_run_global_analyses(**kwargs) -> None:
    runner = GlobalAnalysesRunner(**kwargs)
    runner.run_analysis()
    runner.create_artifacts()
    runner.save_to_mlflow(identifier_run=runner.resolved_run)