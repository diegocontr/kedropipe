from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..analysis_class import BaseAnalysis  # type: ignore


class SegmentedAnalysesRunner(BaseAnalysis):
    """Segmented analyses orchestrator with minimal subclass surface.

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
        train_df_path: str,
        test_df_path: str,
        target_column: str,
        prediction_column: str,
        old_model_column: Optional[str],
        params: Optional[dict],
        run_id: Optional[str],
        resolved_run_extractor,
        model_metrics: Optional[Any],
    ) -> None:
        """Initialize the segmented analyses runner (path-based ingestion)."""
        from model_monitoring.plotting.core import set_plot_theme

        super().__init__(artifact_root="segmented_analyses")
        self.analysis_name = "Segmented Analyses"
        self.train_df_path = train_df_path
        self.test_df_path = test_df_path
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.old_model_column = old_model_column
        self.params = params or {}
        self.run_id = run_id
        self.model_metrics = model_metrics
        self._extract_run_id = resolved_run_extractor

        theme = self.params.get("plot_theme") or {}
        if theme:
            set_plot_theme(theme)

        # Determine column availability using parquet schema (no full data load)
        import pyarrow.parquet as pq

        def _schema_cols(path: str) -> set[str]:
            try:
                return set(pq.ParquetFile(path).schema.names)
            except Exception:
                return set()

        self._train_cols = _schema_cols(self.train_df_path)
        self._test_cols = _schema_cols(self.test_df_path)

        weight_col = self.params.get("weight_column")
        if weight_col and weight_col not in self._train_cols:
            weight_col = None  # silently drop invalid
        self.weight_col = weight_col  # may be None

        self.resolved_run = self._extract_run_id(self.run_id, self.model_metrics)

        # Build prediction dictionary - using simple structure for segmented analysis
        self.pred_dict = {
            "target_model": {
                "sel_col": weight_col or "weight",  # fallback to "weight" if None
                "pred_col": prediction_column,
                "target_col": target_column,
            }
        }

        # If old model is available, add it to pred_dict
        if (
            old_model_column
            and old_model_column in self._train_cols
            and old_model_column in self._test_cols
        ):
            self.pred_dict["old_model"] = {
                "sel_col": weight_col or "weight",
                "pred_col": old_model_column,
                "target_col": target_column,
            }

        # Build segmentation strategies
        self.segments = self._build_segments()

        # Build statistics configuration
        self.func_dict = self._build_func_dict()

        # Build report panels configuration
        self.report_panels = self._build_report_panels()

    def _build_segments(self) -> List[Any]:
        """Build segmentation strategies based on available columns and params."""
        from model_monitoring import SegmentCategorical, SegmentCustom

        segments = []
        segment_configs = self.params.get("segments", {})

        # Age group segment
        age_config = segment_configs.get("age_group", {})
        if "age" in self._train_cols and "age" in self._test_cols:
            segments.append(
                SegmentCustom(
                    seg_col="age",
                    seg_name="age_group",
                    bins=age_config.get("bins", [18, 30, 45, 60, 75]),
                    bin_labels=age_config.get("bin_labels", ["18-29", "30-44", "45-59", "60+"]),
                )
            )

        # Income level segment
        income_config = segment_configs.get("income_level", {})
        if "income" in self._train_cols and "income" in self._test_cols:
            segments.append(
                SegmentCustom(
                    seg_col="income",
                    seg_name="income_level",
                    bins=income_config.get("bins", 5),
                )
            )

        # Credit score segment
        credit_config = segment_configs.get("credit_score_group", {})
        if "credit_score" in self._train_cols and "credit_score" in self._test_cols:
            segments.append(
                SegmentCustom(
                    seg_col="credit_score",
                    seg_name="credit_score_group",
                    bins=credit_config.get("bins", [300, 500, 650, 750, 850]),
                    bin_labels=credit_config.get("bin_labels", ["Poor", "Fair", "Good", "Excellent"]),
                )
            )

        # Region segment (categorical)
        if "region" in self._train_cols and "region" in self._test_cols:
            segments.append(
                SegmentCategorical(seg_col="region", seg_name="region_segment")
            )

        return segments

    def _build_func_dict(self) -> Dict[str, Any]:
        """Build statistics configuration dictionary."""
        # Primary model column
        pred_col = self.prediction_column
        target_col = self.target_column
        weight_col = self.weight_col or "weight"

        func_dict = {
            "aggregations": {
                pred_col: ("weighted_mean", [pred_col, weight_col, weight_col]),
                target_col: ("observed_charge", [target_col, weight_col, weight_col]),
                "gini": ("gini", [target_col, pred_col, weight_col, weight_col]),
                "exposure(k)": (lambda df, e: df[e].sum() / 1000, [weight_col]),
            },
            "post_aggregations": {
                "S/PP": ("division", [target_col, pred_col]),
            },
        }

        # Add old model statistics if available
        if "old_model" in self.pred_dict:
            old_pred_col = self.old_model_column
            func_dict["aggregations"][f"{old_pred_col}"] = (
                "weighted_mean",
                [old_pred_col, weight_col, weight_col],
            )
            func_dict["post_aggregations"]["S/PP_old"] = (
                "division",
                [target_col, old_pred_col],
            )

        return func_dict

    def _build_report_panels(self) -> List[Dict[str, Any]]:
        """Build report panels configuration."""
        pred_col = self.prediction_column
        target_col = self.target_column
        
        pred_cols = [pred_col]
        spp_cols = ["S/PP"]
        
        # Add old model columns if available
        if "old_model" in self.pred_dict:
            pred_cols.append(self.old_model_column)
            spp_cols.append("S/PP_old")

        report_panels = [
            {
                "title": "Prediction vs. Target",
                "type": "pred_vs_target",
                "pred_col": pred_cols,
                "target_col": target_col,
                "plot_type": "line",
                "show_mean_line": "all",
            },
            {
                "title": "S/PP (Observed/Predicted Ratio)",
                "type": "spp",
                "spp_col": spp_cols,
                "plot_type": "line",
                "show_mean_line": False,
            },
            {
                "title": "Prediction vs. Target",  # Twin plot on same axis
                "type": "exposure",
                "metric_col": "exposure(k)",
                "plot_type": "bar",
                "colors": ["#9F9DAA"],
            },
            {
                "title": "Gini",  # Twin plot on same axis
                "type": "metric",
                "metric_col": "gini",
                "plot_type": "line",
                "colors": ["#3D27B9"],
            },
        ]

        return report_panels

    # ---- analysis execution (no artifact logging here) --------------------
    def run_analysis(self) -> None:
        """Execute segmented analyses directly from parquet paths (no DataFrame inputs)."""
        from model_monitoring import AnalysisDataBuilder, calculate_statistics

        self._subset_results = {}

        path_map = {"train": self.train_df_path, "test": self.test_df_path}

        for subset_label, data_path in path_map.items():
            # Get extra columns needed for analysis
            extra_cols = []
            if self.weight_col:
                extra_cols.append(self.weight_col)
            extra_cols.extend([self.prediction_column, self.target_column])
            if self.old_model_column:
                extra_cols.append(self.old_model_column)
            
            # Add segmentation columns
            for segment in self.segments:
                if hasattr(segment, 'seg_col'):
                    extra_cols.append(segment.seg_col)

            # Remove duplicates
            extra_cols = list(set(extra_cols))

            # Initialize AnalysisDataBuilder
            builder = AnalysisDataBuilder(data=data_path, extra_cols=extra_cols)

            # Add segments
            for segment in self.segments:
                builder.add_segment(segment)

            # Load data and apply all defined steps
            builder.load_data()
            builder.apply_treatments()
            builder.apply_segments()

            # Calculate statistics
            dict_stats, agg_stats = calculate_statistics(
                builder, self.func_dict, bootstrap=False
            )

            self._subset_results[subset_label] = {
                "builder": builder,
                "dict_stats": dict_stats,
                "agg_stats": agg_stats,
            }

    # ---- artifact materialization ----------------------------------------
    def create_artifacts(self) -> None:
        """Create segmented panel figures per subset and store artifacts."""
        from model_monitoring.plotting import plot_segment_statistics

        self._figures_by_subset = {}

        for subset_label, payload in self._subset_results.items():
            dict_stats = payload["dict_stats"]
            agg_stats = payload["agg_stats"]

            figures_by_segment = {}
            tables_by_segment = {}

            # Generate plots for each segment
            for segment_name, stats_df in dict_stats.items():
                try:
                    plot_obj = plot_segment_statistics(
                        stats_df, panel_configs=self.report_panels, agg_stats=agg_stats, show=False
                    )
                    # Extract the figure from the plot object
                    if isinstance(plot_obj, tuple) and len(plot_obj) >= 1:
                        fig = plot_obj[0]  # Usually returns (fig, axes)
                    else:
                        fig = plot_obj
                    
                    figures_by_segment[segment_name] = fig

                    # Convert stats_df to serializable format
                    tables_by_segment[segment_name] = {
                        "statistics": stats_df.to_dict(orient="records"),
                        "metadata": {
                            "segment_name": segment_name,
                            "n_rows": len(stats_df),
                            "columns": list(stats_df.columns),
                        },
                    }
                except Exception as e:
                    print(f"Warning: Could not generate plot for segment {segment_name}: {e}")

            # Store aggregate statistics
            aggregate_tables = {
                "aggregate_statistics": agg_stats.to_dict() if agg_stats is not None else {},
                "metadata": {
                    "subset": subset_label,
                    "n_segments": len(dict_stats),
                    "segment_names": list(dict_stats.keys()),
                },
            }

            # Combine tables
            all_tables = {
                **tables_by_segment,
                "aggregate": aggregate_tables,
            }

            self._figures_by_subset[subset_label] = {
                "figures": figures_by_segment,
                "dict_stats": dict_stats,
                "agg_stats": agg_stats,
            }

            self.add_artifact(
                f"segmented_panels_{subset_label}",
                figures=figures_by_segment,
                tables=all_tables,
                include_config=True,
                config={"panels": self.report_panels, "segments": [s.seg_name for s in self.segments]},
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


def build_and_run_segmented_analyses(**kwargs) -> None:
    """Build and run segmented analyses with the provided configuration."""
    runner = SegmentedAnalysesRunner(**kwargs)
    runner.run_analysis()
    runner.create_artifacts()
    runner.save_to_mlflow(identifier_run=runner.resolved_run)
