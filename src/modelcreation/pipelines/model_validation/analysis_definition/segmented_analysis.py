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
        self._extract_run_id = resolved_run_extractor

        theme = self.params.get("plot_theme") or {}
        if theme:
            set_plot_theme(theme)

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
            weight_col = None
        self.weight_col = weight_col

        self.resolved_run = self._extract_run_id(self.run_id)

        self.pred_dict = {
            "target_model": {
                "sel_col": weight_col or "weight",
                "pred_col": prediction_column,
                "target_col": target_column,
            }
        }
        if old_model_column and old_model_column in self._train_cols and old_model_column in self._test_cols:
            self.pred_dict["old_model"] = {
                "sel_col": weight_col or "weight",
                "pred_col": old_model_column,
                "target_col": target_column,
            }

        self.segments = self._build_segments()
        self.func_dict = self._build_func_dict()
        self.report_panels = self._build_report_panels()

    def _build_segments(self) -> List[Any]:
        """Build segmentation strategies based on available columns and params."""
        from model_monitoring import SegmentCustom, SegmentCategorical

        segments = []
        
        # Get feature columns from data preparation parameters
        feature_columns = self.params.get("data_preparation", {}).get("feature_columns", [])
        if not feature_columns:
            # Fallback: infer feature columns from available columns
            excluded_cols = {
                self.target_column, 
                self.prediction_column, 
                self.old_model_column, 
                self.weight_col, 
                'weight'
            }
            feature_columns = [
                col for col in self._train_cols 
                if col not in excluded_cols and col in self._test_cols
            ]
        
        # Get categorical features list
        categorical_features = self.params.get("categorical_features", [])
        
        # Get segment configurations from parameters
        segment_configs = self.params.get("segments", {})
        feature_binning = self.params.get("feature_binning", {})
        default_bins = feature_binning.get("default_bins", 5)
        
        # Build segments from explicit segment configurations first
        for segment_name, segment_config in segment_configs.items():
            # Extract column name from segment name (remove suffix like _group, _level, etc.)
            potential_col_names = [
                segment_name,
                segment_name.replace("_group", ""),
                segment_name.replace("_level", ""),
                segment_name.replace("_segment", ""),
            ]
            
            # Find the actual column that exists in data
            seg_col = None
            for col_name in potential_col_names:
                if col_name in self._train_cols and col_name in self._test_cols:
                    seg_col = col_name
                    break
            
            if seg_col:
                bins = segment_config.get("bins", default_bins)
                bin_labels = segment_config.get("bin_labels")
                
                if seg_col in categorical_features:
                    # Use categorical segmentation for categorical features
                    mapping = segment_config.get("mapping")  # Optional mapping for categories
                    segments.append(
                        SegmentCategorical(
                            seg_col=seg_col,
                            seg_name=segment_name,
                            mapping=mapping,
                        )
                    )
                else:
                    # Use numeric segmentation for continuous features
                    segments.append(
                        SegmentCustom(
                            seg_col=seg_col,
                            seg_name=segment_name,
                            bins=bins,
                            bin_labels=bin_labels,
                        )
                    )
        
        # If no explicit segments configured, create default segments for feature columns
        if not segments:
            for feature_col in feature_columns:
                if feature_col in self._train_cols and feature_col in self._test_cols:
                    # Check if there's specific binning configuration for this feature
                    feature_config = feature_binning.get(feature_col, {})
                    bins = feature_config.get("bins", default_bins)
                    bin_labels = feature_config.get("labels")
                    
                    # Check if column is categorical or numeric
                    if feature_col in categorical_features:
                        # Use categorical segmentation for categorical features
                        mapping = feature_config.get("mapping")  # Optional mapping
                        segments.append(
                            SegmentCategorical(
                                seg_col=feature_col,
                                seg_name=f"{feature_col}_segment",
                                mapping=mapping,
                            )
                        )
                    else:
                        # Use numeric segmentation for continuous features
                        segments.append(
                            SegmentCustom(
                                seg_col=feature_col,
                                seg_name=f"{feature_col}_segment",
                                bins=bins,
                                bin_labels=bin_labels,
                            )
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
