from __future__ import annotations

from typing import Any, Dict, Optional


class PDPAnalysesRunner:
    """PDP analyses orchestrator with minimal subclass surface.

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
        weight_column: Optional[str] = None,
        params: Optional[dict],
        run_id: Optional[str],
        resolved_run_extractor,
        trained_model: Any,
    ) -> None:
        """Initialize the PDP analyses runner (path-based ingestion)."""
        from predlab.plotting.core import set_plot_theme

        self._artifact_root = "pdp_analyses"
        self.analysis_name = "PDP Analyses"
        self.artifacts = {}  # Store artifacts for get_artifacts method
        self.train_df_path = train_df_path
        self.test_df_path = test_df_path
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.old_model_column = old_model_column
        self.weight_column = weight_column
        self.params = params or {}
        self.run_id = run_id
        self.trained_model = trained_model
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

        self.resolved_run = self._extract_run_id(self.run_id)

        # Get feature columns from parameters first, then model, then infer
        feature_cols = self.params.get("data_preparation", {}).get("feature_columns")
        if not feature_cols:
            # Try to get from model
            feature_cols = getattr(self.trained_model, 'feature_names_', None)
        if not feature_cols:
            # Fallback: infer from available columns
            excluded_cols = {
                self.target_column,
                self.prediction_column,
                self.old_model_column,
                self.weight_col,
                'weight'
            }
            feature_cols = [
                col for col in self._train_cols
                if col not in excluded_cols and col in self._test_cols
            ]
        self.feature_cols = feature_cols

        # Build segmentation strategies
        self.segments = self._build_segments()

        # Build PDP configuration
        self.func_dict_pdp = self._build_pdp_config()

        # Build report panels configuration
        self.report_panels = self._build_report_panels()

    def _build_segments(self):
        """Build segmentation strategies based on available columns and params."""
        from predlab import SegmentCustom

        segments = []
        
        # Get feature columns from data preparation parameters
        feature_columns = self.params.get("data_preparation", {}).get("feature_columns", [])
        if not feature_columns:
            # Fallback: use feature columns from model or infer from available columns
            feature_columns = self.feature_cols
        
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
                    
                    segments.append(
                        SegmentCustom(
                            seg_col=feature_col,
                            seg_name=f"{feature_col}_segment",
                            bins=bins,
                            bin_labels=bin_labels,
                        )
                    )

        return segments

    def _build_pdp_config(self) -> Dict[str, Any]:
        """Build PDP configuration dictionary."""
        return {
            "model": {
                "model": self.trained_model,
                "name": "trained_model",
                "feature_cols": self.feature_cols,
            },
            "target_col": self.target_column,
            "weight_col": self.weight_col or "weight",
        }

    def _build_report_panels(self):
        """Build report panels configuration for PDP plots."""
        return [
            {
                "title": "Partial Dependence Plot",
                "type": "metric",
                "metric_col": "pdp",
                "plot_type": "line",
            },
        ]

    # ---- analysis execution (no artifact logging here) --------------------
    def run_analysis(self) -> None:
        """Execute PDP analyses directly from parquet paths (no DataFrame inputs)."""
        from predlab.model_analyses import ModelAnalysisDataBuilder

        self._subset_results = {}

        path_map = {"train": self.train_df_path, "test": self.test_df_path}

        for subset_label, data_path in path_map.items():
            # Get extra columns needed for analysis
            extra_cols = []
            if self.weight_col:
                extra_cols.append(self.weight_col)
            extra_cols.extend([self.target_column])
            if self.old_model_column:
                extra_cols.append(self.old_model_column)
            
            # Add feature columns
            extra_cols.extend(self.feature_cols)
            
            # Add segmentation columns
            for segment in self.segments:
                if hasattr(segment, 'seg_col'):
                    extra_cols.append(segment.seg_col)

            # Remove duplicates
            extra_cols = list(set(extra_cols))

            # Initialize ModelAnalysisDataBuilder
            builder = ModelAnalysisDataBuilder(extra_cols=extra_cols)

            # Register PDP analysis
            builder.add_analysis("PDP", self.func_dict_pdp)

            # Add segments (used to derive PDP grids for the corresponding features)
            for segment in self.segments:
                builder.add_segment(segment)

            # Load data and apply treatments
            builder.load_data(data=data_path)
            builder.apply_treatments()
            builder.apply_segments()

            # Calculate PDP analysis (not statistics - actually run PDP algorithm)
            builder.calculate(analysis_name="PDP")
            
            # Get PDP results
            pdp_analysis = builder.get_analysis("PDP")
            pdp_data = pdp_analysis._data  # This should contain the PDP results
            
            # Group PDP data by segment for plotting
            dict_stats = {}
            if not pdp_data.empty:
                for segment_name in pdp_data['segment'].unique():
                    segment_data = pdp_data[pdp_data['segment'] == segment_name].copy()
                    dict_stats[segment_name] = segment_data
            
            # No aggregate stats for PDP
            agg_stats = None

            self._subset_results[subset_label] = {
                "builder": builder,
                "dict_stats": dict_stats,
                "agg_stats": agg_stats,
            }

    def _create_pdp_plot(self, stats_df, segment_name):
        """Create a PDP plot for a given segment's data."""
        import matplotlib.pyplot as plt
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # PDP data should have columns: ["segment", "bin", "eval_value", "prediction", "pdp"]
            if 'eval_value' in stats_df.columns and 'pdp' in stats_df.columns:
                # Sort by eval_value for proper line plotting
                plot_data = stats_df.sort_values('eval_value')
                
                ax.plot(plot_data['eval_value'], plot_data['pdp'], 
                       marker='o', linewidth=2, markersize=6)
                
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'Partial Dependence Plot - {segment_name}')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                return fig
            else:
                # Fallback: create a simple text plot indicating missing data
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, f'PDP data structure issue for {segment_name}\nColumns: {list(stats_df.columns)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'PDP Plot - {segment_name} (Data Issue)')
                return fig
                
        except Exception as e:
            # Create error plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Error creating PDP plot for {segment_name}:\n{e!s}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'PDP Plot - {segment_name} (Error)')
            return fig

    def add_artifact(self, name: str, figures: Any, tables: Any, include_config: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
        """Add an artifact to the collection."""
        # Add config to tables if requested
        if include_config and config:
            if tables is None:
                tables = {}
            if isinstance(tables, dict):
                tables = {"_config": config, **tables}
            else:
                # Convert to dict first if it has to_dict method
                if hasattr(tables, "to_dict"):
                    tables = {"_config": config, **tables.to_dict()}
                else:
                    tables = {"_config": config, "data": tables}
        
        self.artifacts[name] = {"figures": figures, "tables": tables}

    # ---- artifact materialization ----------------------------------------
    def create_artifacts(self) -> None:
        """Create PDP panel figures per subset and store artifacts."""
        self._figures_by_subset = {}

        for subset_label, payload in self._subset_results.items():
            dict_stats = payload["dict_stats"]
            agg_stats = payload["agg_stats"]

            figures_by_segment = {}
            tables_by_segment = {}

            # Generate plots for each segment
            for segment_name, stats_df in dict_stats.items():
                try:
                    fig = self._create_pdp_plot(stats_df, segment_name)
                    figures_by_segment[segment_name] = fig

                    # Convert stats_df to serializable format
                    tables_by_segment[segment_name] = {
                        "pdp_data": stats_df.to_dict(orient="records"),
                        "metadata": {
                            "segment_name": segment_name,
                            "n_rows": len(stats_df),
                            "columns": list(stats_df.columns),
                        },
                    }
                except Exception as e:
                    print(f"Warning: Could not generate PDP plot for segment {segment_name}: {e}")
                    # Create error figure as fallback
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.text(0.5, 0.5, f'Error: {e!s}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'PDP Plot Error - {segment_name}')
                    figures_by_segment[segment_name] = fig

            # Store metadata
            metadata_tables = {
                "metadata": {
                    "subset": subset_label,
                    "n_segments": len(dict_stats),
                    "segment_names": list(dict_stats.keys()),
                    "feature_columns": self.feature_cols,
                    "model_name": self.trained_model.__class__.__name__ if self.trained_model else 'Unknown',
                },
            }

            # Combine tables
            all_tables = {
                **tables_by_segment,
                "metadata": metadata_tables,
            }

            self._figures_by_subset[subset_label] = {
                "figures": figures_by_segment,
                "dict_stats": dict_stats,
            }

            self.add_artifact(
                f"pdp_panels_{subset_label}",
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
            k: {"figures": v["figures"], "tables": v["tables"]}
            for k, v in self.artifacts.items()
        }


def build_and_run_pdp_analyses(mlflow_saver, **kwargs) -> None:
    """Build and run PDP analyses with the provided configuration."""
    runner = PDPAnalysesRunner(**kwargs)
    runner.run_analysis()
    runner.create_artifacts()
    mlflow_saver.save_to_mlflow(runner)
