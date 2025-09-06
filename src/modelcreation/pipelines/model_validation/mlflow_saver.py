from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow
from .serialization import serialize_tables


class MLflowArtifactSaver:
    """Class responsible for saving analysis artifacts to MLflow.
    
    This class handles the MLflow integration separately from analysis logic,
    allowing analysis classes to focus on their core responsibilities.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        *,
        figure_formats: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize the MLflow artifact saver.

        Parameters
        ----------
        run_id : str, optional
            MLflow run ID to use for saving artifacts. If None, will use active run or create new one.
        figure_formats : iterable[str], optional
            Image formats to persist figures as (default: ["png"]).
        """
        self.run_id = run_id
        self.figure_formats = list(figure_formats or ["png"])

    def save_to_mlflow(
        self, 
        analysis_runner: Any,
        base_artifact_path: Optional[str] = None
    ) -> None:
        """Save artifacts from an analysis runner to MLflow.

        Parameters
        ----------
        analysis_runner : Any
            Analysis runner instance that has a get_artifacts() method.
        base_artifact_path : str, optional
            Base path under which artifacts are logged. If None, uses analysis_runner's
            artifact_root or a default based on analysis_name.
        """
        if not hasattr(analysis_runner, 'get_artifacts'):
            raise ValueError("Analysis runner must have a get_artifacts() method")

        # Get artifacts from the runner
        artifacts = analysis_runner.get_artifacts()
        
        # Determine artifact root path
        if base_artifact_path:
            artifact_root = base_artifact_path
        elif hasattr(analysis_runner, '_artifact_root'):
            artifact_root = analysis_runner._artifact_root
        elif hasattr(analysis_runner, 'analysis_name'):
            artifact_root = analysis_runner.analysis_name.lower().replace(" ", "_")
        else:
            artifact_root = "analysis"

        # Ensure we have an active MLflow run
        self._ensure_mlflow_run()

        # Save all artifacts
        for artifact_name, artifact_data in artifacts.items():
            sub_path = f"{artifact_root}/{artifact_name}"
            
            figures = artifact_data.get("figures", {})
            tables = artifact_data.get("tables", {})
            
            # Normalize artifacts before logging
            normalized_figures = self._normalize_figures(figures)
            normalized_tables = self._normalize_tables(tables)
            
            self._log_figures(normalized_figures, sub_path, analysis_runner)
            self._log_tables(normalized_tables, artifact_name, sub_path, analysis_runner)

    def _normalize_figures(self, figures: Any) -> Dict[str, Any]:
        """Normalize figures to a standard dictionary format."""
        if figures is None:
            return {}
        if hasattr(figures, "savefig"):
            return {"plot": figures}
        if isinstance(figures, dict):
            return figures
        raise TypeError("figures must be None, a matplotlib Figure, or a dict[str, Figure]")

    def _normalize_tables(self, tables: Any) -> Dict[str, Any]:
        """Normalize tables to a standard dictionary format using shared serializer."""
        return serialize_tables(tables)

    def _ensure_mlflow_run(self) -> None:
        """Ensure there's an active MLflow run, creating or restarting one if needed."""
        # If no active run but we have a specific run_id, try to restart that run
        if not mlflow.active_run() and self.run_id:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                run = client.get_run(self.run_id)
                if run:
                    exp = client.get_experiment(run.info.experiment_id)
                    mlflow.set_experiment(exp.name)
                    mlflow.start_run(run_id=self.run_id)
            except Exception as exc:
                import logging
                logging.debug("Could not restart run %s: %s", self.run_id, exc)
                mlflow.start_run()
        elif not mlflow.active_run():
            mlflow.start_run()

    def _log_figures(
        self, 
        figures: Dict[str, Any], 
        artifact_sub_path: str, 
        analysis_runner: Any
    ) -> None:
        """Log figure artifacts to MLflow."""
        analysis_name = getattr(analysis_runner, 'analysis_name', 'analysis')
        
        for fig_key, fig in figures.items():
            if not hasattr(fig, "savefig"):
                # Skip non-figure objects silently (could be metadata)
                continue
                
            for fmt in self.figure_formats:
                filename = f"{analysis_name.lower().replace(' ', '_')}_{fig_key}.{fmt}"
                local_path = Path.cwd() / filename
                try:
                    fig.savefig(local_path, bbox_inches="tight")
                    mlflow.log_artifact(str(local_path), artifact_path=artifact_sub_path)
                finally:
                    if local_path.exists():
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass

    def _log_tables(
        self, 
        tables: Dict[str, Any], 
        name: str, 
        artifact_sub_path: str, 
        analysis_runner: Any
    ) -> None:
        """Log table artifacts to MLflow."""
        analysis_name = getattr(analysis_runner, 'analysis_name', 'analysis')
        
        table_file = f"{analysis_name.lower().replace(' ', '_')}_{name}_tables.json"
        table_path = Path.cwd() / table_file
        try:
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(tables, f, indent=2)
            mlflow.log_artifact(str(table_path), artifact_path=artifact_sub_path)
        finally:
            if table_path.exists():
                try:
                    os.remove(table_path)
                except OSError:
                    pass
