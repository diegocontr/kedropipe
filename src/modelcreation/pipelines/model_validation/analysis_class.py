from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow


@dataclass
class _ArtifactPayload:
    figures: Dict[str, Any]
    tables: Dict[str, Any]


class BaseAnalysis:
    """Abstract base for analyses that want MLflow artifact management.

    Subclasses should implement:
      - run(): populate internal artifacts via self.add_artifact(call_name, figures, tables)
    Optionally override analysis_name property dynamically.
    """

    analysis_name: str = "analysis"

    def __init__(
        self,
        *,
        artifact_root: Optional[str] = None,
        figure_formats: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize base analysis container.

        Parameters
        ----------
        artifact_root : str, optional
            Root folder name for MLflow artifacts.
        figure_formats : iterable[str], optional
            Image formats to persist figures as.
        """
        self._artifact_root = artifact_root or self.analysis_name.lower().replace(" ", "_")
        self.figure_formats = list(figure_formats or ["png"])
        self.artifacts: Dict[str, _ArtifactPayload] = {}

    # API for subclasses --------------------------------------------------
    def add_artifact(self, name: str, figures: Any, tables: Any, include_config: bool = False, config: Optional[Dict[str, Any]] = None) -> None:
        fig_dict = self._normalize_figures(figures)
        table_dict = self._normalize_tables(tables, include_config=include_config, config=config)
        self.artifacts[name] = _ArtifactPayload(figures=fig_dict, tables=table_dict)

    def run(self, *args, **kwargs):  # pragma: no cover - abstract placeholder
        raise NotImplementedError

    # Reuse normalization & logging helpers from concrete manager
    def _normalize_figures(self, figures: Any) -> Dict[str, Any]:
        if figures is None:
            return {}
        if hasattr(figures, "savefig"):
            return {"plot": figures}
        if isinstance(figures, dict):
            return figures
        raise TypeError("figures must be None, a matplotlib Figure, or a dict[str, Figure]")

    def _normalize_tables(self, tables: Any, *, include_config: bool, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if tables is None:
            tables = {}
        if isinstance(tables, dict):
            out = tables
        else:
            if hasattr(tables, "to_dict"):
                out = tables.to_dict()  # type: ignore
            else:
                raise TypeError("tables must be dict-like or have to_dict()")
        if include_config and config:
            out = {"_config": config, **out}
        return out

    # Logging (same as ModelAnalysis simplified) -------------------------
    def save_to_mlflow(self, identifier_run: Optional[str] = None, base_artifact_path: Optional[str] = None) -> None:
        artifact_root = base_artifact_path or self._artifact_root
        
        # If no active run but we have a specific run_id, try to restart that validation run
        if not mlflow.active_run() and identifier_run:
            try:
                # Find validation experiment and restart the run
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                run = client.get_run(identifier_run)
                if run:
                    exp = client.get_experiment(run.info.experiment_id)
                    mlflow.set_experiment(exp.name)
                    mlflow.start_run(run_id=identifier_run)
            except Exception as exc:
                import logging
                logging.debug("Could not restart validation run %s: %s", identifier_run, exc)
                mlflow.start_run()
        elif not mlflow.active_run():
            mlflow.start_run()
            
        for name, payload in self.artifacts.items():
            sub_path = f"{artifact_root}/{name}"
            self._log_figures(payload.figures, sub_path)
            self._log_tables(payload.tables, name, sub_path)

    # Shared primitives
    def _log_figures(self, figures: Dict[str, Any], artifact_sub_path: str) -> None:
        for fig_key, fig in figures.items():
            if not hasattr(fig, "savefig"):
                continue
            for fmt in self.figure_formats:
                filename = f"{self.analysis_name.lower().replace(' ', '_')}_{fig_key}.{fmt}"
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

    def _log_tables(self, tables: Dict[str, Any], name: str, artifact_sub_path: str) -> None:
        table_file = f"{self.analysis_name.lower().replace(' ', '_')}_{name}_tables.json"
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


class ModelAnalysis:
    """Container to execute one or more analysis functions and log artifacts to MLflow.

    Backward compatible with previous implementation but adds:
    - Extensible payload dataclass
    - Support for multiple figure formats (future-proof)
    - Single-shot or incremental MLflow logging via save_to_mlflow / log_all
    - Clear normalization hooks for figures & tables

    Callable contract for `func` passed to constructor:
        func(**kwargs) -> (figures, tables)
    where figures is either None, a single matplotlib Figure, or a dict[str, Figure].
    Tables must be JSON serializable (dict) after normalization.
    """

    def __init__(
        self,
        func,
        analysis_name: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        artifact_root: Optional[str] = None,
        figure_formats: Optional[Iterable[str]] = None,
    ) -> None:
        """Create a ModelAnalysis wrapper.

        Parameters
        ----------
        func : Callable
            Callable returning (figures, tables) when executed.
        analysis_name : str
            Human friendly name for the analysis (used for artifact folder & filenames).
        config : dict, optional
            Configuration influencing analysis logic (persisted into tables under `_config`).
        artifact_root : str, optional
            Base artifact folder (defaults to a slugified version of analysis_name).
        figure_formats : Iterable[str], optional
            Iterable of file extensions to persist figures as (default: ["png"]).
        """
        self.analysis_name = analysis_name
        self.func = func
        self.config: Dict[str, Any] = config or {}
        self._artifact_root = (
            artifact_root or analysis_name.lower().replace(" ", "_")
        )
        self.figure_formats = list(figure_formats or ["png"])  # future extension
        # artifacts[name] = _ArtifactPayload
        self.artifacts: Dict[str, _ArtifactPayload] = {}

    # --------- Public API -------------------------------------------------
    def run_analysis(
        self, name: str, include_config: bool = True, **kwargs: Any
    ) -> None:
        """Execute the underlying callable and store normalized artifacts under `name`."""
        figures, tables = self.func(**kwargs)
        fig_dict = self._normalize_figures(figures)
        table_dict = self._normalize_tables(tables, include_config=include_config)
        self.artifacts[name] = _ArtifactPayload(figures=fig_dict, tables=table_dict)

    def get_artifacts(self) -> Dict[str, Dict[str, Any]]:
        # Return a shallow copy in legacy shape for backwards compatibility
        return {
            k: {"figures": v.figures, "tables": v.tables} for k, v in self.artifacts.items()
        }

    def save_to_mlflow(
        self,
        identifier_run: Optional[str] = None,
        base_artifact_path: Optional[str] = None,
    ) -> None:
        """Persist currently stored artifacts to MLflow.

        Parameters
        ----------
        identifier_run : str, optional
            Existing MLflow run id; if absent uses active run or creates new one.
        base_artifact_path : str, optional
            Base path under which artifacts are logged (defaults to sanitized analysis name).
        """
        self._log_to_mlflow(identifier_run, base_artifact_path)

    # Convenience alias for clarity in multi-analysis scenarios
    log_all = save_to_mlflow

    # --------- Internal helpers -------------------------------------------
    def _normalize_figures(self, figures: Any) -> Dict[str, Any]:
        if figures is None:
            return {}
        if hasattr(figures, "savefig"):
            return {"plot": figures}
        if isinstance(figures, dict):
            # Ensure all values look like figures (have savefig) or are serializable placeholders
            return figures
        raise TypeError(
            "figures must be None, a matplotlib Figure, or a dict[str, Figure]"
        )

    def _normalize_tables(
    self, tables: Any, *, include_config: bool
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if tables is None:
            tables = {}
        if isinstance(tables, dict):
            out = tables
        else:  # Allow returning arbitrary object with to_dict
            if hasattr(tables, "to_dict"):
                out = tables.to_dict()  # type: ignore[assignment]
            else:
                raise TypeError("tables must be dict-like or have to_dict()")
        if include_config and self.config:
            out = {"_config": self.config, **out}
        return out

    def _log_to_mlflow(
        self, identifier_run: Optional[str], base_artifact_path: Optional[str]
    ) -> None:
        """Log artifacts into a single shared active MLflow run.

        If a run is active, reuse it. If not and we have identifier_run, restart that run.
        Otherwise start a new run. This ensures all validation artifacts go to the same run.
        """
        artifact_root = base_artifact_path or self._artifact_root
        
        # If no active run but we have a specific run_id, try to restart that validation run
        if not mlflow.active_run() and identifier_run:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                run = client.get_run(identifier_run)
                if run:
                    exp = client.get_experiment(run.info.experiment_id)
                    mlflow.set_experiment(exp.name)
                    mlflow.start_run(run_id=identifier_run)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("Could not restart validation run %s: %s", identifier_run, exc)
                mlflow.start_run()
        elif not mlflow.active_run():
            mlflow.start_run()
            
        for name, payload in self.artifacts.items():
            sub_path = f"{artifact_root}/{name}"
            self._log_figures(payload.figures, sub_path)
            self._log_tables(payload.tables, name, sub_path)

    # Logging primitives ---------------------------------------------------
    def _log_figures(self, figures: Dict[str, Any], artifact_sub_path: str) -> None:
        for fig_key, fig in figures.items():
            if not hasattr(fig, "savefig"):
                # Skip non-figure objects silently (could be metadata)
                continue
            for fmt in self.figure_formats:
                filename = (
                    f"{self.analysis_name.lower().replace(' ', '_')}_"
                    f"{fig_key}.{fmt}"
                )
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
        self, tables: Dict[str, Any], name: str, artifact_sub_path: str
    ) -> None:
        table_file = (
            f"{self.analysis_name.lower().replace(' ', '_')}_{name}_tables.json"
        )
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
