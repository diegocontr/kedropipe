from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


class ModelAnalysis:
    """Run analyses, collect artifacts, and persist them as MLflow artifacts.

    Contract for `func` provided in constructor:
    - Inputs: arbitrary keyword args forwarded from `run_analysis`.
    - Output: tuple (figures, tables)
        - figures: can be a single Matplotlib Figure or a dict[str, Figure]
        - tables: a dict-like structure suitable for JSON serialization
    """

    def __init__(self, func, analysis_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the analysis wrapper.

        Parameters:
            func: Callable implementing the analysis, returning (figures, tables).
            analysis_name: A human-friendly name for this analysis.
            config: Optional dict of configuration parameters (thresholds, bin counts, etc.)
                    that influence the analysis but are not necessarily passed as direct
                    function args. Persisted in self.config and can be added to tables.
        """
        self.analysis_name = analysis_name
        self.func = func
        self.config: Dict[str, Any] = config or {}
        # artifacts[name] = {"figures": dict[str, Figure], "tables": dict[str, Any]}
        self.artifacts: Dict[str, Dict[str, Any]] = {}

    def run_analysis(self, name: str, include_config: bool = True, **kwargs: Any) -> None:
        figures, tables = self.func(**kwargs)

        # Normalize figures to a dict[str, Figure]
        if figures is None:
            fig_dict = {}
        elif hasattr(figures, "savefig"):
            fig_dict = {"plot": figures}  # single figure
        elif isinstance(figures, dict):
            fig_dict = figures
        else:
            raise TypeError(
                "figures must be a matplotlib Figure or a dict[str, Figure]"
            )

        table_dict = tables or {}
        if include_config and self.config:
            # Attach config under a reserved key to avoid collision
            table_dict = {"_config": self.config, **table_dict}

        self.artifacts[name] = {"figures": fig_dict, "tables": table_dict}

    def get_artifacts(self) -> Dict[str, Dict[str, Any]]:
        return self.artifacts

    def save_to_mlflow(
        self,
        identifier_run: Optional[str] = None,
        base_artifact_path: Optional[str] = None,
    ) -> None:
        """Persist collected figures and tables into MLflow as artifacts.
        - identifier_run: existing MLflow run_id to use, or None to use active run or create a new one.
        - base_artifact_path: folder under which to place artifacts (defaults to sanitized analysis_name).
        """
        artifact_root = base_artifact_path or self.analysis_name.lower().replace(" ", "_")

        # Determine run context
        active = mlflow.active_run()
        need_close = False

        if identifier_run:
            if not active or active.info.run_id != identifier_run:
                mlflow.start_run(run_id=identifier_run)
            need_close = False
        elif not active:
            mlflow.start_run()
            need_close = True

        try:
            for name, payload in self.artifacts.items():
                sub_path = f"{artifact_root}/{name}"
                figures = payload.get("figures", {})
                for fig_key, fig in figures.items():
                    filename = f"{self.analysis_name.lower().replace(' ', '_')}_{name}_{fig_key}.png"
                    local_path = Path.cwd() / filename
                    try:
                        fig.savefig(local_path, bbox_inches="tight")
                        mlflow.log_artifact(str(local_path), artifact_path=sub_path)
                    finally:
                        if local_path.exists():
                            try:
                                os.remove(local_path)
                            except OSError:
                                pass
                tables = payload.get("tables", {})
                table_file = (
                    f"{self.analysis_name.lower().replace(' ', '_')}_{name}_tables.json"
                )
                table_path = Path.cwd() / table_file
                try:
                    with open(table_path, "w", encoding="utf-8") as f:
                        json.dump(tables, f, indent=2)
                    mlflow.log_artifact(str(table_path), artifact_path=sub_path)
                finally:
                    if table_path.exists():
                        try:
                            os.remove(table_path)
                        except OSError:
                            pass
        finally:
            if need_close:
                mlflow.end_run()
