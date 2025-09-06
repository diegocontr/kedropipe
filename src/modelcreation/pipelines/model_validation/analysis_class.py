"""Base analysis runner utilities.

Provides the shared ``add_artifact`` helper so concrete analysis runner
classes can stay focused on their domain logic.

Shape stored in ``self.artifacts``::

    {artifact_name: {"figures": <figure-or-dict>, "tables": <dict-or-list>}}

Sub-classes may set up ``self.artifacts`` before calling ``super().__init__``
or just rely on the base initialiser.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["BaseAnalysisRunner"]


class BaseAnalysisRunner:
    """Lightweight base class exposing a unified ``add_artifact`` method."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        """Initialise artifacts container if not already present."""
        if not hasattr(self, "artifacts"):
            self.artifacts: Dict[str, Dict[str, Any]] = {}

    def add_artifact(
        self,
        name: str,
        figures: Any,
        tables: Any,
        include_config: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an artifact under ``name``.

        Parameters
        ----------
        name: Artifact key.
        figures: Figure or mapping of figures.
        tables: Dict / list / dataframe-like object (converted if possible).
        include_config: Inject config under ``_config`` key when True.
        config: Optional configuration metadata.
        """
        if include_config and config:
            if tables is None:
                tables = {}
            if isinstance(tables, dict):
                tables = {"_config": config, **tables}
            else:
                if hasattr(tables, "to_dict"):
                    try:
                        tables = tables.to_dict()
                    except Exception:  # pragma: no cover
                        tables = {"data": tables}
                else:
                    tables = {"data": tables}
                tables = {"_config": config, **tables}

        if not hasattr(self, "artifacts"):
            self.artifacts = {}
        self.artifacts[name] = {"figures": figures, "tables": tables}

