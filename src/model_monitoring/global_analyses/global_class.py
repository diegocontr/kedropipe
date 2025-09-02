from __future__ import annotations

from typing import Dict, List, Optional

from ..monitoring import AnalysisDataBuilder
from .analyses import ANALYSIS_REGISTRY, GlobalAnalysis


class GlobalAnalysisDataBuilder(AnalysisDataBuilder):
    """Extension of AnalysisDataBuilder to support global analyses (e.g., Lorenz curve).

    Usage:
    - add_analysis(name, config_dict)
    - load_data() picks necessary columns for all analyses plus extra_cols/treatments/segments
    - calculate() runs all registered analyses; calculate(name) runs a specific one
    - get_analysis(name) returns the analysis instance, which exposes get_data()
    """

    def __init__(self, data, extra_cols: Optional[List[str]] = None, treatements=None):
        """Initialize the builder for global analyses.

        Args:
            data: Same as AnalysisDataBuilder (path, DataFrame, or dict for TripleParquetDataLoader).
            extra_cols: Optional list of additional columns to load.
            treatements: Optional list of treatment instances to apply.
        """
        super().__init__(data=data, extra_cols=extra_cols, treatements=treatements)
        self._analyses_config: Dict[str, Dict] = {}
        self._analyses_instances: Dict[str, GlobalAnalysis] = {}

    @staticmethod
    def list_available_analyses() -> List[str]:
        return list(ANALYSIS_REGISTRY.keys())

    def add_analysis(self, analysis_name: str, config: Dict):
        """Register a global analysis with its configuration and prepare required columns."""
        if analysis_name not in ANALYSIS_REGISTRY:
            raise ValueError(
                f"Unknown analysis '{analysis_name}'. Available: {list(ANALYSIS_REGISTRY.keys())}"
            )
        self._analyses_config[analysis_name] = config

        # Ask the analysis class for required columns and add them
        analysis_cls = ANALYSIS_REGISTRY[analysis_name]
        for col in analysis_cls.required_columns(config):
            self.add_col(col)

    def load_data(self):
        """Override to ensure columns required by added analyses are included.
        Uses the same loading mechanism but relies on add_col calls done in add_analysis.
        """
        return super().load_data()

    def calculate(self, analysis_name: Optional[str] = None):
        """Execute analyses.

        - If analysis_name is provided, run only that analysis.
        - Otherwise, run all registered analyses.
        """
        if self.db is None:
            self.load_data()

        names = [analysis_name] if analysis_name else list(self._analyses_config.keys())
        for name in names:
            if name not in self._analyses_config:
                raise ValueError(
                    f"Analysis '{name}' not registered. Registered: {list(self._analyses_config.keys())}"
                )
            analysis_cls = ANALYSIS_REGISTRY[name]
            instance = analysis_cls(self._analyses_config[name])
            instance.run(self.db)
            self._analyses_instances[name] = instance
        return self

    def get_analysis(self, analysis_name: str) -> GlobalAnalysis:
        if analysis_name not in self._analyses_instances:
            raise ValueError(
                f"Analysis '{analysis_name}' has not been calculated yet. Call calculate('{analysis_name}') first."
            )
        return self._analyses_instances[analysis_name]

    def get_analyses_objects(self) -> Dict[str, GlobalAnalysis]:
        """Return a dict with all calculated analyses keyed by their names.

        Raises if calculate() has not been called yet or no analyses are available.
        """
        if not self._analyses_instances:
            raise ValueError(
                "No analyses have been calculated yet. Call calculate() first."
            )
        return dict(self._analyses_instances)
