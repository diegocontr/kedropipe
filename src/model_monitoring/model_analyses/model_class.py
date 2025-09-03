from __future__ import annotations

from typing import Dict, List, Optional

from ..monitoring import AnalysisDataBuilder
from .analyses import ANALYSIS_REGISTRY, ModelAnalysis


class ModelAnalysisDataBuilder(AnalysisDataBuilder):
    """Extension of AnalysisDataBuilder to support model-based analyses (e.g., PDP).

    Usage:
    - add_analysis(name, config_dict)
    - load_data() picks necessary columns for all analyses plus extra_cols/treatments/segments
    - calculate() runs all registered analyses; calculate(name) runs a specific one
    - get_analysis(name) returns the analysis instance, which exposes get_data()
    """

    def __init__(self, data, extra_cols: Optional[List[str]] = None, treatements=None):
        """Initialize the builder and store analysis configs/instances."""
        super().__init__(data=data, extra_cols=extra_cols, treatements=treatements)
        self._analyses_config: Dict[str, Dict] = {}
        self._analyses_instances: Dict[str, ModelAnalysis] = {}

    @staticmethod
    def list_available_analyses() -> List[str]:
        return list(ANALYSIS_REGISTRY.keys())

    def add_analysis(self, analysis_name: str, config: Dict):
        """Register a model analysis with its configuration and prepare required columns."""
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
        """Ensure columns required by added analyses are included; then defer to parent loader."""
        return super().load_data()

    def calculate(self, analysis_name: Optional[str] = None):
        """Execute analyses. If a name is provided, run only that analysis."""
        if self.db is None:
            self.load_data()

        # Ensure segments exist but do not need to be applied (PDP uses bin specs)
        names = [analysis_name] if analysis_name else list(self._analyses_config.keys())
        for name in names:
            if name not in self._analyses_config:
                raise ValueError(
                    f"Analysis '{name}' not registered. Registered: {list(self._analyses_config.keys())}"
                )
            analysis_cls = ANALYSIS_REGISTRY[name]
            instance = analysis_cls(self._analyses_config[name])
            instance.run(self.db, self.segments)
            self._analyses_instances[name] = instance
        return self

    def get_analysis(self, analysis_name: str) -> ModelAnalysis:
        if analysis_name not in self._analyses_instances:
            raise ValueError(
                f"Analysis '{analysis_name}' has not been calculated yet. Call calculate('{analysis_name}') first."
            )
        return self._analyses_instances[analysis_name]

    def get_analyses_objects(self):
        """Return DataFrame if a single analysis exists, else dict of name->DataFrame."""
        if not self._analyses_instances:
            raise ValueError(
                "No analyses have been calculated yet. Call calculate() first."
            )
        if len(self._analyses_instances) == 1:
            only = next(iter(self._analyses_instances.values()))
            return only.get_data()
        return {
            name: inst.get_data() for name, inst in self._analyses_instances.items()
        }
