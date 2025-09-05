"""Pre-configured analysis classes for model monitoring.

This module provides ready-to-use analysis classes that wrap the model monitoring
functionality with standardized outputs for integration with larger reporting systems.
"""

from .base_analysis import BaseAnalysis
from .basic_analysis import BasicModelAnalysis
from .insurance_analysis import InsuranceModelAnalysis

__all__ = [
    "BaseAnalysis",
    "BasicModelAnalysis",
    "InsuranceModelAnalysis",
]
