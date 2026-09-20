"""GRN–CCI alpha sensitivity analysis for natural hierarchy EI."""

from .config import SensitivityConfig
from .workflow import run_sensitivity_analysis

__all__ = ["SensitivityConfig", "run_sensitivity_analysis"]
