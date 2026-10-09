"""Source-only GRN structural enhancements; Legacy remains the default."""
from .config import GRNFeatureConfig
from .encoder import GRNPairEnhancer

__all__ = ["GRNFeatureConfig", "GRNPairEnhancer"]
