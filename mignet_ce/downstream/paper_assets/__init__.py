"""The sole active publication-asset package for downstream results."""

from .config import AssetConfig
from .workflow import build_paper_assets

__all__ = ["AssetConfig", "build_paper_assets"]
