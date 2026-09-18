"""Paper-facing tables and figures built from persisted downstream results."""

from .config import AssetConfig
from .workflow import build_paper_assets

__all__ = ["AssetConfig", "build_paper_assets"]
