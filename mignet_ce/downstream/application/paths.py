"""Application output location; all training inputs are supplied externally."""
from pathlib import Path

APPLICATION_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = APPLICATION_ROOT / "output"
