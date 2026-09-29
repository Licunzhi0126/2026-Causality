"""Controlled PIJ feature/transition ablation package.

This package is intentionally separate from ``mignet_ce.pij.compare`` so paper
ablations cannot silently change production/full methods.
"""

from .config import AblationConfig
from .methods import METHOD_SPECS, AblationMethodSpec
from .registry import ABLATION_METHOD_REGISTRY, create_ablation_method, get_ablation_spec
from .runner import append_and_write, evaluate_context

__all__ = [
    "AblationConfig",
    "AblationMethodSpec",
    "METHOD_SPECS",
    "ABLATION_METHOD_REGISTRY",
    "create_ablation_method",
    "get_ablation_spec",
    "evaluate_context",
    "append_and_write",
]
