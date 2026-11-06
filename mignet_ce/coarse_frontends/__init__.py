"""Coarse frontend API with lightweight method metadata imports."""

from mignet_ce.coarse_frontends.method_specs import (
    COARSE_METHOD_SPECS,
    DYNAMIC_CLOSURE_TWO_STAGE,
    LEGACY_SINGLE_STAGE,
    CoarseMethodSpec,
    get_coarse_method_spec,
    training_mode_for_method,
)

__all__ = [
    "COARSE_FRONTEND_REGISTRY",
    "COARSE_METHOD_SPECS",
    "CoarseMethodSpec",
    "CoarseFrontendRequest",
    "DYNAMIC_CLOSURE_TWO_STAGE",
    "LEGACY_SINGLE_STAGE",
    "get_coarse_method_spec",
    "prepare_coarse_input",
    "training_mode_for_method",
]


def __getattr__(name: str):
    if name == "CoarseFrontendRequest":
        from mignet_ce.coarse_frontends._common import CoarseFrontendRequest

        return CoarseFrontendRequest
    if name in {"COARSE_FRONTEND_REGISTRY", "prepare_coarse_input"}:
        from mignet_ce.coarse_frontends.registry import (
            COARSE_FRONTEND_REGISTRY,
            prepare_coarse_input,
        )

        return {
            "COARSE_FRONTEND_REGISTRY": COARSE_FRONTEND_REGISTRY,
            "prepare_coarse_input": prepare_coarse_input,
        }[name]
    raise AttributeError(name)
