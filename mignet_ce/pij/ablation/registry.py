from __future__ import annotations

from .config import AblationConfig
from .method import ControlledFeatureAblationMethod
from .methods import METHOD_BY_ID, METHOD_SPECS, AblationMethodSpec


ABLATION_METHOD_REGISTRY = dict(METHOD_BY_ID)


def get_ablation_spec(method_id: str) -> AblationMethodSpec:
    try:
        return ABLATION_METHOD_REGISTRY[str(method_id)]
    except KeyError as exc:
        raise KeyError(
            f"Unknown controlled ablation method {method_id!r}; "
            f"choose from {list(ABLATION_METHOD_REGISTRY)}"
        ) from exc


def create_ablation_method(
    method_id: str,
    *,
    config: AblationConfig | None = None,
) -> ControlledFeatureAblationMethod:
    return ControlledFeatureAblationMethod(get_ablation_spec(method_id), config)


__all__ = [
    "ABLATION_METHOD_REGISTRY", "METHOD_SPECS", "create_ablation_method", "get_ablation_spec",
]
