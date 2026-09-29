from __future__ import annotations

from dataclasses import dataclass

from mignet_ce.pij.compare._shared.ng_kl_ot import (
    CANONICAL_ALPHA_CCI,
    CANONICAL_FEATURE_BETA_G,
    CANONICAL_FEATURE_BETA_N,
    CANONICAL_TEMPERATURE,
    CANONICAL_TRANSITION_PROTOCOL,
)


DEFAULT_BETA = CANONICAL_FEATURE_BETA_N
DEFAULT_ALPHA_CCI = CANONICAL_ALPHA_CCI
DEFAULT_TEMPERATURE = CANONICAL_TEMPERATURE


@dataclass(frozen=True)
class AblationConfig:
    """Scientific contract for the controlled PIJ ablation.

    The experiment deliberately separates three choices:
    1) feature representation (N / L / G / N+G / L+G),
    2) raw-KL convex fusion (alpha controls CCI vs GRN only),
    3) transition construction (row-normalized kernel vs balanced OT).
    """

    beta_n: float = DEFAULT_BETA
    beta_l: float = DEFAULT_BETA
    beta_g: float = CANONICAL_FEATURE_BETA_G
    alpha_cci: float = DEFAULT_ALPHA_CCI
    temperature: float = DEFAULT_TEMPERATURE
    sinkhorn_max_iter: int = 2000
    sinkhorn_tolerance: float = 1e-9
    sinkhorn_check_every: int = 10

    def validate(self) -> None:
        for name, value in (
            ("beta_n", self.beta_n),
            ("beta_l", self.beta_l),
            ("beta_g", self.beta_g),
            ("temperature", self.temperature),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive; got {value}.")
        if not 0.0 <= float(self.alpha_cci) <= 1.0:
            raise ValueError("alpha_cci must lie in [0, 1].")
        if int(self.sinkhorn_max_iter) < 1 or int(self.sinkhorn_check_every) < 1:
            raise ValueError("Sinkhorn iteration settings must be positive.")
        if float(self.sinkhorn_tolerance) <= 0.0:
            raise ValueError("sinkhorn_tolerance must be positive.")
        fixed = {
            "beta_n": (self.beta_n, CANONICAL_FEATURE_BETA_N),
            "beta_l": (self.beta_l, DEFAULT_BETA),
            "beta_g": (self.beta_g, CANONICAL_FEATURE_BETA_G),
            "alpha_cci": (self.alpha_cci, CANONICAL_ALPHA_CCI),
            "temperature": (self.temperature, CANONICAL_TEMPERATURE),
        }
        mismatched = {
            name: {"received": received, "expected": expected}
            for name, (received, expected) in fixed.items()
            if abs(float(received) - float(expected)) > 1e-12
        }
        if mismatched:
            raise ValueError(f"Controlled ablation constants are fixed: {mismatched}")

    def contract(self) -> dict[str, object]:
        self.validate()
        return {
            "protocol": "controlled_pij_feature_ablation_v1",
            "transition_protocol": CANONICAL_TRANSITION_PROTOCOL,
            "component_cost": "raw_pairwise_feature_KL",
            "component_normalization": "none",
            "combined_cost_clipping": False,
            "combined_cost_clipped": False,
            "combined_scale_control": "none",
            "beta_n": float(self.beta_n),
            "beta_l": float(self.beta_l),
            "beta_g": float(self.beta_g),
            "alpha_cci": float(self.alpha_cci),
            "nominal_grn_weight": 1.0 - float(self.alpha_cci),
            "nominal_cci_weight": float(self.alpha_cci),
            "temperature": float(self.temperature),
            "ot_policy": "same_KL_cost_then_balanced_uniform_sinkhorn_when_enabled",
        }
