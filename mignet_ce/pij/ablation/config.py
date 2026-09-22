from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BETA = 0.05
DEFAULT_ALPHA_CCI = 0.01
DEFAULT_TEMPERATURE = 0.8


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
    beta_g: float = DEFAULT_BETA
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

    def contract(self) -> dict[str, object]:
        self.validate()
        return {
            "protocol": "controlled_pij_feature_ablation_v1",
            "component_cost": "raw_pairwise_feature_KL",
            "component_normalization": "none",
            "combined_cost_clipping": False,
            "beta_n": float(self.beta_n),
            "beta_l": float(self.beta_l),
            "beta_g": float(self.beta_g),
            "alpha_cci": float(self.alpha_cci),
            "nominal_grn_weight": 1.0 - float(self.alpha_cci),
            "nominal_cci_weight": float(self.alpha_cci),
            "temperature": float(self.temperature),
            "ot_policy": "same_KL_cost_then_balanced_uniform_sinkhorn_when_enabled",
        }
