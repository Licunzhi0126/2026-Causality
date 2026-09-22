from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationMethodSpec:
    method_id: str
    input_group: str
    cci_representation: str | None
    use_grn: bool
    use_ot: bool
    feature_method: str
    pij_construction: str

    @property
    def uses_cci(self) -> bool:
        return self.cci_representation is not None


METHOD_SPECS: tuple[AblationMethodSpec, ...] = (
    AblationMethodSpec("N_KL", "CCI", "N", False, False, "NMF", "KL"),
    AblationMethodSpec("N_KL_OT", "CCI", "N", False, True, "NMF", "KL + OT"),
    AblationMethodSpec("L_KL", "CCI", "L", False, False, "Laplacian", "KL"),
    AblationMethodSpec("L_KL_OT", "CCI", "L", False, True, "Laplacian", "KL + OT"),
    AblationMethodSpec("G_KL", "GRN", None, True, False, "Dual-end gating", "KL"),
    AblationMethodSpec("G_KL_OT", "GRN", None, True, True, "Dual-end gating", "KL + OT"),
    AblationMethodSpec(
        "NG_KL", "CCI + GRN", "N", True, False,
        "NMF + dual-end gating", "KL",
    ),
    AblationMethodSpec(
        "NG_KL_OT", "CCI + GRN", "N", True, True,
        "NMF + dual-end gating", "KL + OT",
    ),
    AblationMethodSpec(
        "LG_KL", "CCI + GRN", "L", True, False,
        "Laplacian + dual-end gating", "KL",
    ),
    AblationMethodSpec(
        "LG_KL_OT", "CCI + GRN", "L", True, True,
        "Laplacian + dual-end gating", "KL + OT",
    ),
)

METHOD_BY_ID = {spec.method_id: spec for spec in METHOD_SPECS}
