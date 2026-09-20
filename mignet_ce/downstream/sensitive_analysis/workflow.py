from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mignet_ce.pij.compare._shared.ng_kl_ot import (
    CANONICAL_TRANSITION_PROTOCOL,
    canonical_transition_contract,
)

from .analysis import build_hierarchy_sensitivity, build_layer_ei_table, build_requested_wide_table
from .config import DEFAULT_COMPARISONS, SensitivityConfig
from .plots import plot_alpha_sensitivity


def run_sensitivity_analysis(cfg: SensitivityConfig) -> dict[str, Path]:
    cfg = cfg.normalized()
    cfg.validate()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    layer_ei = build_layer_ei_table(cfg)
    sensitivity = build_hierarchy_sensitivity(layer_ei)
    wide = build_requested_wide_table(sensitivity, time_pair_order=cfg.time_pairs)

    expected_layer_rows = len(cfg.time_pairs) * len(cfg.layers) * len(cfg.alphas)
    expected_sensitivity_rows = len(cfg.time_pairs) * len(DEFAULT_COMPARISONS) * len(cfg.alphas)
    if len(layer_ei) != expected_layer_rows:
        raise RuntimeError(
            f"Sensitivity layer table has {len(layer_ei)} rows; expected {expected_layer_rows}."
        )
    if len(sensitivity) != expected_sensitivity_rows:
        raise RuntimeError(
            f"Sensitivity comparison table has {len(sensitivity)} rows; "
            f"expected {expected_sensitivity_rows}."
        )
    numeric_columns = [
        "EI_bits",
        "actual_grn_cost_share",
        "mixed_robust_span",
        "sinkhorn_residual",
    ]
    if not np.isfinite(layer_ei[numeric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Sensitivity diagnostics contain non-finite values.")
    for (time_pair, alpha), group in sensitivity.groupby(["time_pair", "alpha"]):
        values = group.set_index("hierarchy_pair")["delta_EI_bits"]
        left = float(values["spot:seuratK40"])
        right = float(values["spot:seuratK150"] + values["seuratK150:seuratK40"])
        if not np.isclose(left, right, rtol=0.0, atol=1e-10):
            raise RuntimeError(
                f"Hierarchy DeltaEI identity failed for {time_pair}, alpha={alpha}: "
                f"{left} vs {right}."
            )

    layer_path = out / "alpha_layer_ei_long.csv"
    long_path = out / "alpha_hierarchy_sensitivity_long.csv"
    table_path = out / "alpha_hierarchy_sensitivity_table.csv"
    figure_path = out / "alpha_sensitivity_12.5_13.5.png"
    metadata_path = out / "sensitivity_metadata.json"

    layer_ei.to_csv(layer_path, index=False)
    sensitivity.to_csv(long_path, index=False)
    wide.to_csv(table_path, index=False)
    plot_alpha_sensitivity(
        sensitivity,
        figure_path,
        time_pair="12.5->13.5",
        canonical_alpha=cfg.canonical_alpha,
    )

    metadata = {
        "protocol": "GRN_CCI_alpha_sensitivity_v2_canonical_ng",
        "transition_protocol": CANONICAL_TRANSITION_PROTOCOL,
        "transition_contract": canonical_transition_contract(),
        "analysis_role": "sensitivity_sweep_not_parameter_optimization",
        "parameter_optimization_performed": False,
        "same_shared_production_transition_constructor_used": True,
        "shared_constructor": (
            "mignet_ce.pij.compare._shared.ng_kl_ot.canonical_ng_pij_from_cost_numpy"
        ),
        "formula": "C=((1-alpha)*Robust5_95(D_G)+alpha*Robust5_95(D_N))/RobustSpan5_95(C_pre_scale)",
        "component_clipping": "Robust5_95 clips each component to [0,1]",
        "combined_cost_clipping": False,
        "alpha_role": "relative GRN/CCI role only",
        "tau_role": "transition sharpness only",
        "canonical_alpha": cfg.canonical_alpha,
        "canonical_production_alpha": cfg.canonical_alpha,
        "tau": cfg.tau,
        "canonical_production_tau": cfg.tau,
        "beta_n": cfg.beta_n,
        "beta_g": cfg.beta_g,
        "time_pairs": list(cfg.time_pairs),
        "layers": list(cfg.layers),
        "alphas": list(cfg.alphas),
        "nmf_components": cfg.nmf_components,
        "nmf_max_iter": cfg.nmf_max_iter,
        "random_seed": cfg.random_seed,
        "production_profile_aligned": bool(layer_ei["production_profile_aligned"].all()),
        "outputs": {
            "layer_ei": str(layer_path),
            "long_sensitivity": str(long_path),
            "wide_table": str(table_path),
            "figure": str(figure_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "layer_ei": layer_path,
        "long_sensitivity": long_path,
        "wide_table": table_path,
        "figure": figure_path,
        "metadata": metadata_path,
    }
