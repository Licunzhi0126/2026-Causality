from __future__ import annotations

import json
from pathlib import Path

from .analysis import build_hierarchy_sensitivity, build_layer_ei_table, build_requested_wide_table
from .config import SensitivityConfig
from .plots import plot_alpha_sensitivity


def run_sensitivity_analysis(cfg: SensitivityConfig) -> dict[str, Path]:
    cfg = cfg.normalized()
    cfg.validate()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    layer_ei = build_layer_ei_table(cfg)
    sensitivity = build_hierarchy_sensitivity(layer_ei)
    wide = build_requested_wide_table(sensitivity, time_pair_order=cfg.time_pairs)

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
        "protocol": "GRN_CCI_alpha_sensitivity_v1",
        "formula": "C=((1-alpha)*Robust5_95(D_G)+alpha*Robust5_95(D_N))/RobustSpan5_95(C_pre_scale)",
        "component_clipping": "Robust5_95 clips each component to [0,1]",
        "combined_cost_clipping": False,
        "alpha_role": "relative GRN/CCI role only",
        "tau_role": "transition sharpness only",
        "canonical_alpha": cfg.canonical_alpha,
        "tau": cfg.tau,
        "beta_n": cfg.beta_n,
        "beta_g": cfg.beta_g,
        "time_pairs": list(cfg.time_pairs),
        "layers": list(cfg.layers),
        "alphas": list(cfg.alphas),
        "nmf_components": cfg.nmf_components,
        "nmf_max_iter": cfg.nmf_max_iter,
        "random_seed": cfg.random_seed,
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
