from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig
from mignet_ce.metrics import effective_information
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import TimePair

from .config import AblationConfig
from .methods import METHOD_SPECS
from .registry import create_ablation_method


FEATURE_ABLATION_COLUMNS = (
    "method_id", "input_group", "feature_method", "pij_construction",
    "organ", "lower_layer", "upper_layer", "hierarchy", "time_pair",
    "EI_lower", "EI_upper", "delta_EI", "alpha_cci", "beta_n", "beta_l",
    "beta_g", "temperature", "ot_enabled",
)


def hierarchy_label(lower: str, upper: str) -> str:
    labels = {
        ("spot", "seurat_k150"): "Spot -> K150",
        ("seurat_k150", "seurat_k40"): "K150 -> K40",
        ("seurat_k40", "seurat_k10"): "K40 -> K10",
        ("spot", "seurat_k40"): "Spot -> K40",
        ("spot", "seurat_k10"): "Spot -> K10",
        ("seurat_k150", "seurat_k10"): "K150 -> K10",
    }
    return labels.get((lower, upper), f"{lower} -> {upper}")


def evaluate_context(
    context: NetworkContext,
    temporal_cfg: TemporalRunConfig,
    pairs: Sequence[TimePair],
    *,
    ablation_cfg: AblationConfig | None = None,
    method_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    scientific_cfg = ablation_cfg or AblationConfig()
    scientific_cfg.validate()
    selected = list(method_ids) if method_ids is not None else [spec.method_id for spec in METHOD_SPECS]
    rows: list[dict[str, object]] = []
    for method_id in selected:
        method = create_ablation_method(method_id, config=scientific_cfg)
        _result, kernels = method.run(context, temporal_cfg, pairs)
        spec = method.spec
        for pair in pairs:
            label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            lower_pij = np.asarray(kernels.p_lower[pair], dtype=float)
            upper_pij = np.asarray(kernels.p_upper[pair], dtype=float)
            ei_lower = float(effective_information(lower_pij.copy()))
            ei_upper = float(effective_information(upper_pij.copy()))
            rows.append({
                "method_id": spec.method_id,
                "input_group": spec.input_group,
                "feature_method": spec.feature_method,
                "pij_construction": spec.pij_construction,
                "organ": context.organ,
                "lower_layer": context.pair.lower_layer,
                "upper_layer": context.pair.upper_layer,
                "hierarchy": hierarchy_label(context.pair.lower_layer, context.pair.upper_layer),
                "time_pair": label,
                "EI_lower": ei_lower,
                "EI_upper": ei_upper,
                "delta_EI": ei_upper - ei_lower,
                "alpha_cci": scientific_cfg.alpha_cci,
                "beta_n": scientific_cfg.beta_n,
                "beta_l": scientific_cfg.beta_l,
                "beta_g": scientific_cfg.beta_g,
                "temperature": scientific_cfg.temperature,
                "ot_enabled": spec.use_ot,
            })
    return pd.DataFrame(rows)


def append_and_write(
    frames: Iterable[pd.DataFrame],
    output_root: Path,
    *,
    config: AblationConfig | None = None,
) -> tuple[Path, Path]:
    cfg = config or AblationConfig()
    cfg.validate()
    materialized = list(frames)
    if not materialized:
        raise ValueError("At least one feature-ablation frame is required.")
    table = pd.concat(materialized, ignore_index=True)
    missing = set(FEATURE_ABLATION_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Feature-ablation output is missing columns: {sorted(missing)}")
    identity = ["method_id", "organ", "lower_layer", "upper_layer", "time_pair"]
    if table.duplicated(identity).any():
        duplicate = table.loc[table.duplicated(identity, keep=False), identity].iloc[0]
        raise ValueError(f"Duplicate feature-ablation result: {duplicate.to_dict()}")
    known_method_order = [spec.method_id for spec in METHOD_SPECS]
    present_methods = set(table["method_id"].astype(str))
    unknown_methods = present_methods - set(known_method_order)
    if unknown_methods:
        raise ValueError(f"Unknown controlled ablation methods: {sorted(unknown_methods)}")
    manifest_methods = [method_id for method_id in known_method_order if method_id in present_methods]
    table = table.loc[:, FEATURE_ABLATION_COLUMNS]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    long_path = output_root / "feature_ablation_long.csv"
    manifest_path = output_root / "feature_ablation_manifest.json"
    table.to_csv(long_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": "controlled_pij_feature_ablation_v1",
                "methods": manifest_methods,
                "contract": cfg.contract(),
                "rows": int(len(table)),
                "output": str(long_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return long_path, manifest_path
