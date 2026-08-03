from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import ei_decomposition, row_normalize


def _blend_rows(matrix: np.ndarray, indices: np.ndarray, dose: float) -> np.ndarray:
    output = row_normalize(matrix).copy()
    mean_row = output.mean(axis=0)
    output[indices] = (1.0 - dose) * output[indices] + dose * mean_row[None, :]
    return row_normalize(output)


def build_perturbation_curves(
    cfg: DownstreamConfig,
    state_table: pd.DataFrame,
    grn: pd.DataFrame,
    cci: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed + 17)
    rows: list[dict[str, object]] = []
    doses = np.linspace(0.0, 1.0, 6)
    for pair in cfg.adjacent_pairs:
        source, _ = pair.split("->")
        transition = row_normalize(load_pij(cfg.pair_archive, pair, "upper"))
        units = load_units(cfg.pair_archive, "upper", source)
        target_count = max(1, int(round(0.2 * len(units))))
        state_scores = state_table[
            (state_table["time_pair"] == pair) & (state_table["layer"] == cfg.upper_layer)
        ].set_index("state")["state_ei"].reindex(units)
        cci_scores = cci[cci["time"] == source].set_index("state")["cci_out_log"].reindex(units)
        grn_scores = grn[grn["time"] == source].set_index("state")["grn_concentration"].reindex(units)
        if state_scores.isna().any() or cci_scores.isna().any() or grn_scores.isna().any():
            raise ValueError(f"Perturbation score alignment failed for {pair}")
        selections = {
            "high_state_ei": np.argsort(state_scores.to_numpy())[-target_count:],
            "high_cci_out": np.argsort(cci_scores.to_numpy())[-target_count:],
            "high_grn_concentration": np.argsort(grn_scores.to_numpy())[-target_count:],
        }
        baseline_ei = ei_decomposition(transition)["EI"]
        for target, indices in selections.items():
            for dose in doses:
                drop = baseline_ei - ei_decomposition(_blend_rows(transition, indices, float(dose)))["EI"]
                rows.append(
                    {
                        "time_pair": pair,
                        "target": target,
                        "dose": float(dose),
                        "ei_drop_mean": float(drop),
                        "ei_drop_low": float(drop),
                        "ei_drop_high": float(drop),
                        "perturbation": "Pij_row_homogenization",
                    }
                )
        for dose in doses:
            drops = []
            for _ in range(cfg.perturb_random_repeats):
                indices = rng.choice(len(units), size=target_count, replace=False)
                drops.append(
                    baseline_ei - ei_decomposition(_blend_rows(transition, indices, float(dose)))["EI"]
                )
            rows.append(
                {
                    "time_pair": pair,
                    "target": "matched_random",
                    "dose": float(dose),
                    "ei_drop_mean": float(np.mean(drops)),
                    "ei_drop_low": float(np.quantile(drops, 0.025)),
                    "ei_drop_high": float(np.quantile(drops, 0.975)),
                    "perturbation": "Pij_row_homogenization",
                }
            )
    return pd.DataFrame(rows)
