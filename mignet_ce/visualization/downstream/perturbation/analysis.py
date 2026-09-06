from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DownstreamConfig
from ..io import load_pij, load_units
from ..metrics import ei_decomposition, row_normalize
from ..dynamic_closure.analysis import effective_information, state_level_ei


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


def build_unified_perturbation_curves(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
    mechanism: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.profile.random_seed + 17)
    doses = np.linspace(0.0, 1.0, 6)
    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            transition = row_normalize(record.q_model_full)
            mechanism_states = mechanism[
                (mechanism["mapping"] == mapping) & (mechanism["time_pair"] == pair)
            ].set_index("state_index")
            state_count = transition.shape[0]
            target_count = max(1, int(round(0.2 * state_count)))
            state_ei_values = state_level_ei(transition)
            cci_values = mechanism_states["cci_out_log"].reindex(range(state_count)).to_numpy(dtype=float)
            grn_values = mechanism_states["grn_concentration"].reindex(range(state_count)).to_numpy(dtype=float)
            if not np.isfinite(cci_values).all() or not np.isfinite(grn_values).all():
                raise ValueError(f"Unified perturbation score alignment failed for {mapping} {pair}")
            selections = {
                "high_state_ei": np.argsort(state_ei_values)[-target_count:],
                "high_cci_out": np.argsort(cci_values)[-target_count:],
                "high_grn_concentration": np.argsort(grn_values)[-target_count:],
            }
            baseline = effective_information(transition)
            for target, indices in selections.items():
                for dose in doses:
                    drop = baseline - effective_information(_blend_rows(transition, indices, float(dose)))
                    rows.append(
                        {
                            "mapping": mapping,
                            "time_pair": pair,
                            "target": target,
                            "dose": float(dose),
                            "ei_drop_mean": float(drop),
                            "ei_drop_low": float(drop),
                            "ei_drop_high": float(drop),
                            "perturbation": "Pij_row_homogenization",
                            "analysis_space": "full_model",
                        }
                    )
            for dose in doses:
                drops = []
                for _ in range(cfg.profile.perturb_random_repeats):
                    indices = rng.choice(state_count, size=target_count, replace=False)
                    drops.append(
                        baseline - effective_information(_blend_rows(transition, indices, float(dose)))
                    )
                rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "target": "matched_random",
                        "dose": float(dose),
                        "ei_drop_mean": float(np.mean(drops)),
                        "ei_drop_low": float(np.quantile(drops, 0.025)),
                        "ei_drop_high": float(np.quantile(drops, 0.975)),
                        "perturbation": "Pij_row_homogenization",
                        "analysis_space": "full_model",
                    }
                )
    return pd.DataFrame(rows)
