from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..io import (
    cci_index_path,
    cci_path,
    grn_path,
    layer_h5ad,
    read_h5ad_expression,
    read_index,
)
from ..metrics import entropy, row_normalize
from ..dynamic_closure.analysis import entropy_rows, state_level_ei


def _spot_grn_activity(cfg, time: str) -> tuple[np.ndarray, list[str], list[str]]:
    matrix, units, genes = read_h5ad_expression(
        layer_h5ad(cfg.data_root, "spot", time, cfg.organ)
    )
    matrix = matrix.tocsr().astype(float)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    matrix = sp.diags(1.0 / np.maximum(totals, 1.0)) @ matrix * 1.0e4
    matrix.data = np.log1p(matrix.data)
    grn = pd.read_csv(grn_path(cfg.data_root, "spot", time, cfg.organ))
    required = {"regulator", "target", "weight"}
    missing = required - set(grn.columns)
    if missing:
        raise ValueError(f"Spot GRN at {time} is missing columns {sorted(missing)}")
    grn = grn.copy()
    grn["regulator"] = grn["regulator"].astype(str)
    grn["target"] = grn["target"].astype(str)
    grn["weight"] = pd.to_numeric(grn["weight"], errors="coerce").abs()
    grn = (
        grn.dropna(subset=["weight"])
        .sort_values(["regulator", "weight"], ascending=[True, False])
        .groupby("regulator", group_keys=False)
        .head(50)
    )
    gene_lookup = {str(gene): index for index, gene in enumerate(genes)}
    grn = grn[grn["target"].isin(gene_lookup)].copy()
    regulators = sorted(grn["regulator"].unique())
    if not regulators:
        raise ValueError(f"No spot GRN targets overlap expression genes at {time}")
    regulator_lookup = {regulator: index for index, regulator in enumerate(regulators)}
    target_indices = grn["target"].map(gene_lookup).to_numpy(dtype=int)
    regulator_indices = grn["regulator"].map(regulator_lookup).to_numpy(dtype=int)
    weights = grn["weight"].to_numpy(dtype=float)
    totals = np.bincount(regulator_indices, weights=weights, minlength=len(regulators))
    weights /= np.maximum(totals[regulator_indices], 1e-12)
    projection = sp.coo_matrix(
        (weights, (target_indices, regulator_indices)),
        shape=(len(genes), len(regulators)),
    ).tocsr()
    return (matrix @ projection).toarray(), units.astype(str).tolist(), regulators


def build_unified_mechanism_table(
    cfg,
    records_by_pair: dict[tuple[str, str], object],
) -> pd.DataFrame:
    """Aggregate GRN activity and CCI roles on the same spot substrate for all mappings."""

    rows: list[dict[str, object]] = []
    for pair in cfg.adjacent_pairs:
        source, _target = pair.split("->")
        activity, activity_units, regulators = _spot_grn_activity(cfg, source)
        activity_lookup = {unit: index for index, unit in enumerate(activity_units)}
        cci = sp.load_npz(cci_path(cfg.data_root, "spot", source, cfg.organ)).tocsr().astype(float)
        cci_units = read_index(cci_index_path(cfg.data_root, "spot", source, cfg.organ))
        cci_lookup = {unit: index for index, unit in enumerate(cci_units)}
        for mapping in cfg.mapping_names:
            record = records_by_pair[(mapping, pair)]
            missing_activity = [unit for unit in record.spots_s if unit not in activity_lookup]
            missing_cci = [unit for unit in record.spots_s if unit not in cci_lookup]
            if missing_activity or missing_cci:
                raise ValueError(
                    f"Spot substrate alignment failed for {mapping} {pair}: "
                    f"activity={missing_activity[:5]}, cci={missing_cci[:5]}"
                )
            activity_order = np.asarray([activity_lookup[unit] for unit in record.spots_s], dtype=int)
            ordered_activity = activity[activity_order]
            assignment = np.asarray(record.source_assignment, dtype=float)
            sizes = assignment.sum(axis=0)
            state_activity = (assignment.T @ ordered_activity) / np.maximum(sizes[:, None], 1.0)
            cci_order = np.asarray([cci_lookup[unit] for unit in record.spots_s], dtype=int)
            ordered_cci = cci[cci_order][:, cci_order]
            macro_cci = (
                sp.csr_matrix(assignment).T @ ordered_cci @ sp.csr_matrix(assignment)
            ).toarray()
            outgoing = macro_cci.sum(axis=1)
            incoming = macro_cci.sum(axis=0)
            outgoing_prob = row_normalize(macro_cci)
            incoming_prob = row_normalize(macro_cci.T)
            log_capacity = np.log2(max(macro_cci.shape[0], 2))
            state_ei_values = state_level_ei(record.q_direct)
            transition_entropy = entropy_rows(record.q_direct)
            for state in range(assignment.shape[1]):
                values = np.maximum(state_activity[state], 0.0)
                total = float(values.sum())
                probabilities = values / max(total, 1e-12)
                h_value = float(entropy(probabilities))
                concentration = 1.0 - h_value / np.log2(max(len(probabilities), 2))
                top = np.argsort(values)[-5:][::-1]
                rows.append(
                    {
                        "mapping": mapping,
                        "time_pair": pair,
                        "source_time": source,
                        "state_index": state,
                        "state_ei": float(state_ei_values[state]),
                        "transition_entropy": float(transition_entropy[state]),
                        "grn_total_activity": total,
                        "grn_concentration": float(concentration),
                        "grn_top10_share": float(np.sort(probabilities)[-10:].sum()),
                        "top_regulators": ";".join(regulators[index] for index in top),
                        "cci_out_strength": float(outgoing[state]),
                        "cci_in_strength": float(incoming[state]),
                        "cci_out_log": float(np.log1p(outgoing[state])),
                        "cci_in_log": float(np.log1p(incoming[state])),
                        "cci_out_entropy_norm": float(entropy(outgoing_prob[state]) / log_capacity),
                        "cci_in_entropy_norm": float(entropy(incoming_prob[state]) / log_capacity),
                        "soft_mass": float(sizes[state]),
                    }
                )
    return pd.DataFrame(rows)
