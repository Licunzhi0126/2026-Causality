from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

from ..config import DownstreamConfig
from ..io import (
    cci_path,
    grn_path,
    layer_h5ad,
    load_pij,
    load_units,
    read_h5ad_expression,
)
from ..metrics import ei_decomposition, entropy, row_normalize


def _safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    count = int(mask.sum())
    if count < 4 or np.isclose(np.std(x_values[mask]), 0) or np.isclose(np.std(y_values[mask]), 0):
        return np.nan, np.nan, np.nan, np.nan, count
    pearson = pearsonr(x_values[mask], y_values[mask])
    spearman = spearmanr(x_values[mask], y_values[mask])
    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
        count,
    )


def _benjamini_hochberg(values: pd.Series) -> np.ndarray:
    p_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _reorder_rows(matrix: sp.csr_matrix, observed_units: Sequence[str], expected_units: Sequence[str]) -> sp.csr_matrix:
    observed = list(map(str, observed_units))
    expected = list(map(str, expected_units))
    if observed == expected:
        return matrix
    lookup = {unit: index for index, unit in enumerate(observed)}
    missing = [unit for unit in expected if unit not in lookup]
    if missing:
        raise ValueError(f"H5AD is missing archive units: {missing[:5]}")
    return matrix[[lookup[unit] for unit in expected]]


def build_grn_metrics(cfg: DownstreamConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time in cfg.times[:-1]:
        matrix, h5_units, genes = read_h5ad_expression(layer_h5ad(cfg.data_root, cfg.upper_layer, time, cfg.organ))
        units = load_units(cfg.pair_archive, "upper", time)
        matrix = _reorder_rows(matrix.tocsr().astype(float), h5_units, units)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        matrix = sp.diags(1.0 / np.maximum(totals, 1.0)) @ matrix * 1.0e4
        matrix.data = np.log1p(matrix.data)
        grn = pd.read_csv(grn_path(cfg.data_root, cfg.upper_layer, time, cfg.organ))
        required = {"regulator", "target", "weight"}
        missing = required - set(grn.columns)
        if missing:
            raise ValueError(f"GRN is missing columns {sorted(missing)}")
        grn["regulator"] = grn["regulator"].astype(str)
        grn["target"] = grn["target"].astype(str)
        grn["weight"] = pd.to_numeric(grn["weight"], errors="coerce").abs()
        grn = (
            grn.dropna(subset=["weight"])
            .sort_values(["regulator", "weight"], ascending=[True, False])
            .groupby("regulator", group_keys=False)
            .head(50)
        )
        gene_index = pd.Series(np.arange(len(genes)), index=genes.astype(str))
        grn = grn[grn["target"].isin(gene_index.index)].copy()
        regulators = sorted(grn["regulator"].unique().tolist())
        if not regulators:
            raise ValueError(f"No GRN targets overlap H5AD genes for {time}")
        regulator_index = {regulator: index for index, regulator in enumerate(regulators)}
        target_indices = grn["target"].map(gene_index).to_numpy(dtype=int)
        regulator_indices = grn["regulator"].map(regulator_index).to_numpy(dtype=int)
        weights = grn["weight"].to_numpy(dtype=float)
        sums = np.bincount(regulator_indices, weights=weights, minlength=len(regulators))
        weights = weights / np.maximum(sums[regulator_indices], 1e-12)
        projection = sp.coo_matrix(
            (weights, (target_indices, regulator_indices)),
            shape=(len(genes), len(regulators)),
        ).tocsr()
        activity = (matrix @ projection).toarray().astype(float, copy=False)
        for index, state in enumerate(units):
            values = np.maximum(activity[index], 0.0)
            total = values.sum()
            probabilities = values / max(total, 1e-12)
            h_value = float(entropy(probabilities))
            h_normalized = h_value / np.log2(max(len(probabilities), 2))
            top_indices = np.argsort(values)[-5:][::-1]
            rows.append(
                {
                    "time": time,
                    "state": state,
                    "grn_total_activity": float(total),
                    "grn_concentration": float(1.0 - h_normalized),
                    "grn_entropy_norm": float(h_normalized),
                    "grn_effective_regulators": float(2.0**h_value),
                    "grn_top10_share": float(np.sort(probabilities)[-10:].sum()),
                    "top_regulators": ";".join(regulators[item] for item in top_indices),
                }
            )
    return pd.DataFrame(rows)


def build_cci_metrics(cfg: DownstreamConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time in cfg.times[:-1]:
        archive_units = load_units(cfg.pair_archive, "upper", time)
        _, h5_units, _ = read_h5ad_expression(
            layer_h5ad(cfg.data_root, cfg.upper_layer, time, cfg.organ),
            prefer_counts=False,
        )
        matrix = sp.load_npz(cci_path(cfg.data_root, cfg.upper_layer, time, cfg.organ)).toarray().astype(float)
        matrix = np.maximum(matrix, 0.0)
        observed_units = list(map(str, h5_units))
        if observed_units != archive_units:
            lookup = {unit: index for index, unit in enumerate(observed_units)}
            missing = [unit for unit in archive_units if unit not in lookup]
            if missing:
                raise ValueError(f"CCI/H5AD is missing archive units for {time}: {missing[:5]}")
            indices = [lookup[unit] for unit in archive_units]
            matrix = matrix[np.ix_(indices, indices)]
        if matrix.shape != (len(archive_units), len(archive_units)):
            raise ValueError(f"CCI shape mismatch for {time}: {matrix.shape}")
        out_strength = matrix.sum(axis=1)
        in_strength = matrix.sum(axis=0)
        out_probabilities = row_normalize(matrix)
        in_probabilities = row_normalize(matrix.T)
        log_capacity = np.log2(max(len(archive_units), 2))
        for index, state in enumerate(archive_units):
            rows.append(
                {
                    "time": time,
                    "state": state,
                    "cci_out_strength": float(out_strength[index]),
                    "cci_in_strength": float(in_strength[index]),
                    "cci_out_log": float(np.log1p(out_strength[index])),
                    "cci_in_log": float(np.log1p(in_strength[index])),
                    "cci_out_entropy_norm": float(entropy(out_probabilities[index]) / log_capacity),
                    "cci_in_entropy_norm": float(entropy(in_probabilities[index]) / log_capacity),
                }
            )
    return pd.DataFrame(rows)


def build_mechanism_table(
    cfg: DownstreamConfig,
    state_table: pd.DataFrame,
    spatial: pd.DataFrame,
    grn: pd.DataFrame,
    cci: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = state_table[state_table["layer"] == cfg.upper_layer].copy()
    states = states[states["source_time"].isin(grn["time"].unique())]
    merged = states.merge(
        grn,
        left_on=["source_time", "state"],
        right_on=["time", "state"],
        how="left",
        suffixes=("", "_grn"),
    )
    merged = merged.merge(
        cci,
        left_on=["source_time", "state"],
        right_on=["time", "state"],
        how="left",
        suffixes=("", "_cci"),
    )
    spatial_keep = spatial[spatial["layer"] == cfg.upper_layer][
        ["time", "state", "spot_count", "boundary_ratio", "moran_i", "fragmentation"]
    ]
    merged = merged.merge(
        spatial_keep,
        left_on=["source_time", "state"],
        right_on=["time", "state"],
        how="left",
        suffixes=("", "_spatial"),
    )
    variables = [
        "grn_concentration",
        "grn_top10_share",
        "cci_out_log",
        "cci_in_log",
        "cci_out_entropy_norm",
        "boundary_ratio",
        "moran_i",
        "spot_count",
    ]
    correlation_rows: list[dict[str, object]] = []
    for outcome in ("state_ei", "transition_entropy"):
        for variable in variables:
            pearson_r, pearson_p, spearman_r, spearman_p, count = _safe_corr(merged[variable], merged[outcome])
            correlation_rows.append(
                {
                    "outcome": outcome,
                    "predictor": variable,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n": count,
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    correlations["pearson_q"] = _benjamini_hochberg(correlations["pearson_p"])
    correlations["spearman_q"] = _benjamini_hochberg(correlations["spearman_p"])

    predictors = [
        "grn_concentration",
        "cci_out_log",
        "cci_in_log",
        "cci_out_entropy_norm",
        "boundary_ratio",
        "moran_i",
        "spot_count",
    ]
    model = merged.dropna(subset=["state_ei", *predictors]).copy()
    if len(model) <= len(predictors) + 1:
        coefficients = pd.DataFrame(
            {"predictor": predictors, "coefficient": np.nan, "model_r2": np.nan, "n": len(model)}
        )
    else:
        numerical = StandardScaler().fit_transform(model[predictors])
        time_dummies = pd.get_dummies(model["source_time"], prefix="time", drop_first=True, dtype=float)
        design = np.column_stack([np.ones(len(model)), numerical, time_dummies.to_numpy(dtype=float)])
        response = StandardScaler().fit_transform(model[["state_ei"]]).ravel()
        beta = np.linalg.lstsq(design, response, rcond=None)[0]
        prediction = design @ beta
        r_squared = 1.0 - np.sum((response - prediction) ** 2) / max(
            np.sum((response - response.mean()) ** 2),
            1e-12,
        )
        coefficients = pd.DataFrame(
            {"predictor": predictors, "coefficient": beta[1 : 1 + len(predictors)]}
        )
        coefficients["model_r2"] = float(r_squared)
        coefficients["n"] = int(len(model))
    return merged, correlations, coefficients
