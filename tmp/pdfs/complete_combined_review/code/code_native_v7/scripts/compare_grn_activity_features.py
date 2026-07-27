#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float).ravel()
    y = np.asarray(right, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def effective_rank(values: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    total = float(singular_values.sum())
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return float(np.exp(entropy))


def feature_specificity(values: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape[0] < 2:
        mean_distance = 0.0
    else:
        distances = cosine_distances(matrix)
        mean_distance = float(distances[np.triu_indices(matrix.shape[0], k=1)].mean())
    return {
        "mean_pairwise_cosine_distance": mean_distance,
        "feature_variance_across_units": float(np.var(matrix, axis=0).mean()) if matrix.size else 0.0,
        "effective_rank": effective_rank(matrix),
    }


def compare_feature_tables(feature_a: pd.DataFrame, feature_b: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    common_units = feature_a.index.intersection(feature_b.index)
    common_features = feature_a.columns.intersection(feature_b.columns)
    if common_units.empty:
        raise ValueError("The feature tables have no common unit IDs.")
    if common_features.empty:
        raise ValueError("The feature tables have no common feature columns.")
    a = feature_a.loc[common_units, common_features].to_numpy(dtype=float)
    b = feature_b.loc[common_units, common_features].to_numpy(dtype=float)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    unit_cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    per_unit = pd.DataFrame(
        {
            "unit_id": common_units.astype(str),
            "feature_cosine": unit_cosine,
            "feature_norm_a": np.linalg.norm(a, axis=1),
            "feature_norm_b": np.linalg.norm(b, axis=1),
        }
    )
    similarity_a = cosine_similarity(a)
    similarity_b = cosine_similarity(b)
    triangle = np.triu_indices(len(common_units), k=1)
    summary: dict[str, object] = {
        "common_unit_count": int(len(common_units)),
        "common_feature_count": int(len(common_features)),
        "mean_same_unit_feature_cosine": float(unit_cosine.mean()),
        "unit_similarity_structure_correlation": _safe_correlation(
            similarity_a[triangle],
            similarity_b[triangle],
        ),
        "method_a": feature_specificity(a),
        "method_b": feature_specificity(b),
    }
    return per_unit, summary


def _row_entropy(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums > 0,
    )
    return -np.sum(probabilities * np.log2(probabilities + 1e-12), axis=1)


def compare_pij(path_a: Path, path_b: Path) -> dict[str, object]:
    a = sp.load_npz(path_a).toarray()
    b = sp.load_npz(path_b).toarray()
    if a.shape != b.shape:
        return {
            "comparable": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "reason": "matrix_shapes_differ",
        }
    entropy_a = _row_entropy(a)
    entropy_b = _row_entropy(b)
    return {
        "comparable": True,
        "shape": list(a.shape),
        "pij_correlation": _safe_correlation(a, b),
        "mean_row_entropy_a": float(entropy_a.mean()),
        "mean_row_entropy_b": float(entropy_b.mean()),
        "row_entropy_correlation": _safe_correlation(entropy_a, entropy_b),
    }


def compare_metrics(path_a: Path, path_b: Path) -> pd.DataFrame:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    key_candidates = ["organ", "lower_layer", "upper_layer", "time_pair", "pij_method"]
    keys = [key for key in key_candidates if key in a.columns and key in b.columns]
    value_columns = [
        column
        for column in ["EI_lower", "EI_upper", "EI_gain", "DI", "TE"]
        if column in a.columns and column in b.columns
    ]
    if not keys:
        a = a.reset_index(names="row_index")
        b = b.reset_index(names="row_index")
        keys = ["row_index"]
    merged = a[keys + value_columns].merge(
        b[keys + value_columns],
        on=keys,
        suffixes=("_a", "_b"),
        how="inner",
    )
    for column in value_columns:
        merged[f"{column}_difference"] = merged[f"{column}_b"] - merged[f"{column}_a"]
    return merged


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare expression-weighted and unit-specific/local GRN feature outputs."
    )
    parser.add_argument("--feature-a", type=Path, required=True)
    parser.add_argument("--feature-b", type=Path, required=True)
    parser.add_argument("--label-a", default="method_a")
    parser.add_argument("--label-b", default="method_b")
    parser.add_argument("--pij-a", type=Path, default=None)
    parser.add_argument("--pij-b", type=Path, default=None)
    parser.add_argument("--metrics-a", type=Path, default=None)
    parser.add_argument("--metrics-b", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    feature_a = pd.read_csv(args.feature_a, index_col=0)
    feature_b = pd.read_csv(args.feature_b, index_col=0)
    per_unit, summary = compare_feature_tables(feature_a, feature_b)
    summary["label_a"] = args.label_a
    summary["label_b"] = args.label_b
    per_unit.to_csv(args.output_root / "unit_feature_cosine.csv", index=False)

    if args.pij_a is not None and args.pij_b is not None:
        summary["pij"] = compare_pij(args.pij_a, args.pij_b)
    if args.metrics_a is not None and args.metrics_b is not None:
        compare_metrics(args.metrics_a, args.metrics_b).to_csv(
            args.output_root / "metrics_comparison.csv",
            index=False,
        )
    with (args.output_root / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
