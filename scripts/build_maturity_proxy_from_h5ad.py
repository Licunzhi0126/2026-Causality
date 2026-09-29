#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.io.h5ad_h5py import read_h5ad_axis_names, read_h5ad_csr_matrix  # noqa: E402


@dataclass(frozen=True)
class StageInput:
    label: str
    h5ad: Path
    output_csv: Path


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    low = float(arr.min())
    high = float(arr.max())
    if high <= low:
        return np.zeros_like(arr, dtype=float)
    return (arr - low) / (high - low)


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), 0.5, dtype=float)
    ranks = series.rank(method="average", ascending=True)
    return ((ranks - 1.0) / max(1, len(series) - 1)).to_numpy(dtype=float)


def _expression_entropy(matrix: sp.csr_matrix) -> np.ndarray:
    csr = matrix.tocsr(copy=True).astype(float)
    csr.data = np.clip(np.nan_to_num(csr.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    features = int(csr.shape[1])
    if features <= 1:
        return np.zeros(csr.shape[0], dtype=float)
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    x_log_x = csr.copy()
    x_log_x.data = x_log_x.data * np.log(x_log_x.data + 1e-12)
    sum_x_log_x = np.asarray(x_log_x.sum(axis=1)).ravel()
    entropy = np.zeros(csr.shape[0], dtype=float)
    mask = row_sums > 0
    entropy[mask] = (
        np.log(row_sums[mask] + 1e-12) - (sum_x_log_x[mask] / row_sums[mask])
    ) / np.log(features)
    return np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_counts(matrix: sp.csr_matrix) -> sp.csr_matrix:
    csr = matrix.tocsr(copy=True).astype(np.float32)
    csr.data = np.clip(np.nan_to_num(csr.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    row_sum = np.asarray(csr.sum(axis=1)).ravel()
    scale = np.divide(1e4, row_sum, out=np.zeros_like(row_sum, dtype=float), where=row_sum > 0)
    norm = sp.diags(scale.astype(np.float32)) @ csr
    norm.data = np.log1p(norm.data)
    norm.eliminate_zeros()
    return norm.tocsr()


def _parse_stage_specs(values: list[str]) -> list[StageInput]:
    if len(values) % 3 != 0:
        raise ValueError("--stage entries must repeat: LABEL H5AD OUTPUT_CSV.")
    return [
        StageInput(str(values[index]), Path(values[index + 1]), Path(values[index + 2]))
        for index in range(0, len(values), 3)
    ]


def build_maturity_proxy(
    stages: list[StageInput],
    *,
    output_metadata: Path,
    n_components: int,
    within_stage_weight: float,
    mode: str,
    seed: int,
) -> None:
    if len(stages) < 2:
        raise ValueError("At least two stages are required for a stage-aware maturity proxy.")
    if not 0.0 <= within_stage_weight <= 1.0:
        raise ValueError("within_stage_weight must be between 0 and 1.")
    if mode not in {"stage_aware", "within_stage_rank"}:
        raise ValueError("mode must be one of stage_aware, within_stage_rank.")

    loaded: list[dict[str, object]] = []
    common_genes: set[str] | None = None
    for stage in stages:
        units, genes = read_h5ad_axis_names(stage.h5ad)
        matrix = read_h5ad_csr_matrix(stage.h5ad, layer="count")
        if matrix.shape != (len(units), len(genes)):
            raise ValueError(f"Matrix shape does not match axes for {stage.h5ad}.")
        loaded.append({"stage": stage, "units": units, "genes": genes, "matrix": matrix})
        gene_set = set(genes)
        common_genes = gene_set if common_genes is None else common_genes & gene_set
    if not common_genes:
        raise ValueError("No common genes across input stages.")
    ordered_common = [gene for gene in loaded[0]["genes"] if gene in common_genes]

    matrices: list[sp.csr_matrix] = []
    entropies: list[np.ndarray] = []
    for item in loaded:
        genes = list(item["genes"])
        matrix = sp.csr_matrix(item["matrix"])
        index = {gene: idx for idx, gene in enumerate(genes)}
        subset = matrix[:, [index[gene] for gene in ordered_common]]
        matrices.append(_normalize_counts(subset))
        entropies.append(_expression_entropy(subset))

    x_all = sp.vstack(matrices, format="csr")
    n_fit = min(int(n_components), x_all.shape[0] - 1, x_all.shape[1] - 1)
    if n_fit < 1:
        raise ValueError(f"Maturity SVD requires at least a 2x2 matrix; got {x_all.shape}.")
    embedding = TruncatedSVD(n_components=n_fit, random_state=seed).fit_transform(x_all)
    local_order = _rank01(embedding[:, 0])

    numeric_stages = np.asarray([float(item["stage"].label) for item in loaded], dtype=float)
    stage_scores = _minmax(numeric_stages)
    expanded_stage = np.concatenate(
        [
            np.full(len(item["units"]), stage_scores[index], dtype=float)
            for index, item in enumerate(loaded)
        ]
    )
    if mode == "stage_aware":
        pseudotime = _minmax(
            (1.0 - float(within_stage_weight)) * expanded_stage
            + float(within_stage_weight) * local_order
        )
    else:
        pseudotime = np.zeros(x_all.shape[0], dtype=float)
        start = 0
        for item in loaded:
            stop = start + len(item["units"])
            pseudotime[start:stop] = _rank01(embedding[start:stop, 0])
            start = stop
    sr = _minmax(np.concatenate(entropies))
    potency = 1.0 - pseudotime

    start = 0
    rows: list[dict[str, object]] = []
    for item in loaded:
        stage = item["stage"]
        stop = start + len(item["units"])
        table = pd.DataFrame(
            {
                "spot_id": item["units"],
                "maturity": pseudotime[start:stop],
                "pseudotime": pseudotime[start:stop],
                "sr": sr[start:stop],
                "potency_score": potency[start:stop],
                "stage_label": stage.label,
                "maturity_source": "factory_proxy_stage_expression_svd",
            }
        )
        stage.output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(stage.output_csv, index=False)
        rows.append(
            {
                "stage": stage.label,
                "h5ad": str(stage.h5ad),
                "output_csv": str(stage.output_csv),
                "n_units": len(item["units"]),
            }
        )
        start = stop

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(
            {
                "feature_mode": "factory_proxy",
                "semantic_status": "proxy_not_externally_validated_pseudotime",
                "maturity_mode": mode,
                "direction": (
                    "higher maturity means later developmental stage/order"
                    if mode == "stage_aware"
                    else "higher maturity means later within-slice expression-SVD order"
                ),
                "training_semantics_note": (
                    "When each stage CSV is independently min-max normalized and L_dev is "
                    "computed within timepoint, the effective training signal is within-stage "
                    "expression-SVD order rather than the cross-stage offset."
                ),
                "n_common_genes": len(ordered_common),
                "n_components_requested": int(n_components),
                "n_components_fit": int(n_fit),
                "within_stage_weight": float(within_stage_weight),
                "seed": int(seed),
                "stages": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build maturity proxy CSVs for complete-combined coarse graining."
    )
    parser.add_argument("--stage", nargs="+", required=True, help="Repeat triples: LABEL H5AD OUTPUT_CSV.")
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--within-stage-weight", type=float, default=0.15)
    parser.add_argument(
        "--mode",
        choices=["stage_aware", "within_stage_rank"],
        default="stage_aware",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    build_maturity_proxy(
        _parse_stage_specs(args.stage),
        output_metadata=args.output_metadata,
        n_components=args.n_components,
        within_stage_weight=args.within_stage_weight,
        mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
