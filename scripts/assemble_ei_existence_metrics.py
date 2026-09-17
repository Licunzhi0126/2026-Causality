#!/usr/bin/env python3
"""Combine the established three-layer EI results with the Seurat K10 additions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_PAIRS = (
    ("spot", "seurat_k150"),
    ("seurat_k150", "seurat_k40"),
    ("spot", "seurat_k40"),
)
K10_PAIRS = (
    ("seurat_k40", "seurat_k10"),
    ("seurat_k150", "seurat_k10"),
    ("spot", "seurat_k10"),
)
KEY_COLUMNS = ("network_method", "pij_method", "organ", "lower_layer", "upper_layer", "time_pair")
EI_COLUMNS = ("EI_lower", "EI_upper", "EI_gain")


def _read_and_check(
    path: Path,
    *,
    pairs: tuple[tuple[str, str], ...],
    time_pairs: tuple[str, ...],
    organ: str,
    network_method: str,
    pij_method: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing_columns = set(KEY_COLUMNS + EI_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{path} contains duplicate EI metric keys")
    for column, expected in (("organ", organ), ("network_method", network_method), ("pij_method", pij_method)):
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"{path} has unexpected {column}; expected only {expected!r}")
    actual_grid = set(zip(frame["lower_layer"], frame["upper_layer"], frame["time_pair"]))
    expected_grid = {(lower, upper, time_pair) for lower, upper in pairs for time_pair in time_pairs}
    if actual_grid != expected_grid or len(frame) != len(expected_grid):
        missing = sorted(expected_grid - actual_grid)
        extra = sorted(actual_grid - expected_grid)
        raise ValueError(f"{path} has an incomplete EI grid: missing={missing}, extra={extra}")
    values = frame.loc[:, list(EI_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains nonfinite EI values")
    if not np.allclose(values[:, 1] - values[:, 0], values[:, 2], rtol=1e-8, atol=1e-8):
        raise ValueError(f"{path} has EI_gain values inconsistent with EI_upper - EI_lower")
    return frame


def _check_layer_ei_consistency(frame: pd.DataFrame) -> None:
    observations: dict[tuple[str, str], list[float]] = {}
    for row in frame.itertuples(index=False):
        observations.setdefault((str(row.time_pair), str(row.lower_layer)), []).append(float(row.EI_lower))
        observations.setdefault((str(row.time_pair), str(row.upper_layer)), []).append(float(row.EI_upper))
    for (time_pair, layer), values in observations.items():
        if not np.allclose(values, values[0], rtol=1e-6, atol=1e-6):
            raise ValueError(f"EI for {layer} at {time_pair} changes across layer comparisons: {values}")


def assemble_metrics(
    base_path: Path,
    addition_path: Path,
    *,
    organ: str = "heart",
    network_method: str = "light_cci_grn",
    pij_method: str = "NG_KLot",
    time_points: tuple[str, ...] = ("11.5", "12.5", "13.5", "14.5"),
) -> pd.DataFrame:
    if len(time_points) != len(set(time_points)) or len(time_points) < 2:
        raise ValueError("time_points must contain at least two unique values")
    time_pairs = tuple(f"{source}->{target}" for source, target in itertools.combinations(time_points, 2))
    common = dict(
        time_pairs=time_pairs,
        organ=organ,
        network_method=network_method,
        pij_method=pij_method,
    )
    base = _read_and_check(base_path, pairs=BASE_PAIRS, **common)
    addition = _read_and_check(addition_path, pairs=K10_PAIRS, **common)
    combined = pd.concat([base, addition], ignore_index=True)
    if combined.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Combined EI metrics contain duplicate keys")
    _check_layer_ei_consistency(combined)
    pair_order = {pair: position for position, pair in enumerate(BASE_PAIRS + K10_PAIRS)}
    time_order = {pair: position for position, pair in enumerate(time_pairs)}
    combined = combined.assign(
        _pair_order=[pair_order[(lower, upper)] for lower, upper in zip(combined.lower_layer, combined.upper_layer)],
        _time_order=combined.time_pair.map(time_order),
    )
    return combined.sort_values(["_pair_order", "_time_order"]).drop(columns=["_pair_order", "_time_order"]).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Existing three-layer metrics.csv")
    parser.add_argument("--addition", type=Path, required=True, help="New K10-pair metrics.csv")
    parser.add_argument("--output", type=Path, required=True, help="New combined metrics.csv")
    parser.add_argument("--organ", default="heart")
    parser.add_argument("--network-method", default="light_cci_grn")
    parser.add_argument("--pij-method", default="NG_KLot")
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5", "13.5", "14.5"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing metrics: {output}")
    manifest_path = output.parent / "assembly_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace existing assembly manifest: {manifest_path}")
    base = args.base.resolve()
    addition = args.addition.resolve()
    combined = assemble_metrics(
        base,
        addition,
        organ=args.organ,
        network_method=args.network_method,
        pij_method=args.pij_method,
        time_points=tuple(args.time_points),
    )
    provenance = {
        "base": {"path": str(base), "sha256": _sha256(base)},
        "addition": {"path": str(addition), "sha256": _sha256(addition)},
        "organ": args.organ,
        "network_method": args.network_method,
        "pij_method": args.pij_method,
        "time_points": args.time_points,
        "rows": len(combined),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    manifest_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(combined)} EI metric rows to {output}")


if __name__ == "__main__":
    main()
