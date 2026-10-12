"""Preflight checks for Plan 1 inputs and WYT runner parameters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_files(paths: dict[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"Optimal CG required inputs are missing: {missing}")


def validate_maturity(path: Path, spot_ids: list[str]) -> None:
    table = pd.read_csv(path)
    if not {"spot_id", "maturity"}.issubset(table.columns):
        raise ValueError(f"Maturity CSV must contain spot_id and maturity: {path}")
    ids = table["spot_id"].astype(str)
    if ids.duplicated().any() or set(ids) != set(map(str, spot_ids)):
        raise ValueError(f"Maturity IDs must exactly cover the CCI index: {path}")
    if table["maturity"].isna().any():
        raise ValueError(f"Maturity values must not be missing: {path}")


def validate_run(*, inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path, spot_ids_t: list[str], spot_ids_tp: list[str], k: int, out_dir: Path, device: str) -> None:
    require_files(inputs)
    if k <= 1:
        raise ValueError("Optimal CG K must be greater than one.")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported device: {device}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Runner output directory must be empty: {out_dir}")
    validate_maturity(maturity_t, spot_ids_t)
    validate_maturity(maturity_tp, spot_ids_tp)

