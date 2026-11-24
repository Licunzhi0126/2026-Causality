"""Read-only preflight checks for preprocessed Optimal CG inputs."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def require_files(paths: dict[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"Optimal CG required inputs are missing: {missing}")


def read_index(path: Path) -> list[str]:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if table.shape[1] != 1:
        raise ValueError(f"CCI index must have one headered Spot ID column: {path}")
    ids = table.iloc[:, 0].tolist()
    if not ids or any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"CCI index must contain nonempty unique Spot IDs: {path}")
    return ids


def validate_maturity(path: Path, spot_ids: list[str]) -> None:
    table = pd.read_csv(path, dtype={"spot_id": str}, keep_default_na=False)
    if not {"spot_id", "maturity"}.issubset(table.columns):
        raise ValueError(f"Maturity CSV must contain spot_id and maturity: {path}")
    ids = table["spot_id"]
    if ids.duplicated().any() or set(ids) != set(map(str, spot_ids)):
        raise ValueError(f"Maturity IDs must exactly cover the CCI index: {path}")
    values = pd.to_numeric(table["maturity"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Maturity values must be finite numbers: {path}")


def validate_stage(inputs: dict[str, Path], stage: str) -> list[str]:
    from anndata import read_h5ad
    from mignet_ce.io.loaders import standardize_grn_edges

    ids = read_index(inputs[f"index_{stage}"])
    matrix = sparse.load_npz(inputs[f"cci_{stage}"])
    if matrix.shape != (len(ids), len(ids)):
        raise ValueError(f"CCI shape must match its index for {stage}: {matrix.shape}")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f"CCI values must be finite and nonnegative for {stage}.")
    adata = read_h5ad(inputs[f"h5ad_{stage}"], backed="r")
    try:
        spots = adata.obs_names.astype(str).tolist()
        if not adata.n_vars or len(set(spots)) != len(spots) or set(spots) != set(ids):
            raise ValueError(f"H5AD Spot IDs must exactly cover the CCI index for {stage}.")
    finally:
        adata.file.close()
    edges = standardize_grn_edges(pd.read_csv(inputs[f"grn_{stage}"]))
    if edges.empty or not np.isfinite(edges["weight"].to_numpy()).all():
        raise ValueError(f"GRN must contain readable finite weighted edges for {stage}.")
    return ids


def validate_run(*, inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path,
                 k: int, out_dir: Path, device: str, protected_roots=(),
                 spot_ids_t=None, spot_ids_tp=None) -> None:
    require_files({**inputs, "maturity_t": maturity_t, "maturity_tp": maturity_tp})
    if k <= 1:
        raise ValueError("Optimal CG K must be greater than one.")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; select cpu or auto.")
    root = out_dir.parent.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Optimal CG output directory must be empty: {root}")
    for protected in protected_roots:
        protected = Path(protected).resolve()
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError(f"Output and input directories must be separate: {protected}")
    for path in (*inputs.values(), maturity_t, maturity_tp):
        if root == path.resolve() or root in path.resolve().parents:
            raise ValueError(f"Output directory contains an input file: {path}")
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    import os
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise PermissionError(f"Output parent must be writable: {ancestor}")
    for stage, maturity, supplied_ids in (("t", maturity_t, spot_ids_t), ("tp", maturity_tp, spot_ids_tp)):
        ids = validate_stage(inputs, stage)
        if supplied_ids is not None and list(supplied_ids) != ids:
            raise ValueError(f"Supplied Spot ID order differs from index for {stage}.")
        validate_maturity(maturity, ids)
