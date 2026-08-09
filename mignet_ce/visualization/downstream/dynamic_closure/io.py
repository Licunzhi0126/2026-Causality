from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


LAYER_PREFIX = {
    "spot": "spot",
    "seurat_k150": "seurat150",
    "seurat_k40": "seurat",
}


def layer_stem(layer: str, organ: str, time: str) -> str:
    try:
        prefix = LAYER_PREFIX[layer]
    except KeyError as exc:
        raise ValueError(f"Unsupported closure layer: {layer}") from exc
    return f"{prefix}_{organ}_{time}"


def layer_paths(data_root: Path, organ: str, layer: str, time: str) -> dict[str, Path]:
    root = Path(data_root)
    stem = layer_stem(layer, organ, time)
    return {
        "h5ad": root / layer / organ / f"{stem}.h5ad",
        "cci": root / "cci" / layer / f"{stem}_CCI_total.npz",
        "index": root / "cci" / layer / f"{stem}_index.tsv",
        "grn": root / "grn" / layer / stem / "grn_edges.csv",
        "map": root / layer / organ / f"{stem}_spot_domain_map.csv",
    }


def read_index(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")
    return frame.iloc[:, 0].astype(str).tolist()


def load_matrix(path: Path) -> np.ndarray:
    path = Path(path)
    try:
        return np.asarray(sp.load_npz(path).toarray(), dtype=np.float64)
    except (ValueError, OSError, KeyError):
        loaded = np.load(path)
        if isinstance(loaded, np.ndarray):
            return np.asarray(loaded, dtype=np.float64)
        try:
            if "pij" in loaded.files:
                return np.asarray(loaded["pij"], dtype=np.float64)
            if len(loaded.files) == 1:
                return np.asarray(loaded[loaded.files[0]], dtype=np.float64)
            raise ValueError(f"Cannot infer matrix in {path}; keys={loaded.files}")
        finally:
            loaded.close()


def validate_raw_inputs(
    data_root: Path,
    organ: str,
    time_points: Iterable[str],
) -> None:
    missing: list[Path] = []
    for layer in LAYER_PREFIX:
        for time in time_points:
            paths = layer_paths(data_root, organ, layer, str(time))
            required = [paths["h5ad"], paths["cci"], paths["index"], paths["grn"]]
            if layer != "spot":
                required.append(paths["map"])
            missing.extend(path for path in required if not path.is_file())
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing dynamic-closure inputs: {preview}{suffix}")


def write_manifest(output_root: Path, configuration: dict[str, object]) -> Path:
    import json

    output_root = Path(output_root)
    payload = {
        "configuration": configuration,
        "tables": [
            str(path.relative_to(output_root))
            for path in sorted(output_root.rglob("*.csv"))
        ],
        "matrices": [
            str(path.relative_to(output_root))
            for path in sorted(output_root.rglob("*.npy")) + sorted(output_root.rglob("*.npz"))
        ],
        "figures": [
            str(path.relative_to(output_root))
            for path in sorted(output_root.rglob("*.png")) + sorted(output_root.rglob("*.pdf"))
        ],
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


__all__ = [
    "LAYER_PREFIX",
    "layer_paths",
    "layer_stem",
    "load_matrix",
    "read_index",
    "validate_raw_inputs",
    "write_manifest",
]
