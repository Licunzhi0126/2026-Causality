"""Immutable method-specific model artifact loading and integrity checks."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from dataclasses import field
from .baseline import baseline_pair_dir
from ..metrics import row_normalize

def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

@dataclass(frozen=True)
class BaselineArtifact:
    record: "BaselineRecord"
    root: Path
    checksums: dict[str, str]
    method: str
    checkpoint_name: str


@dataclass(frozen=True)
class BaselineRecord:
    source_assignment: np.ndarray
    target_assignment: np.ndarray
    p: np.ndarray
    q_direct: np.ndarray
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    target_ids: tuple[str, ...] = field(default_factory=tuple)

def load_baseline(baseline_root: Path, pair: str, method: str) -> BaselineArtifact:
    from ..preparation import required_optimized_outputs
    from mignet_ce.coarse_frontends.method_specs import DYNAMIC_CLOSURE_TWO_STAGE, get_coarse_method_spec

    root = baseline_pair_dir(baseline_root, pair, method)
    required = required_optimized_outputs(method)
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete {method} baseline artifact {root}: {missing}")
    source_assignment, target_assignment = row_normalize(np.load(root / "S_t.npy")), row_normalize(np.load(root / "S_tp.npy"))
    p, q_direct = row_normalize(np.load(root / "PIJ_micro_train.npy")), row_normalize(np.load(root / "PIJ_macro_train.npy"))
    source_frame, target_frame = pd.read_csv(root / "assignments_t.csv"), pd.read_csv(root / "assignments_tp.csv")
    source_id = "spot_id" if "spot_id" in source_frame else source_frame.columns[0]
    target_id = "spot_id" if "spot_id" in target_frame else target_frame.columns[0]
    record = BaselineRecord(source_assignment, target_assignment, p, q_direct, tuple(source_frame[source_id].astype(str)), tuple(target_frame[target_id].astype(str)))
    if not np.allclose(record.source_assignment.sum(axis=1), 1.0) or not np.allclose(record.target_assignment.sum(axis=1), 1.0):
        raise ValueError("Frozen assignments must be row-normalized")
    if not np.allclose(record.p.sum(axis=1), 1.0) or not np.allclose(record.q_direct.sum(axis=1), 1.0):
        raise ValueError("Frozen baseline PIJs must be row-normalized")
    checkpoint_name = (
        "best_joint.pt"
        if get_coarse_method_spec(method).training_mode == DYNAMIC_CLOSURE_TWO_STAGE
        else "best_model.pt"
    )
    return BaselineArtifact(
        record,
        root,
        {name: checksum(root / name) for name in required},
        method,
        checkpoint_name,
    )
