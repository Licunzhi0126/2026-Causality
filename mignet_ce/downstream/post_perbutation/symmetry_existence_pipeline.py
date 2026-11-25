from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import Settings, read_json
from .formal import read_matrices
from .symmetry import entropies, existence_rows, hard_labels, seurat_labels


METHODS = ("MiGNet-CE", "Seurat K40")


def batches(settings: Settings, batch_size: int = 4) -> list[list[tuple[str, int]]]:
    jobs = [(name, repeat) for name in METHODS for repeat in range(settings.random_partitions + 1)]
    return [jobs[start:start + batch_size] for start in range(0, len(jobs), batch_size)]


@lru_cache(maxsize=1)
def _baseline_inputs(baseline: Path, seurat: tuple[Path, Path]):
    p, _, source, target = read_matrices(baseline / "training")
    metadata = read_json(baseline / "prepared" / "metadata.json")
    labels_t = {METHODS[0]: hard_labels(source), METHODS[1]: seurat_labels(seurat[0], metadata["unit_ids_t"])}
    labels_tp = {METHODS[0]: hard_labels(target), METHODS[1]: seurat_labels(seurat[1], metadata["unit_ids_tp"])}
    with np.load(baseline / "prepared" / "arrays.npz", allow_pickle=False) as arrays:
        features_t, features_tp = arrays["encoder_features_t"], arrays["encoder_features_tp"]
    return p, labels_t, labels_tp, features_t, features_tp, entropies(p)


def calculate_batch(baseline: Path, seurat: tuple[Path, Path], settings: Settings, jobs,
                    *, device: str = "cpu", cache: dict | None = None) -> dict:
    started = perf_counter()
    p, labels_t, labels_tp, features_t, features_tp, entropy = _baseline_inputs(baseline, seurat)
    matrix = None
    metrics = {}
    if device != "cpu":
        import torch

        torch.cuda.reset_peak_memory_stats(device)
        key = (baseline, device)
        if cache.get("existence_key") != key:
            cache.update(existence_key=key, existence_matrix=torch.tensor(p, device=device))
        matrix = cache["existence_matrix"]
    rows = existence_rows(p, labels_t, labels_tp, features_t, features_tp, settings, jobs, matrix=matrix, entropy=entropy)
    if matrix is not None:
        metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return {"rows": rows, "seconds": perf_counter() - started, "metrics": metrics}
