from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import xlogy

from .config import Settings, seed_for


@dataclass(frozen=True)
class Reference:
    source: np.ndarray
    target: np.ndarray
    perm_source: np.ndarray
    perm_target: np.ndarray
    pairs: np.ndarray

    def save(self, path: Path) -> None:
        np.savez(path, source=self.source, target=self.target, perm_source=self.perm_source,
                 perm_target=self.perm_target, pairs=self.pairs)

    @classmethod
    def load(cls, path: Path) -> Reference:
        with np.load(path, allow_pickle=False) as values:
            return cls(**{key: values[key] for key in values.files})


def groups(labels: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def hard_labels(assignment: np.ndarray) -> np.ndarray:
    return np.unique(np.argmax(assignment, axis=1), return_inverse=True)[1]


def seurat_labels(path: Path, unit_ids: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"spot_id", "domain_id"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Seurat map requires spot_id and domain_id: {path}")
        mapping = {}
        for row in reader:
            spot, domain = row["spot_id"], row["domain_id"]
            if spot in mapping or not spot or not domain:
                raise ValueError(f"Duplicate or empty Seurat mapping in {path}: {spot}")
            mapping[spot] = domain
    missing = set(unit_ids) - mapping.keys()
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} spots, including {sorted(missing)[:5]}")
    return np.unique([mapping[spot] for spot in unit_ids], return_inverse=True)[1]


def within_cluster_pairs(labels: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    samples = []
    for members in groups(labels):
        n = len(members)
        total = n * (n - 1) // 2
        if not total:
            continue
        codes = np.sort(rng.choice(total, min(limit, total), replace=False))
        row_starts = np.r_[0, np.cumsum(np.arange(n - 1, 0, -1, dtype=np.int64))]
        left = np.searchsorted(row_starts, codes, side="right") - 1
        right = left + 1 + codes - row_starts[left]
        samples.append(np.column_stack((members[left], members[right])))
    return np.concatenate(samples) if samples else np.empty((0, 2), dtype=np.int64)


def permutation(labels: np.ndarray, features: np.ndarray, cyclic: bool = False) -> np.ndarray:
    perm = np.arange(len(labels))
    for members in groups(labels):
        if len(members) < 2:
            continue
        if cyclic:
            block = features[members]
            axis = int(np.argmax(np.var(block, axis=0)))
            ordered = members[np.argsort(block[:, axis], kind="stable")]
            perm[ordered] = np.roll(ordered, -1)
            continue
        remaining = members.copy()
        while len(remaining) > 1:
            first = remaining[0]
            delta = features[remaining[1:]] - features[first]
            offset = int(np.argmin(np.einsum("ij,ij->i", delta, delta))) + 1
            second = remaining[offset]
            perm[first], perm[second] = second, first
            remaining = np.delete(remaining[1:], offset - 1)
    return perm


def matched_permutation(perm: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffle = np.arange(len(perm))
    for members in groups(labels):
        shuffle[members] = rng.permutation(members)
    return shuffle[perm[np.argsort(shuffle)]]


def reference(source: np.ndarray, target: np.ndarray, features_t: np.ndarray, features_tp: np.ndarray,
              settings: Settings, seed: int) -> Reference:
    pairs = within_cluster_pairs(source, settings.pairs_per_cluster, np.random.default_rng(seed))
    return Reference(source, target, permutation(source, features_t), permutation(target, features_tp), pairs)


def global_distance(p: np.ndarray, source: np.ndarray, target: np.ndarray) -> float | None:
    if np.array_equal(source, np.arange(len(source))) and np.array_equal(target, np.arange(len(target))):
        return None
    total = 0.0
    for start in range(0, len(source), 256):
        rows = source[start:start + 256]
        total += np.abs(p[start:start + len(rows)] - p[np.ix_(rows, target)]).sum(dtype=np.float64)
    return float(total / (2 * len(source)))


def target_mass(p: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([p[:, group].sum(axis=1, dtype=np.float64) for group in columns])


def _pair_summary(values: np.ndarray, labels: np.ndarray, pairs: np.ndarray) -> tuple[float | None, float | None]:
    if not len(pairs):
        return None, None
    _, cluster = np.unique(labels, return_inverse=True)
    size = np.bincount(cluster)
    owner = cluster[pairs[:, 0]]
    counts = np.bincount(owner, minlength=len(size))
    has_pairs = counts > 0
    per_cluster = np.bincount(owner, weights=values, minlength=len(size))[has_pairs] / counts[has_pairs]
    return float(np.average(per_cluster, weights=size[has_pairs])), float(np.mean(per_cluster))


def partition_distances(p: np.ndarray, columns: list[np.ndarray], pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = pairs.T
    local = np.zeros(len(pairs), dtype=np.float64)
    for group in columns:
        ranked = np.sort(p[:, group].astype(np.float64, copy=False), axis=1)
        local += np.abs(ranked[left] - ranked[right]).sum(axis=1, dtype=np.float64) / 2
    masses = target_mass(p, columns)
    weak = np.abs(masses[left] - masses[right]).sum(axis=1) / 2
    if np.any(weak - local > 1e-9):
        raise ArithmeticError("Weak error exceeded local error for the same pair")
    return local, weak


def _torch_global_distance(matrix, source: np.ndarray, target: np.ndarray) -> float | None:
    import torch

    if np.array_equal(source, np.arange(len(source))) and np.array_equal(target, np.arange(len(target))):
        return None
    row_indices = torch.as_tensor(source, device=matrix.device)
    col_indices = torch.as_tensor(target, device=matrix.device)
    total = torch.zeros((), dtype=torch.float64, device=matrix.device)
    for start in range(0, len(source), 256):
        rows = row_indices[start:start + 256]
        moved = matrix[rows[:, None], col_indices[None, :]]
        total += (matrix[start:start + len(rows)] - moved).abs().sum(dtype=torch.float64)
    return float(total.item() / (2 * len(source)))


def _torch_partition_distances(matrix, columns, pairs):
    import torch

    left, right = torch.as_tensor(pairs.T.copy(), device=matrix.device)
    local = torch.zeros(len(pairs), dtype=torch.float64, device=matrix.device)
    weak = torch.zeros_like(local)
    for group in columns:
        values = matrix.index_select(1, torch.as_tensor(group, device=matrix.device))
        ranked = values.sort(dim=1).values
        local += (ranked[left].double() - ranked[right].double()).abs().sum(dim=1) / 2
        masses = values.sum(dim=1, dtype=torch.float64)
        weak += (masses[left] - masses[right]).abs() / 2
    local, weak = torch.stack((local, weak)).cpu().numpy()
    if np.any(weak - local > 1e-9):
        raise ArithmeticError("Weak error exceeded local error for the same pair")
    return local, weak


def symmetry_metrics(p: np.ndarray, ref: Reference, *, matrix=None) -> dict:
    columns = groups(ref.target)
    if matrix is None:
        local, weak = partition_distances(p, columns, ref.pairs)
        global_score = global_distance(p, ref.perm_source, ref.perm_target)
    else:
        local, weak = _torch_partition_distances(matrix, columns, ref.pairs)
        global_score = _torch_global_distance(matrix, ref.perm_source, ref.perm_target)
    outcome = {
        "global_distance": global_score,
        "source_coverage": float(np.mean(ref.perm_source != np.arange(p.shape[0]))),
        "target_coverage": float(np.mean(ref.perm_target != np.arange(p.shape[1]))),
        "pairs": len(ref.pairs), "source_clusters": len(np.unique(ref.source)),
        "target_clusters": len(np.unique(ref.target)),
    }
    for name, values in (("local", local), ("weak", weak), ("gap", np.maximum(local - weak, 0))):
        outcome[f"{name}_distance"], outcome[f"{name}_distance_equal_clusters"] = _pair_summary(values, ref.source, ref.pairs)
    return outcome


def entropies(p: np.ndarray) -> dict:
    marginal = p.mean(axis=0, dtype=np.float64)
    marginal_entropy = -float(xlogy(marginal, marginal).sum()) / np.log(2)
    conditional_entropy = -float(xlogy(p, p).sum(dtype=np.float64) / p.shape[0]) / np.log(2)
    return {
        "marginal_entropy_bits": marginal_entropy,
        "conditional_entropy_bits": conditional_entropy,
        "micro_ei_from_entropy_bits": marginal_entropy - conditional_entropy,
    }


def existence_rows(p: np.ndarray, labels_t: dict, labels_tp: dict, features_t: np.ndarray,
                   features_tp: np.ndarray, settings: Settings, jobs=None, *, matrix=None, entropy=None) -> list[dict]:
    rows = []
    entropy = entropies(p) if entropy is None else entropy
    distance, data = (global_distance, p) if matrix is None else (_torch_global_distance, matrix)
    if jobs is None:
        jobs = ((name, repeat) for name in labels_t for repeat in range(settings.random_partitions + 1))
    for name, repeat in jobs:
        setting_seed = seed_for(settings.perturbation_seed, "existence", name, repeat)
        rng = np.random.default_rng(setting_seed)
        source = rng.permutation(labels_t[name]) if repeat else labels_t[name]
        target = rng.permutation(labels_tp[name]) if repeat else labels_tp[name]
        ref = reference(source, target, features_t, features_tp, settings, setting_seed)
        pair_score = symmetry_metrics(p, ref, matrix=matrix)
        cycles_t = permutation(source, features_t, cyclic=True)
        cycles_tp = permutation(target, features_tp, cyclic=True)
        rows.append({
            "partition": "Random" if repeat else name, "matched_to": name, "repeat": repeat,
            **pair_score, **entropy,
            "global_cycle_distance": distance(data, cycles_t, cycles_tp),
            "cycle_source_coverage": float(np.mean(cycles_t != np.arange(len(source)))),
            "cycle_target_coverage": float(np.mean(cycles_tp != np.arange(len(target)))),
            "global_matched_permutation_distance": distance(
                data, matched_permutation(ref.perm_source, source, rng), matched_permutation(ref.perm_target, target, rng)),
            "global_cycle_matched_permutation_distance": distance(
                data, matched_permutation(cycles_t, source, rng), matched_permutation(cycles_tp, target, rng)),
        })
    return rows
