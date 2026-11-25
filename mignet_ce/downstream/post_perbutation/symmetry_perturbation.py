from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import LEVELS, VARIANTS, Settings, seed_for, write_json
from .symmetry import Reference, groups, symmetry_metrics, target_mass


KEEP_TOLERANCE = 1e-6
MIN_CHANGE = 1e-9
MARGINAL_TOLERANCE = 5e-6


def moved_mass(original: np.ndarray, altered: np.ndarray) -> float:
    return float(np.abs(altered - original).sum(dtype=np.float64) / (2 * len(original)))


def _global_endpoint(p: np.ndarray, ref: Reference) -> np.ndarray:
    target = (p + p[np.ix_(ref.perm_source, ref.perm_target)]) / 2
    row_mass, col_mass = p.sum(axis=1), p.sum(axis=0)
    for _ in range(500):
        target *= (row_mass / target.sum(axis=1))[:, None]
        current = target.sum(axis=0)
        np.multiply(target, np.divide(col_mass, current, out=np.zeros_like(current), where=current > 0), out=target)
        if np.max(np.abs(target.sum(axis=1) - row_mass)) < 1e-10:
            return target
    raise ValueError("Global enhancement did not converge to the original marginals")


def _weak_endpoint(p: np.ndarray, ref: Reference) -> np.ndarray:
    target = p.copy()
    for cluster in groups(ref.source):
        block = p[cluster]
        row_mass = block.sum(axis=1, keepdims=True)
        target[cluster] = row_mass * (block.sum(axis=0) / row_mass.sum())
    return target


class PairObjective:
    """Score a candidate edit using only sampled pairs incident on the candidate rows."""

    def __init__(self, ref: Reference):
        self.pairs = ref.pairs
        self.target = ref.target
        self.columns = groups(ref.target)
        cluster = ref.source[self.pairs[:, 0]]
        sizes = np.bincount(ref.source)
        counts = np.bincount(cluster, minlength=len(sizes))
        self.weight = sizes[cluster] / (counts[cluster] * sizes[counts > 0].sum())
        self.incident = [[] for _ in ref.source]
        for i, (left, right) in enumerate(self.pairs):
            self.incident[left].append(i)
            self.incident[right].append(i)

    def delta(self, p: np.ndarray, rows: np.ndarray, cols: np.ndarray, change: np.ndarray,
              *, local: bool, masses: np.ndarray | None) -> float:
        row_ids = np.unique(rows)
        indices = np.union1d(self.incident[row_ids[0]], self.incident[row_ids[-1]]).astype(int)
        if not len(indices):
            return 0.0
        used, inverse = np.unique(self.pairs[indices], return_inverse=True)
        pair_positions = inverse.reshape(-1, 2)
        if local:
            columns = self.columns[self.target[cols[0]]]
            values = p[np.ix_(used, columns)]
            positions = np.searchsorted(columns, cols)
        else:
            values = masses[used]
            positions = self.target[cols]
        original = np.sort(values, axis=1) if local else values.copy()
        row_positions = np.searchsorted(used, rows)
        present = (row_positions < len(used)) & np.isin(rows, used)
        np.add.at(values, (row_positions[present], positions[present]), change[present])
        updated = np.sort(values, axis=1) if local else values
        left, right = pair_positions.T
        difference = (np.abs(updated[left] - updated[right]) - np.abs(original[left] - original[right])).sum(axis=1)
        return float(np.dot(difference, self.weight[indices]) / 2)


def _global_delta(p: np.ndarray, ref: Reference, rows: np.ndarray, cols: np.ndarray, change: np.ndarray) -> float:
    n_cols = p.shape[1]
    keys = rows * n_cols + cols
    transformed = ref.perm_source[rows] * n_cols + ref.perm_target[cols]
    affected = np.union1d(keys, transformed)
    a, b = affected // n_cols, affected % n_cols
    old = p[a, b] - p[ref.perm_source[a], ref.perm_target[b]]
    delta = np.zeros(len(affected))
    np.add.at(delta, np.searchsorted(affected, keys), change)
    np.add.at(delta, np.searchsorted(affected, transformed), -change)
    return float((np.abs(old + delta) - np.abs(old)).sum() / (2 * len(p)))


def _balanced_endpoint(p: np.ndarray, ref: Reference, family: str, variant: str,
                       rng: np.random.Generator) -> tuple[np.ndarray, int]:
    output = p.copy()
    if family != "global" and not len(ref.pairs):
        return output, 0
    columns = groups(ref.target)
    within = [group for group in columns if len(group) >= 2]
    same_target = family == "local" or (family == "weak" and variant == "keep")
    if (same_target and not within) or (family == "weak" and variant != "keep" and len(columns) < 2):
        return output, 0

    objective = PairObjective(ref) if (family == "local" and variant != "random") or (family == "weak" and variant == "break") else None
    masses = target_mass(output, columns) if family == "weak" and objective is not None else None
    n_rows, n_cols = p.shape
    if min(n_rows, n_cols) < 2:
        return output, 0
    total_delta = 0.0
    accepted = 0
    for _ in range(max(512, min(16000, 8 * n_rows))):
        if family == "global":
            a, b = rng.choice(n_rows, 2, replace=False)
        else:
            a, b = ref.pairs[rng.integers(len(ref.pairs))]
        if same_target:
            c, d = rng.choice(within[rng.integers(len(within))], 2, replace=False)
        elif family == "weak":
            u, v = rng.choice(len(columns), 2, replace=False)
            c, d = rng.choice(columns[u]), rng.choice(columns[v])
        else:
            c, d = rng.choice(n_cols, 2, replace=False)
        rows, cols = np.asarray([a, a, b, b]), np.asarray([c, d, c, d])
        coefficients = np.array([1., -1., -1., 1.])
        if family == "global" and variant in {"keep", "break"}:
            keys = rows * n_cols + cols
            transformed = ref.perm_source[rows] * n_cols + ref.perm_target[cols]
            combined, inverse = np.unique(np.r_[keys, transformed], return_inverse=True)
            sign = 1 if variant == "keep" else -1
            coefficients = np.bincount(inverse, weights=np.r_[coefficients, sign * coefficients])
            nonzero = coefficients != 0
            rows, cols, coefficients = combined[nonzero] // n_cols, combined[nonzero] % n_cols, coefficients[nonzero]
        if not len(coefficients):
            continue

        # Random controls sample both directions without selecting by the symmetry score.
        scored = objective is not None or (family == "global" and variant == "break")
        signs = (1, -1) if scored else (rng.choice((-1, 1)),)
        for sign in signs:
            direction = sign * coefficients
            donors = direction < 0
            capacity = float(np.min(output[rows[donors], cols[donors]] / -direction[donors]))
            change = 0.75 * capacity * direction
            if capacity <= 1e-12:
                continue
            delta = 0.0
            if objective is not None:
                delta = objective.delta(output, rows, cols, change, local=family == "local", masses=masses)
            elif scored:
                delta = _global_delta(output, ref, rows, cols, change)
            if variant == "enhance" and delta >= -1e-13:
                continue
            if variant == "break" and delta <= 1e-13:
                continue
            if objective is not None and variant == "keep" and abs(total_delta + delta) > KEEP_TOLERANCE / 2:
                # Full transfers often cross ranks. Fit a smaller transfer within the keep band.
                allowed = KEEP_TOLERANCE / 2 - np.sign(delta) * total_delta
                change *= min(0.5, 0.8 * max(0, allowed) / abs(delta))
                if np.max(np.abs(change)) <= 1e-12:
                    continue
                delta = objective.delta(output, rows, cols, change, local=True, masses=None)
                if abs(total_delta + delta) > KEEP_TOLERANCE / 2:
                    continue
            output[rows, cols] += change
            if masses is not None:
                np.add.at(masses, (rows, ref.target[cols]), change)
            total_delta += delta
            accepted += 1
            break
    return output, accepted


def audit_transition(p: np.ndarray, altered: np.ndarray, ref: Reference, family: str) -> dict:
    if altered.shape != p.shape or not np.isfinite(altered).all() or np.any(altered < 0):
        raise ValueError("Perturbed transition is invalid")
    row_error = float(np.max(np.abs(p.sum(axis=1, dtype=np.float64) - altered.sum(axis=1, dtype=np.float64))))
    col_error = float(np.max(np.abs(p.sum(axis=0, dtype=np.float64) - altered.sum(axis=0, dtype=np.float64))))
    if max(row_error, col_error) > MARGINAL_TOLERANCE:
        raise ValueError(f"Perturbation changed row/column marginals: {row_error}, {col_error}")
    result = {"rho": moved_mass(p, altered), "row_mass_error": row_error, "column_mass_error": col_error}
    if family == "local":
        columns = groups(ref.target)
        error = float(np.max(np.abs(target_mass(p, columns) - target_mass(altered, columns))))
        if error > MARGINAL_TOLERANCE:
            raise ValueError(f"Local perturbation changed destination-cluster masses: {error}")
        result["target_cluster_mass_error"] = error
    return result


def mix_transition(p: np.ndarray, endpoint: np.ndarray, fraction: float) -> np.ndarray:
    return ((1 - fraction) * p.astype(np.float64) + fraction * endpoint).astype(np.float32)


def _direction_valid(base: float | None, score: float | None, variant: str) -> bool:
    if base is None or score is None:
        return False
    delta = score - base
    return {"enhance": delta < -MIN_CHANGE, "keep": abs(delta) <= KEEP_TOLERANCE,
            "break": delta > MIN_CHANGE, "random": True}[variant]


def plan_endpoint(p_path: Path, reference_path: Path, family: str, variant: str, settings: Settings,
                  directory: Path) -> dict:
    p = np.load(p_path, mmap_mode="r", allow_pickle=False)
    ref = Reference.load(reference_path)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed_for(settings.perturbation_seed, family, variant))
    output = np.asarray(p, dtype=np.float64)
    try:
        if family == "global" and variant == "enhance":
            endpoint, moves = _global_endpoint(output, ref), None
        elif family == "weak" and variant == "enhance":
            endpoint, moves = _weak_endpoint(output, ref), None
        else:
            endpoint, moves = _balanced_endpoint(output, ref, family, variant, rng)
    except ValueError as error:
        return {"rho": 0.0, "error": str(error)}
    path = directory / f"{variant}.npy"
    np.save(path, endpoint)
    return {"rho": moved_mass(p, endpoint), "path": str(path), "accepted_moves": moves}


def calibrate_family(p_path: Path, reference_path: Path, family: str, endpoints: dict,
                     directory: Path, levels: tuple[int, ...]) -> dict:
    p = np.load(p_path, mmap_mode="r", allow_pickle=False)
    ref = Reference.load(reference_path)
    metric = f"{family}_distance"
    baseline = symmetry_metrics(p, ref)

    common_budget = min(item["rho"] for item in endpoints.values())
    planned = {}
    for level in levels:
        for variant, item in endpoints.items():
            key = f"sym-{family}-{variant}-{level:02d}"
            rho = common_budget * level / 100
            if rho <= 1e-8 or baseline[metric] is None:
                planned[key] = {"error": "No positive common budget or no eligible comparison"}
                continue
            fraction = rho / item["rho"]
            endpoint = np.load(item["path"], mmap_mode="r", allow_pickle=False)
            altered = mix_transition(p, endpoint, fraction)
            try:
                audit = audit_transition(p, altered, ref, family)
            except ValueError as error:
                planned[key] = {"error": str(error)}
                continue
            scores = symmetry_metrics(altered, ref)
            if abs(audit["rho"] - rho) > max(1e-8, rho * 1e-3):
                planned[key] = {"error": "The float32 transition did not reach the common budget", **audit}
            elif not _direction_valid(baseline[metric], scores[metric], variant):
                planned[key] = {"error": "Target symmetry change not achieved at the common budget",
                                "baseline_distance": baseline[metric], "perturbed_distance": scores[metric], **audit}
            else:
                planned[key] = {"path": item["path"], "fraction": fraction,
                                "audit": {**audit, "rho_target": rho, **{f"fixed_{k}": v for k, v in scores.items()}}}
    result = {"family": family, "baseline": baseline, "common_budget": common_budget,
              "endpoints": endpoints, "conditions": planned}
    write_json(directory / "plan.json", result)
    return result


def plan_family(p_path: Path, reference_path: Path, family: str, settings: Settings,
                directory: Path, levels: tuple[int, ...] = LEVELS) -> dict:
    endpoints = {variant: plan_endpoint(p_path, reference_path, family, variant, settings, directory)
                 for variant in VARIANTS}
    return calibrate_family(p_path, reference_path, family, endpoints, directory, levels)
