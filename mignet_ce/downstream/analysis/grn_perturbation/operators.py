"""Deterministic source-GRN topology perturbation operators."""

from __future__ import annotations

import hashlib
import numpy as np

METHODS = (
    "random_grn_edge_deletion", "random_grn_edge_attenuation", "hub_regulator_deletion",
    "regulon_deletion", "regulatory_module_deletion", "random_grn_edge_addition",
    "random_grn_edge_amplification", "regulon_addition", "regulatory_module_addition",
    "cross_module_edge_addition", "degree_preserving_rewire", "cross_module_rewire",
)


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def fixed_targets(adjacency: np.ndarray) -> dict[str, object]:
    a = np.asarray(adjacency, dtype=float)
    out = np.abs(a).sum(axis=1)
    hub = int(np.argmax(out))
    neighbours = np.flatnonzero(a[hub] != 0)
    # A compact, deterministic module partition is preferable to recomputing a
    # community layout for every strength/repeat.
    module = np.where(out >= np.median(out), 1, 0).astype(int)
    return {"hub": hub, "regulon": neighbours, "module": int(module[hub]), "modules": module}


def apply_operator(adjacency: np.ndarray, method: str, strength: float, seed: int, targets: dict[str, object]):
    if method not in METHODS:
        raise ValueError(f"Unknown GRN perturbation method: {method}")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    a = np.asarray(adjacency, dtype=float).copy()
    rng = np.random.default_rng(seed)
    present = np.argwhere(a != 0)
    missing = np.argwhere((a == 0) & ~np.eye(a.shape[0], dtype=bool))
    count = int(round(strength * len(present)))
    chosen = present[rng.permutation(len(present))[:count]] if len(present) else np.empty((0, 2), int)
    hub, modules = int(targets["hub"]), np.asarray(targets["modules"])
    if method == "hub_regulator_deletion": chosen = np.argwhere(a[hub] != 0); chosen = np.column_stack([np.full(len(chosen), hub), chosen[:, 0]]) if len(chosen) else chosen
    elif method.startswith("regulon_"):
        candidates = np.argwhere(a[hub] != 0); candidates = np.column_stack([np.full(len(candidates), hub), candidates[:, 0]]) if len(candidates) else candidates
        chosen = candidates[:int(round(strength * len(candidates)))]
    elif method.startswith("regulatory_module_"):
        chosen = np.argwhere((a != 0) & (modules[:, None] == int(targets["module"])))
        chosen = chosen[:int(round(strength * len(chosen)))]
    elif method.startswith("cross_module_"):
        candidates = np.argwhere((a == 0) & (modules[:, None] != modules[None, :]) & ~np.eye(a.shape[0], dtype=bool))
        chosen = candidates[rng.permutation(len(candidates))[:count]] if len(candidates) else candidates
    if method in {"random_grn_edge_deletion", "hub_regulator_deletion", "regulon_deletion", "regulatory_module_deletion"}:
        a[chosen[:, 0], chosen[:, 1]] = 0.0
    elif method == "random_grn_edge_attenuation": a[chosen[:, 0], chosen[:, 1]] *= 1.0 - strength
    elif method == "random_grn_edge_amplification": a[chosen[:, 0], chosen[:, 1]] *= 1.0 + strength
    elif "addition" in method:
        if method in {"random_grn_edge_addition", "regulon_addition", "regulatory_module_addition"}:
            if method == "regulon_addition": missing = np.argwhere((a[hub] == 0)); missing = np.column_stack([np.full(len(missing), hub), missing[:, 0]]) if len(missing) else missing
            elif method == "regulatory_module_addition": missing = np.argwhere((a == 0) & (modules[:, None] == int(targets["module"])) & ~np.eye(a.shape[0], dtype=bool))
            chosen = missing[rng.permutation(len(missing))[:count]] if len(missing) else missing
        value = float(np.mean(np.abs(a[a != 0]))) if np.any(a != 0) else 1.0
        a[chosen[:, 0], chosen[:, 1]] = value
    else:  # rewiring preserves selected edge weights and total edge count.
        for source, target in chosen:
            candidates = np.flatnonzero((a[source] == 0) & (np.arange(a.shape[0]) != source))
            if method == "cross_module_rewire": candidates = candidates[modules[candidates] != modules[source]]
            if len(candidates):
                replacement = int(rng.choice(candidates)); a[source, replacement], a[source, target] = a[source, target], 0.0
    return a, {"target_id": hub, "candidate_edge_count": int(len(present)), "operated_edge_count": int(len(chosen)), "shortfall": max(0, count - len(chosen))}

