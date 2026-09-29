from __future__ import annotations

"""Assignment diagnostics migrated from WYT train_feature_align_deltaei_v40.py.

Adaptation: adds real unit IDs and export-ready cluster diagnostics.
"""

import math

import numpy as np
import torch


EPS = 1e-12


def usage_stats(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    usage = torch.clamp(assignment.mean(dim=0), min=EPS)
    entropy = -(usage * torch.log(usage)).sum()
    return usage, torch.exp(entropy)


def assignment_entropy(assignment: torch.Tensor) -> torch.Tensor:
    k = assignment.shape[1]
    safe = torch.clamp(assignment, min=EPS)
    return (-(safe * torch.log(safe)).sum(dim=1).mean()) / math.log(k)


def assignment_rows(unit_ids: list[str], assignment: np.ndarray) -> list[dict[str, object]]:
    values = np.asarray(assignment, dtype=float)
    hard = values.argmax(axis=1)
    maximum = values.max(axis=1)
    entropy = -(np.clip(values, EPS, None) * np.log(np.clip(values, EPS, None))).sum(axis=1)
    return [
        {
            "spot_id": unit,
            "hard_cluster": int(hard[index]),
            "max_probability": float(maximum[index]),
            "assignment_entropy": float(entropy[index]),
        }
        for index, unit in enumerate(unit_ids)
    ]
