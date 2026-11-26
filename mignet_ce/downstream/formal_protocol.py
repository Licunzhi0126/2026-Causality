"""Formal Core controls shared by the downstream experiments."""
from __future__ import annotations

import csv
import os
import random

from mignet_ce.coarse_frontends.method_specs import FORMAL_LOCAL_GRAPH_MODE


FORMAL_SEED = 42
FORMAL_KNN_K = 30
FORMAL_ALPHA = 0.20
FRONTEND = {"alpha_cci": FORMAL_ALPHA, "cci_feature_method": "laplacian_hks",
            "grn_feature_method": "legacy_nmf"}
TRAINING = {"local_graph_mode": FORMAL_LOCAL_GRAPH_MODE, "knn_k": FORMAL_KNN_K}


def resolve_options(frontend=None, training=None, *, seed=FORMAL_SEED, strict=False):
    """Supply explicit defaults and reject conflicts for a production Formal run."""
    resolved = []
    for supplied, defaults in ((frontend, FRONTEND), (training, TRAINING)):
        options = dict(supplied or {})
        for name, value in {**defaults, "seed": seed}.items():
            if (strict or name == "seed") and name in options and options[name] != value:
                raise ValueError(f"Formal Core requires {name}={value!r}; got {options[name]!r}")
            options.setdefault(name, value)
        resolved.append(options)
    if resolved[1].get("stage1_source") is not None or resolved[1].get("stage1_only", False):
        raise ValueError("Formal Core requires both training stages from scratch")
    return tuple(resolved)


def maturity_columns(paths):
    """Recognize the Formal export and the previously supported maturity CSVs."""
    conventions = (("unit_id", "pseudotime"), ("spot_id", "maturity"))
    columns = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns.append(set(next(csv.reader(stream), ())))
    for id_column, value_column in conventions:
        if all({id_column, value_column} <= header for header in columns):
            return {"maturity_id_column": id_column, "maturity_column": value_column}
    raise ValueError("Maturity files must use a matching unit_id/pseudotime or spot_id/maturity schema")


def reset_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
