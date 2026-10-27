"""Direct-prediction Jensen-Shannon divergence (bits) for formal Core.

There are TWO distinct scientific inputs:
* Natural hierarchy: reuse measured `closure_JS_direct_bits` from formal sensitivity.
* Trained optimal: compare P_micro_train @ S_tp with S_t @ PIJ_macro_train.

Never use induced/oracle Q or the information-retention CQ as a substitute for
actual independently trained macro PIJ when evaluating direct prediction JS.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from mignet_ce.downstream.analysis.dynamic_closure.analysis import js_rows

NATURAL_JS_COLUMNS = (
    ("Spot -> K150", "spot->seurat_k150"),
    ("Spot -> K40", "spot->seurat_k40"),
    ("K150 -> K40", "seurat_k150->seurat_k40"),
    ("Spot -> K10*", "spot->seurat_k10"),
)


def _finite_distribution(array: np.ndarray, name: str, *, tolerance: float = 1e-4) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError(f"{name}: expected nonempty 2-D array, got {values.shape}")
    if not np.isfinite(values).all() or np.min(values) < -1e-8:
        raise ValueError(f"{name}: non-finite/negative probabilities")
    if not np.allclose(values.sum(axis=1), 1., rtol=0, atol=tolerance):
        raise ValueError(f"{name}: rows are not stochastic")
    return values


def direct_js_bits(p_micro: np.ndarray, s_t: np.ndarray, s_tp: np.ndarray,
                   q_macro_trained: np.ndarray) -> float:
    """Mean across source units; uses exact same js_rows as formal sensitivity.

    P_micro shape (n_t, n_tp), assignments (n_t, k_t)/(n_tp, k_tp),
    Q_macro_train shape (k_t, k_tp). No implicit reordering or K padding.
    """
    micro = _finite_distribution(p_micro, "P_micro_train")
    source = _finite_distribution(s_t, "S_t")
    target = _finite_distribution(s_tp, "S_tp")
    trained = _finite_distribution(q_macro_trained, "PIJ_macro_train")
    expected = (source.shape[0], target.shape[0])
    if micro.shape != expected:
        raise ValueError(f"Micro shape {micro.shape} disagrees with assignment rows {expected}")
    required = (source.shape[1], target.shape[1])
    if trained.shape != required:
        raise ValueError(f"Trained macro shape {trained.shape} disagrees with assignment columns {required}")
    observed = micro @ target
    predicted = source @ trained
    samples = js_rows(observed, predicted)
    if not np.isfinite(samples).all() or np.min(samples) < -1e-8 or np.max(samples) > 1. + 1e-8:
        raise ArithmeticError("Invalid direct Jensen-Shannon divergence in bits")
    return float(np.mean(samples))


def read_optimal_direct_js(run_dir: Path) -> float:
    """Require persisted checkpoint arrays; do not silently fall back to induced Q."""
    root = Path(run_dir)
    names = ("PIJ_micro_train.npy", "S_t.npy", "S_tp.npy", "PIJ_macro_train.npy")
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Direct JS requires checkpoint matrices: {missing}")
    arrays = [np.load(root / name, allow_pickle=False) for name in names]
    return direct_js_bits(*arrays)


def natural_js_grid(cq: pd.DataFrame, *, pairs: tuple[str, ...],
                    allow_missing_k10: bool = False) -> pd.DataFrame:
    """Read selected formal sensitivity results, preserving fixed 4-column layout.

    With allow_missing_k10, *only* truly absent K10 can be blank; all other
    scientific measurements remain mandatory. Exploratory output must be labeled.
    """
    required = {"time_pair", "hierarchy", "closure_JS_direct_bits"}
    if not required.issubset(cq.columns):
        raise ValueError(f"Missing natural direct JS columns {sorted(required - set(cq))}")
    records = []
    for pair in pairs:
        row: dict[str, str | float] = {"Time pair": pair}
        for heading, hierarchy in NATURAL_JS_COLUMNS:
            matches = cq.loc[cq.time_pair.eq(pair) & cq.hierarchy.eq(hierarchy)]
            if len(matches) == 0 and hierarchy == "spot->seurat_k10" and allow_missing_k10:
                row[heading] = float("nan")
                continue
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one measured JS: {pair}, {hierarchy}; got {len(matches)}")
            match = matches.iloc[0]
            number = float(match["closure_JS_direct_bits"])
            if not np.isfinite(number) or number < -1e-8 or number > 1. + 1e-8:
                raise ValueError(f"Invalid measured JS bits {number}: {pair}/{hierarchy}")
            if hierarchy == "seurat_k150->seurat_k40":
                if str(match.get("mapping_method")) != "spot_overlap_soft_contingency":
                    raise ValueError(f"K150->K40 requires actual soft-overlap mapping provenance at {pair}")
            row[heading] = number
        records.append(row)
    return pd.DataFrame.from_records(records, columns=["Time pair", *(x[0] for x in NATURAL_JS_COLUMNS)])
