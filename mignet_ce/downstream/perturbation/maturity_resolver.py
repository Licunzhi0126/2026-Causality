"""Resolve the already-produced maturity inputs used by the main pipeline."""
from __future__ import annotations

from pathlib import Path

MATURITY_ID_COLUMN = "unit_id"
MATURITY_VALUE_COLUMN = "pseudotime"


def resolve_maturity(*, supplied_root: Path | None) -> tuple[Path, Path, dict[str, object]]:
    """Return official source/target maturity files without creating a proxy."""
    if supplied_root is None:
        raise FileNotFoundError(
            "Perturbation training requires --maturity-root containing source.csv and target.csv; "
            "no maturity proxy is generated."
        )
    root = Path(supplied_root)
    maturity_t, maturity_tp = root / "source.csv", root / "target.csv"
    missing = [str(path) for path in (maturity_t, maturity_tp) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official maturity inputs are missing: {missing}")
    return maturity_t, maturity_tp, {
        "source": "supplied",
        "paths": [str(maturity_t), str(maturity_tp)],
        "id_column": MATURITY_ID_COLUMN,
        "maturity_column": MATURITY_VALUE_COLUMN,
    }
