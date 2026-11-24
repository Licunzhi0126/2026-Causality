"""Locate existing maturity tables without generating or modifying inputs."""
from pathlib import Path


def resolve_maturity(*, supplied_root: Path) -> tuple[Path, Path, dict[str, object]]:
    paths = (supplied_root / "source.csv", supplied_root / "target.csv")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required maturity CSV files are missing: {missing}")
    return paths[0], paths[1], {"source": "user", "paths": [str(path) for path in paths]}
