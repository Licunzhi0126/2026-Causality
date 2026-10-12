"""Select supplied maturity tables or invoke the established proxy builder."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROXY_SCRIPT = PROJECT_ROOT / "scripts" / "build_maturity_proxy_from_h5ad.py"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def resolve_maturity(*, source_sample: str, target_sample: str, h5ad_t: Path, h5ad_tp: Path, output_root: Path, supplied_root: Path | None, runner: Runner = subprocess.run) -> tuple[Path, Path, dict[str, object]]:
    if supplied_root is not None:
        paths = (supplied_root / "source.csv", supplied_root / "target.csv")
        return paths[0], paths[1], {"source": "user", "paths": [str(path) for path in paths]}
    maturity_root = output_root / "maturity"
    maturity_t, maturity_tp = maturity_root / "source.csv", maturity_root / "target.csv"
    metadata = maturity_root / "metadata.json"
    # The authoritative proxy requires numeric stage labels for stage-aware ordering.
    command = [sys.executable, str(PROXY_SCRIPT), "--stage", "7", str(h5ad_t), str(maturity_t), "21", str(h5ad_tp), str(maturity_tp), "--output-metadata", str(metadata)]
    completed = runner(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Maturity proxy failed for {source_sample} -> {target_sample}: {completed.stderr}")
    return maturity_t, maturity_tp, {"source": "proxy", "command": command, "returncode": completed.returncode, "metadata": str(metadata)}
