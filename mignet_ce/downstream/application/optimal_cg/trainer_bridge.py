"""Subprocess adapter for the repository's authoritative WYT runner."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAINER_SCRIPT = PROJECT_ROOT / "scripts" / "run_wyt_deltaei_coarse_grain.py"
Runner = Callable[..., subprocess.CompletedProcess[str]]
REQUIRED_OUTPUTS = ("summary.json", "S_t.npy", "S_tp.npy", "PIJ_micro_train.npy", "PIJ_macro_train.npy", "assignments_t.csv", "assignments_tp.csv", "best_ei.pt", "best_joint.pt")


def run_training(*, python: str, inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path, out_dir: Path, k: int = 40, device: str = "cpu", runner: Runner = subprocess.run) -> dict[str, object]:
    command = [python, str(TRAINER_SCRIPT), "--method", "maturity_cci_grn_two_stage", "--h5ad-t", str(inputs["h5ad_t"]), "--h5ad-tp", str(inputs["h5ad_tp"]), "--cci-t", str(inputs["cci_t"]), "--cci-tp", str(inputs["cci_tp"]), "--cci-index-t", str(inputs["index_t"]), "--cci-index-tp", str(inputs["index_tp"]), "--grn-t", str(inputs["grn_t"]), "--grn-tp", str(inputs["grn_tp"]), "--maturity-t", str(maturity_t), "--maturity-tp", str(maturity_tp), "--k", str(k), "--out-dir", str(out_dir), "--device", device]
    completed = runner(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    record = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    if completed.returncode != 0:
        raise RuntimeError(f"Optimal CG trainer failed: {completed.stderr}")
    missing = [name for name in REQUIRED_OUTPUTS if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"WYT runner did not produce required outputs: {missing}")
    return record

