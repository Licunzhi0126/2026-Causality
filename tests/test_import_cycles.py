from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_modules_are_safe_to_import_in_fresh_processes() -> None:
    statements = (
        "from mignet_ce.metrics import pairwise_shared_core_directed_nmf",
        "import mignet_ce.pipelines.vertical",
    )
    for statement in statements:
        completed = subprocess.run(
            [sys.executable, "-c", statement],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_analysis_layer_does_not_import_visualization() -> None:
    analysis_root = PROJECT_ROOT / "mignet_ce" / "downstream" / "analysis"
    offenders = []
    for path in analysis_root.rglob("*.py"):
        if "visualization" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if ".visualization" in source or "mignet_ce.downstream.analysis.visualization" in source:
            offenders.append(path)
    assert offenders == []
