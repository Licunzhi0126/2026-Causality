from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_downstream_analysis.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_downstream_analysis_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_cli_exposes_analyze_and_render_subcommands() -> None:
    module = _load_script_module()
    parser = module.build_argparser()
    analyze = parser.parse_args(
        ["analyze", "--metrics-csv", "metrics.csv", "--pair-archive", "pair_archive"]
    )
    render = parser.parse_args(["render"])
    assert analyze.command == "analyze"
    assert render.command == "render"
    assert tuple(analyze.time_points) == module.DEFAULT_TIMES
