from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script_module(name: str):
    script = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_unified_cli_exposes_only_full_run_inputs() -> None:
    module = _load_script_module("run_unified_downstream_analysis")
    args = module.build_parser().parse_args(
        ["--data-root", "data", "--cache-root", "cache", "--output-dir", "out"]
    )
    assert tuple(args.time_points) == ("11.5", "12.5", "13.5", "14.5")
    assert not hasattr(args, "epochs")


def test_grn_perturbation_cli_defaults_to_two_stage_method() -> None:
    module = _load_script_module("run_grn_perturbation")
    args = module.build_parser().parse_args(
        ["--data-root", "data", "--cache-root", "cache", "--output-dir", "out"]
    )
    assert args.method == "maturity_cci_grn_two_stage"
