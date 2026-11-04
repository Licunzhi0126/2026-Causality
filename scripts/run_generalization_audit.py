#!/usr/bin/env python3
"""Run one method's organ generalization experiment and always retain its table."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.generalization_audit.runner import (
    GeneralizationConfig, HIERARCHIES, METHOD_ALIASES, run,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=tuple(METHOD_ALIASES))
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory"))
    p.add_argument("--developmental-root", type=Path,
                   default=Path("/home/jovyan/work/2026 Causality/output/developmental_features"))
    p.add_argument("--output-root", type=Path, required=True,
                   help="Parent directory; two_stage/ or shr_method_10/ is appended automatically.")
    p.add_argument("--organs", nargs="+", choices=("lung", "brain", "heart"), default=["lung", "brain"])
    p.add_argument("--pairs", nargs="+", default=["11.5:12.5", "12.5:13.5", "11.5:13.5"])
    p.add_argument("--hierarchies", nargs="+", choices=tuple(HIERARCHIES), default=list(HIERARCHIES))
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--nmf-max-iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--mode", choices=("run", "resume", "render"), default="run",
                   help="resume revalidates results and preserves incomplete attempts; render only rebuilds reports.")
    p.add_argument("--dry-run", action="store_true", help="Check paths and print commands without creating files.")
    p.add_argument("--runner-extra-args", nargs=argparse.REMAINDER, default=[],
                   help="Explicit trainer overrides; identity, data and fixed feature options cannot be overridden.")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    config = GeneralizationConfig(
        method=a.method, data_root=a.data_root, developmental_root=a.developmental_root,
        output_root=a.output_root, organs=tuple(a.organs), pairs=tuple(a.pairs),
        hierarchies=tuple(a.hierarchies), epochs=a.epochs, nmf_max_iter=a.nmf_max_iter,
        seed=a.seed, device=a.device, dpi=a.dpi, runner_extra_args=tuple(a.runner_extra_args),
    )
    result = run(config, mode=a.mode, dry_run=a.dry_run)
    print(f"[generalization] status={result['status']} table_root={result.get('table_root', '')}")
    # A partial run still exports CSV/PNG/PDF. A nonzero status keeps failures visible to shell callers.
    return 0 if result["status"] in {"complete", "ready"} else 130 if result["status"] == "interrupted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
