#!/usr/bin/env python3
from __future__ import annotations

"""Re-render formal figures from existing unified analysis tables."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-render unified downstream figures from tables only.")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from mignet_ce.downstream.analysis.visualization.workflow import render_unified_downstream_figures

    outputs = render_unified_downstream_figures(args.results_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
