"""Render formal Core table layouts without loading data or training models.

python -m mignet_ce.downstream.formal_core_analysis.preview --output-root PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .methods import METHODS, METHOD_LABELS, expected_conditions
from .plots import _save_triplet_layout, _triplet_columns, save_table
from .tables import PAIRS, SCALES


def generate_preview(output_root: Path) -> dict:
    """Use text placeholders and shared renderers; never emit scientific CSVs."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Preview output must be empty; existing files will not be overwritten: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [{"Method": method, "Input scale": scale,
             **{pair: "—" for pair in PAIRS}} for method in METHODS for scale in SCALES]
    triplets = pd.DataFrame([
        {"Method": method, "Input scale": scale,
         **{column: "—" for column in _triplet_columns()[2:]}}
        for method in METHODS for scale in SCALES
    ], columns=_triplet_columns())
    names = ["preview_optimal_trained_deltaei_keff", "preview_optimal_induced_cq",
             "preview_optimal_closure_js_direct_bits"]
    _save_triplet_layout(triplets, output_root / names[0],
                         title="PREVIEW · Trained ΔEI and effective state usage",
                         subtitle="Layout only · no experiments run · — = not computed\n"
                                  "Within each time pair: ΔEI (wide), source Keff, target Keff (narrow).")
    for name, title, metric in (
        (names[1], "PREVIEW · Optimal induced CQ", "CQ = I retained / I available"),
        (names[2], "PREVIEW · Closure JS Direct (bits)", "Direct prediction JS in bits · lower is better"),
    ):
        save_table(pd.DataFrame(rows), output_root / name, title=title,
                   subtitle=f"Layout only · no experiments run · — = not computed\n{metric}",
                   group_cols=("Method",))
    manifest = {
        "pipeline": "formal_core_analysis", "status": "layout_preview_only",
        "models_trained": False, "contains_experimental_measurements": False,
        "methods": [METHOD_LABELS[method] for method in METHODS],
        "method_ids": list(METHODS), "scales": list(SCALES), "pairs": list(PAIRS),
        "rows_per_table": len(rows), "expected_formal_conditions": expected_conditions(SCALES, PAIRS),
        "files": [name + extension for name in names for extension in (".png", ".pdf")],
    }
    (output_root / "preview_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=(
        Path(__file__).resolve().parents[3] / "output" / "formal_core_analysis_shr_preview"))
    args = parser.parse_args()
    print(json.dumps(generate_preview(args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
