"""Preview report layouts and optional real-coordinate figures without training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .methods import METHODS, METHOD_LABELS, ORGANS, expected_conditions
from .tables import PAIRS, SCALES, LEVELS, OPTIMAL_TABLES, NATURAL_TABLE, NATURAL_HIERARCHIES, NATURAL_METRICS
from .plots import save_optimal_table, save_natural_table, draw_natural_figure, draw_optimal_figure


def generate_preview(output_root: Path, *, data_root: Path | None = None,
                     optimal_run: Path | None = None, optimal_scale="seurat_k150",
                     optimal_pair="11.5->12.5", figure_pair="12.5->13.5") -> dict:
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Preview output must be empty; existing files will not be overwritten: {output_root}")
    if optimal_run is not None and data_root is None:
        raise ValueError("Checkpoint figure preview requires --data-root")
    output_root.mkdir(parents=True, exist_ok=True)
    files = []
    natural_columns = ["Organ", "Hierarchy"] + [f"{pair}::{metric}" for pair in PAIRS for metric in NATURAL_METRICS]
    natural = pd.DataFrame([{"Organ": organ, "Hierarchy": hierarchy,
                             **{column: "—" for column in natural_columns[2:]}}
                            for organ in ORGANS for hierarchy in NATURAL_HIERARCHIES], columns=natural_columns)
    save_natural_table(natural, output_root / NATURAL_TABLE, preview=True)
    files.extend(NATURAL_TABLE + extension for extension in (".png", ".pdf"))
    for name, metrics in OPTIMAL_TABLES.items():
        columns = ["Organ", "Input scale"] + [f"{pair}::{metric}" for pair in PAIRS for metric in metrics]
        frame = pd.DataFrame([{"Organ":organ, "Input scale":scale,
                              **{column:"—" for column in columns[2:]}}
                             for organ in ORGANS for scale in SCALES], columns=columns)
        save_optimal_table(frame, output_root / name, metrics=metrics, preview=True)
        files.extend(name + extension for extension in (".png", ".pdf"))
    manifest = {"pipeline":"formal_core_analysis", "status":"layout_preview_only",
                "models_trained":False, "contains_experimental_measurements":False,
                "methods":[METHOD_LABELS[m] for m in METHODS], "method_ids":list(METHODS),
                "organs":list(ORGANS), "scales":list(SCALES), "pairs":list(PAIRS),
                "natural_table_rows": len(natural), "natural_metrics": ["JS", "CQ", "deltaEI"],
                "rows_per_table":len(ORGANS) * len(SCALES),
                "expected_formal_conditions":expected_conditions(SCALES, PAIRS),
                "table_values":"placeholders only; no scientific CSVs generated", "files":files}
    if data_root is not None:
        grid = pd.DataFrame([{"Time pair":figure_pair, **{name:np.nan for name, _, _ in LEVELS[:3]}}])
        name = "figure1_natural"
        draw_natural_figure(data_root, grid, figure_pair, output_root / name,
                            require_k10=False,
                            preview_notice="LAYOUT PREVIEW | Real heart coordinates | ΔEI not computed")
        files.extend(name + extension for extension in (".png", ".pdf"))
        manifest["figure1"] = {"data_root":str(Path(data_root).resolve()), "pair":figure_pair,
                               "layers":["spot", "seurat_k150", "seurat_k40"],
                               "delta_ei":"not computed", "spatial_coordinates":"real local maps"}
    if optimal_run is not None:
        summary = json.loads((Path(optimal_run) / "summary.json").read_text(encoding="utf-8"))
        method = summary["method"]
        notice = "CHECKPOINT PREVIEW | " + method + " | " + optimal_pair.replace("->", " to ")
        if method != "formal_two_stage" or summary.get("loss_protocol") != "formal_two_stage_main":
            notice += " | Historical method; not formal_two_stage_main"
        draw_optimal_figure(data_root, optimal_scale, optimal_pair, optimal_run,
                            float(summary.get("trained_delta_EI", summary["delta_EI_best_checkpoint"])),
                            output_root / "figure2_optimal_NO_ROI_ENLARGE",
                            method_label=METHOD_LABELS.get(method, method), preview_notice=notice)
        files.extend("figure2_optimal_NO_ROI_ENLARGE" + extension for extension in (".png", ".pdf"))
        manifest["contains_experimental_measurements"] = True
        manifest["figure2"] = {"run_dir":str(Path(optimal_run).resolve()), "method":method,
                               "pair":optimal_pair, "input_scale":optimal_scale,
                               "loss_protocol":summary.get("loss_protocol")}
    (output_root / "preview_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output/formal_core_analysis_preview"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--optimal-run", type=Path)
    parser.add_argument("--optimal-scale", choices=SCALES, default="seurat_k150")
    parser.add_argument("--optimal-pair", choices=PAIRS, default="11.5->12.5")
    parser.add_argument("--figure-pair", choices=PAIRS, default="12.5->13.5")
    args = parser.parse_args()
    print(json.dumps(generate_preview(args.output_root, data_root=args.data_root,
                                      optimal_run=args.optimal_run, optimal_scale=args.optimal_scale,
                                      optimal_pair=args.optimal_pair, figure_pair=args.figure_pair),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
