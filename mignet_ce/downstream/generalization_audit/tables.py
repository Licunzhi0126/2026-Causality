"""Audit selected training kernels and export partial or complete organ tables.

Missing cells are displayed explicitly as em dashes; the numeric long CSV and
failure CSV distinguish OOM, missing inputs, invalid checkpoints and pending jobs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

TABLE_COLUMNS = ("Spot -> K150", "K150 -> K40", "Spot -> K40")
SUCCESS_METRICS = ("trained_delta_EI", "EI_micro_recomputed", "EI_macro_recomputed", "Keff_t", "Keff_tp", "best_epoch")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _checked_matrix(path: Path, shape: tuple[int, int] | None = None):
    import numpy as np
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or (shape is not None and matrix.shape != shape):
        raise ValueError(f"Unexpected array shape at {path}: {matrix.shape}; expected {shape}")
    return matrix


def _matrix_stats(matrix) -> tuple[float, object]:
    """Uniform-source EI in bits, streamed in bounded blocks over a read-only mmap."""
    import numpy as np
    effect = np.zeros(matrix.shape[1], dtype=np.float64)
    conditional = 0.0
    step = max(1, min(256, 1_000_000 // matrix.shape[1]))
    for start in range(0, matrix.shape[0], step):
        block = np.array(matrix[start:start + step], dtype=np.float64, copy=True)
        sums = block.sum(axis=1)
        if (not np.isfinite(block).all() or np.any(block < -1e-8)
                or not np.allclose(sums, 1.0, atol=1e-4, rtol=0)):
            raise ValueError("Persisted PIJ/assignment must be finite, nonnegative, and row-stochastic.")
        block = np.maximum(block, 0.0)
        block /= block.sum(axis=1, keepdims=True)
        effect += block.sum(axis=0)
        conditional -= float(np.sum(block * np.log2(np.maximum(block, 1e-12))))
    effect /= matrix.shape[0]
    ei = -float(np.sum(effect * np.log2(np.maximum(effect, 1e-12)))) - conditional / matrix.shape[0]
    return ei, effect


def inspect_training_run(job: dict, contract: dict, method_root: Path) -> dict:
    import numpy as np
    root = Path(job["run_dir"]).resolve()
    root.relative_to((method_root / "runs" / job["job_id"]).resolve())
    summary, config = _read_json(root / "summary.json"), _read_json(root / "config.json")
    feature = _read_json(root / "feature_manifest.json")
    if summary.get("method") != contract["method"] or feature.get("method") != contract["method"]:
        raise ValueError("Saved model method disagrees with the requested method.")
    for key, wanted in (("k", job["K"]), ("epochs", contract["epochs"]), ("seed", contract["seed"])):
        if config.get(key) != wanted:
            raise ValueError(f"Saved config {key}={config.get(key)!r}; expected {wanted!r}")
    if summary.get("K") != job["K"]:
        raise ValueError("Summary K disagrees with the requested hierarchy.")
    provenance = feature.get("provenance", {})
    required_features = {"grn_feature_method": "legacy_nmf", "cci_feature_method": "laplacian_hks",
                         "alpha_cci": .20, "nmf_components": 5, "nmf_max_iter_used": contract["nmf_max_iter"]}
    for key, value in required_features.items():
        if provenance.get(key) != value:
            raise ValueError(f"Saved feature protocol mismatch: {key}={provenance.get(key)!r}; expected {value}")
    for side in ("t", "tp"):
        recorded = provenance.get(f"h5ad_{side}")
        if recorded is None or Path(recorded).resolve() != Path(job["inputs"][f"h5ad_{side}"]).resolve():
            raise ValueError(f"Saved input organ/scale/time provenance mismatch: h5ad_{side}")
    micro = _checked_matrix(root / "PIJ_micro_train.npy")
    k = int(job["K"])
    if k >= min(micro.shape):
        raise ValueError("The saved result is not a genuine coarse graining: K >= micro state count.")
    macro = _checked_matrix(root / "PIJ_macro_train.npy", (k, k))
    ei_micro, _ = _matrix_stats(micro)
    ei_macro, _ = _matrix_stats(macro)
    delta = ei_macro - ei_micro
    reported = summary.get("trained_delta_EI", summary.get("delta_EI_best_checkpoint"))
    if "trained_delta_EI" in summary and "delta_EI_best_checkpoint" in summary:
        if not np.isclose(float(summary["trained_delta_EI"]), float(summary["delta_EI_best_checkpoint"]), atol=1e-6, rtol=0):
            raise ValueError("Saved trained_delta_EI and delta_EI_best_checkpoint aliases disagree.")
    for actual, expected in ((ei_micro, summary.get("EI_micro_fixed")),
                             (ei_macro, summary.get("EI_macro_best_checkpoint")), (delta, reported)):
        if expected is None or not np.isfinite(float(expected)) or not np.isclose(actual, float(expected), atol=2e-4, rtol=0):
            raise ValueError(f"Persisted training PIJ disagrees with saved EI/DeltaEI: {actual} vs {expected}")
    for label, count in (("t", micro.shape[0]), ("tp", micro.shape[1])):
        assignment = _checked_matrix(root / f"S_{label}.npy", (count, k))
        _, usage = _matrix_stats(assignment)
        usage = np.maximum(usage, 1e-12)
        usage /= usage.sum()
        keff = float(np.exp(-np.sum(usage * np.log(usage))))
        if not np.isclose(keff, float(summary.get(f"Keff_{label}", float("nan"))), atol=1e-3, rtol=0):
            raise ValueError(f"Saved Keff_{label} disagrees with learned soft assignments.")
    epoch = int(summary.get("best_epoch", 0))
    if not 1 <= epoch <= contract["epochs"]:
        raise ValueError("Saved best_epoch is outside this training run.")
    checkpoint = "best_joint.pt" if contract["method"] == "maturity_cci_grn_two_stage" else "best_model.pt"
    if not (root / checkpoint).is_file():
        raise FileNotFoundError(f"Missing selected checkpoint: {root / checkpoint}")
    if contract["method"] == "shr_method_10":
        floor = config.get("checkpoint_keff_min")
        overrides = contract.get("runner_extra_args", [])
        # Explicit test overrides are recorded; formal runs use the approved 1e-6 floor.
        if "--checkpoint-keff-min" not in overrides and not any(x.startswith("--checkpoint-keff-min=") for x in overrides):
            if floor != 1e-6:
                raise ValueError("Method 10 formal checkpoint_keff_min must be 1e-6.")
        with (root / "training_history.csv").open(encoding="utf-8-sig", newline="") as handle:
            matches = [row for row in csv.DictReader(handle) if int(row["epoch"]) == epoch]
        if len(matches) != 1:
            raise ValueError("Method 10 selected epoch is not unique in training history.")
        row = matches[0]
        if row["checkpoint_eligible"].lower() not in {"true", "1"} or float(reported) <= 0:
            raise ValueError("Method 10 selected checkpoint violates its native positive-DeltaEI eligibility rule.")
        if not np.isclose(float(row["delta_EI"]), float(reported), atol=2e-4, rtol=0):
            raise ValueError("Method 10 selected history DeltaEI disagrees with final checkpoint.")
        for label in ("t", "tp"):
            if not (float(row[f"Keff_{label}"]) >= float(floor) - 1e-8):
                raise ValueError("Method 10 selected history violates checkpoint Keff floor.")
    return {"trained_delta_EI": float(reported), "EI_micro_recomputed": ei_micro,
            "EI_macro_recomputed": ei_macro, "Keff_t": float(summary["Keff_t"]),
            "Keff_tp": float(summary["Keff_tp"]), "best_epoch": epoch}


def wide_rows(state: dict) -> list[dict]:
    import math
    from .runner import HIERARCHIES
    indexed = {}
    for job in state["jobs"]:
        key = (job["organ"], job["time_pair"], job["column"])
        if key in indexed:
            raise ValueError(f"Duplicate experiment cell: {key}")
        indexed[key] = job
    expected = {(organ, "->".join(pair), HIERARCHIES[hierarchy][0])
                for organ in state["contract"]["organs"]
                for pair in state["contract"]["pairs"]
                for hierarchy in state["contract"]["hierarchies"]}
    if set(indexed) != expected:
        raise ValueError("Manifest jobs do not match the declared organ/time/hierarchy grid.")
    rows = []
    for organ in state["contract"]["organs"]:
        for pair in state["contract"]["pairs"]:
            pair_label = "->".join(pair)
            row = {"Organ": organ.title(), "Time pair": pair_label}
            for label in TABLE_COLUMNS:
                job = indexed.get((organ, pair_label, label))
                value = job.get("trained_delta_EI") if job and job["status"] == "success" else None
                if job and job["status"] == "success" and (value is None or not math.isfinite(float(value))):
                    raise ValueError(f"Successful cell has no finite trained DeltaEI: {job['job_id']}")
                row[label] = f"{float(value):+.4f}" if value is not None else "—"
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_tables(root: Path, state: dict, *, dpi: int = 300) -> None:
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = wide_rows(state)
    stem = table_dir / "table_generalization_train_deltaei"
    _write_csv(stem.with_suffix(".csv"), rows, ["Organ", "Time pair", *TABLE_COLUMNS])
    fields = ["job_id", "organ", "method", "time_pair", "column", "input_scale", "K", "status",
              *SUCCESS_METRICS, "run_dir", "log_path", "error"]
    long_rows = [{**job, **({key: None for key in SUCCESS_METRICS} if job["status"] != "success" else {})}
                 for job in state["jobs"]]
    _write_csv(root / "generalization_long.csv", long_rows, fields)
    _write_csv(root / "failures.csv", [j for j in long_rows if j["status"] not in {"success", "pending", "running"}], fields)
    _render(rows, stem, state, dpi=dpi)


def _render(rows: list[dict], stem: Path, state: dict, *, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    method_label = "Two-stage" if state["method"] == "maturity_cci_grn_two_stage" else "SHR Method 10"
    done = sum(j["status"] == "success" for j in state["jobs"])
    footnote = ("Independent within-organ training; K150/K40 denote learned target prototype counts.\n"
                "— = not selected/pending/unavailable/failed; see generalization_long.csv and failures.csv for reasons.")
    with rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(13.6, max(3.1, 1.6 + .42 * len(rows))))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        title = f"{method_label}  |  Organ generalization  |  Train ΔEI (bits)"
        if not state["contract"].get("formal_profile", False):
            title += "  |  SHORT-RUN VALIDATION"
        fig.text(.035, .955, title, fontsize=13, va="top")
        fig.text(.965, .955, f"{done}/{len(state['jobs'])} verified", fontsize=10, ha="right", va="top")
        labels = ["Organ", "Time pair", *[c.replace("->", "→") for c in TABLE_COLUMNS]]
        body = [[str(row[c]).replace("->", "→") for c in ("Organ", "Time pair", *TABLE_COLUMNS)] for row in rows]
        table = ax.table(cellText=body, colLabels=labels, cellLoc="center", colLoc="center",
                         colWidths=[.15, .19, .22, .22, .22], bbox=[0, .09, 1, .85])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("#C9C9C9")
            cell.set_linewidth(.55)
            cell.PAD = .025
            cell.get_text().set_fontweight("bold" if r == 0 else "normal")
        # One merged-looking organ cell per group, matching the reference layout.
        n_pairs = len(state["contract"]["pairs"])
        for group in range(len(state["contract"]["organs"])):
            first = group * n_pairs + 1
            for offset in range(n_pairs):
                cell = table[first + offset, 0]
                if offset != n_pairs // 2:
                    cell.get_text().set_text("")
                cell.visible_edges = "TLRB" if n_pairs == 1 else "TLR" if offset == 0 else "BLR" if offset == n_pairs - 1 else "LR"
        fig.text(.035, .018, footnote, fontsize=8, va="bottom", color="#555555")
        fig.subplots_adjust(left=.035, right=.965, top=.88, bottom=.12)
        try:
            for suffix in (".png", ".pdf"):
                target = stem.with_suffix(suffix)
                temporary = target.with_name(target.stem + ".tmp" + suffix)
                fig.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
                temporary.replace(target)
        finally:
            plt.close(fig)
