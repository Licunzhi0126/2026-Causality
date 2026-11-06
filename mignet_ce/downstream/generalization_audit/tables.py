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


def inspect_training_run(job: dict, contract: dict, method_root: Path, *, allowed_root: Path | None = None) -> dict:
    import numpy as np
    root = Path(job["run_dir"]).resolve()
    boundary = allowed_root if allowed_root is not None else method_root / "runs" / job["job_id"]
    root.relative_to(boundary.resolve())
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


def induced_cq_from_saved(run_dir: Path) -> dict:
    """Induced information retention from saved P/S, without an N x K x K tensor.

This is the same uniform-Spot information budget used by dynamic_closure.analysis.
Only the N x K observed distributions and target assignments are held in RAM;
the potentially large N x M micro kernel is accessed through bounded mmap blocks.
"""
    import numpy as np
    from mignet_ce.information_thresholds import closure_signal_threshold_bits
    p = _checked_matrix(run_dir / "PIJ_micro_train.npy")
    source = _checked_matrix(run_dir / "S_t.npy")
    target = _checked_matrix(run_dir / "S_tp.npy")
    if p.shape != (source.shape[0], target.shape[0]) or source.shape[1] != target.shape[1]:
        raise ValueError("CQ micro PIJ and soft assignment dimensions disagree")
    k = source.shape[1]
    t = np.array(target, dtype=np.float64, copy=True)
    t /= t.sum(axis=1, keepdims=True)
    observed = np.empty((p.shape[0], k), dtype=np.float64)
    source_mass = np.zeros(k, dtype=np.float64)
    joint = np.zeros((k, k), dtype=np.float64)
    step = max(1, min(256, 1_000_000 // p.shape[1]))
    for start in range(0, p.shape[0], step):
        block = np.array(p[start:start + step], dtype=np.float64, copy=True)
        block /= block.sum(axis=1, keepdims=True)
        future = block @ t
        future /= future.sum(axis=1, keepdims=True)
        membership = np.array(source[start:start + step], dtype=np.float64, copy=True)
        membership /= membership.sum(axis=1, keepdims=True)
        observed[start:start + step] = future
        source_mass += membership.sum(axis=0)
        joint += membership.T @ future
    source_mass /= p.shape[0]
    joint /= p.shape[0]
    induced = np.divide(joint, source_mass[:, None], out=np.full_like(joint, 1.0 / k),
                        where=source_mass[:, None] > 1e-12)
    induced /= induced.sum(axis=1, keepdims=True)
    available, _ = _matrix_stats(observed)
    effect = source_mass @ induced
    entropy = lambda x: -np.sum(x * np.log2(np.maximum(x, 1e-12)), axis=-1)
    retained = float(entropy(effect) - np.sum(source_mass * entropy(induced)))
    if retained < -1e-8 or retained > available + 1e-7:
        raise ValueError(f"Induced information budget failed: retained={retained}, available={available}")
    retained = max(0.0, retained)
    quality = retained / available if available > 1e-12 else None
    if quality is not None and not -1e-7 <= quality <= 1.000001:
        raise ValueError(f"CQ outside [0,1]: {quality}")
    threshold = closure_signal_threshold_bits(k)
    return {"CQ_induced": quality, "I_available_bits": available, "I_retained_bits": retained,
            "signal_status": "informative" if available >= threshold else "low-signal",
            "cq_status": "defined" if quality is not None else "undefined_zero_available_information"}


def inspect_spot40_run(job: dict, contract: dict, method_root: Path, allowed_root: Path) -> dict:
    import numpy as np
    checked = inspect_training_run(job, contract, method_root, allowed_root=allowed_root)
    run_dir = Path(job["run_dir"])
    inputs = _read_json(run_dir / "input_manifest.json")
    provenance = _read_json(run_dir / "feature_manifest.json")["provenance"]
    for side in ("t", "tp"):
        for key in ("cci", "cci_index"):
            raw = provenance.get(f"{key}_{side}")
            if raw is None or Path(raw).resolve() != Path(job["inputs"][f"{key}_{side}"]).resolve():
                raise ValueError(f"Saved {key}_{side} source path differs from the requested input")
        raw = provenance.get(f"stage_{side}", {}).get("grn_path")
        if raw is None or Path(raw).resolve() != Path(job["inputs"][f"grn_{side}"]).resolve():
            raise ValueError(f"Saved grn_{side} source path differs from the requested input")
        maturity = inputs.get("maturity_provenance", {}).get(side, {})
        raw = maturity.get("path")
        if raw is None or Path(raw).resolve() != Path(job["inputs"][f"maturity_{side}"]).resolve():
            raise ValueError(f"Saved maturity_{side} source differs from the requested developmental CSV")
        for key, expected in (("id_column", "unit_id"), ("maturity_column", "pseudotime"),
                              ("normalization", "minmax"), ("direction", "higher_is_more_mature")):
            if maturity.get(key) != expected:
                raise ValueError(f"Saved maturity {key} differs from the requested protocol")
    budget = induced_cq_from_saved(run_dir)
    summary = _read_json(run_dir / "summary.json")
    for key, measured in (("I_available", budget["I_available_bits"]), ("I_retained", budget["I_retained_bits"])):
        if key in summary and not np.isclose(float(summary[key]), measured, atol=2e-4, rtol=0):
            raise ValueError(f"Selected checkpoint {key} disagrees with the saved P/S information budget")
    if budget["I_available_bits"] > 1e-6 and "closure_quality" in summary:
        if not np.isclose(float(summary["closure_quality"]), budget["CQ_induced"], atol=2e-3, rtol=0):
            raise ValueError("Selected checkpoint CQ disagrees with independently recomputed induced CQ")
    return {**checked, **budget}


def spot40_wide_rows(state: dict, metric: str) -> list[dict]:
    import math
    expected = {(organ, "->".join(pair)) for organ in state["contract"]["organs"]
                for pair in state["contract"]["pairs"]}
    indexed = {}
    for job in state["jobs"]:
        key = (job["organ"], job["time_pair"])
        if key in indexed or job["hierarchy"] != "spot_to_k40" or job["K"] != 40:
            raise ValueError(f"Duplicate or non-Spot/K40 cell in Spot40 report: {key}")
        indexed[key] = job
    if set(indexed) != expected:
        raise ValueError("Spot40 report grid differs from the declared organs/time pairs")
    rows = []
    for organ in state["contract"]["organs"]:
        row = {"Organ": organ.title()}
        for pair in state["contract"]["pairs"]:
            label = "->".join(pair)
            job = indexed[organ, label]
            value = job.get(metric) if job["status"] == "success" else None
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite {metric}: {job['job_id']}")
            row[label] = "—" if value is None else (f"{value:+.4f}" if metric == "trained_delta_EI" else f"{value:.4f}")
        rows.append(row)
    return rows


def export_spot40_tables(root: Path, state: dict, *, dpi: int = 300) -> dict:
    """One authoritative output directory for both tables, their figure and source audit."""
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pairs = ["->".join(pair) for pair in state["contract"]["pairs"]]
    specs = (("train_deltaei", "trained_delta_EI", "Train ΔEI (bits)"),
             ("induced_cq", "CQ_induced", "Induced information-retention CQ"))
    rows_by_metric = {}
    counts = {}
    for name, metric, _ in specs:
        rows = spot40_wide_rows(state, metric)
        rows_by_metric[metric] = rows
        counts[metric] = sum(row[pair] != "—" for row in rows for pair in pairs)
        _write_csv(table_dir / f"table_spot40_{name}.csv", rows, ["Organ", *pairs])
    fields = ["job_id", "organ", "method", "time_pair", "status", "trained_delta_EI", "CQ_induced",
              "I_available_bits", "I_retained_bits", "signal_status", "cq_status", "best_epoch", "Keff_t", "Keff_tp",
              "action", "run_dir", "source_root", "log_path", "error"]
    _write_csv(root / "spot40_sources.csv", state["jobs"], fields)
    failures = [{**j, "error": j.get("error") or j.get("cq_status", "")}
                for j in state["jobs"] if j["status"] != "success" or j.get("CQ_induced") is None]
    _write_csv(root / "spot40_failures.csv", failures, fields)
    rejected = [dict(job_id=j["job_id"], **item) for j in state["jobs"] for item in j.get("rejected_sources", [])]
    _write_csv(root / "spot40_rejected_sources.csv", rejected, ["job_id", "run_dir", "reason"])
    _render_spot40(rows_by_metric, table_dir, state, counts, dpi=dpi)
    return counts


def _render_spot40(rows_by_metric: dict, table_dir: Path, state: dict, counts: dict, *, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    label = "Two-stage" if state["method"] == "maturity_cci_grn_two_stage" else "SHR Method 10"
    n = len(state["jobs"])
    base_title = f"{label} | Spot input | K=40"
    if not state["contract"].get("formal_profile", False):
        base_title += " | SHORT-RUN VALIDATION"
    if counts["trained_delta_EI"] == 0:
        base_title += " | NO VERIFIED RESULTS"
    elif min(counts.values()) < n:
        base_title += " | PARTIAL"
    pairs = ["->".join(p) for p in state["contract"]["pairs"]]
    specs = (("train_deltaei", "trained_delta_EI", "Train ΔEI (bits)"),
             ("induced_cq", "CQ_induced", "Induced information-retention CQ"))

    def draw(ax, metric, title):
        ax.axis("off")
        ax.set_title(f"{title} | {counts[metric]}/{n} values", loc="left", fontsize=12, pad=8)
        body = [[row["Organ"], *[row[p] for p in pairs]] for row in rows_by_metric[metric]]
        widths = [.18, *([.82 / len(pairs)] * len(pairs))]
        table = ax.table(cellText=body, colLabels=["Organ", *[p.replace("->", "→") for p in pairs]],
                         cellLoc="center", colLoc="center", colWidths=widths, bbox=[0, .08, 1, .80])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("white")
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(.6)
            cell.get_text().set_fontweight("bold" if r == 0 else "normal")

    def save(fig, stem):
        try:
            for suffix in ("png", "pdf"):
                path = table_dir / f"{stem}.{suffix}"
                temporary = table_dir / f"{stem}.tmp.{suffix}"
                fig.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
                temporary.replace(path)
        finally:
            plt.close(fig)

    with rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42}):
        for name, metric, title in specs:
            fig, ax = plt.subplots(figsize=(12, 2.8))
            fig.suptitle(base_title, fontsize=12)
            draw(ax, metric, title)
            fig.subplots_adjust(top=.77, bottom=.12, left=.04, right=.96)
            save(fig, f"table_spot40_{name}")
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.5))
        fig.suptitle(base_title, fontsize=13)
        for ax, (_, metric, title) in zip(axes, specs):
            draw(ax, metric, title)
        fig.text(.04, .018, "— = unavailable/failed/undefined CQ. Finite low-signal CQ values are retained. See spot40_sources.csv.", fontsize=8)
        fig.subplots_adjust(top=.88, bottom=.08, hspace=.40, left=.04, right=.96)
        save(fig, "figure_spot40_deltaei_cq")
