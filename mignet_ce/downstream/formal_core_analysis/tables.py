"""Traceable table assembly from measured EI/CQ and strict coarse-grain outputs.

Never label an incomplete or proxy-input smoke run as a complete PNAS result.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import re
import pandas as pd
import numpy as np

LEVELS = (("Spot -> K150", "spot", "seurat_k150"),
          ("K150 -> K40", "seurat_k150", "seurat_k40"),
          ("Spot -> K40", "spot", "seurat_k40"),
          ("K40 -> K10", "seurat_k40", "seurat_k10"),
          ("K150 -> K10", "seurat_k150", "seurat_k10"),
          ("Spot -> K10", "spot", "seurat_k10"))
SCALES = ("spot", "seurat_k150", "seurat_k40")
PAIRS = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
METHODS = ("complete_combined_coarse_maturity_cci_grn", "maturity_cci_grn_two_stage")
STRICT_DELTA_KEY = "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G"


def _json(p: Path) -> dict:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Not a JSON object: {p}")
    return obj


def _exact_one(frame: pd.DataFrame, **fields) -> float:
    hits = frame
    for key, value in fields.items():
        hits = hits[hits[key].eq(value)]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one scientific measurement for {fields}; got {len(hits)}")
    value = float(hits.iloc[0]["EI_bits"])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite EI value for {fields}")
    return value


def build_ei_grid(ei: pd.DataFrame, *, require_k10: bool = True, pairs=PAIRS) -> pd.DataFrame:
    required = {"time_pair", "layer", "EI_bits"}
    if not required <= set(ei):
        raise ValueError(f"EI CSV missing {sorted(required-set(ei))}")
    available = set(ei.layer.unique())
    if require_k10 and "seurat_k10" not in available:
        raise ValueError("Formal full EI existence table requires real K10. No K10 values may be fabricated.")
    levels = LEVELS if require_k10 else LEVELS[:3]
    rows = []
    for pair in pairs:
        row = {"Time pair": pair}
        for name, lower, upper in levels:
            row[name] = _exact_one(ei, time_pair=pair, layer=upper) - _exact_one(ei, time_pair=pair, layer=lower)
        rows.append(row)
    return pd.DataFrame(rows)


def build_cq_grid(cq: pd.DataFrame, *, require_k10: bool = True, pairs=PAIRS) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"time_pair", "hierarchy", "CQ", "signal_status", "I_available_bits", "I_retained_bits"}
    if not required <= set(cq):
        raise ValueError(f"CQ CSV missing {sorted(required-set(cq))}")
    hierarchy = (("Spot -> K150", "spot->seurat_k150"),
                 ("K150 -> K40", "seurat_k150->seurat_k40"),
                 ("Spot -> K40", "spot->seurat_k40"))
    if require_k10:
        hierarchy += (("Spot -> K10", "spot->seurat_k10"),)
    rows, audit = [], []
    for pair in pairs:
        row = {"Time pair": pair}
        for title, name in hierarchy:
            hit = cq.loc[cq.time_pair.eq(pair) & cq.hierarchy.eq(name)]
            if len(hit) != 1:
                raise ValueError(f"Expected one CQ entry for {name} {pair}; got {len(hit)}")
            record = hit.iloc[0].to_dict()
            value = float(record["CQ"])
            if np.isfinite(value) and not (0 <= value <= 1 + 1e-6):
                raise ValueError(f"Invalid CQ at {name} {pair}: {value}")
            row[title] = value if np.isfinite(value) else np.nan
            audit.append({"Time pair": pair, "Hierarchy": title, "raw_CQ": value,
                          "shown_CQ": value if np.isfinite(value) else None, "signal_status": record["signal_status"],
                          "I_available_bits": record["I_available_bits"], "I_retained_bits": record["I_retained_bits"]})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(audit)


def selected_candidate_sensitivity(root: Path, alpha: float, grn_method: str, transition: str = "KL+OT"):
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError("Formal sensitivity completion manifest is required")
    manifest = _json(root / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Sensitivity run did not finish")
    ei = pd.read_csv(root / "alpha_layer_ei.csv")
    cq = pd.read_csv(root / "alpha_natural_cq.csv")
    def select(f):
        s = f.loc[f.grn_feature.eq(grn_method) & f.transition.eq(transition) & np.isclose(f.alpha, alpha)].copy()
        if s.empty:
            raise ValueError(f"No actual computed {grn_method}, α={alpha}, {transition} measurements")
        return s
    return select(ei), select(cq), manifest


def _infer_run_descriptors(run: Path, summary: dict, config: dict) -> dict:
    """Trust explicit saved metadata only; never infer a paper row from folder spelling."""
    merged = {**config, **summary}
    method = merged.get("method")
    if method not in METHODS:
        raise ValueError(f"No valid method in saved summary: {run}")
    scale = next((merged[k] for k in ("input_scale", "layer", "coarse_scale") if k in merged), None)
    pair = next((merged[k] for k in ("time_pair", "pair") if k in merged), None)
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        pair = f"{pair[0]}->{pair[1]}"
    if pair is not None:
        pair = str(pair).replace(":", "->").replace("_to_", "->")
    return {"method": method, "input_scale": scale, "time_pair": pair,
            "organ": merged.get("organ")}


def inspect_optimal_run(run_dir: Path) -> dict:
    p = Path(run_dir)
    if not (p / "summary.json").is_file():
        raise FileNotFoundError(f"Missing summary: {p}")
    summary = _json(p / "summary.json")
    strict_path = p / "strict_complete_combined_maturity_evaluation.json"
    if not strict_path.is_file():
        raise FileNotFoundError(f"Missing strict evaluation; cannot report training-only Delta EI: {p}")
    strict = _json(strict_path)
    manifest_path = p / "feature_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature manifest; method cannot be audited: {p}")
    feature = _json(manifest_path)
    config_path = next((p/f for f in ("pilot_config.json", "config.json") if (p/f).is_file()), None)
    config = _json(config_path) if config_path else {}
    desc = _infer_run_descriptors(p, summary, config)
    provenance = feature.get("provenance", {})
    input_t = provenance.get("h5ad_t", "")
    input_tp = provenance.get("h5ad_tp", "")
    # Explicit context produced by the multiscale runner is preferred; provenance
    # paths are used to validate organ identity, never the output directory name.
    context_path = p / "experiment_context.json"
    context = _json(context_path) if context_path.is_file() else {}
    if desc["organ"] is None:
        desc["organ"] = context.get("organ")
    mt = re.search(r"_(heart|lung|brain)_(\d+(?:\.\d+)?)\.h5ad$", str(input_t))
    mp = re.search(r"_(heart|lung|brain)_(\d+(?:\.\d+)?)\.h5ad$", str(input_tp))
    if mt and mp:
        if mt.group(1) != mp.group(1):
            raise ValueError(f"Cross-organ input accidentally mixed: {p}")
        if desc["organ"] is not None and desc["organ"] != mt.group(1):
            raise ValueError(f"Organ metadata disagrees with H5AD provenance: {p}")
        desc["organ"] = mt.group(1)
        if desc["time_pair"] is None:
            desc["time_pair"] = f"{mt.group(2)}->{mp.group(2)}"
    if desc["organ"] not in {"heart", "lung", "brain"}:
        raise ValueError(f"Could not verify organ from input provenance/metadata: {p}")
    if desc["input_scale"] is None:
        desc["input_scale"] = next((scale for scale in SCALES if f"/{scale}/" in str(input_t).replace("\\", "/")), None)
    if desc["input_scale"] not in SCALES or desc["time_pair"] not in PAIRS:
        raise ValueError(f"Could not verify source input scale/time pair from saved manifest: {p}")
    cci = strict.get("cci_feature_method")
    alpha = strict.get("alpha_cci")
    raw = strict.get(STRICT_DELTA_KEY)
    if cci != "laplacian_hks" or provenance.get("grn_feature_method") != "legacy_nmf" or alpha is None or not np.isclose(float(alpha), .20):
        raise ValueError(f"Strict protocol does not match frozen CCI=Lap/α=0.20: {p}")
    if raw is None or not np.isfinite(float(raw)):
        raise ValueError(f"Missing finite strict raw Delta EI: {p}")
    if "EI_micro_fixed" in summary and "EI_micro_canonical_ng" in strict:
        if not np.isclose(float(summary["EI_micro_fixed"]), float(strict["EI_micro_canonical_ng"]), atol=1e-4):
            raise ValueError(f"Input EI/strict baseline differ: {p}")
    # Explicitly compute information-retention CQ from the persisted micro PIJ and
    # learned soft assignments; this is not CQ from the strict macro PIJ.
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget
    micro_path = p / "PIJ_micro_train.npy"
    st_path, stp_path = p / "S_t.npy", p / "S_tp.npy"
    for required_path in (micro_path, st_path, stp_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Cannot recompute induced CQ without saved micro PIJ/S: {required_path}")
    micro, st, stp = (np.load(fp, allow_pickle=False) for fp in (micro_path, st_path, stp_path))
    if micro.shape != (st.shape[0], stp.shape[0]):
        raise ValueError(f"Micro PIJ/S shape mismatch: {p}")
    if not np.all(np.isfinite(micro)) or not np.all(np.isfinite(st)) or not np.all(np.isfinite(stp)):
        raise ValueError(f"Non-finite micro PIJ or learned assignment: {p}")
    budget = information_closure_budget(micro, st, stp)
    if budget["information_identity_error"] > 1e-5:
        raise ValueError(f"Information budget identity failed: {p}")
    return {"method": desc["method"], "input_scale": desc["input_scale"], "time_pair": desc["time_pair"],
            "organ": desc["organ"],
            "seed": summary.get("seed", config.get("seed")), "epochs": summary.get("epochs", config.get("epochs")),
            "K": summary.get("K"), "delta_EI_strict_raw": float(raw),
            "delta_EI_train": summary.get("delta_EI_best_checkpoint"),
            "hardK_t": summary.get("hardK_t"), "hardK_tp": summary.get("hardK_tp"),
            "CQ_induced": float(budget["closure_quality"]),
            "I_available_bits": float(budget["I_available_bits"]),
            "I_retained_bits": float(budget["I_macro_retained_bits"]),
            "signal_status": budget["signal_status"],
            "CQ_display": float(budget["closure_quality"]),
            "CCI": cci, "alpha": float(alpha), "run_dir": str(p.resolve()),
            "provenance": "saved_run_audit_requires_formal_maturity_validation"}


def collect_optimal_runs(root: Path) -> tuple[pd.DataFrame, list[str]]:
    root = Path(root)
    if not root.exists():
        return pd.DataFrame(), [f"Optimal runs directory does not exist: {root}"]
    if not root.is_dir():
        return pd.DataFrame(), [f"Optimal runs root is not a directory: {root}"]
    runs = [root] if (root / "summary.json").is_file() else sorted({p.parent for p in root.rglob("summary.json")})
    if not runs:
        return pd.DataFrame(), [
            f"Found 0 summary.json under {root}. GRN/CCI input files are not trained optimal runs. "
            "Run both coarse-graining frameworks first and point --optimal-runs-root "
            "to their completed output directory; alternatively use --stage natural "
            "to generate only natural EI/CQ tables."
        ]
    good, errors = [], []
    for run in runs:
        try:
            good.append(inspect_optimal_run(run))
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"{run}: {exc}")
    return pd.DataFrame(good), errors


def build_optimal_grid(frame: pd.DataFrame, measure: str, *, require_complete: bool = True) -> pd.DataFrame:
    """Two frameworks x three scales; no best-of-seed / best-of-epoch selection."""
    if frame.empty:
        raise ValueError("No validated optimal runs")
    needed = {"method", "input_scale", "time_pair", measure}
    if not needed <= set(frame):
        raise ValueError(f"Optimal measurements lack {sorted(needed-set(frame))}")
    rows = []
    for method in METHODS:
        for scale in SCALES:
            row = {"Method": method, "Input scale": scale}
            for pair in PAIRS:
                selected = frame.loc[frame.method.eq(method) & frame.input_scale.eq(scale) & frame.time_pair.eq(pair)]
                if len(selected) > 1:
                    raise ValueError(f"Multiple runs found for {method}, {scale}, {pair}. Select seed/epochs explicitly; never cherry-pick")
                if not len(selected):
                    if require_complete:
                        raise ValueError(f"Missing {method} / {scale} / {pair}")
                    row[pair] = np.nan
                else:
                    row[pair] = float(selected.iloc[0][measure])
            rows.append(row)
    return pd.DataFrame(rows)


# The four cross-organ formal tables are built without requiring absent
# measurements to be fabricated. Missing cells are NaN and audited separately.
ORGANS = ("heart", "lung", "brain")
NATURAL_COLUMNS = ("Spot -> K150", "K150 -> K40", "Spot -> K40")


def selected_organ_sensitivity(root: Path, organ: str, alpha: float = .20):
    """Read a single organ's completed scientific source, or signal missing."""
    root = Path(root)
    candidate = root / organ
    if (candidate / "manifest.json").is_file():
        src = candidate
    elif organ == "heart" and (root / "manifest.json").is_file():
        src = root  # backwards compatible with existing heart-only runs
    else:
        return None
    ei, cq, manifest = selected_candidate_sensitivity(src, alpha, "legacy_nmf")
    source_organ = manifest.get("organ", "heart")
    if source_organ != organ:
        raise ValueError(f"Organ manifest mismatch: expected {organ}, got {source_organ}: {src}")
    return ei, cq, src


def build_multiorgan_natural(source_root: Path, *, organs=ORGANS, pairs=PAIRS):
    """Return EI grid, CQ grid, CQ audit and provenance/coverage audit."""
    ei_rows, cq_rows, audit, coverage = [], [], [], []
    for organ in organs:
        source = selected_organ_sensitivity(source_root, organ)
        ei, cq, src = source if source is not None else (None, None, None)
        for pair in pairs:
            ei_row = {"Organ": organ.title(), "Time pair": pair,
                      **{c: np.nan for c in NATURAL_COLUMNS}}
            cq_row = dict(ei_row)
            for name, lower, upper in LEVELS[:3]:
                if ei is not None:
                    a = ei.loc[ei.time_pair.eq(pair) & ei.layer.eq(lower)]
                    b = ei.loc[ei.time_pair.eq(pair) & ei.layer.eq(upper)]
                    if len(a) > 1 or len(b) > 1:
                        raise ValueError(f"Duplicate measured EI: {organ}/{pair}/{name}")
                    if len(a) == len(b) == 1:
                        va, vb = float(a.iloc[0].EI_bits), float(b.iloc[0].EI_bits)
                        if not (np.isfinite(va) and np.isfinite(vb)):
                            raise ValueError(f"Invalid EI: {organ}/{pair}/{name}")
                        ei_row[name] = vb - va
                if cq is not None:
                    target = f"{lower}->{upper}"
                    hits = cq.loc[cq.time_pair.eq(pair) & cq.hierarchy.eq(target)]
                    if len(hits) > 1:
                        raise ValueError(f"Duplicate measured CQ: {organ}/{pair}/{name}")
                    if len(hits) == 1:
                        hit = hits.iloc[0]
                        value = float(hit.CQ)
                        if np.isfinite(value) and not (-1e-6 <= value <= 1+1e-6):
                            raise ValueError(f"Invalid CQ: {organ}/{pair}/{name}: {value}")
                        cq_row[name] = value if np.isfinite(value) else np.nan
                        audit.append({"Organ": organ, "Time pair": pair, "Hierarchy": name,
                                      "CQ": value, "signal_status": hit.signal_status,
                                      "I_available_bits": hit.I_available_bits,
                                      "I_retained_bits": hit.I_retained_bits,
                                      "mapping_method": getattr(hit, "mapping_method", "spot_domain_map")})
                for metric, row in (("EI", ei_row), ("CQ", cq_row)):
                    coverage.append({"Organ": organ, "Time pair": pair, "Hierarchy": name,
                                     "Metric": metric, "available": bool(pd.notna(row[name])),
                                     "source": str(src) if src else ""})
            ei_rows.append(ei_row)
            cq_rows.append(cq_row)
    return (pd.DataFrame(ei_rows), pd.DataFrame(cq_rows),
            pd.DataFrame(audit), pd.DataFrame(coverage))


def build_multiorgan_optimal(runs: pd.DataFrame, measure: str, *, organs=ORGANS, pairs=PAIRS) -> pd.DataFrame:
    rows = []
    for organ in organs:
        row = {"Organ": organ.title()}
        for pair in pairs:
            if runs.empty:
                row[pair] = np.nan
                continue
            match = runs.loc[runs.organ.eq(organ) & runs.time_pair.eq(pair) &
                             runs.method.eq("maturity_cci_grn_two_stage") & runs.input_scale.eq("spot")]
            if len(match) > 1:
                raise ValueError(f"Multiple strict runs for {organ}/{pair}; refuse cherry-picking")
            if len(match) == 1:
                if int(match.iloc[0].K) != 40:
                    raise ValueError(f"Unexpected K for {organ}/{pair}; formal protocol requires 40")
                value = float(match.iloc[0][measure])
                row[pair] = value if np.isfinite(value) else np.nan
            else:
                row[pair] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
