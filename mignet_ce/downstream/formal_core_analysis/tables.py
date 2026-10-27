"""Traceable table assembly from measured EI/CQ and saved trained coarse-grain outputs.

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
TRAINED_DELTA_KEY = "trained_delta_EI"


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
    hierarchy = (("Spot -> K150", "spot->seurat_k150"), ("Spot -> K40", "spot->seurat_k40"),
                 ("Spot -> K10", "spot->seurat_k10"))
    if not require_k10:
        hierarchy = hierarchy[:2]
    rows, audit = [], []
    for pair in pairs:
        row = {"Time pair": pair}
        for title, name in hierarchy:
            hit = cq.loc[cq.time_pair.eq(pair) & cq.hierarchy.eq(name)]
            if len(hit) != 1:
                raise ValueError(f"Expected one CQ entry for {name} {pair}; got {len(hit)}")
            record = hit.iloc[0].to_dict()
            valid = str(record["signal_status"]) == "informative"
            value = float(record["CQ"])
            if not (np.isfinite(value) and 0 <= value <= 1 + 1e-6):
                raise ValueError(f"Invalid CQ at {name} {pair}: {value}")
            row[title] = value  # Do not suppress measured low-signal CQ.
            audit.append({"Time pair": pair, "Hierarchy": title, "raw_CQ": value,
                          "shown_CQ": value, "signal_status": record["signal_status"],
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
    if manifest.get("cci_feature") != "laplacian_hks":
        raise ValueError("Sensitivity manifest must identify CCI=laplacian_hks")
    def select(f):
        required = {"grn_feature", "transition", "alpha"}
        if not required <= set(f):
            raise ValueError(f"Sensitivity CSV lacks selector columns: {sorted(required-set(f))}")
        # Sensitivity CQ CSV intentionally omits per-row cci_feature;
        # the completed manifest records the single CCI representation.
        select_cci = f.cci_feature.eq("laplacian_hks") if "cci_feature" in f.columns else True
        s = f.loc[f.grn_feature.eq(grn_method) & select_cci & f.transition.eq(transition) & np.isclose(f.alpha, alpha)].copy()
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
    return {"method": method, "input_scale": scale, "time_pair": pair}


def inspect_optimal_run(run_dir: Path) -> dict:
    p = Path(run_dir)
    if not (p / "summary.json").is_file():
        raise FileNotFoundError(f"Missing summary: {p}")
    summary = _json(p / "summary.json")
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
    if desc["input_scale"] is None:
        desc["input_scale"] = next((scale for scale in SCALES if f"/{scale}/" in str(input_t).replace("\\", "/")), None)
    if desc["time_pair"] is None and input_t and input_tp:
        # Exact time points taken from paths recorded by the completed source loader.
        match_t = re.search(r"_heart_(\d+(?:\.\d+)?)\.h5ad$", str(input_t))
        match_tp = re.search(r"_heart_(\d+(?:\.\d+)?)\.h5ad$", str(input_tp))
        if match_t and match_tp:
            desc["time_pair"] = f"{match_t.group(1)}->{match_tp.group(1)}"
    if desc["input_scale"] not in SCALES or desc["time_pair"] not in PAIRS:
        raise ValueError(f"Could not verify source input scale/time pair from saved manifest: {p}")
    cci = provenance.get("cci_feature_method")
    alpha = provenance.get("alpha_cci")
    if cci != "laplacian_hks" or provenance.get("grn_feature_method") != "legacy_nmf" or alpha is None or not np.isclose(float(alpha), .20):
        raise ValueError(f"Saved feature protocol does not match frozen CCI=Lap/GRN=NMF/alpha=0.20: {p}")
    raw = summary.get(TRAINED_DELTA_KEY, summary.get("delta_EI_best_checkpoint"))
    if raw is None or not np.isfinite(float(raw)):
        raise ValueError(f"Missing finite trained Delta EI in summary.json: {p}")
    for field in ("EI_macro_best_checkpoint", "EI_micro_fixed", "Keff_t", "Keff_tp"):
        if field not in summary or not np.isfinite(float(summary[field])):
            raise ValueError(f"Missing finite {field} in summary.json: {p}")
    if not np.isclose(float(raw), float(summary["EI_macro_best_checkpoint"])-float(summary["EI_micro_fixed"]), atol=1e-4, rtol=0):
        raise ValueError(f"Saved trained Delta EI disagrees with EI_macro - EI_micro: {p}")
    # Independently check the persisted training transition matrix against summary.
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import effective_information
    macro_path = p / "PIJ_macro_train.npy"
    if not macro_path.is_file():
        raise FileNotFoundError(f"Missing trained macro PIJ: {macro_path}")
    macro_pij = np.load(macro_path, allow_pickle=False)
    if macro_pij.ndim != 2 or not np.isfinite(macro_pij).all() or np.any(macro_pij < -1e-8):
        raise ValueError(f"Invalid trained macro PIJ: {macro_path}")
    if not np.allclose(macro_pij.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Trained macro PIJ not row stochastic: {macro_path}")
    calculated_macro = effective_information(macro_pij)
    if not np.isclose(calculated_macro, float(summary["EI_macro_best_checkpoint"]), atol=2e-4, rtol=0):
        raise ValueError(f"Saved trained PIJ / EI_macro summary mismatch ({calculated_macro} vs {summary['EI_macro_best_checkpoint']}): {p}")
    # Explicitly compute information-retention CQ from the persisted micro PIJ and
    # learned soft assignments; this is an assignment-induced CQ, not a transition-prediction metric.
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
    for arr, label in ((st, "t"), (stp, "tp")):
        if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"Assignment not row stochastic at {label}: {p}")
        usage = np.maximum(arr.mean(axis=0), 1e-12)
        usage /= usage.sum()
        verified_keff = float(np.exp(-np.sum(usage*np.log(usage))))
        if not np.isclose(verified_keff, float(summary[f"Keff_{label}"]), atol=1e-3, rtol=0):
            raise ValueError(f"Recomputed Keff_{label} mismatch at {p}")
    budget = information_closure_budget(micro, st, stp)
    # Direct prediction uses the persisted TRAINED macro kernel, not induced Q.
    from .closure_js import direct_js_bits
    direct_js = direct_js_bits(micro, st, stp, macro_pij)
    if budget["information_identity_error"] > 1e-5:
        raise ValueError(f"Information budget identity failed: {p}")
    return {"method": desc["method"], "input_scale": desc["input_scale"], "time_pair": desc["time_pair"],
            "seed": summary.get("seed", config.get("seed")), "epochs": summary.get("epochs", config.get("epochs")),
            "K": summary.get("K"), "trained_delta_EI": float(raw),
            "Keff_t": float(summary["Keff_t"]), "Keff_tp": float(summary["Keff_tp"]),
            "hardK_t": summary.get("hardK_t"), "hardK_tp": summary.get("hardK_tp"),
            "CQ_induced": float(budget["closure_quality"]),
            "I_available_bits": float(budget["I_available_bits"]),
            "I_retained_bits": float(budget["I_macro_retained_bits"]),
            "signal_status": budget["signal_status"],
            "CQ_display": float(budget["closure_quality"]),
            "closure_JS_direct_bits": direct_js,
            "CCI": cci, "alpha": float(alpha), "run_dir": str(p.resolve()),
            "provenance": "persisted_train_PIJ_checkpoint_and_assignments_verified"}


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


def build_optimal_triplet_grid(frame: pd.DataFrame, *, require_complete: bool=True) -> pd.DataFrame:
    """Three compact numeric subcells per time pair: DeltaEI | Keff_t | Keff_tp.

    No invented results or silent missing placeholders. The renderer groups columns
    visually under each time pair. Original source/target values are preserved.
    """
    if frame.empty:
        raise ValueError("No trained optimal runs")
    required={"method", "input_scale", "time_pair", "trained_delta_EI", "Keff_t", "Keff_tp"}
    if not required <= set(frame):
        raise ValueError(f"Missing trained optimal fields: {sorted(required - set(frame))}")
    rows=[]
    for method in METHODS:
        for scale in SCALES:
            row={"Method":method, "Input scale":scale}
            for pair in PAIRS:
                hit=frame.loc[frame.method.eq(method)&frame.input_scale.eq(scale)&frame.time_pair.eq(pair)]
                if len(hit)>1:
                    raise ValueError(f"Duplicate runs for {method}/{scale}/{pair}")
                if len(hit)==0:
                    if require_complete:
                        raise ValueError(f"Missing run for {method}/{scale}/{pair}")
                    continue
                for key in ("trained_delta_EI", "Keff_t", "Keff_tp"):
                    val=float(hit.iloc[0][key])
                    if not np.isfinite(val):
                        raise ValueError(f"Non-finite {key} at {method}/{scale}/{pair}")
                    row[f"{pair}::{key}"]=val
            rows.append(row)
    return pd.DataFrame(rows)
