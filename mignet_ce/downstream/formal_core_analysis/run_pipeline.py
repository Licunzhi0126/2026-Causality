"""Full formal Core orchestration using the project's NATIVE maturity builder.

Scientific code is delegated without reimplementing maturity, PIJ, coarse graining,
or independent trained-PIJ verification. Natural measurements use fixed alpha=.20
from original inputs; sensitivity results are not a prerequisite.

Dry run checks inputs and emits exact commands without creating any output files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .methods import METHODS, ORGANS, expected_conditions
from .natural_analysis import NaturalAnalysisConfig, inspect_source_inputs, load_natural_results
from mignet_ce.coarse_frontends.method_specs import (
    LOCAL_GRAPH_MODES, FORMAL_LOCAL_GRAPH_MODE, graph_configuration, validate_graph_configuration,
)
SCALES = ("spot", "seurat_k150", "seurat_k40")
TIMES = ("11.5", "12.5", "13.5")
PAIRS = ("11.5:12.5", "12.5:13.5", "11.5:13.5")
K_BY_SCALE = ("spot=40", "seurat_k150=40", "seurat_k40=10")
STEMS = {"spot": "spot", "seurat_k150": "seurat150", "seurat_k40": "seurat"}
REPO = Path(__file__).resolve().parents[3]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native maturity -> formal two-stage across organs -> formal Core output")
    p.add_argument("--data-root", type=Path, required=True, help="Factory root with spot/seurat_k150/seurat_k40")
    p.add_argument("--grn-root", type=Path, required=True)
    p.add_argument("--cci-root", type=Path, required=True, help="True CCI matrices AND corresponding *_index.tsv")
    p.add_argument("--developmental-root", type=Path, required=True,
                   help="Source/output of native per-layer developmental feature CSV files")
    p.add_argument("--sensitivity-root", type=Path, help="Legacy argument retained; ignored by this fixed-alpha pipeline")
    p.add_argument("--stage", choices=("all", "optimal"), default="all")
    p.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--maturity-mode", choices=("existing", "native"), default="existing",
                   help="existing: read per-organ/per-layer CSVs; native: use scripts/build_developmental_features.py when missing")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--local-graph-mode", choices=LOCAL_GRAPH_MODES, default=FORMAL_LOCAL_GRAPH_MODE)
    p.add_argument("--knn-k", type=int, default=30)
    p.add_argument("--local-dims", type=int, default=2)
    p.add_argument("--figure-pair", choices=("11.5->12.5", "12.5->13.5", "11.5->13.5"), default="12.5->13.5")
    p.add_argument("--dry-run", action="store_true", help="Read-only checks and full command plan; exit nonzero if prerequisites missing")
    p.add_argument("--resume", action="store_true", help="Reuse valid completed runs")
    p.add_argument("--resume-incomplete", action="store_true",
                   help="With --resume, back up and rerun only incomplete/invalid run directories; valid completed runs are reused")
    p.add_argument("--skip-render", action="store_true", help="Only train models; do not run final paper table/figure collector")
    return p


def maturity_files(a: argparse.Namespace) -> dict[tuple[str, str, str], Path]:
    return {(organ, scale, t): a.developmental_root / scale / f"{organ}_{t}_features.csv"
            for organ in a.organs for scale in SCALES for t in TIMES}


def _native_build_command(a: argparse.Namespace, layers: tuple[str, ...]) -> list[str]:
    return [sys.executable, str(REPO / "scripts" / "build_developmental_features.py"),
            "--data-root", str(a.data_root),
            "--output-root", str(a.developmental_root),
            "--organs", *a.organs, "--time-points", *TIMES,
            "--layers", *layers,
            "--seed", str(a.seed)]


def _maturity_plan(a: argparse.Namespace) -> tuple[list[str] | None, list[str], list[str]]:
    """Return native builder command, materialized paths, and actionable issues.

    The project-native builder itself uses mean-over-member-spots for domain
    features.  We do NOT substitute a newly written algorithm or silently claim
    that K150/K40 pseudotimes were inferred independently from domain H5AD.
    """
    files = maturity_files(a)
    missing = [key for key, path in files.items() if not path.is_file()]
    if not missing:
        return None, [], []
    if a.maturity_mode == "existing":
        return None, [], [f"Missing precomputed layer maturity: {files[key]}" for key in missing]
    have_spot = [path.is_file() for (_, scale, _), path in files.items() if scale == "spot"]
    have_domain = [path.is_file() for (_, scale, _), path in files.items() if scale != "spot"]
    if all(have_spot) and not any(have_domain):
        # Native builder's explicit domain-only mode consumes existing SPOT files,
        # never overwrites or recalculates them.
        layers = ("seurat_k150", "seurat_k40")
    elif not any(have_spot) and not any(have_domain):
        layers = SCALES
    else:
        return None, [], [
            "Partial/stale developmental-root. Native builder refuses overwrites. "
            "Use all selected organ/layer/time CSVs (--maturity-mode existing), or a NEW empty "
            "--developmental-root for --maturity-mode native. No silent overwrite."
        ]
    return _native_build_command(a, layers), [str(files[k]) for k in missing], []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sensitivity_source_hashes(a: argparse.Namespace, manifest: dict) -> tuple[int, list[str]]:
    """Reject mixing previously computed EI/CQ with DIFFERENT GRN/CCI files.

    No assumptions about the old absolute input paths: map by scale/stage.
    The optional manifest hashes are used only when recorded by formal sensitivity.
    """
    checked = 0
    errors: list[str] = []
    recorded = manifest.get("input_sources", {})
    if not isinstance(recorded, dict):
        return checked, ["Invalid sensitivity input_sources provenance: expected object"]
    for scale in SCALES:
        for t in TIMES:
            entry = recorded.get(f"{scale}:{t}")
            if entry is None:
                continue
            if not isinstance(entry, dict):
                errors.append(f"Invalid sensitivity provenance entry {scale}:{t}")
                continue
            sample = f"{STEMS[scale]}_heart_{t}"
            for key, current in (
                ("grn_sha256", a.grn_root / scale / sample / "grn_edges.csv"),
                ("cci_sha256", a.cci_root / scale / f"{sample}_CCI_total.npz"),
            ):
                expected = entry.get(key)
                if expected is None or not current.is_file():
                    continue
                checked += 1
                if _sha256(current) != expected:
                    errors.append(
                        f"STALE sensitivity results: {scale}/{t} {key} does not match latest {current}. "
                        "Rerun formal_sensitive_analysis before final Core."
                    )
    return checked, errors



def _missing_natural_k150_k40_js(path: Path) -> list[str]:
    """Require the three measured alpha=.20 K150->K40 direct-JS cells before training."""
    path = Path(path)
    if not path.is_file():
        return [f"Missing natural CQ source for K150->K40 direct JS: {path}"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"grn_feature", "transition", "alpha", "time_pair", "hierarchy",
                "mapping_method", "closure_JS_direct_bits"}
    if not rows or not required <= set(rows[0]):
        missing_columns = required - set(rows[0]) if rows else required
        return [f"Natural CQ source lacks K150->K40 direct-JS fields: {sorted(missing_columns)}"]
    issues = []
    for pair in ("11.5->12.5", "12.5->13.5", "11.5->13.5"):
        selected = []
        for row in rows:
            try:
                alpha = float(row["alpha"])
                value = float(row["closure_JS_direct_bits"])
            except (TypeError, ValueError):
                continue
            if (row["grn_feature"] == "legacy_nmf" and row["transition"] == "KL+OT"
                    and abs(alpha - 0.20) < 1e-9
                    and row["time_pair"] == pair
                    and row["hierarchy"] == "seurat_k150->seurat_k40"):
                selected.append((row, value))
        if len(selected) != 1:
            issues.append(f"Natural K150->K40 direct JS requires exactly one alpha=.20 row for {pair}; found {len(selected)}")
            continue
        row, value = selected[0]
        if row.get("mapping_method") != "spot_overlap_soft_contingency":
            issues.append(f"Natural K150->K40 direct JS has invalid mapping provenance for {pair}")
        if not (0.0 <= value <= 1.0):
            issues.append(f"Natural K150->K40 direct JS is outside [0,1] for {pair}: {value}")
    return issues

def _inspect_legacy_sensitivity_inputs(a: argparse.Namespace, missing: list[str]) -> int:
    # Output of separate independent formal sensitivity analysis is a required
    # scientific source for the natural hierarchy table; never fill missing K10.
    hashes_checked = 0
    if not a.skip_render and a.stage == "all":
        for basename in ("manifest.json", "alpha_layer_ei.csv", "alpha_natural_cq.csv"):
            p = a.sensitivity_root / basename
            if not p.is_file():
                missing.append(str(p))
        manifest_path = a.sensitivity_root / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "complete":
                    missing.append(f"Sensitivity manifest is not complete: {manifest_path}")
                hashes_checked, stale_errors = _check_sensitivity_source_hashes(a, manifest)
                missing.extend(stale_errors)
            except (OSError, ValueError, TypeError) as exc:
                missing.append(f"Unreadable sensitivity manifest: {exc}")
        natural_cq_csv = a.sensitivity_root / "alpha_natural_cq.csv"
        if natural_cq_csv.is_file():
            missing.extend(_missing_natural_k150_k40_js(natural_cq_csv))
        ei_csv = a.sensitivity_root / "alpha_layer_ei.csv"
        if ei_csv.is_file():
            try:
                with ei_csv.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                covered = {(r["time_pair"], r["layer"]) for r in rows
                           if r["grn_feature"] == "legacy_nmf"
                           and r["cci_feature"] == "laplacian_hks"
                           and r["transition"] == "KL+OT"
                           and abs(float(r["alpha"]) - 0.20) < 1e-9}
                for pair in ("11.5->12.5", "12.5->13.5", "11.5->13.5"):
                    for layer in (*SCALES, "seurat_k10"):
                        if (pair, layer) not in covered:
                            missing.append(f"Missing sensitivity EI: {pair} {layer} legacy_nmf/laplacian_hks/alpha=.20/KL+OT")
            except (KeyError, ValueError, OSError, TypeError) as exc:
                missing.append(f"Invalid sensitivity EI CSV: {exc}")
    return hashes_checked


def inspect_inputs(a: argparse.Namespace) -> dict:
    missing: list[str] = []
    for organ in a.organs:
        for t in TIMES:
            for scale in SCALES:
                sample = f"{STEMS[scale]}_{organ}_{t}"
                paths = [
                    a.data_root / scale / organ / f"{sample}.h5ad",
                    a.grn_root / scale / sample / "grn_edges.csv",
                    a.cci_root / scale / f"{sample}_CCI_total.npz",
                    a.cci_root / scale / f"{sample}_index.tsv",
                ]
                if scale != "spot":
                    paths.append(a.data_root / scale / organ / f"{sample}_spot_domain_map.csv")
                for path in paths:
                    if not path.is_file():
                        missing.append(str(path))
    builder, to_generate, maturity_issues = _maturity_plan(a)
    missing.extend(maturity_issues)
    natural = None
    if not a.skip_render and a.stage == "all":
        cfg = NaturalAnalysisConfig(data_root=a.data_root, output_root=a.output_root / "natural_analysis",
                                    grn_root=a.grn_root, cci_root=a.cci_root, alpha=a.alpha)
        natural = inspect_source_inputs(cfg, figure_pair=a.figure_pair)
        missing.extend(natural["missing_inputs"])
        missing.extend(natural["issues"])
        if cfg.output_root.exists() and any(cfg.output_root.iterdir()):
            try:
                if not a.resume:
                    raise FileExistsError("Natural output exists; use --resume or a new output-root")
                load_natural_results(cfg.output_root, cfg)
                natural["measurements_ready"] = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing.append(f"Cannot reuse natural results: {exc}")
        else:
            natural["measurements_ready"] = False
    missing = list(dict.fromkeys(missing))
    return {
        "missing_input_count": len(missing),
        "sensitivity_required": False,
        "natural_inputs": natural,
        "missing_inputs": missing,
        "natural_k150_k40_js_ready": bool(natural and natural.get("measurements_ready")),
        "maturity_to_generate": to_generate,
        "native_maturity_builder_command": builder,
        "maturity_mode": a.maturity_mode,
        "maturity_builder_semantics": (
            "PROJECT NATIVE factory_proxy: spot pseudotime from expression; "
            "K150/K40 outputs use project-native mean_over_member_spots. "
            "No independently inferred domain pseudotime is claimed."
        ),
    }


def make_commands(a: argparse.Namespace, native_builder: list[str] | None) -> list[dict]:
    graph = graph_configuration("formal_two_stage", local_graph_mode=a.local_graph_mode,
                                 knn_k=a.knn_k, local_dims=a.local_dims)
    jobs: list[dict] = []
    if native_builder:
        jobs.append({"stage": "native_developmental_builder", "command": native_builder})
    if not a.skip_render and a.stage == "all":
        natural_command = [sys.executable, "-m", "mignet_ce.downstream.formal_core_analysis.natural_analysis",
                           "--data-root", str(a.data_root), "--grn-root", str(a.grn_root),
                           "--cci-root", str(a.cci_root),
                           "--output-root", str(a.output_root / "natural_analysis"), "--alpha", "0.20"]
        if a.resume:
            natural_command.append("--resume")
        jobs.append({"stage": "natural_fixed_alpha", "command": natural_command})
    run_root = a.output_root / "optimal_runs"
    for organ in a.organs:
        for method in METHODS:
            command = [
                sys.executable, str(REPO / "scripts" / "run_multiscale_maturity_cci_grn.py"),
                "--method", method,
                "--spot-root", str(a.data_root / "spot"),
                "--seurat-k40-root", str(a.data_root / "seurat_k40"),
                "--seurat-k150-root", str(a.data_root / "seurat_k150"),
                "--cci-root", str(a.cci_root),
                "--grn-root", str(a.grn_root),
                "--developmental-root", str(a.developmental_root),
                "--out-root", str(run_root / organ),
                "--organ", organ, "--time-points", *TIMES,
                "--pairs", *PAIRS, "--scales", *SCALES,
                "--k-by-scale", *K_BY_SCALE, "--seeds", str(a.seed),
            ]
            if a.resume:
                command.append("--resume")
            if a.resume_incomplete:
                command.append("--resume-incomplete")
            command.extend([
                "--runner-extra-args",
                "--grn-feature-method", "legacy_nmf",
                "--cci-feature-method", "laplacian_hks",
                "--alpha-cci", "0.20",
                "--local-graph-mode", graph["local_graph_mode"],
                "--knn-k", str(graph["knn_k"]), "--local-dims", str(graph["local_dims"]),
                "--epochs", str(a.epochs), "--device", a.device,
            ])
            jobs.append({"stage": "train_" + method + "_" + organ, "method": method, "organ": organ, "expected_runs": 9, "command": command})
    if not a.skip_render:
        jobs.append({
            "stage": "formal_core_collect_and_plot",
            "command": [sys.executable, "-m", "mignet_ce.downstream.formal_core_analysis",
                        "--stage", a.stage, "--organs", *a.organs,
                        *(["--natural-root", str(a.output_root / "natural_analysis")] if a.stage == "all" else []),
                        "--grn-root", str(a.grn_root), "--cci-root", str(a.cci_root),
                        "--data-root", str(a.data_root),
                        "--optimal-runs-root", str(run_root),
                        "--output-root", str(a.output_root),
                        "--alpha", "0.20", "--grn-method", "legacy_nmf",
                        "--local-graph-mode", graph["local_graph_mode"],
                        "--knn-k", str(graph["knn_k"]), "--local-dims", str(graph["local_dims"]),
                        "--figure-pair", a.figure_pair,
                        "--figure-organ", "heart" if "heart" in a.organs else a.organs[0]],
        })
    return jobs


def audit_finished_method(output_root: Path, method: str, organ: str = "heart", *,
                          local_graph_mode=FORMAL_LOCAL_GRAPH_MODE, knn_k=30, local_dims=2) -> list[dict]:
    """Require 9 trained-PIJ verified, provenance-matched runs BEFORE rendering.

    Evaluates the existing project formal_core_analysis audit.  This is deliberately
    more rigorous than the multiscale runner's resume criterion (summary.json alone).
    """
    from .tables import inspect_optimal_run
    graph = graph_configuration(method, local_graph_mode=local_graph_mode, knn_k=knn_k, local_dims=local_dims)

    rows = []
    for scale, k in (("spot", 40), ("seurat_k150", 40), ("seurat_k40", 10)):
        for source, target in (("11.5", "12.5"), ("12.5", "13.5"), ("11.5", "13.5")):
            run_dir = output_root / "optimal_runs" / organ / method / scale / f"K{k}" / f"{source}_to_{target}"
            try:
                item = inspect_optimal_run(run_dir)
                validate_graph_configuration(item, graph, context=run_dir)
            except Exception as exc:
                raise RuntimeError(f"Trained-PIJ verification failed for {run_dir}: {exc}") from exc
            if (item["organ"], item["method"], item["input_scale"], item["time_pair"]) != (organ, method, scale, f"{source}->{target}"):
                raise RuntimeError(f"Recorded scientific condition does not match {run_dir}: {item}")
            rows.append({"run_dir": str(run_dir), "trained_deltaEI_bits": item["trained_delta_EI"],
                         **{key:item[key] for key in graph},
                         "checkpoint_keff_min": item.get("checkpoint_keff_min"),
                         "keff_min_resolved": item.get("keff_min_resolved")})
    return rows


def main(argv: list[str] | None = None) -> int:
    a = build_argparser().parse_args(argv)
    graph = graph_configuration("formal_two_stage", local_graph_mode=a.local_graph_mode,
                                 knn_k=a.knn_k, local_dims=a.local_dims)
    if len(a.organs) != len(set(a.organs)):
        raise SystemExit("--organs must be unique")
    if a.resume_incomplete and not a.resume:
        raise SystemExit("--resume-incomplete requires --resume")
    if a.epochs <= 0:
        raise SystemExit("--epochs must be >= 1")
    if abs(a.alpha - 0.20) > 1e-12:
        raise SystemExit("The frozen formal protocol requires --alpha 0.20")
    report = inspect_inputs(a)
    jobs = make_commands(a, report["native_maturity_builder_command"])
    plan = {
        "status": "dry_run" if a.dry_run else "preflight",
        "schema_version": 3,
        "output_root": str(a.output_root),
        "optimal_runs_root": str(a.output_root / "optimal_runs"),
        "training_conditions": expected_conditions(SCALES, PAIRS, a.organs),
        "methods": list(METHODS), "organs": a.organs, "stage": a.stage, "scales": list(SCALES), "pairs": list(PAIRS),
        "grn_feature": "legacy_nmf", "cci_feature": "laplacian_hks",
        "transition": "KL+OT", "alpha_cci": 0.20,
        **graph,
        "epochs": a.epochs, "seed": a.seed, "resume": a.resume,
        "resume_incomplete": a.resume_incomplete,
        "provenance_note": "Training K150/K40 uses native per-layer CSV paths. No bespoke maturity aggregator.",
        **report,
        "commands": jobs,
    }
    if a.dry_run or report["missing_input_count"]:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0 if not report["missing_input_count"] else 2
    a.output_root.mkdir(parents=True, exist_ok=True)
    audit = a.output_root / "formal_full_pipeline_execution.json"
    plan["status"] = "running"
    audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for idx, job in enumerate(jobs, 1):
        print(f"[formal core {idx}/{len(jobs)}] {job['stage']}\n$ {shlex.join(job['command'])}", flush=True)
        result = subprocess.run(job["command"], cwd=REPO, check=False)
        job["exit_code"] = result.returncode
        if result.returncode:
            plan["status"] = "failed"
            plan["failed_stage"] = job["stage"]
            audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"FAILED: {job['stage']}, exit={result.returncode}; see audit at {audit}", file=sys.stderr)
            return int(result.returncode)
        if job["stage"].startswith("train_"):
            try:
                job["trained_audit"] = audit_finished_method(a.output_root, job["method"], job["organ"], **graph)
            except Exception as exc:
                job["trained_audit_error"] = str(exc)
                plan["status"] = "failed"
                plan["failed_stage"] = job["stage"] + "_trained_audit"
                audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"FAILED trained-PIJ audit: {exc}", file=sys.stderr)
                return 3
        audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan["status"] = "complete"
    audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Formal Core pipeline complete: {a.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
