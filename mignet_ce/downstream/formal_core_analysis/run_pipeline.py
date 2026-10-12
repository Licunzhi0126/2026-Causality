"""Full formal Core orchestration using the project's NATIVE maturity builder.

Scientific code is delegated without reimplementing maturity, PIJ, coarse graining,
or strict reconstruction.  The formal sensitivity pipeline must already have run.

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

METHODS = (
    "complete_combined_coarse_maturity_cci_grn",
    "maturity_cci_grn_two_stage",
)
SCALES = ("spot", "seurat_k150", "seurat_k40")
TIMES = ("11.5", "12.5", "13.5")
PAIRS = ("11.5:12.5", "12.5:13.5", "11.5:13.5")
K_BY_SCALE = ("spot=40", "seurat_k150=40", "seurat_k40=10")
STEMS = {"spot": "spot", "seurat_k150": "seurat150", "seurat_k40": "seurat"}
REPO = Path(__file__).resolve().parents[3]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native maturity -> 18 strict optimal runs -> formal Core output")
    p.add_argument("--data-root", type=Path, required=True, help="Factory root with spot/seurat_k150/seurat_k40")
    p.add_argument("--grn-root", type=Path, required=True)
    p.add_argument("--cci-root", type=Path, required=True, help="True CCI matrices AND corresponding *_index.tsv")
    p.add_argument("--developmental-root", type=Path, required=True,
                   help="Source/output of native per-layer developmental feature CSV files")
    p.add_argument("--sensitivity-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--maturity-mode", choices=("existing", "native"), default="existing",
                   help="existing: ONLY read all 9 direct per-layer CSVs; native: use scripts/build_developmental_features.py when missing")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--figure-pair", choices=("11.5->12.5", "12.5->13.5", "11.5->13.5"), default="12.5->13.5")
    p.add_argument("--dry-run", action="store_true", help="Read-only checks and full command plan; exit nonzero if prerequisites missing")
    p.add_argument("--resume", action="store_true", help="Reuse valid completed runs; never overwrite incomplete runs")
    p.add_argument("--skip-render", action="store_true", help="Only train models; do not run final paper table/figure collector")
    return p


def maturity_files(a: argparse.Namespace) -> dict[tuple[str, str], Path]:
    return {(scale, t): a.developmental_root / scale / f"heart_{t}_features.csv"
            for scale in SCALES for t in TIMES}


def _native_build_command(a: argparse.Namespace, layers: tuple[str, ...]) -> list[str]:
    return [sys.executable, str(REPO / "scripts" / "build_developmental_features.py"),
            "--data-root", str(a.data_root),
            "--output-root", str(a.developmental_root),
            "--organs", "heart", "--time-points", *TIMES,
            "--layers", *layers,
            "--seed", str(a.seed)]


def _maturity_plan(a: argparse.Namespace) -> tuple[list[str] | None, list[str], list[str]]:
    """Return native builder command, materialized paths, and actionable issues.

    The project-native builder itself uses mean-over-member-spots for domain
    features.  We do NOT substitute a newly written algorithm or silently claim
    that K150/K40 pseudotimes were inferred independently from domain H5AD.
    """
    files = maturity_files(a)
    missing = [(scale, t) for (scale, t), path in files.items() if not path.is_file()]
    if not missing:
        return None, [], []
    if a.maturity_mode == "existing":
        return None, [], [f"Missing precomputed layer maturity: {files[key]}" for key in missing]
    have_spot = [files[("spot", t)].is_file() for t in TIMES]
    have_domain = [files[(s, t)].is_file() for s in SCALES[1:] for t in TIMES]
    if all(have_spot) and not any(have_domain):
        # Native builder's explicit domain-only mode consumes existing SPOT files,
        # never overwrites or recalculates them.
        layers = ("seurat_k150", "seurat_k40")
    elif not any(have_spot) and not any(have_domain):
        layers = SCALES
    else:
        return None, [], [
            "Partial/stale developmental-root. Native builder refuses overwrites. "
            "Use all 9 existing CSVs (--maturity-mode existing), or a NEW empty "
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


def inspect_inputs(a: argparse.Namespace) -> dict:
    missing: list[str] = []
    for t in TIMES:
        for scale in SCALES:
            sample = f"{STEMS[scale]}_heart_{t}"
            paths = [
                a.data_root / scale / "heart" / f"{sample}.h5ad",
                a.grn_root / scale / sample / "grn_edges.csv",
                a.cci_root / scale / f"{sample}_CCI_total.npz",
                a.cci_root / scale / f"{sample}_index.tsv",
            ]
            if scale != "spot":
                paths.append(a.data_root / scale / "heart" / f"{sample}_spot_domain_map.csv")
            for path in paths:
                if not path.is_file():
                    missing.append(str(path))
    builder, to_generate, maturity_issues = _maturity_plan(a)
    missing.extend(maturity_issues)
    # Output of separate independent formal sensitivity analysis is a required
    # scientific source for the natural hierarchy table; never fill missing K10.
    hashes_checked = 0
    if not a.skip_render:
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
    return {
        "missing_input_count": len(missing),
        "sensitivity_input_hashes_compared": hashes_checked,
        "missing_inputs": missing,
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
    jobs: list[dict] = []
    if native_builder:
        jobs.append({"stage": "native_developmental_builder", "command": native_builder})
    run_root = a.output_root / "optimal_runs"
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
            "--out-root", str(run_root),
            "--organ", "heart", "--time-points", *TIMES,
            "--pairs", *PAIRS, "--scales", *SCALES,
            "--k-by-scale", *K_BY_SCALE, "--seeds", str(a.seed),
        ]
        if a.resume:
            command.append("--resume")
        command.extend([
            "--runner-extra-args",
            "--grn-feature-method", "legacy_nmf",
            "--cci-feature-method", "laplacian_hks",
            "--alpha-cci", "0.20",
            "--epochs", str(a.epochs), "--device", a.device,
        ])
        jobs.append({"stage": "train_" + method, "expected_runs": 9, "command": command})
    if not a.skip_render:
        jobs.append({
            "stage": "formal_core_collect_and_plot",
            "command": [sys.executable, "-m", "mignet_ce.downstream.formal_core_analysis",
                        "--stage", "all", "--sensitivity-root", str(a.sensitivity_root),
                        "--data-root", str(a.data_root),
                        "--optimal-runs-root", str(run_root),
                        "--output-root", str(a.output_root),
                        "--alpha", "0.20", "--grn-method", "legacy_nmf",
                        "--figure-pair", a.figure_pair],
        })
    return jobs


def audit_finished_method(output_root: Path, method: str) -> list[dict]:
    """Require 9 strictly reconstructed, provenance-matched runs BEFORE rendering.

    Evaluates the existing project formal_core_analysis audit.  This is deliberately
    stricter than the multiscale runner's resume criterion (summary.json alone).
    """
    from .tables import inspect_optimal_run

    rows = []
    for scale, k in (("spot", 40), ("seurat_k150", 40), ("seurat_k40", 10)):
        for source, target in (("11.5", "12.5"), ("12.5", "13.5"), ("11.5", "13.5")):
            run_dir = output_root / "optimal_runs" / method / scale / f"K{k}" / f"{source}_to_{target}"
            try:
                item = inspect_optimal_run(run_dir)
            except Exception as exc:
                raise RuntimeError(f"Strict verification failed for {run_dir}: {exc}") from exc
            if (item["method"], item["input_scale"], item["time_pair"]) != (method, scale, f"{source}->{target}"):
                raise RuntimeError(f"Recorded scientific condition does not match {run_dir}: {item}")
            rows.append({"run_dir": str(run_dir), "strict_deltaEI_bits": item["delta_EI_strict_raw"],
                         "CQ_signal_status": item["signal_status"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    a = build_argparser().parse_args(argv)
    if a.epochs <= 0:
        raise SystemExit("--epochs must be >= 1")
    if abs(a.alpha - 0.20) > 1e-12:
        raise SystemExit("The frozen formal protocol requires --alpha 0.20")
    report = inspect_inputs(a)
    jobs = make_commands(a, report["native_maturity_builder_command"])
    plan = {
        "status": "dry_run" if a.dry_run else "preflight",
        "schema_version": 2,
        "output_root": str(a.output_root),
        "optimal_runs_root": str(a.output_root / "optimal_runs"),
        "training_conditions": 18,
        "methods": list(METHODS), "scales": list(SCALES), "pairs": list(PAIRS),
        "grn_feature": "legacy_nmf", "cci_feature": "laplacian_hks",
        "transition": "KL+OT", "alpha_cci": 0.20,
        "epochs": a.epochs, "seed": a.seed, "resume": a.resume,
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
                job["strict_audit"] = audit_finished_method(a.output_root, job["stage"].removeprefix("train_"))
            except Exception as exc:
                job["strict_audit_error"] = str(exc)
                plan["status"] = "failed"
                plan["failed_stage"] = job["stage"] + "_strict_audit"
                audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"FAILED strict audit: {exc}", file=sys.stderr)
                return 3
        audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan["status"] = "complete"
    audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Formal Core pipeline complete: {a.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
