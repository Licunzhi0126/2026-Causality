"""Formal multi-organ orchestration: natural existence and spot Two-stage only.

Reuses the canonical sensitivity and multiscale training entrypoints; deliberately
never duplicates EI/CQ/OT/Two-stage scientific implementations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from mignet_ce.io.organ_paths import resolve_organ_paths

REPO = Path(__file__).resolve().parents[3]
ORGANS = ("heart", "lung", "brain")
TIMES = ("11.5", "12.5", "13.5")
PAIRS = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
SCALES = ("spot", "seurat_k150", "seurat_k40")
METHOD = "maturity_cci_grn_two_stage"


def build_argparser():
    p = argparse.ArgumentParser(description="Run formal 3-organ natural EI/CQ and 9 spot Two-stage experiments")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cci-root", type=Path, help="Defaults to DATA/cci, NOT cci_clean")
    p.add_argument("--grn-root", type=Path, help="Defaults to DATA/grn")
    p.add_argument("--developmental-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS))
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--alpha", type=float, default=.20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-missing", action="store_true", help="Skip organs lacking inputs, render missing table cells as dashes")
    p.add_argument("--skip-render", action="store_true")
    return p


def _paths(a, organ):
    cci = a.cci_root or a.data_root / "cci"
    grn = a.grn_root or a.data_root / "grn"
    results = []
    for scale in SCALES:
        for time in TIMES:
            paths = resolve_organ_paths(a.data_root, scale, organ, time, cci_root=cci, grn_root=grn)
            for key, path in paths.items():
                if path is not None and not path.is_file():
                    results.append(f"{organ}/{scale}/{time}/{key}: {path}")
            if scale == "spot":
                maturity = a.developmental_root / "spot" / f"{organ}_{time}_features.csv"
                if not maturity.is_file():
                    results.append(f"{organ}/{time}/maturity: {maturity}")
    return results


def make_plan(a):
    if abs(a.alpha-.20)>1e-9 or a.epochs <= 0:
        raise ValueError("Frozen protocol requires alpha .20 and epochs >= 1")
    if len(a.organs) != len(set(a.organs)):
        raise ValueError("Duplicate organ arguments")
    cci = a.cci_root or a.data_root / "cci"
    grn = a.grn_root or a.data_root / "grn"
    jobs = []
    missing = {}
    for organ in a.organs:
        missing[organ] = _paths(a, organ)
        if missing[organ]:
            continue
        sensitivity = a.output_root / "sensitivity" / organ
        natural_command = [sys.executable, "-m", "mignet_ce.downstream.formal_sensitive_analysis",
                           "--data-root", str(a.data_root), "--organ", organ,
                           "--grn-root", str(grn), "--cci-root", str(cci), "--cci-index-root", str(cci),
                           "--output-root", str(sensitivity), "--layers", *SCALES,
                           "--time-pairs", *PAIRS, "--alphas", "0.20",
                           "--grn-methods", "legacy_nmf", "--transitions", "KL+OT", "--skip-plots"]
        optimal_root = a.output_root / "optimal_runs" / organ
        optimal_command = [sys.executable, str(REPO / "scripts" / "run_multiscale_maturity_cci_grn.py"),
                           "--method", METHOD,
                           "--spot-root", str(a.data_root / "spot"),
                           "--seurat-k40-root", str(a.data_root / "seurat_k40"),
                           "--seurat-k150-root", str(a.data_root / "seurat_k150"),
                           "--cci-root", str(cci), "--grn-root", str(grn),
                           "--developmental-root", str(a.developmental_root),
                           "--out-root", str(optimal_root), "--organ", organ,
                           "--time-points", *TIMES,
                           "--pairs", *[pair.replace("->", ":") for pair in PAIRS],
                           "--scales", "spot", "--k-by-scale", "spot=40", "--seeds", str(a.seed)]
        if a.resume:
            optimal_command.append("--resume")
        optimal_command += ["--runner-extra-args", "--grn-feature-method", "legacy_nmf",
                            "--cci-feature-method", "laplacian_hks", "--alpha-cci", "0.20",
                            "--epochs", str(a.epochs), "--device", a.device]
        jobs.extend([{"organ": organ, "stage": "natural", "command": natural_command},
                     {"organ": organ, "stage": "optimal", "command": optimal_command}])
    if not a.skip_render:
        jobs.append({"stage": "tables", "command": [sys.executable, "-m",
                     "mignet_ce.downstream.formal_core_analysis",
                     "--sensitivity-root", str(a.output_root / "sensitivity"),
                     "--optimal-runs-root", str(a.output_root / "optimal_runs"),
                     "--output-root", str(a.output_root), "--organs", *a.organs]})
    return {"status": "dry_run" if a.dry_run else "planned", "organs": list(a.organs),
            "missing_inputs_by_organ": missing, "jobs": jobs,
            "method": METHOD, "scales_optimal": ["spot"], "K": 40,
            "expected_optimal_runs": 3*sum(not problems for problems in missing.values()),
            "final_table_conditions": 3*len(a.organs), "cci_root": str(cci),
            "grn_root": str(grn), "maturity_mode": "existing spot features"}


def _verify_cached_sensitivity(a, organ: str, manifest: dict) -> None:
    """Verify saved alpha/method AND physical input hashes on resume."""
    if (manifest.get("status") != "complete" or manifest.get("organ", "heart") != organ
        or list(manifest.get("alphas", [])) != [.20]
        or list(manifest.get("grn_methods", [])) != ["legacy_nmf"]
        or list(manifest.get("transitions", [])) != ["KL+OT"]):
        raise ValueError(f"Incompatible completed sensitivity manifest for {organ}")
    sources = manifest.get("input_sources", {})
    for scale in SCALES:
        for t in TIMES:
            entry = sources.get(f"{scale}:{t}")
            if not isinstance(entry, dict):
                raise ValueError(f"Sensitivity source hash is unavailable: {organ}/{scale}/{t}")
            paths = resolve_organ_paths(a.data_root, scale, organ, t,
                    cci_root=a.cci_root or a.data_root / "cci",
                    grn_root=a.grn_root or a.data_root / "grn")
            for key, filename in (("grn_sha256", "grn"), ("cci_sha256", "cci")):
                expected = entry.get(key)
                if not expected:
                    raise ValueError(f"No input integrity hash: {organ}/{scale}/{t}/{key}")
                h = hashlib.sha256()
                with paths[filename].open("rb") as stream:
                    for block in iter(lambda: stream.read(4*1024*1024), b""):
                        h.update(block)
                if h.hexdigest() != expected:
                    raise ValueError(f"Stale cached {organ}/{scale}/{t}/{filename}: regenerate sensitivity")


def main(argv=None):
    a = build_argparser().parse_args(argv)
    plan = make_plan(a)
    incomplete = {organ: issues for organ, issues in plan["missing_inputs_by_organ"].items() if issues}
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 2 if incomplete and not a.skip_missing else 0
    if incomplete and not a.skip_missing:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("Inputs missing; fix them or use --skip-missing for partial tables", file=sys.stderr)
        return 2
    a.output_root.mkdir(parents=True, exist_ok=True)
    audit = a.output_root / "formal_multiorgan_execution.json"
    plan["status"] = "running"
    for job in plan["jobs"]:
        if job["stage"] == "natural":
            root = a.output_root / "sensitivity" / job["organ"]
            if a.resume and (root / "manifest.json").is_file():
                # A cached run is only reusable if parameters and provenance agree.
                cached = json.loads((root / "manifest.json").read_text(encoding="utf8"))
                try:
                    _verify_cached_sensitivity(a, job["organ"], cached)
                except (ValueError, OSError) as exc:
                    print(f"Incompatible/stale cached sensitivity: {root}: {exc}", file=sys.stderr)
                    return 3
                print(f"[formal] reuse completed sensitivity {job['organ']}", flush=True)
                job["exit_code"] = 0
                continue
        print(f"[formal] {job['stage']} {job.get('organ', '')}: {shlex.join(job['command'])}", flush=True)
        result = subprocess.run(job["command"], cwd=REPO, check=False)
        job["exit_code"] = result.returncode
        if result.returncode:
            plan["status"] = "failed"
            audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
            return result.returncode
        audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
    plan["status"] = "complete" if not incomplete else "partial"
    audit.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
