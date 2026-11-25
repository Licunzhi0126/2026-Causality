from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from .config import (
    DATA_ROOT, DEFAULT_LEVELS, DEFAULT_STRENGTHS, FAMILIES, LEVELS, PIPELINES, STRENGTHS, TOPOLOGIES,
    Settings, default_cpu_workers, tasks_for,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CCI P0–P5 and rectangular Pij symmetry experiments")
    p.add_argument("--experiment", choices=("all", "cci", "symmetry-existence", "symmetry-perturbation", "report"), default="all")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--cci-root", type=Path)
    p.add_argument("--grn-root", type=Path)
    p.add_argument("--developmental-root", type=Path, help="Directory containing spot/<organ>_<time>_features.csv")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--organ", default="heart")
    p.add_argument("--layer", choices=("spot",), default="spot")
    p.add_argument("--source-time", default="11.5")
    p.add_argument("--target-time", default="12.5")
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--perturbation-seed", type=int, default=42)
    p.add_argument("--backbone-fraction", type=float, default=0.10)
    p.add_argument("--random-partitions", type=int, default=100)
    p.add_argument("--pairs-per-cluster", type=int, default=256)
    p.add_argument("--cci-topologies", nargs="+", choices=TOPOLOGIES, default=TOPOLOGIES)
    p.add_argument("--cci-strengths", nargs="+", type=int, choices=STRENGTHS, default=DEFAULT_STRENGTHS, metavar="PERCENT")
    p.add_argument("--symmetry-families", nargs="+", choices=FAMILIES, default=FAMILIES)
    p.add_argument("--symmetry-levels", nargs="+", type=int, choices=LEVELS, default=DEFAULT_LEVELS, metavar="PERCENT")
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    p.add_argument("--workers-per-gpu", type=int, default=3)
    p.add_argument("--max-workers-per-gpu", type=int, help="Permit measured expansion up to this count; default 6")
    p.add_argument("--gpu-budget-gib", type=float, default=18.0)
    p.add_argument("--cpu-workers", type=int, default=default_cpu_workers())
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    from .scheduler import Scheduler, device_tokens, gpu_probe, run_lock, worker_loop

    argv = sys.argv[1:] if argv is None else argv
    if argv == ["_probe"]:
        gpu_probe()
        return 0
    if len(argv) == 2 and argv[0] == "_worker" and argv[1] in {"cpu", "cuda:0"}:
        worker_loop(argv[1])
        return 0
    args = parser().parse_args(argv)
    common_root = args.output_root.resolve()
    if args.experiment == "report":
        from .reporting import report_pipelines

        with run_lock(common_root):
            report_pipelines(common_root)
        return 0

    from .formal import preflight, resolve_inputs, resolve_seurat

    settings = Settings(organ=args.organ, source_time=args.source_time, target_time=args.target_time,
                        k=args.k, epochs=args.epochs, train_seed=args.train_seed, perturbation_seed=args.perturbation_seed,
                        backbone_fraction=args.backbone_fraction, random_partitions=args.random_partitions,
                        pairs_per_cluster=args.pairs_per_cluster)
    settings.validate()
    if min(args.workers_per_gpu, args.cpu_workers) < 1:
        raise ValueError("Worker counts must be positive")
    maximum = args.max_workers_per_gpu if args.max_workers_per_gpu is not None else max(6, args.workers_per_gpu)
    if maximum < args.workers_per_gpu:
        raise ValueError("--max-workers-per-gpu must be at least --workers-per-gpu")
    if not math.isfinite(args.gpu_budget_gib) or args.gpu_budget_gib <= 2:
        raise ValueError("--gpu-budget-gib must be finite and exceed 2")
    root = common_root / PIPELINES[args.experiment] if args.experiment in PIPELINES else common_root
    data_root = args.data_root.resolve()
    development = (args.developmental_root or data_root / "developmental_features").resolve()
    inputs = resolve_inputs(data_root, args.cci_root.resolve() if args.cci_root else None,
                            args.grn_root.resolve() if args.grn_root else None, development, settings)
    seurat = resolve_seurat(data_root, settings) if args.experiment in {"all", "symmetry-existence"} else None
    preflight_audit = preflight(inputs, settings, seurat)
    tasks = tasks_for(args.experiment, *[tuple(dict.fromkeys(values)) for values in (
        args.cci_topologies, args.cci_strengths, args.symmetry_families, args.symmetry_levels,
    )])
    if args.dry_run:
        from .cci import discover_regions, load_adjacency

        if any(task.family == "cci" for task in tasks):
            _, _, preflight_audit["cci_selection"] = discover_regions(load_adjacency(inputs.cci_t), settings)
        print(json.dumps({**preflight_audit, "settings": asdict(settings), "tasks": [task.id for task in tasks],
                          "formal": {"method": "formal_two_stage", "grn_features": "legacy_nmf",
                                     "cci_features": "laplacian_hks", "alpha_cci": 0.20},
                          "requested_trainings": len(tasks), "devices": args.devices,
                          "workers_per_gpu": args.workers_per_gpu, "max_workers_per_gpu": maximum,
                          "gpu_budget_gib": args.gpu_budget_gib, "cpu_workers": args.cpu_workers,
                          "pipeline_output_root": str(root),
                          "existence_groups": 2 * (settings.random_partitions + 1) if seurat else 0}, indent=2))
        return 0
    tokens = device_tokens(args.devices)
    with run_lock(root):
        return Scheduler(root, settings, inputs, tasks, seurat, tokens,
                         args.workers_per_gpu, args.cpu_workers, args.resume,
                         shared_root=common_root, max_workers_per_gpu=maximum, budget_gib=args.gpu_budget_gib).run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
