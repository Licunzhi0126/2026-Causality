"""Run the selected species factory with immutable inputs and audited resume."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "data_factory" / "lib")]
from mignet_ce.io.experiment_spec import load_experiment
from mignet_ce.io.organ_paths import sample_stem

LAYERS = ("spot", "seurat_k150", "seurat_k40")


def layer_file(root, spec, layer, stage, suffix=".h5ad"):
    return root / layer / (spec.organs[0] if spec.output_layout == "nested" else "") / (sample_stem(layer, spec.organs[0], stage) + suffix)


def validate_domain(root, spec, layer, stage):
    import numpy as np
    import pandas as pd
    from species_adapter import ad
    spot = ad.read_h5ad(layer_file(root, spec, "spot", stage))
    domain = ad.read_h5ad(layer_file(root, spec, layer, stage))
    frame = pd.read_csv(layer_file(root, spec, layer, stage, "_spot_domain_map.csv"), dtype=str)
    expected = 150 if layer == "seurat_k150" else 40
    if domain.n_obs != expected or not domain.obs_names.is_unique:
        raise ValueError("Wrong domain count/IDs")
    if (not {"spot_id", "domain_id", "x", "y"} <= set(frame) or frame.spot_id.duplicated().any()
            or set(frame.spot_id) != set(spot.obs_names) or set(frame.domain_id) != set(domain.obs_names)):
        raise ValueError("Domain map does not cover the exact spot/domain axes")
    if not np.isfinite(frame[["x", "y"]].to_numpy(float)).all():
        raise ValueError("Domain map coordinates are not finite")
    if list(domain.var_names) != list(spot.var_names):
        raise ValueError("Seurat changed canonical gene IDs/order")
    def counts(a):
        return a.layers["counts"] if "counts" in a.layers else a.X
    import scipy.sparse as sp
    assignments = frame.set_index("spot_id").loc[spot.obs_names, "domain_id"]
    column = {value: i for i, value in enumerate(domain.obs_names)}
    membership = sp.csr_matrix((np.ones(spot.n_obs), (np.arange(spot.n_obs), assignments.map(column))),
                               shape=(spot.n_obs, domain.n_obs))
    difference = sp.csr_matrix(membership.T @ counts(spot) - counts(domain))
    if difference.nnz and not np.allclose(difference.data, 0, atol=1e-8):
        raise ValueError("Domain counts disagree with the actual member-spot sums")
    return {"stage": stage, "layer": layer, "n_domains": expected,
            "min_domain_spots": int(frame.domain_id.value_counts().min()),
            "max_domain_spots": int(frame.domain_id.value_counts().max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path)
    p.add_argument("--steps", nargs="+", choices=("inputs", "domains", "cci", "grn", "developmental"),
                   default=["inputs", "domains", "cci", "grn", "developmental"])
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grn-threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    spec = load_experiment(a.experiment_config)
    root = a.data_root.resolve()
    refs = a.reference_root or root / "references"
    issues = [f"Missing raw input: {root / f}" for f in spec.raw_files.values() if not (root / f).is_file()]
    if "domains" in a.steps and not shutil.which(a.rscript):
        issues.append(f"Rscript executable not found: {a.rscript}")
    elif "domains" in a.steps:
        probe = subprocess.run([a.rscript, "--vanilla", "-e",
            'p<-c("Seurat","zellkonverter","SingleCellExperiment"); b<-p[!sapply(p,requireNamespace,quietly=TRUE)]; '
            'if(length(b)){cat(paste(b,collapse=", "));quit(status=2)}'], capture_output=True, text=True)
        if probe.returncode:
            issues.append("Missing R dependencies: " + (probe.stdout.strip() or probe.stderr.strip()))
    ref_manifest = refs / "reference_manifest.json"
    if set(a.steps) & {"cci", "grn"}:
        if not ref_manifest.is_file():
            issues.append(f"Prepare the curated LR/TF bundle first: {ref_manifest}")
        else:
            from species_adapter import sha256
            ref = json.loads(ref_manifest.read_text(encoding="utf-8"))
            if ref.get("status") != "complete" or ref.get("experiment") != spec.metadata():
                issues.append("Reference bundle profile mismatch/incomplete")
            for name, digest in ref.get("outputs", {}).items():
                if not (refs / name).is_file() or sha256(refs / name) != digest:
                    issues.append(f"Reference hash mismatch: {name}")
    plan = {"experiment": spec.metadata(), "data_root": str(root), "reference_root": str(refs),
            "steps": a.steps, "issues": issues, "expected_layer_samples": len(LAYERS) * len(spec.time_points),
            "status": "dry_run" if a.dry_run else "preflight"}
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run or issues:
        return 2 if issues else 0
    from species_adapter import prepare_inputs, sha256, write_new_json, ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    log_path = root / "manifests" / "species_factory_execution.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {"experiment": spec.metadata(), "jobs": {}}
    if log["experiment"] != spec.metadata() or (log_path.exists() and not a.resume):
        raise ValueError("Existing factory log requires matching --resume")

    def save_log():
        # The execution journal is mutable; raw inputs and completed outputs are immutable.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    def execute(key, inputs, outputs, run, validate=lambda: None):
        signature = {str(path.resolve()): sha256(path) for path in inputs}
        old = log["jobs"].get(key)
        if old:
            if old["inputs"] != signature or any(not q.is_file() or old["outputs"].get(str(q.resolve())) != sha256(q) for q in outputs):
                raise ValueError(f"Stale/incomplete factory outputs: {key}; preserve them and use a new data-root")
            validate()
            print(f"[reuse] {key}", flush=True)
            return
        if any(q.exists() for q in outputs):
            raise FileExistsError(f"Unverified partial/existing job: {key}")
        print(f"[run] {key}", flush=True)
        run()
        qc = validate()
        log["jobs"][key] = {"inputs": signature, "outputs": {str(q.resolve()): sha256(q) for q in outputs}, "qc": qc}
        save_log()

    if "inputs" in a.steps:
        prepare_inputs(spec, root, resume=a.resume)
    if "domains" in a.steps:
        for layer, k in (("seurat_k150", 150), ("seurat_k40", 40)):
            for stage in spec.time_points:
                command = [a.rscript, str(REPO / "data_factory/scripts/03_run_seurat_k40.R"),
                           "--spot-root", str(root / "spot"), "--output-root", str(root / layer),
                           "--output-layout", spec.output_layout, "--k", str(k),
                           "--sample-names", sample_stem("spot", spec.organs[0], stage),
                           "--manifest-name", f"domain_manifest_{layer}_{stage}.csv"]
                outputs = [layer_file(root, spec, layer, stage, suffix) for suffix in
                           (".h5ad", "_spot_domain_map.csv", "_spots_with_domain.h5ad")]
                execute(f"{layer}:{stage}", [layer_file(root, spec, "spot", stage)], outputs,
                        lambda cmd=command: subprocess.run(cmd, check=True, cwd=REPO),
                        lambda l=layer, t=stage: validate_domain(root, spec, l, t))
    for step in ("cci", "grn"):
        if step not in a.steps:
            continue
        for layer in LAYERS:
            for stage in spec.time_points:
                stem = sample_stem(layer, spec.organs[0], stage)
                h5ad = layer_file(root, spec, layer, stage)
                output = root / step / layer
                if step == "cci":
                    from commot_layer_runner import run_commot_layer
                    outputs = [output / (stem + suffix) for suffix in ("_CCI_total.npz", "_index.tsv")]
                    def run():
                        run_commot_layer(root / layer, output, "spot" if layer == "spot" else "domain",
                            f"cci_manifest_{layer}_{stage}.csv", commot_reference_dir=REPO / "reference/COMMOT",
                            workers=a.workers, manifest_root=root / "manifests", sample_names=[stem],
                            dis_thr=spec.distance_threshold, species=spec.species, lr_table=refs / "lr_pairs.tsv",
                            min_cell_pct=spec.min_cell_pct, expression_layer="counts")
                    def validate():
                        units = ad.read_h5ad(h5ad, backed="r")
                        try:
                            mat = sp.load_npz(outputs[0]); idx = pd.read_csv(outputs[1], sep="\t").iloc[:, 0].astype(str)
                            if (mat.shape != (units.n_obs, units.n_obs) or list(idx) != list(units.obs_names)
                                    or not np.isfinite(mat.data).all() or np.any(mat.data < 0) or not np.any(mat.data > 0)):
                                raise ValueError("Invalid/empty CCI or index mismatch")
                            return {"n_units": units.n_obs, "nnz": int(mat.nnz), "total_weight": float(mat.sum()),
                                    "zero_sender_fraction": float(np.mean(np.asarray(mat.sum(axis=1)).ravel() == 0)),
                                    "zero_receiver_fraction": float(np.mean(np.asarray(mat.sum(axis=0)).ravel() == 0)),
                                    "weak_components": int(sp.csgraph.connected_components(
                                        mat, directed=True, connection="weak", return_labels=False))}
                        finally:
                            units.file.close()
                else:
                    from grn_layer_runner import run_grn_layer
                    outputs = [output / stem / "grn_edges.csv"]
                    def run():
                        run_grn_layer(root / layer, output, f"grn_manifest_{layer}_{stage}.csv",
                            sample_names=[stem], threads=a.grn_threads, tf_list=refs / "tf_list.txt",
                            strict_tf=True, manifest_root=root / "manifests")
                    def validate():
                        frame = pd.read_csv(outputs[0])
                        if frame.empty or not {"regulator", "target", "weight"} <= set(frame):
                            raise ValueError("Invalid/empty GRN")
                        if not np.isfinite(frame.weight).all() or (frame.weight < 0).any():
                            raise ValueError("Invalid GRN weights")
                        units = ad.read_h5ad(h5ad, backed="r")
                        try:
                            if not (set(frame.regulator) | set(frame.target)) <= set(units.var_names):
                                raise ValueError("GRN gene IDs do not match canonical expression genes")
                        finally:
                            units.file.close()
                        curated = {g.strip().upper() for g in (refs / "tf_list.txt").read_text(encoding="utf-8").splitlines() if g.strip()}
                        if not set(frame.regulator.str.upper()) <= curated:
                            raise ValueError("GRN includes regulators outside the curated TF list")
                        return {"n_edges": len(frame), "n_regulators": frame.regulator.nunique(),
                                "n_targets": frame.target.nunique(), "all_gene_fallback": False}
                execute(f"{step}:{layer}:{stage}",
                        [h5ad, ref_manifest, refs / ("lr_pairs.tsv" if step == "cci" else "tf_list.txt")],
                        outputs, run, validate)
    if "developmental" in a.steps:
        from mignet_ce.io.developmental_feature_builder import DevelopmentalFeatureBuildConfig, build_developmental_features
        outputs = [root / "developmental_features" / layer / f"{spec.organs[0]}_{t}_features.csv"
                   for layer in LAYERS for t in spec.time_points]
        inputs = [layer_file(root, spec, layer, t) for layer in LAYERS for t in spec.time_points]
        inputs += [layer_file(root, spec, layer, t, "_spot_domain_map.csv") for layer in LAYERS[1:] for t in spec.time_points]
        execute("developmental", inputs, outputs, lambda: build_developmental_features(
            DevelopmentalFeatureBuildConfig(root, root / "developmental_features", organs=spec.organs,
                                            time_points=spec.time_points, layers=LAYERS)))
    log["status"] = "selected_steps_complete"
    save_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
