"""Export or verify the single-section H5ADs named in the regeneration profile."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from prepare_zebrafish_regenerate_h5ad import MANIFEST_PATH, STAGES, ad, sha256, verify_stage
from mignet_ce.io.experiment_spec import load_experiment

SELECTED_DIR = Path("h5ad/selected")
SECTION_MANIFEST = Path("manifests/selected_sections.tsv")


def selected_paths(spec):
    if spec.dataset_id != "zebrafish_regenerate" or spec.species != "zebrafish" or spec.organs != ("heart",):
        raise ValueError("Expected the zebrafish regeneration heart profile")
    paths = {stage: Path(spec.raw_files[stage]) for stage in spec.time_points}
    if len(set(paths.values())) != len(paths) or any(
        path.parent != SELECTED_DIR or path.suffix != ".h5ad" for path in paths.values()
    ):
        raise ValueError("Each stage must specify a distinct section H5AD in h5ad/selected")
    return paths


def verify_selected(root, spec, paths):
    records = pd.read_csv(root / SECTION_MANIFEST, sep="\t", dtype=str)
    if (records["stage"].tolist() != list(spec.time_points)
            or set(records["profile_sha256"]) != {spec.fingerprint}):
        raise ValueError("Selection manifest does not match the experiment profile")
    for row in records.itertuples(index=False):
        relpath = paths[row.stage]
        path = root / relpath
        if (row.section_id != relpath.stem or row.selected_path != relpath.as_posix()
                or row.selected_sha256 != sha256(path)):
            raise ValueError(f"Selected section path or hash mismatch: {path}")
    return records


def export_sections(root, spec, paths):
    for relative in (SELECTED_DIR, SECTION_MANIFEST):
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or path.exists():
            raise FileExistsError(f"Output must be new and inside the data root: {path}")

    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest["dataset_id"] != spec.dataset_id or manifest["status"] != "complete":
        raise ValueError("Full-stage preprocessing manifest is incomplete or belongs to another dataset")
    full_stages = {entry["stage"]: entry for entry in manifest["stages"]}
    records = []
    with tempfile.TemporaryDirectory(prefix=".zebrafish_selection_", dir=root) as temp_dir:
        work = Path(temp_dir)
        (work / SELECTED_DIR).mkdir(parents=True)
        for stage, output in paths.items():
            entry = full_stages[stage]
            source_path = root / entry["path"]
            if not source_path.resolve().is_relative_to(root.resolve()) or sha256(source_path) != entry["sha256"]:
                raise ValueError(f"Full-stage source path or hash mismatch: {source_path}")
            source = ad.read_h5ad(source_path)
            section_id = output.stem
            section = source[source.obs["orig.ident"].eq(section_id)].copy()
            if section.n_obs == 0 or not section.obs_names.is_unique or not np.isfinite(section.obsm["spatial"]).all():
                raise ValueError(f"Invalid or empty section: {section_id}")
            if (STAGES[entry["time_point"]] != stage or not section.obs["stage_day"].eq(float(stage)).all()
                    or not section.obs["time_points"].eq(entry["time_point"]).all()
                    or section.obs["isolate"].nunique() != 1):
                raise ValueError(f"Stage or isolate metadata disagrees for {section_id}")

            path = work / output
            section.write_h5ad(path, compression="gzip")
            n_umi = verify_stage(path, section.layers["counts"], section.obs, section.var, section.obsm["spatial"])
            records.append({"stage": stage, "time_point": entry["time_point"], "section_id": section_id,
                            "isolate": section.obs["isolate"].iloc[0], "n_spots": section.n_obs,
                            "n_genes": section.n_vars, "total_umis": n_umi,
                            "source_path": entry["path"], "source_sha256": entry["sha256"],
                            "selected_path": output.as_posix(), "selected_sha256": sha256(path),
                            "profile_sha256": spec.fingerprint})
        (work / SECTION_MANIFEST).parent.mkdir(parents=True)
        pd.DataFrame(records).to_csv(work / SECTION_MANIFEST, sep="\t", index=False, lineterminator="\n")
        verify_selected(work, spec, paths)
        for relative in (SELECTED_DIR, SECTION_MANIFEST):
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (work / relative).rename(root / relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    spec = load_experiment(args.experiment_config)
    paths = selected_paths(spec)
    if not args.verify_only:
        export_sections(args.data_root, spec, paths)
    result = verify_selected(args.data_root, spec, paths)
    print(result[["time_point", "section_id", "isolate", "n_spots"]].to_string(index=False))
    print(f"Verified {len(result)} selected sections with {result['n_spots'].astype(int).sum():,} spots")


if __name__ == "__main__":
    main()
