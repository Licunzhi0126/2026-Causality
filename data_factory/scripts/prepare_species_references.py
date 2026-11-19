"""Prepare traceable LR/TF bundles; never infer orthology from display labels.

An optional mapping TSV requires source_gene,target_gene,relationship,evidence.
Bijective orthology and explicitly identified unique author annotations are
distinct evidence types; the latter never claims phylogenetic orthology.
"""
from pathlib import Path
import argparse
import json
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "data_factory" / "lib")]
import numpy as np
import pandas as pd
from mignet_ce.io.experiment_spec import load_experiment
from species_adapter import read_counts, sha256, write_new_json


def load_lr(path):
    if Path(path).suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(path, sep="\t")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]
    if {"ligand", "receptor", "pathway"} <= set(frame):
        frame = frame[["ligand", "receptor", "pathway"]]
    else:
        frame = frame.iloc[:, :3].copy()
        frame.columns = ["ligand", "receptor", "pathway"]
    if frame[["ligand", "receptor"]].isna().any().any() or frame.empty:
        raise ValueError("LR table must contain complete ligand/receptor records")
    # The official COMMOT CellPhoneDB table has no pathway annotations. This is
    # an explicit unknown category, not an inferred biological pathway.
    frame["pathway"] = frame["pathway"].fillna("Unannotated")
    return frame.astype(str).drop_duplicates().reset_index(drop=True)


def read_mapping(path):
    frame = pd.read_csv(path, sep="\t", dtype=str)
    required = {"source_gene", "target_gene", "relationship", "evidence"}
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Curated map requires source_gene,target_gene,relationship,evidence")
    if frame.evidence.str.strip().eq("").any() or not frame.relationship.isin(
            ["one_to_one", "author_annotation_unique"]).all():
        raise ValueError("Require evidence-backed one_to_one or author_annotation_unique mapping")
    frame = frame.drop_duplicates(["source_gene", "target_gene"])
    if frame.source_gene.duplicated().any() or frame.target_gene.duplicated().any():
        raise ValueError("Ambiguous orthology: mapping must be bijective")
    return dict(zip(frame.source_gene, frame.target_gene))


def artista_mapping(path):
    """Unique author annotations; this does NOT establish phylogenetic orthology.

    Require explicit human Ensembl IDs and symbols. Discard every ambiguous
    association across the full annotation, before looking at selected samples.
    No matching on free-form gene descriptions or partial symbols is performed.
    """
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not {"Axolotl_ID", "hs_ID", "hs_gene"} <= set(frame):
        raise ValueError("ARTISTA annotation requires Axolotl_ID, hs_ID, hs_gene")
    candidates = frame.loc[frame.Axolotl_ID.str.fullmatch(r"AMEX60DD\d+")
        & frame.hs_ID.str.fullmatch(r"ENSG\d+")
        & frame.hs_gene.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
        & ~frame.hs_gene.isin(["NA", "N/A", "nan"])].copy()
    candidates = candidates.drop_duplicates(["Axolotl_ID", "hs_ID", "hs_gene"])
    keep = ~(candidates.Axolotl_ID.duplicated(keep=False) | candidates.hs_ID.duplicated(keep=False)
             | candidates.hs_gene.duplicated(keep=False))
    unique = candidates.loc[keep]
    mapping = pd.DataFrame({"source_gene": unique.hs_gene, "target_gene": unique.Axolotl_ID,
        "source_ensembl": unique.hs_ID, "relationship": "author_annotation_unique",
        "evidence": "ARTISTA v1.1.0 Summary_Gene_Annotaion_0707; explicit hs_ID/hs_gene; globally unique association"})
    return mapping, {"annotation_rows": len(frame), "candidate_rows": len(candidates),
                     "ambiguous_rows_excluded": int((~keep).sum()), "unique_associations": len(mapping),
                     "basis": "author_annotation_unique", "phylogenetic_one_to_one_orthology_claimed": False}


def map_references(lr, tfs, mapping):
    def complex_id(value):
        mapped = [mapping.get(g) for g in value.split("_")]
        return "_".join(mapped) if all(mapped) else None
    out = lr.copy()
    for key in ("ligand", "receptor"):
        out[key] = out[key].map(complex_id)
    return out.dropna().drop_duplicates(), sorted({mapping[g] for g in tfs if g in mapping})


def coverage(spec, data_root, lr, tfs):
    rows = []
    for stage in spec.time_points:
        a = read_counts(Path(data_root) / spec.raw_files[stage], spec)
        counts = a.layers["counts"]
        detected = np.asarray((counts > 0).sum(axis=0)).ravel()
        fractions = dict(zip(a.var_names, detected / a.n_obs))
        for cutoff in sorted({.01, .02, .05, spec.min_cell_pct}):
            def present(value):
                return all(fractions.get(g, 0) >= cutoff for g in value.split("_"))
            keep = lr.ligand.map(present) & lr.receptor.map(present)
            rows.append({"stage": stage, "n_spots": a.n_obs, "min_cell_pct": cutoff,
                         "lr_records": int(keep.sum()), "n_detected_tf": sum(fractions.get(g, 0) > 0 for g in tfs),
                         "selected_cutoff": cutoff == spec.min_cell_pct})
    return pd.DataFrame(rows)


def prepare_references(spec, data_root, *, lr_source=None, tf_source=None,
                       ortholog_map=None, artista_annotation=None, output_root=None, dry_run=False, resume=False):
    default_lr = REPO / "reference/COMMOT/commot/preprocessing/_data/LRdatabase/CellChat/CellChatDB.ligrec.zebrafish.csv"
    lr_source = Path(lr_source) if lr_source else default_lr if spec.species == "zebrafish" else None
    tf_source = Path(tf_source) if tf_source else None
    missing = []
    if lr_source is None or not lr_source.is_file():
        missing.append("Curated LR source required (--lr-source); salamandrid IDs must be AMEX or mapped with --ortholog-map")
    if tf_source is None or not tf_source.is_file():
        missing.append("Curated species TF list required (--tf-source); one gene ID per line")
    report = {"experiment": spec.metadata(), "status": "missing_references" if missing else "ready",
              "issues": missing, "source_hashes": {},
              "lr_missing_pathway_policy": "explicit_Unannotated_category"}
    lr = load_lr(lr_source) if lr_source and lr_source.is_file() else None
    tfs = sorted({line.strip() for line in tf_source.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")}) if tf_source and tf_source.is_file() else []
    mapping_frame = None
    if ortholog_map and artista_annotation:
        raise ValueError("Select --ortholog-map or --artista-annotation, not both")
    if artista_annotation:
        mapping_frame, report["mapping_audit"] = artista_mapping(artista_annotation)
        mapping = dict(zip(mapping_frame.source_gene, mapping_frame.target_gene))
    elif ortholog_map:
        mapping = read_mapping(ortholog_map)
        mapping_frame = pd.read_csv(ortholog_map, sep="\t", dtype=str)
    if lr is not None and mapping_frame is not None:
        report["source_lr_records"] = len(lr)
        lr, tfs = map_references(lr, tfs, mapping)
        report["mapped_lr_records"] = len(lr)
        report["mapped_tf_count"] = len(tfs)
    for key, path in (("lr", lr_source), ("tf", tf_source), ("ortholog_map", ortholog_map),
                      ("artista_annotation", artista_annotation)):
        if path and Path(path).is_file():
            report["source_hashes"][key] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    if lr is not None:
        qc = coverage(spec, data_root, lr, tfs)
        report["coverage"] = qc.to_dict("records")
        if (qc.loc[qc.selected_cutoff, "lr_records"] < 1).any():
            missing.append("Some selected stages have no supported LR records")
        if not tfs or (qc.loc[qc.selected_cutoff, "n_detected_tf"] < 1).any():
            missing.append("Each stage must detect curated TFs; GRN also rejects zero matches after HVG filtering")
    if missing:
        report["status"] = "missing_references"
        return report
    if not dry_run:
        output = Path(output_root) if output_root else Path(data_root) / "references"
        manifest = output / "reference_manifest.json"
        if manifest.is_file():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            # Source paths are historical provenance; a verified bundle may be
            # relocated intact between the local machine and the server.
            digests = lambda obj: {k: v["sha256"] for k, v in obj.items()}
            if (not resume or old["experiment"] != report["experiment"] or old["coverage"] != report["coverage"]
                    or digests(old["source_hashes"]) != digests(report["source_hashes"])):
                raise ValueError("Existing reference bundle differs; use a new --output-root")
            for name, digest in old["outputs"].items():
                if sha256(output / name) != digest:
                    raise ValueError("Reference output hash mismatch")
            return old
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Reference bundle output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        lr.to_csv(output / "lr_pairs.tsv", sep="\t", index=False, mode="x")
        with (output / "tf_list.txt").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(tfs) + "\n")
        qc.to_csv(output / "lr_tf_coverage.csv", index=False, mode="x")
        report["outputs"] = {name: sha256(output / name) for name in ("lr_pairs.tsv", "tf_list.txt", "lr_tf_coverage.csv")}
        if mapping_frame is not None:
            mapping_frame.to_csv(output / "gene_mapping.tsv", sep="\t", index=False, mode="x")
            report["outputs"]["gene_mapping.tsv"] = sha256(output / "gene_mapping.tsv")
        report["status"] = "complete"
        write_new_json(manifest, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    for flag in ("lr-source", "tf-source", "ortholog-map", "artista-annotation", "output-root"):
        p.add_argument("--" + flag, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    report = prepare_references(load_experiment(a.experiment_config), a.data_root,
        lr_source=a.lr_source, tf_source=a.tf_source, ortholog_map=a.ortholog_map, artista_annotation=a.artista_annotation,
        output_root=a.output_root, dry_run=a.dry_run, resume=a.resume)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
