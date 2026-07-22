from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

# anndata 0.12.6 imports xarray code that still references this NumPy 1.x alias.
# The compatibility alias is process-local and does not modify the Python environment.
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
from scipy import sparse


STAGES = ("12.5", "13.5")
LAYERS = {
    "spot": {"stem": "spot_heart_{stage}", "unit_column": "spot_id", "sample_units": True},
    "seurat_k40": {
        "stem": "seurat_heart_{stage}",
        "unit_column": "domain_id",
        "sample_units": False,
    },
    "seurat_k150": {
        "stem": "seurat150_heart_{stage}",
        "unit_column": "domain_id",
        "sample_units": False,
    },
}
N_SPOTS = 500
N_GENES = 4000
N_LR_PAIRS = 30
GRN_REQUIRED_EDGE_ROWS = 200
RANDOM_SEED = 20260722
MAX_ZIP_BYTES = 512 * 1024 * 1024


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_mean_by_gene(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X[:]
    means = np.asarray(matrix.mean(axis=0)).ravel()
    return np.nan_to_num(means, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def select_spots(source: Path) -> tuple[dict[str, list[str]], dict[str, pd.DataFrame]]:
    selections: dict[str, list[str]] = {}
    maps: dict[str, pd.DataFrame] = {}
    for stage in STAGES:
        map_path = source / "seurat_k150" / "heart" / f"seurat150_heart_{stage}_spot_domain_map.csv"
        frame = pd.read_csv(map_path, dtype={"spot_id": str})
        if frame["spot_id"].duplicated().any():
            raise ValueError(f"Duplicate spot IDs in {map_path}")
        rng = np.random.default_rng(RANDOM_SEED + int(float(stage) * 10))
        chosen: set[str] = set()
        for _, group in frame.groupby("domain_id", sort=True):
            choices = group["spot_id"].astype(str).to_numpy()
            chosen.add(str(rng.choice(choices)))
        if len(chosen) > N_SPOTS:
            raise ValueError(f"{stage} has more domains ({len(chosen)}) than the spot cap ({N_SPOTS})")
        remaining = frame.loc[~frame["spot_id"].isin(chosen), "spot_id"].astype(str).to_numpy()
        fill_count = min(N_SPOTS - len(chosen), len(remaining))
        if fill_count:
            chosen.update(str(value) for value in rng.choice(remaining, size=fill_count, replace=False))
        ordered = frame.loc[frame["spot_id"].isin(chosen), "spot_id"].astype(str).tolist()
        selections[stage] = ordered
        maps[stage] = frame
        log(f"Selected {len(ordered)} spots for E{stage}, covering {frame.loc[frame['spot_id'].isin(chosen), 'domain_id'].nunique()} k150 domains")
    return selections, maps


def select_lr_manifests(source: Path) -> dict[tuple[str, str], pd.DataFrame]:
    selected: dict[tuple[str, str], pd.DataFrame] = {}
    for layer, config in LAYERS.items():
        for stage in STAGES:
            stem = config["stem"].format(stage=stage)
            path = source / "cci" / layer / f"{stem}_COMMOT_lr_pairs.tsv"
            frame = pd.read_csv(path, sep="\t")
            frame = frame.sort_values(["nnz", "lr_key"], ascending=[False, True]).head(N_LR_PAIRS).copy()
            frame.insert(0, "sample_rank", np.arange(1, len(frame) + 1))
            selected[(layer, stage)] = frame
    return selected


def collect_required_genes(
    source: Path,
    lr_manifests: dict[tuple[str, str], pd.DataFrame],
) -> set[str]:
    required: set[str] = set()
    for frame in lr_manifests.values():
        for column in ("ligand", "receptor"):
            for value in frame[column].dropna().astype(str):
                required.update(token for token in value.split("_") if token)
    grn_dirs = {
        "spot": "spot_heart_{stage}",
        "seurat_k40": "seurat_heart_{stage}",
        "seurat_k150": "seurat150_heart_{stage}",
    }
    for layer, template in grn_dirs.items():
        for stage in STAGES:
            path = source / "grn" / layer / template.format(stage=stage) / "grn_edges.csv"
            edges = pd.read_csv(path, nrows=GRN_REQUIRED_EDGE_ROWS, usecols=["regulator", "target"])
            required.update(edges["regulator"].dropna().astype(str))
            required.update(edges["target"].dropna().astype(str))
    return required


def select_genes(
    source: Path,
    required: set[str],
) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    gene_lists: dict[str, list[str]] = {}
    mean_by_stage: dict[str, dict[str, float]] = {}
    source_gene_counts: dict[str, int] = {}
    for stage in STAGES:
        path = source / "seurat_k150" / "heart" / f"seurat150_heart_{stage}_spots_with_domain.h5ad"
        adata = ad.read_h5ad(path, backed="r")
        genes = adata.var_names.astype(str).tolist()
        means = matrix_mean_by_gene(adata)
        adata.file.close()
        gene_lists[stage] = genes
        mean_by_stage[stage] = dict(zip(genes, means, strict=True))
        source_gene_counts[stage] = len(genes)
    common = set(gene_lists[STAGES[0]])
    for stage in STAGES[1:]:
        common.intersection_update(gene_lists[stage])
    scores = {
        gene: float(sum(mean_by_stage[stage][gene] for stage in STAGES) / len(STAGES))
        for gene in common
    }
    required_common = required.intersection(common)
    required_ranked = sorted(required_common, key=lambda gene: (-scores[gene], gene))
    expression_ranked = sorted(common.difference(required_common), key=lambda gene: (-scores[gene], gene))
    selected = (required_ranked + expression_ranked)[: min(N_GENES, len(common))]
    selected_set = set(selected)
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected) + 1),
            "gene": selected,
            "combined_mean_expression": [scores[gene] for gene in selected],
            "selection_reason": ["LR_or_top_GRN" if gene in required_common else "top_expression" for gene in selected],
        }
    )
    if not required_common.issubset(selected_set):
        log(f"Warning: required-gene union exceeded the {N_GENES}-gene cap; lowest-ranked required genes were omitted")
    log(f"Selected {len(selected)} shared genes ({sum(table['selection_reason'] == 'LR_or_top_GRN')} LR/top-GRN genes)")
    return selected, table, source_gene_counts


def h5ad_specs() -> list[tuple[str, Path, bool]]:
    specs: list[tuple[str, Path, bool]] = []
    for stage in STAGES:
        for layer, prefix in (("seurat_k40", "seurat"), ("seurat_k150", "seurat150")):
            base = Path(layer) / "heart" / f"{prefix}_heart_{stage}"
            specs.append((stage, Path(str(base) + ".h5ad"), False))
            specs.append((stage, Path(str(base) + "_spots_with_domain.h5ad"), True))
        specs.append((stage, Path("cci") / "spot" / f"spot_heart_{stage}_COMMOT.h5ad", True))
        specs.append((stage, Path("cci") / "seurat_k40" / f"seurat_heart_{stage}_COMMOT.h5ad", False))
        specs.append((stage, Path("cci") / "seurat_k150" / f"seurat150_heart_{stage}_COMMOT.h5ad", False))
    return specs


def preflight_h5ad(
    source: Path,
    selected_spots: dict[str, list[str]],
    selected_genes: list[str],
) -> None:
    for stage, relative, is_spot in h5ad_specs():
        path = source / relative
        adata = ad.read_h5ad(path, backed="r")
        missing_genes = set(selected_genes).difference(adata.var_names.astype(str))
        missing_spots = set(selected_spots[stage]).difference(adata.obs_names.astype(str)) if is_spot else set()
        adata.file.close()
        if missing_genes or missing_spots:
            raise ValueError(
                f"Preflight failed for {relative}: missing_genes={len(missing_genes)}, missing_spots={len(missing_spots)}"
            )


def write_h5ad_samples(
    source: Path,
    output: Path,
    selected_spots: dict[str, list[str]],
    selected_genes: list[str],
    shapes: dict[str, dict[str, list[int]]],
) -> None:
    for stage, relative, is_spot in h5ad_specs():
        source_path = source / relative
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        adata = ad.read_h5ad(source_path)
        obs_ids = selected_spots[stage] if is_spot else adata.obs_names.astype(str).tolist()
        sampled = adata[obs_ids, selected_genes].copy()
        sampled.uns["sampledata"] = {
            "source_relative_path": str(relative).replace("\\", "/"),
            "stages": list(STAGES),
            "spot_sampling": "one per k150 domain, then deterministic random fill" if is_spot else "all domains retained",
            "gene_sampling": "shared LR/top-GRN genes plus highest combined mean expression",
            "random_seed": RANDOM_SEED,
        }
        sampled.write_h5ad(destination, compression="gzip")
        shapes[str(relative).replace("\\", "/")] = {
            "source": [int(adata.n_obs), int(adata.n_vars)],
            "sample": [int(sampled.n_obs), int(sampled.n_vars)],
        }
        log(f"Wrote {relative}: {sampled.n_obs} x {sampled.n_vars}")
        del adata, sampled
        gc.collect()


def write_seurat_companions(
    source: Path,
    output: Path,
    selected_spots: dict[str, list[str]],
) -> None:
    for stage in STAGES:
        for layer, prefix in (("seurat_k40", "seurat"), ("seurat_k150", "seurat150")):
            source_dir = source / layer / "heart"
            destination_dir = output / layer / "heart"
            destination_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{prefix}_heart_{stage}"
            map_path = source_dir / f"{stem}_spot_domain_map.csv"
            frame = pd.read_csv(map_path, dtype={"spot_id": str})
            sampled = frame.set_index("spot_id").loc[selected_spots[stage]].reset_index()
            sampled.to_csv(destination_dir / map_path.name, index=False)
            counts = pd.crosstab(sampled["domain_id"], sampled["annotation"])
            counts.to_csv(destination_dir / f"{stem}_domain_organ_counts.csv")
            for suffix in ("_spot_domains.png", "_spot_organs.png"):
                shutil.copy2(source_dir / f"{stem}{suffix}", destination_dir / f"{stem}{suffix}")


def subset_square(matrix: sparse.spmatrix, positions: np.ndarray | None) -> sparse.csr_matrix:
    sampled = matrix if positions is None else matrix.tocsr()[positions][:, positions]
    return sampled.tocsr()


def write_cci_layer(
    source: Path,
    output: Path,
    layer: str,
    stage: str,
    selected_spots: dict[str, list[str]],
    selected_manifest: pd.DataFrame,
) -> dict[str, int]:
    config = LAYERS[layer]
    stem = config["stem"].format(stage=stage)
    source_dir = source / "cci" / layer
    destination_dir = output / "cci" / layer
    destination_dir.mkdir(parents=True, exist_ok=True)
    index_path = source_dir / f"{stem}_index.tsv"
    index_frame = pd.read_csv(index_path, sep="\t", dtype=str)
    source_ids = index_frame.iloc[:, 0].astype(str).tolist()
    if config["sample_units"]:
        output_ids = selected_spots[stage]
        position_by_id = {unit_id: position for position, unit_id in enumerate(source_ids)}
        missing = set(output_ids).difference(position_by_id)
        if missing:
            raise ValueError(f"{stem} CCI index is missing {len(missing)} sampled spots")
        positions = np.asarray([position_by_id[unit_id] for unit_id in output_ids], dtype=np.int64)
    else:
        output_ids = source_ids
        positions = None
    pd.DataFrame({config["unit_column"]: output_ids}).to_csv(
        destination_dir / index_path.name, sep="\t", index=False
    )

    total_source = source_dir / f"{stem}_CCI_total.npz"
    total_matrix = subset_square(sparse.load_npz(total_source), positions)
    sparse.save_npz(destination_dir / total_source.name, total_matrix, compressed=True)

    pathway_source_dir = source_dir / f"{stem}_COMMOT_by_pathway"
    pathway_destination_dir = destination_dir / pathway_source_dir.name
    pathway_destination_dir.mkdir(parents=True, exist_ok=True)
    pathway_manifest_path = source_dir / f"{stem}_COMMOT_pathways.tsv"
    pathway_manifest = pd.read_csv(pathway_manifest_path, sep="\t")
    for row_index, row in pathway_manifest.iterrows():
        matrix = subset_square(sparse.load_npz(pathway_source_dir / str(row["filename"])), positions)
        sparse.save_npz(pathway_destination_dir / str(row["filename"]), matrix, compressed=True)
        pathway_manifest.loc[row_index, "nnz"] = int(matrix.nnz)
        pathway_manifest.loc[row_index, "shape_0"] = int(matrix.shape[0])
        pathway_manifest.loc[row_index, "shape_1"] = int(matrix.shape[1])
    pathway_manifest.to_csv(destination_dir / pathway_manifest_path.name, sep="\t", index=False)

    lr_source_dir = source_dir / f"{stem}_COMMOT_by_LR"
    lr_destination_dir = destination_dir / lr_source_dir.name
    lr_destination_dir.mkdir(parents=True, exist_ok=True)
    manifest = selected_manifest.copy()
    manifest.insert(manifest.columns.get_loc("nnz"), "source_nnz", manifest["nnz"].astype(int))
    for row_index, row in manifest.iterrows():
        filename = str(row["filename"])
        matrix = subset_square(sparse.load_npz(lr_source_dir / filename), positions)
        sparse.save_npz(lr_destination_dir / filename, matrix, compressed=True)
        manifest.loc[row_index, "nnz"] = int(matrix.nnz)
        manifest.loc[row_index, "shape_0"] = int(matrix.shape[0])
        manifest.loc[row_index, "shape_1"] = int(matrix.shape[1])
    manifest.to_csv(destination_dir / f"{stem}_COMMOT_lr_pairs.tsv", sep="\t", index=False)

    selected_keys = set(manifest["lr_key"].astype(str))
    ligrec_path = source_dir / f"{stem}_COMMOT_ligrec.tsv"
    ligrec = pd.read_csv(ligrec_path, sep="\t")
    ligrec_keys = ligrec["ligand"].astype(str) + "-" + ligrec["receptor"].astype(str)
    ligrec.loc[ligrec_keys.isin(selected_keys)].to_csv(
        destination_dir / ligrec_path.name, sep="\t", index=False
    )

    for direction, prefix in (("sender", "s-"), ("receiver", "r-")):
        summary_path = source_dir / f"{stem}_COMMOT_{direction}_summary.tsv"
        summary = pd.read_csv(summary_path, sep="\t", index_col=0)
        columns = [prefix + key for key in manifest["lr_key"].astype(str) if prefix + key in summary.columns]
        columns.extend(
            column for column in (prefix + "total-total", prefix + "nan") if column in summary.columns
        )
        summary.reindex(output_ids).loc[:, columns].to_csv(
            destination_dir / summary_path.name, sep="\t"
        )

    info_path = source_dir / f"{stem}_COMMOT_info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    info["n_units"] = len(output_ids)
    info["n_genes"] = N_GENES
    info["n_lr_pairs_exported"] = len(manifest)
    info["sampledata"] = {
        "source_n_units": len(source_ids),
        "unit_sampling": "deterministic stratified spot sample" if positions is not None else "all domains retained",
        "lr_sampling": f"top {N_LR_PAIRS} pairs by source nnz",
        "stages": list(STAGES),
        "random_seed": RANDOM_SEED,
    }
    with (destination_dir / info_path.name).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {
        "source_units": len(source_ids),
        "sample_units": len(output_ids),
        "lr_pairs": len(manifest),
        "total_nnz": int(total_matrix.nnz),
    }


def write_cci(
    source: Path,
    output: Path,
    selected_spots: dict[str, list[str]],
    lr_manifests: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, dict[str, int]]:
    metadata: dict[str, dict[str, int]] = {}
    for layer in LAYERS:
        for stage in STAGES:
            key = f"{layer}/E{stage}"
            metadata[key] = write_cci_layer(
                source, output, layer, stage, selected_spots, lr_manifests[(layer, stage)]
            )
            log(f"Wrote CCI {key}: {metadata[key]['sample_units']} units, {metadata[key]['lr_pairs']} LR matrices")
    return metadata


def copy_grn(source: Path, output: Path) -> None:
    templates = {
        "spot": "spot_heart_{stage}",
        "seurat_k40": "seurat_heart_{stage}",
        "seurat_k150": "seurat150_heart_{stage}",
    }
    for layer, template in templates.items():
        for stage in STAGES:
            relative = Path("grn") / layer / template.format(stage=stage)
            source_dir = source / relative
            destination_dir = output / relative
            destination_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("grn_edges.csv", "grn_summary.tsv", "grn_vim.npy"):
                shutil.copy2(source_dir / filename, destination_dir / filename)
            log(f"Copied full GRN files for {relative}")


def write_documentation(
    output: Path,
    source: Path,
    selected_spots: dict[str, list[str]],
    spot_maps: dict[str, pd.DataFrame],
    selected_genes: pd.DataFrame,
    source_gene_counts: dict[str, int],
    h5ad_shapes: dict[str, dict[str, list[int]]],
    cci_metadata: dict[str, dict[str, int]],
) -> None:
    for stage in STAGES:
        chosen = set(selected_spots[stage])
        table = spot_maps[stage].loc[spot_maps[stage]["spot_id"].astype(str).isin(chosen)].copy()
        table.insert(0, "sample_order", np.arange(1, len(table) + 1))
        table.to_csv(output / f"selected_spots_E{stage}.tsv", sep="\t", index=False)
    selected_genes.to_csv(output / "selected_genes.tsv", sep="\t", index=False)
    metadata = {
        "source": str(source),
        "stages": [f"E{stage}" for stage in STAGES],
        "parameters": {
            "random_seed": RANDOM_SEED,
            "spots_per_stage": N_SPOTS,
            "shared_genes": len(selected_genes),
            "lr_pairs_per_layer_stage": N_LR_PAIRS,
            "grn_policy": "complete original GRN edge, summary, and VIM files retained for selected stages",
        },
        "source_gene_counts": source_gene_counts,
        "h5ad_shapes": h5ad_shapes,
        "cci": cci_metadata,
    }
    with (output / "sampling_metadata.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    readme = f"""# E1S1 domain factory GPT sample data

This archive contains deterministic sample data for mouse embryo heart stages **E12.5** and **E13.5**.

## What was sampled

- Spots: {N_SPOTS} per stage. One spot was selected from every k=150 domain, then the sample was filled deterministically using seed {RANDOM_SEED}.
- Genes: {len(selected_genes)} genes shared by both stages. Genes used by retained ligand-receptor pairs and top GRN edges were prioritized, then the list was filled by combined mean expression.
- CCI: all total/pathway matrices plus the top {N_LR_PAIRS} ligand-receptor matrices by source `nnz` for every stage and layer. Spot-level matrices were sliced to the same sampled spot order; k40/k150 domain matrices retain all domains.
- GRN: the original complete `grn_edges.csv`, `grn_summary.tsv`, and `grn_vim.npy` files were retained for all three layers at both stages.
- PNG files: copied unchanged as full-source spatial overview images; CSV maps beside them contain only sampled spots.

## Directory roles

- `seurat_k40/`, `seurat_k150/`: expression/domain H5AD files, sampled spot-domain maps, domain counts, and overview images.
- `cci/`: COMMOT H5AD files, index tables, total/pathway matrices, selected LR matrices, and filtered sender/receiver summaries.
- `grn/`: full GRN outputs for the two stages.
- `selected_spots_E*.tsv`: exact sampled spot order and metadata.
- `selected_genes.tsv`: exact shared gene order, expression score, and selection reason.
- `sampling_metadata.json`: machine-readable source/sample shapes and parameters.
- `MANIFEST.tsv`: SHA-256 and byte size for every payload file except the manifest itself.

## Important index rule

For every CCI `.npz`, row/column order is defined by the adjacent `*_index.tsv`. Spot H5AD files use the same sampled spot order. The H5AD gene order is the order in `selected_genes.tsv`.

This is a derived sample package. The source directory was not modified.
"""
    (output / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.tsv":
            rows.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(output / "MANIFEST.tsv", sep="\t", index=False)


def make_zip(output: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    size = zip_path.stat().st_size
    if size >= MAX_ZIP_BYTES:
        raise RuntimeError(f"ZIP is {size / 1024**2:.2f} MiB, which exceeds the 512 MiB limit")
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP CRC check failed at {bad_file}")
    log(f"ZIP verified: {zip_path} ({size / 1024**2:.2f} MiB)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_source = repo_root / "data" / "mouse_embyro" / "E1S1_domain_factory"
    default_output = repo_root / "data" / "mouse_embyro" / "E1S1_domain_factory_sampledata_E12.5_E13.5"
    parser = argparse.ArgumentParser(description="Build a two-stage GPT sample package from E1S1_domain_factory")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=Path(str(default_output) + ".zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    zip_path = args.zip_path.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {zip_path}")

    log(f"Source: {source}")
    log(f"Output: {output}")
    selected_spots, spot_maps = select_spots(source)
    lr_manifests = select_lr_manifests(source)
    required_genes = collect_required_genes(source, lr_manifests)
    selected_genes, selected_gene_table, source_gene_counts = select_genes(source, required_genes)
    preflight_h5ad(source, selected_spots, selected_genes)
    log("Preflight passed for all 14 H5AD files")

    output.mkdir(parents=True, exist_ok=False)
    h5ad_shapes: dict[str, dict[str, list[int]]] = {}
    write_h5ad_samples(source, output, selected_spots, selected_genes, h5ad_shapes)
    write_seurat_companions(source, output, selected_spots)
    cci_metadata = write_cci(source, output, selected_spots, lr_manifests)
    copy_grn(source, output)
    write_documentation(
        output,
        source,
        selected_spots,
        spot_maps,
        selected_gene_table,
        source_gene_counts,
        h5ad_shapes,
        cci_metadata,
    )
    write_manifest(output)
    make_zip(output, zip_path)
    log("Sample package completed successfully")


if __name__ == "__main__":
    main()
