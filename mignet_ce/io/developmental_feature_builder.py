from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from mignet_ce.io.loaders import LayerDataResolver


SCALAR_COLUMNS = ("pseudotime", "sr", "potency_score")
SR_OBS_CANDIDATES = ("sr", "signaling_entropy", "regulatory_entropy")
SUPPORTED_OUTPUT_LAYERS = ("spot", "seurat_k150", "seurat_k40")
EXPECTED_DOMAIN_COUNTS = {"seurat_k150": 150, "seurat_k40": 40}


@dataclass
class DevelopmentalFeatureBuildConfig:
    data_root: Path
    output_root: Path
    organs: Sequence[str] = ("heart", "brain", "lung")
    time_points: Sequence[str] = ("11.5", "12.5")
    layers: Sequence[str] = ("spot",)
    mode: str = "factory_proxy"
    velocity_components: int = 30
    pseudotime_within_stage_weight: float = 0.15
    sr_source: str = "auto"
    overwrite: bool = False
    skip_missing: bool = False
    seed: int = 42


@dataclass
class DevelopmentalFeatureBuildResult:
    manifest: pd.DataFrame
    output_root: Path


@dataclass
class _StageData:
    organ: str
    stage: str
    path: Path
    units: list[str]
    genes: list[str]
    matrix: object
    obs: pd.DataFrame


def build_developmental_features(cfg: DevelopmentalFeatureBuildConfig) -> DevelopmentalFeatureBuildResult:
    cfg = _normalize_config(cfg)
    rows: list[dict[str, object]] = []
    for organ in cfg.organs:
        rows.extend(build_organ_features(cfg, str(organ)))

    manifest = pd.DataFrame(rows)
    manifest_dir = cfg.output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / "developmental_features_manifest.csv", index=False)
    return DevelopmentalFeatureBuildResult(manifest=manifest, output_root=cfg.output_root)


def build_organ_features(cfg: DevelopmentalFeatureBuildConfig, organ: str) -> list[dict[str, object]]:
    resolver = LayerDataResolver(cfg.data_root)
    stage_data: list[_StageData] = []
    manifest_rows: list[dict[str, object]] = []

    for stage in map(str, cfg.time_points):
        path = resolver.paths("spot", organ, stage).h5ad
        if not path.exists():
            if not cfg.skip_missing:
                raise FileNotFoundError(f"Missing spot h5ad for developmental feature build: {path}")
            manifest_rows.append(
                {
                    "organ": organ,
                    "stage": stage,
                    "n_units": 0,
                    "n_genes": 0,
                    "feature_mode": cfg.mode,
                    "sr_source": "",
                    "velocity_components": cfg.velocity_components,
                    "input_path": str(path),
                    "output_path": "",
                    "status": "missing_input",
                }
            )
            continue
        stage_data.append(_read_stage_data(organ=organ, stage=stage, path=path))

    if not stage_data:
        return manifest_rows

    if "spot" in cfg.layers:
        tables, metadata = build_factory_proxy_features(stage_data, cfg)
    else:
        tables = {}
        for item in stage_data:
            spot_path = cfg.output_root / "spot" / f"{organ}_{item.stage}_features.csv"
            if not spot_path.exists():
                raise FileNotFoundError(
                    f"Domain developmental-feature materialization requires the existing spot feature file "
                    f"{spot_path}, or include 'spot' in layers to generate it in the same run."
                )
            spot_table = pd.read_csv(spot_path)
            validate_output_table(spot_table, cfg.velocity_components)
            tables[item.stage] = spot_table
        metadata = {
            "n_genes": len(_common_genes(stage_data)),
            "sr_source": "precomputed_spot_features",
        }
    for stage in map(str, cfg.time_points):
        if stage not in tables:
            continue
        table = tables[stage]
        input_path = next(item.path for item in stage_data if item.stage == stage)
        spot_output_path = cfg.output_root / "spot" / f"{organ}_{stage}_features.csv"
        if "spot" in cfg.layers:
            output_path = spot_output_path
            _write_feature_table(output_path, table, cfg, layer="spot", organ=organ, stage=stage)
            manifest_rows.append(
                {
                    "layer": "spot",
                    "organ": organ,
                    "stage": stage,
                    "n_units": int(table.shape[0]),
                    "n_genes": int(metadata["n_genes"]),
                    "feature_mode": "factory_proxy",
                    "sr_source": metadata["sr_source"],
                    "velocity_components": cfg.velocity_components,
                    "input_path": str(input_path),
                    "source_feature_path": "",
                    "spot_domain_map": "",
                    "aggregation": "",
                    "n_source_spots": int(table.shape[0]),
                    "expected_units": "",
                    "output_path": str(output_path),
                    "status": "ok",
                }
            )

        domain_source_table = table

        for layer in cfg.layers:
            if layer == "spot":
                continue
            try:
                domain_table, provenance = aggregate_spot_features_to_domain_table(
                    spot_table=domain_source_table,
                    data_root=cfg.data_root,
                    layer=layer,
                    organ=organ,
                    stage=stage,
                )
            except FileNotFoundError as exc:
                if not cfg.skip_missing:
                    raise
                manifest_rows.append(
                    {
                        "layer": layer,
                        "organ": organ,
                        "stage": stage,
                        "n_units": 0,
                        "n_genes": int(metadata["n_genes"]),
                        "feature_mode": "spot_domain_mean",
                        "sr_source": metadata["sr_source"],
                        "velocity_components": cfg.velocity_components,
                        "input_path": str(input_path),
                        "source_feature_path": str(spot_output_path),
                        "spot_domain_map": "",
                        "aggregation": "mean_over_member_spots",
                        "n_source_spots": int(table.shape[0]),
                        "expected_units": EXPECTED_DOMAIN_COUNTS[layer],
                        "output_path": "",
                        "status": f"missing_input: {exc}",
                    }
                )
                continue
            output_path = cfg.output_root / layer / f"{organ}_{stage}_features.csv"
            _write_feature_table(output_path, domain_table, cfg, layer=layer, organ=organ, stage=stage)
            manifest_rows.append(
                {
                    "layer": layer,
                    "organ": organ,
                    "stage": stage,
                    "n_units": int(domain_table.shape[0]),
                    "n_genes": int(metadata["n_genes"]),
                    "feature_mode": "spot_domain_mean",
                    "sr_source": metadata["sr_source"],
                    "velocity_components": cfg.velocity_components,
                    "input_path": str(provenance["domain_h5ad"]),
                    "source_feature_path": str(spot_output_path),
                    "spot_domain_map": str(provenance["spot_domain_map"]),
                    "aggregation": "mean_over_member_spots",
                    "n_source_spots": int(provenance["n_source_spots"]),
                    "expected_units": int(provenance["expected_units"]),
                    "output_path": str(output_path),
                    "status": "ok",
                }
            )
    return manifest_rows


def aggregate_spot_features_to_domain_table(
    *,
    spot_table: pd.DataFrame,
    data_root: Path,
    layer: str,
    organ: str,
    stage: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Materialize strict domain features by averaging existing spot features."""
    if layer not in EXPECTED_DOMAIN_COUNTS:
        raise ValueError(f"Unsupported domain developmental-feature layer {layer!r}.")
    paths = LayerDataResolver(Path(data_root)).paths(layer, organ, str(stage))
    if not paths.h5ad.exists():
        raise FileNotFoundError(f"Missing {layer} h5ad for developmental feature aggregation: {paths.h5ad}")
    if paths.spot_domain_map is None or not paths.spot_domain_map.exists():
        raise FileNotFoundError(f"Missing {layer} spot-domain map: {paths.spot_domain_map}")

    mapping = pd.read_csv(paths.spot_domain_map)
    required_map_columns = {"spot_id", "domain_id"}
    missing_columns = required_map_columns - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"{paths.spot_domain_map} is missing columns {sorted(missing_columns)}.")
    mapping = mapping.copy()
    mapping["spot_id"] = mapping["spot_id"].astype(str)
    mapping["domain_id"] = mapping["domain_id"].astype(str)
    duplicated_spots = mapping.loc[mapping["spot_id"].duplicated(), "spot_id"].unique().tolist()
    if duplicated_spots:
        raise ValueError(
            f"{paths.spot_domain_map} contains duplicate spot_id values, for example {duplicated_spots[:5]}."
        )

    if "unit_id" not in spot_table.columns:
        raise ValueError("Spot developmental feature table must contain unit_id.")
    source = spot_table.copy()
    source["unit_id"] = source["unit_id"].astype(str)
    duplicated_features = source.loc[source["unit_id"].duplicated(), "unit_id"].unique().tolist()
    if duplicated_features:
        raise ValueError(f"Spot developmental features contain duplicate unit_id values: {duplicated_features[:5]}.")
    feature_index = set(source["unit_id"])
    mapping["_feature_unit_id"] = _resolve_mapping_feature_ids(mapping, feature_index)
    missing_spots = mapping.loc[~mapping["_feature_unit_id"].isin(feature_index), "spot_id"].tolist()
    if missing_spots:
        raise ValueError(
            f"{paths.spot_domain_map} has {len(missing_spots)} spots missing from spot developmental features, "
            f"for example {missing_spots[:5]}."
        )

    domain_ids, _ = _read_h5ad_units_and_validate(paths.h5ad)
    expected_count = EXPECTED_DOMAIN_COUNTS[layer]
    if len(domain_ids) != expected_count:
        raise ValueError(f"{layer} expects {expected_count} H5AD units, found {len(domain_ids)} in {paths.h5ad}.")
    map_domains = set(mapping["domain_id"])
    h5ad_domains = set(domain_ids)
    if map_domains != h5ad_domains:
        missing_in_map = sorted(h5ad_domains - map_domains)[:5]
        extra_in_map = sorted(map_domains - h5ad_domains)[:5]
        raise ValueError(
            f"Domain IDs in {paths.spot_domain_map} do not exactly match {paths.h5ad}; "
            f"missing_in_map={missing_in_map}, extra_in_map={extra_in_map}."
        )

    numeric_columns = [column for column in source.columns if column != "unit_id"]
    numeric = source.loc[:, ["unit_id", *numeric_columns]].copy()
    for column in numeric_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if not numeric_columns or not np.isfinite(numeric[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Spot developmental feature numeric columns must be present and finite before aggregation.")
    merged = mapping.merge(numeric, left_on="_feature_unit_id", right_on="unit_id", how="left", validate="one_to_one")
    aggregated = merged.groupby("domain_id", sort=False)[numeric_columns].mean()
    counts = merged.groupby("domain_id", sort=False).size().rename("spot_count")
    aggregated = aggregated.join(counts).reindex(domain_ids)
    if aggregated.isna().any().any() or not np.isfinite(aggregated.to_numpy(dtype=float)).all():
        raise ValueError(f"Generated {layer} developmental features contain missing or non-finite values.")
    output = aggregated.reset_index(names="unit_id")
    return output, {
        "source_spot_features": "in_memory_factory_proxy",
        "spot_domain_map": str(paths.spot_domain_map),
        "domain_h5ad": str(paths.h5ad),
        "layer": layer,
        "organ": organ,
        "stage": str(stage),
        "n_source_spots": int(len(mapping)),
        "n_output_domains": int(len(output)),
        "expected_units": expected_count,
        "aggregation": "mean_over_member_spots",
        "maturity_column": "pseudotime",
    }


def _resolve_mapping_feature_ids(mapping: pd.DataFrame, feature_index: set[str]) -> pd.Series:
    raw = mapping["spot_id"].astype(str)
    if "organ" not in mapping.columns:
        return raw
    prefixed = mapping["organ"].astype(str) + "__" + raw
    return pd.Series(
        [plain if plain in feature_index else pref if pref in feature_index else plain for plain, pref in zip(raw, prefixed)],
        index=mapping.index,
    )


def _read_h5ad_units_and_validate(path: Path) -> tuple[list[str], int]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        units = adata.obs_names.astype(str).tolist()
        gene_count = int(adata.n_vars)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    if len(units) != len(set(units)):
        raise ValueError(f"H5AD observation IDs are not unique: {path}")
    return units, gene_count


def _write_feature_table(
    output_path: Path,
    table: pd.DataFrame,
    cfg: DevelopmentalFeatureBuildConfig,
    *,
    layer: str,
    organ: str,
    stage: str,
) -> None:
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True or --overwrite to replace it.")
    validate_output_table(table, cfg.velocity_components)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    qc_path = (
        cfg.output_root / "qc" / f"{organ}_{stage}_feature_stats.csv"
        if layer == "spot"
        else cfg.output_root / "qc" / layer / f"{organ}_{stage}_feature_stats.csv"
    )
    _write_qc_stats(table, qc_path)


def build_factory_proxy_features(
    stage_data: Sequence[_StageData],
    cfg: DevelopmentalFeatureBuildConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    common_genes = _common_genes(stage_data)
    if not common_genes:
        organs = sorted({item.organ for item in stage_data})
        raise ValueError(f"No common genes found across stages for organ(s) {organs}.")

    matrices = [_prepare_matrix(_subset_matrix(item.matrix, item.genes, common_genes)) for item in stage_data]
    x_all = _vstack_matrices(matrices)
    z_all = _fit_embedding(x_all, cfg.velocity_components, cfg.seed)
    slices = _stage_slices(stage_data)

    pseudotime = _build_pseudotime(stage_data, z_all, cfg.pseudotime_within_stage_weight)
    sr, sr_source = _build_sr(stage_data, matrices, cfg.sr_source)
    potency = 1.0 - pseudotime
    velocity = _build_velocity(stage_data, z_all, slices)

    tables: dict[str, pd.DataFrame] = {}
    for item in stage_data:
        slc = slices[item.stage]
        table = pd.DataFrame({"unit_id": item.units})
        table["pseudotime"] = pseudotime[slc]
        table["sr"] = sr[slc]
        table["potency_score"] = potency[slc]
        for dim in range(cfg.velocity_components):
            table[f"velocity_{dim}"] = velocity[slc, dim]
        _replace_nonfinite(table)
        tables[item.stage] = table
    return tables, {"n_genes": len(common_genes), "sr_source": sr_source}


def build_obs_passthrough_features(*args, **kwargs):
    raise NotImplementedError("obs_passthrough mode is not implemented yet; use factory_proxy.")


def validate_output_table(df: pd.DataFrame, velocity_components: int) -> None:
    required = ["unit_id", *SCALAR_COLUMNS, *[f"velocity_{idx}" for idx in range(velocity_components)]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Developmental feature output is missing required columns {missing}.")
    duplicated = df["unit_id"].astype(str).duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "unit_id"].astype(str).head(5).tolist()
        raise ValueError(f"Developmental feature output contains duplicated unit_id values, for example {examples}.")
    numeric = df.loc[:, required[1:]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Developmental feature output contains NaN or infinite values.")


def _normalize_config(cfg: DevelopmentalFeatureBuildConfig) -> DevelopmentalFeatureBuildConfig:
    cfg.data_root = Path(cfg.data_root)
    cfg.output_root = Path(cfg.output_root)
    cfg.organs = tuple(map(str, cfg.organs))
    cfg.time_points = tuple(map(str, cfg.time_points))
    cfg.layers = tuple(map(str, cfg.layers))
    if cfg.mode != "factory_proxy":
        raise ValueError("Only mode='factory_proxy' is implemented.")
    if cfg.velocity_components <= 0:
        raise ValueError("velocity_components must be positive.")
    if not 0.0 <= cfg.pseudotime_within_stage_weight <= 1.0:
        raise ValueError("pseudotime_within_stage_weight must be between 0 and 1.")
    if cfg.sr_source not in {"auto", "obs", "module", "regulon", "expression"}:
        raise ValueError("sr_source must be one of ['auto', 'obs', 'module', 'regulon', 'expression'].")
    unsupported_layers = sorted(set(cfg.layers) - set(SUPPORTED_OUTPUT_LAYERS))
    if unsupported_layers:
        raise ValueError(
            f"Unsupported output layers {unsupported_layers}; expected a subset of {list(SUPPORTED_OUTPUT_LAYERS)}."
        )
    if not cfg.layers:
        raise ValueError("At least one developmental-feature output layer is required.")
    return cfg


def _read_stage_data(organ: str, stage: str, path: Path) -> _StageData:
    adata = ad.read_h5ad(path)
    matrix, genes = _choose_expression_matrix_and_genes(adata)
    units = adata.obs_names.astype(str).tolist()
    obs = adata.obs.copy()
    obs.index = pd.Index(units, name=adata.obs.index.name)
    return _StageData(
        organ=organ,
        stage=str(stage),
        path=Path(path),
        units=units,
        genes=list(map(str, genes)),
        matrix=matrix,
        obs=obs,
    )


def _choose_expression_matrix_and_genes(adata: ad.AnnData):
    for key in ("count", "counts"):
        if key in adata.layers:
            return adata.layers[key], adata.var_names.astype(str).tolist()
    if adata.raw is not None:
        return adata.raw.X, adata.raw.var_names.astype(str).tolist()
    return adata.X, adata.var_names.astype(str).tolist()


def _common_genes(stage_data: Sequence[_StageData]) -> list[str]:
    gene_sets = [set(item.genes) for item in stage_data]
    shared = set.intersection(*gene_sets)
    return [gene for gene in stage_data[0].genes if gene in shared]


def _subset_matrix(matrix, genes: Sequence[str], common_genes: Sequence[str]):
    index = {gene: idx for idx, gene in enumerate(genes)}
    positions = [index[gene] for gene in common_genes]
    return matrix[:, positions]


def _prepare_matrix(matrix):
    if sp.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(float)
        work.data = np.nan_to_num(work.data, nan=0.0, posinf=0.0, neginf=0.0)
        work.data = np.log1p(np.clip(work.data, 0.0, None))
        work.eliminate_zeros()
        return work
    work = np.asarray(matrix, dtype=float)
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(work, 0.0, None))


def _vstack_matrices(matrices: Sequence):
    if any(sp.issparse(matrix) for matrix in matrices):
        return sp.vstack([matrix if sp.issparse(matrix) else sp.csr_matrix(matrix) for matrix in matrices], format="csr")
    return np.vstack([np.asarray(matrix, dtype=float) for matrix in matrices])


def _fit_embedding(x_all, velocity_components: int, seed: int) -> np.ndarray:
    n_units, n_genes = x_all.shape
    n_fit = min(velocity_components, n_units, n_genes)
    if n_fit <= 0:
        raise ValueError("Cannot build developmental features from an empty expression matrix.")
    model = TruncatedSVD(n_components=n_fit, random_state=seed)
    z_fit = model.fit_transform(x_all)
    z_fit = np.nan_to_num(np.asarray(z_fit, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if n_fit == velocity_components:
        return z_fit
    padded = np.zeros((n_units, velocity_components), dtype=float)
    padded[:, :n_fit] = z_fit
    return padded


def _stage_slices(stage_data: Sequence[_StageData]) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    for item in stage_data:
        stop = start + len(item.units)
        slices[item.stage] = slice(start, stop)
        start = stop
    return slices


def _build_pseudotime(stage_data: Sequence[_StageData], z_all: np.ndarray, within_stage_weight: float) -> np.ndarray:
    numeric_stages = np.asarray([float(item.stage) for item in stage_data], dtype=float)
    min_stage = float(numeric_stages.min())
    max_stage = float(numeric_stages.max())
    if max_stage > min_stage:
        stage_scores = (numeric_stages - min_stage) / (max_stage - min_stage)
    else:
        stage_scores = np.zeros_like(numeric_stages)
    expanded_stage = np.concatenate(
        [np.full(len(item.units), stage_scores[idx], dtype=float) for idx, item in enumerate(stage_data)]
    )
    local_score = _minmax(z_all[:, 0])
    raw = (1.0 - within_stage_weight) * expanded_stage + within_stage_weight * local_score
    return _minmax(raw)


def _build_sr(stage_data: Sequence[_StageData], matrices: Sequence, sr_source: str) -> tuple[np.ndarray, str]:
    if sr_source in {"auto", "obs"} and _all_have_obs_column(stage_data, SR_OBS_CANDIDATES):
        values = np.concatenate([_first_obs_values(item.obs, SR_OBS_CANDIDATES) for item in stage_data])
        return _minmax(values), "obs"
    if sr_source == "obs":
        raise ValueError(f"sr_source='obs' requires one of {list(SR_OBS_CANDIDATES)} in every stage obs.")

    if sr_source in {"auto", "module"} and _all_have_prefixed_columns(stage_data, "Module_", min_count=2):
        values = np.concatenate([_entropy_from_obs_prefix(item.obs, "Module_") for item in stage_data])
        return _minmax(values), "module_entropy"
    if sr_source == "module":
        raise ValueError("sr_source='module' requires at least two Module_* columns in every stage obs.")

    if sr_source in {"auto", "regulon"} and _all_have_prefixed_columns(stage_data, "Regulon - ", min_count=2):
        values = np.concatenate([_entropy_from_obs_prefix(item.obs, "Regulon - ") for item in stage_data])
        return _minmax(values), "regulon_entropy"
    if sr_source == "regulon":
        raise ValueError("sr_source='regulon' requires at least two 'Regulon - ' columns in every stage obs.")

    values = np.concatenate([_entropy_from_expression(matrix) for matrix in matrices])
    return _minmax(values), "expression_entropy_fallback"


def _all_have_obs_column(stage_data: Sequence[_StageData], candidates: Sequence[str]) -> bool:
    return all(any(candidate in item.obs.columns for candidate in candidates) for item in stage_data)


def _first_obs_values(obs: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray:
    for candidate in candidates:
        if candidate in obs.columns:
            values = pd.to_numeric(obs[candidate], errors="coerce").to_numpy(dtype=float)
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    raise KeyError(f"Cannot find any of columns {list(candidates)} in obs.")


def _all_have_prefixed_columns(stage_data: Sequence[_StageData], prefix: str, min_count: int) -> bool:
    return all(len([column for column in item.obs.columns if str(column).startswith(prefix)]) >= min_count for item in stage_data)


def _entropy_from_obs_prefix(obs: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = [column for column in obs.columns if str(column).startswith(prefix)]
    scores = obs.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    row_min = scores.min(axis=1, keepdims=True)
    activity = scores - row_min + 1e-12
    return _row_entropy(activity)


def _entropy_from_expression(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        csr = matrix.tocsr(copy=True).astype(float)
        csr.data = np.clip(np.nan_to_num(csr.data, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        n_features = csr.shape[1]
        if n_features <= 1:
            return np.zeros(csr.shape[0], dtype=float)
        row_sums = np.asarray(csr.sum(axis=1)).ravel()
        data = csr.data
        x_log_x = csr.copy()
        x_log_x.data = data * np.log(data + 1e-12)
        sum_x_log_x = np.asarray(x_log_x.sum(axis=1)).ravel()
        entropy = np.zeros(csr.shape[0], dtype=float)
        mask = row_sums > 0
        entropy[mask] = (np.log(row_sums[mask] + 1e-12) - (sum_x_log_x[mask] / row_sums[mask])) / np.log(n_features)
        return np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    activity = np.clip(np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return _row_entropy(activity)


def _row_entropy(activity: np.ndarray) -> np.ndarray:
    if activity.shape[1] <= 1:
        return np.zeros(activity.shape[0], dtype=float)
    row_sums = activity.sum(axis=1, keepdims=True)
    probabilities = np.divide(activity, row_sums, out=np.zeros_like(activity, dtype=float), where=row_sums > 0)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1) / np.log(activity.shape[1])
    return np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


def _build_velocity(stage_data: Sequence[_StageData], z_all: np.ndarray, slices: dict[str, slice]) -> np.ndarray:
    centroids = {item.stage: z_all[slices[item.stage]].mean(axis=0) for item in stage_data}
    velocities = np.zeros_like(z_all, dtype=float)
    for idx, item in enumerate(stage_data):
        slc = slices[item.stage]
        if len(stage_data) == 1:
            raw = np.zeros((len(item.units), z_all.shape[1]), dtype=float)
        elif idx < len(stage_data) - 1:
            raw = centroids[stage_data[idx + 1].stage] - z_all[slc]
        else:
            raw = z_all[slc] - centroids[stage_data[idx - 1].stage]
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        velocities[slc] = np.divide(raw, norm, out=np.zeros_like(raw, dtype=float), where=norm > 0)
    return np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    low = float(arr[finite].min())
    high = float(arr[finite].max())
    if high <= low:
        return np.zeros_like(arr, dtype=float)
    return (arr - low) / (high - low)


def _replace_nonfinite(df: pd.DataFrame) -> None:
    for column in df.columns:
        if column == "unit_id":
            df[column] = df[column].astype(str)
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        df[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _write_qc_stats(table: pd.DataFrame, path: Path) -> None:
    numeric = table.drop(columns=["unit_id"]).apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame(
        {
            "feature": numeric.columns,
            "min": numeric.min(axis=0).to_numpy(dtype=float),
            "max": numeric.max(axis=0).to_numpy(dtype=float),
            "mean": numeric.mean(axis=0).to_numpy(dtype=float),
            "std": numeric.std(axis=0, ddof=0).to_numpy(dtype=float),
            "missing_count": numeric.isna().sum(axis=0).to_numpy(dtype=int),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)
