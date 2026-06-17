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


@dataclass
class DevelopmentalFeatureBuildConfig:
    data_root: Path
    output_root: Path
    organs: Sequence[str] = ("heart", "brain", "lung")
    time_points: Sequence[str] = ("11.5", "12.5")
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

    tables, metadata = build_factory_proxy_features(stage_data, cfg)
    for stage in map(str, cfg.time_points):
        if stage not in tables:
            continue
        table = tables[stage]
        output_path = cfg.output_root / "spot" / f"{organ}_{stage}_features.csv"
        if output_path.exists() and not cfg.overwrite:
            raise FileExistsError(f"{output_path} already exists; pass overwrite=True or --overwrite to replace it.")
        validate_output_table(table, cfg.velocity_components)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        _write_qc_stats(table, cfg.output_root / "qc" / f"{organ}_{stage}_feature_stats.csv")

        input_path = next(item.path for item in stage_data if item.stage == stage)
        manifest_rows.append(
            {
                "organ": organ,
                "stage": stage,
                "n_units": int(table.shape[0]),
                "n_genes": int(metadata["n_genes"]),
                "feature_mode": "factory_proxy",
                "sr_source": metadata["sr_source"],
                "velocity_components": cfg.velocity_components,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "ok",
            }
        )
    return manifest_rows


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
    if cfg.mode != "factory_proxy":
        raise ValueError("Only mode='factory_proxy' is implemented.")
    if cfg.velocity_components <= 0:
        raise ValueError("velocity_components must be positive.")
    if not 0.0 <= cfg.pseudotime_within_stage_weight <= 1.0:
        raise ValueError("pseudotime_within_stage_weight must be between 0 and 1.")
    if cfg.sr_source not in {"auto", "obs", "module", "regulon", "expression"}:
        raise ValueError("sr_source must be one of ['auto', 'obs', 'module', 'regulon', 'expression'].")
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
