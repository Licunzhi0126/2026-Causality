from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import yaml

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import TimePair
from mignet_ce.pij.feature_versions.distances import matrix_summary
from mignet_ce.pij.feature_versions.recipes import REPO_ROOT, recipe_sha256, recipe_to_dict
from mignet_ce.pij.feature_versions.spec import FeatureRecipe, PairFeatureBundle


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def method_artifact_root(
    cfg: TemporalRunConfig,
    context: NetworkContext,
    method_name: str,
) -> Path:
    return (
        cfg.effective_pij_archive_root()
        / "compare"
        / f"method={method_name}"
        / f"organ={context.organ}"
        / f"pair={context.pair.label()}"
    )


def pair_artifact_directory(
    cfg: TemporalRunConfig,
    context: NetworkContext,
    method_name: str,
    pair: TimePair,
    side: str,
) -> Path:
    source_stage = str(context.time_points[pair[0]])
    target_stage = str(context.time_points[pair[1]])
    return method_artifact_root(cfg, context, method_name) / f"time={source_stage}_to_{target_stage}" / f"side={side}"


def export_run_manifest(
    *,
    cfg: TemporalRunConfig,
    context: NetworkContext,
    recipe: FeatureRecipe,
    input_paths: Sequence[Path],
    entropy_rows: Sequence[Mapping[str, object]],
) -> Path:
    root = method_artifact_root(cfg, context, recipe.entry_method)
    root.mkdir(parents=True, exist_ok=True)
    resolved = recipe_to_dict(recipe)
    (root / "resolved_recipe.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    digest = recipe_sha256(recipe)
    (root / "recipe_sha256.txt").write_text(f"{digest}\n", encoding="utf-8")

    unique_paths = sorted({Path(path).resolve() for path in input_paths if Path(path).exists()}, key=str)
    input_hashes = {str(path): _sha256_file(path) for path in unique_paths if path.is_file()}
    code_zip = REPO_ROOT / "code.zip"
    manifest: dict[str, object] = {
        "entry_method": recipe.entry_method,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": digest,
        "algorithm_version": recipe.algorithm_version,
        "git_commit": _git_commit(),
        "code_zip_path": str(code_zip) if code_zip.exists() else None,
        "code_zip_sha256": _sha256_file(code_zip) if code_zip.exists() else None,
        "input_file_sha256": input_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "seeds": list(recipe.nmf_seeds),
        "projection_seed": int(recipe.projection_seed),
        "full_command": shlex.join(sys.argv),
        "network_method": context.network_method,
        "pij_method": recipe.entry_method,
        "organ": context.organ,
        "layer_pair": context.pair.label(),
        "pairwise_fit": True,
        "transductive_pairwise_fit": True,
        "uses_target_for_nmf_fit": True,
        "uses_old_compare_N_kl_code": False,
    }
    _write_json(root / "run_manifest.json", manifest)
    pd.DataFrame(list(entropy_rows)).to_csv(root / "entropy_decomposition.csv", index=False)
    return root


def _feature_summary(bundle: PairFeatureBundle) -> dict[str, object]:
    blocks: dict[str, object] = {}
    for name in bundle.source_blocks:
        source = np.asarray(bundle.source_blocks[name], dtype=float)
        target = np.asarray(bundle.target_blocks[name], dtype=float)
        blocks[name] = {
            "source": matrix_summary(source),
            "target": matrix_summary(target),
            "source_zero_row_fraction": float(np.mean(np.all(np.abs(source) <= 1e-12, axis=1))) if source.shape[0] else 0.0,
            "target_zero_row_fraction": float(np.mean(np.all(np.abs(target) <= 1e-12, axis=1))) if target.shape[0] else 0.0,
        }
    return {"blocks": blocks, "metadata": bundle.metadata}


def entropy_decomposition(pij: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(pij, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        matrix,
        row_sums,
        out=np.full_like(matrix, 1.0 / matrix.shape[1]) if matrix.shape[1] else np.zeros_like(matrix),
        where=row_sums > 0.0,
    )
    target = probabilities.mean(axis=0) if probabilities.shape[0] else np.zeros(probabilities.shape[1])
    h_j = float(-np.sum(target * np.log2(target + 1e-12)))
    row_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12), axis=1)
    h_j_given_i = float(row_entropy.mean()) if row_entropy.size else 0.0
    return {
        "H_J": h_j,
        "H_J_given_I": h_j_given_i,
        "EI": h_j - h_j_given_i,
        "mean_pij_row_entropy": h_j_given_i,
        "target_marginal_entropy": h_j,
    }


def export_pair_artifacts(
    *,
    cfg: TemporalRunConfig,
    context: NetworkContext,
    recipe: FeatureRecipe,
    pair: TimePair,
    side: str,
    bundle: PairFeatureBundle,
    raw_costs: Mapping[str, np.ndarray],
    normalized_costs: Mapping[str, np.ndarray],
    fused_cost: np.ndarray,
    cost_diagnostics: Mapping[str, object],
    pij: np.ndarray,
    entropy_row: Mapping[str, object],
) -> Path:
    directory = pair_artifact_directory(cfg, context, recipe.entry_method, pair, side)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "feature_blocks_source.npz", **bundle.source_blocks)
    np.savez_compressed(directory / "feature_blocks_target.npz", **bundle.target_blocks)
    array_artifacts = {
        key: value for key, value in bundle.artifacts.items() if isinstance(value, np.ndarray)
    }
    if array_artifacts:
        np.savez_compressed(directory / "model_artifacts.npz", **array_artifacts)
    sp.save_npz(directory / "pij_row_normalized_sparse.npz", sp.csr_matrix(pij))
    _write_json(directory / "feature_block_summary.json", _feature_summary(bundle))
    _write_json(
        directory / "cost_block_summary.json",
        {
            "raw_blocks": {name: matrix_summary(cost) for name, cost in raw_costs.items()},
            "normalized_blocks": {name: matrix_summary(cost) for name, cost in normalized_costs.items()},
            "fused": matrix_summary(fused_cost),
            "diagnostics": dict(cost_diagnostics),
        },
    )
    if cfg.export_feature_diagnostics:
        np.savez_compressed(
            directory / "cost_blocks.npz",
            **{f"raw__{name}": value for name, value in raw_costs.items()},
            **{f"normalized__{name}": value for name, value in normalized_costs.items()},
            fused=fused_cost,
        )
    pd.DataFrame([dict(entropy_row)]).to_csv(directory / "entropy_decomposition.csv", index=False)
    _write_json(
        directory / "metadata.json",
        {
            "entry_method": recipe.entry_method,
            "recipe_id": recipe.recipe_id,
            "recipe_sha256": recipe_sha256(recipe),
            "algorithm_version": recipe.algorithm_version,
            "feature_blocks": list(bundle.source_blocks),
            "distance_per_block": dict(recipe.block_distances),
            "fusion_weights": dict(recipe.fusion_weights),
            "transductive_pairwise_fit": True,
            "uses_target_for_nmf_fit": True,
            "uses_old_compare_N_kl_code": False,
        },
    )
    return directory
