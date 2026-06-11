#!/usr/bin/env python3
"""Global domain-level GRN inference for Louvain less-than-5 outputs."""

from __future__ import annotations

import argparse
import gc
import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


INPUT_DIR = Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/louvain_k1100")
OUTPUT_BASE = Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn/louvain_k1100")
GLOBAL_AUXILIARY_SUFFIX = "_spots_with_domain.h5ad"
_SCRIPT_PATH = Path(__file__).resolve()
TF_CANDIDATE_FILES = [
    Path("/home/jovyan/public/datasets/Mouse-embryo/reference/GENIE3/mouse_tf_list.txt"),
    _SCRIPT_PATH.parent.parent / "reference" / "GENIE3" / "mouse_tf_list.txt",
    _SCRIPT_PATH.parent / "mouse_tf_list.txt",
]


N_THREADS = 16
TREE_METHOD = "ET"
N_TREES = 500
K_PARAM = "sqrt"
TOP_HVG = 2000
TOP_EDGE_COUNT = 500_000
RANDOM_SEED = 2025
MIN_REGULATORS = 100


SHARED_EXPR: Optional[np.ndarray] = None
SHARED_REGULATORS: Optional[List[int]] = None
SHARED_TREE_METHOD: Optional[str] = None
SHARED_K_PARAM: Optional[object] = None
SHARED_N_TREES: Optional[int] = None

TF_SYMBOL_CACHE: Optional[set] = None


def compute_feature_importances(estimator) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    raise AttributeError("Estimator is missing feature_importances_.")


def _determine_max_features(k_param: object, n_inputs: int):
    if k_param == "all":
        return None
    if k_param == "sqrt":
        return "sqrt"
    if isinstance(k_param, int):
        return min(k_param, n_inputs)
    return k_param


def _init_pool(expr_data: np.ndarray, regulators: Sequence[int], tree_method: str, k_param, n_trees: int) -> None:
    global SHARED_EXPR, SHARED_REGULATORS, SHARED_TREE_METHOD, SHARED_K_PARAM, SHARED_N_TREES
    SHARED_EXPR = expr_data
    SHARED_REGULATORS = list(regulators)
    SHARED_TREE_METHOD = tree_method
    SHARED_K_PARAM = k_param
    SHARED_N_TREES = n_trees


def _genie3_worker(target_idx: int) -> Tuple[int, np.ndarray]:
    if SHARED_EXPR is None or SHARED_REGULATORS is None:
        raise RuntimeError("Shared expression matrix was not initialized in worker process.")

    expr_data = SHARED_EXPR
    n_genes = expr_data.shape[1]

    if target_idx in SHARED_REGULATORS:
        regulators = [i for i in SHARED_REGULATORS if i != target_idx]
    else:
        regulators = SHARED_REGULATORS

    if not regulators:
        return target_idx, np.zeros(n_genes, dtype=np.float32)

    output = expr_data[:, target_idx]
    std = float(np.std(output))
    if std == 0.0:
        return target_idx, np.zeros(n_genes, dtype=np.float32)
    output_norm = output / std

    inputs = expr_data[:, regulators]
    max_features = _determine_max_features(SHARED_K_PARAM, len(regulators))

    random_state = RANDOM_SEED + target_idx
    if SHARED_TREE_METHOD == "ET":
        model = ExtraTreesRegressor(
            n_estimators=SHARED_N_TREES,
            max_features=max_features,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=SHARED_N_TREES,
            max_features=max_features,
            random_state=random_state,
            n_jobs=1,
        )

    model.fit(inputs, output_norm)
    importances = compute_feature_importances(model)

    vi = np.zeros(n_genes, dtype=np.float32)
    vi[np.asarray(regulators)] = importances
    return target_idx, vi


def run_genie3(expr_data: np.ndarray, regulator_indices: Sequence[int]) -> np.ndarray:
    regulators = list(regulator_indices)
    if not regulators:
        regulators = list(range(expr_data.shape[1]))

    vim = np.zeros((expr_data.shape[1], expr_data.shape[1]), dtype=np.float32)
    targets = list(range(expr_data.shape[1]))

    with Pool(
        processes=N_THREADS,
        initializer=_init_pool,
        initargs=(expr_data, regulators, TREE_METHOD, K_PARAM, N_TREES),
    ) as pool:
        for idx, vi in tqdm(
            pool.imap_unordered(_genie3_worker, targets),
            total=len(targets),
            desc="GENIE3 targets",
        ):
            vim[idx, :] = vi

    return vim.T


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_h5ad_files(input_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in input_dir.rglob("*.h5ad")
        if p.is_file() and not p.name.endswith(GLOBAL_AUXILIARY_SUFFIX)
    )


def preprocess_adata(adata: sc.AnnData) -> sc.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.var_names_make_unique()

    if TOP_HVG is not None and TOP_HVG < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=TOP_HVG, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata = adata.copy()
    return adata


def extract_expression_matrix(adata: sc.AnnData) -> Tuple[np.ndarray, List[str]]:
    if sp.issparse(adata.X):
        expr = adata.X.toarray()
    else:
        expr = np.asarray(adata.X)
    expr = expr.astype(np.float32, copy=False)
    return expr, adata.var_names.to_list()


def _load_tf_symbols() -> Optional[set]:
    global TF_SYMBOL_CACHE
    if TF_SYMBOL_CACHE is not None:
        return TF_SYMBOL_CACHE

    for path in TF_CANDIDATE_FILES:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            symbols = {
                token.strip().upper()
                for line in handle
                for token in line.replace(",", " ").split()
                if token.strip() and not token.strip().startswith("#")
            }
        if symbols:
            TF_SYMBOL_CACHE = symbols
            print(f"Loaded TF candidate list: {path} ({len(symbols)} TFs)")
            return TF_SYMBOL_CACHE

    warnings.warn(
        "TF candidate list was not found. All genes will be used as potential regulators.",
        RuntimeWarning,
    )
    TF_SYMBOL_CACHE = set()
    return TF_SYMBOL_CACHE


def select_regulator_indices(gene_names: Sequence[str]) -> List[int]:
    tf_symbols = _load_tf_symbols()
    if not tf_symbols:
        return list(range(len(gene_names)))

    gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_names)}
    regulator_indices = [gene_to_idx[symbol] for symbol in tf_symbols if symbol in gene_to_idx]

    if len(regulator_indices) < MIN_REGULATORS:
        warnings.warn(
            f"Only matched {len(regulator_indices)} TFs, below MIN_REGULATORS={MIN_REGULATORS}. "
            "Falling back to all genes as regulators.",
            RuntimeWarning,
        )
        return list(range(len(gene_names)))

    regulator_indices.sort()
    print(f"Matched {len(regulator_indices)} TF regulators.")
    return regulator_indices


def build_edge_table(vim: np.ndarray, gene_names: Sequence[str]) -> pd.DataFrame:
    mask = vim > 0
    regulators, targets = np.where(mask)
    weights = vim[regulators, targets]

    df_edges = pd.DataFrame(
        {
            "regulator": [gene_names[i] for i in regulators],
            "target": [gene_names[j] for j in targets],
            "weight": weights,
        }
    )
    df_edges = df_edges.sort_values("weight", ascending=False)
    if TOP_EDGE_COUNT and df_edges.shape[0] > TOP_EDGE_COUNT:
        df_edges = df_edges.head(TOP_EDGE_COUNT).reset_index(drop=True)
    return df_edges


def summarize_edges(dataset: str, df_edges: pd.DataFrame, total_genes: int, regulators: int) -> pd.DataFrame:
    stats = {
        "dataset": [dataset],
        "total_genes": [total_genes],
        "regulators_used": [regulators],
        "edges_reported": [df_edges.shape[0]],
        "max_weight": [df_edges["weight"].max() if not df_edges.empty else 0.0],
        "min_weight": [df_edges["weight"].min() if not df_edges.empty else 0.0],
        "median_weight": [df_edges["weight"].median() if not df_edges.empty else 0.0],
    }
    return pd.DataFrame(stats)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GENIE3-style GRN inference on Louvain less-than-5 global domain outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_dir = args.input_dir
    output_base = args.output_base

    ensure_dir(output_base)
    np.random.seed(RANDOM_SEED)

    h5ad_files = list_h5ad_files(input_dir)
    if not h5ad_files:
        raise FileNotFoundError(f"Input directory {input_dir} does not contain any usable .h5ad files.")
    print(f"Detected {len(h5ad_files)} h5ad datasets under {input_dir}.")

    summaries: List[pd.DataFrame] = []

    for file_path in tqdm(h5ad_files, desc="GRN datasets", unit="file"):
        dataset_name = file_path.stem
        dataset_output = output_base / dataset_name
        ensure_dir(dataset_output)

        print(f"\n====== Processing dataset {dataset_name} ======")
        adata_raw = sc.read_h5ad(str(file_path))
        adata = preprocess_adata(adata_raw)
        del adata_raw

        print(f"{dataset_name}: genes used for GRN = {adata.n_vars}")
        expr_data, genes = extract_expression_matrix(adata)
        regulator_indices = select_regulator_indices(genes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            vim = run_genie3(expr_data, regulator_indices)

        matrix_path = dataset_output / "grn_vim.npy"
        edges_path = dataset_output / "grn_edges.csv"
        summary_path = dataset_output / "grn_summary.tsv"

        np.save(matrix_path, vim)
        edge_table = build_edge_table(vim, genes)
        edge_table.to_csv(edges_path, index=False)
        summary_df = summarize_edges(dataset_name, edge_table, len(genes), len(regulator_indices))
        summary_df.to_csv(summary_path, sep="\t", index=False)

        print(f"{dataset_name}: exported {edge_table.shape[0]} GRN edges -> {dataset_output}")
        summaries.append(summary_df.assign(input_file=str(file_path), output_dir=str(dataset_output)))

        del adata, expr_data, genes, vim, edge_table, regulator_indices
        gc.collect()

    if summaries:
        overview = pd.concat(summaries, ignore_index=True)
        overview_path = output_base / "summary_overview.tsv"
        overview.to_csv(overview_path, sep="\t", index=False)
        print(f"\nAll dataset summaries written to {overview_path}")
    else:
        print("\nNo valid GRN result was produced. Please inspect the input datasets.")


if __name__ == "__main__":
    main()
