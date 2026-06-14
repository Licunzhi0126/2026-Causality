from __future__ import annotations

from typing import Tuple

import numpy as np

from mignet_ce.transition.cosine import build_cosine_transition_kernel


def _missing_slat_error(exc: Exception) -> ImportError:
    err = ImportError(
        "The slat Pij method requires the real scSLAT runtime and its dependencies "
        "(torch, torch_geometric, torch_scatter, faiss-cpu). Install scSLAT in the "
        "Python environment. The project intentionally does not import from the local "
        "reference/SLAT folder."
    )
    err.__cause__ = exc
    return err


def _make_adata(features: np.ndarray, coords: np.ndarray, prefix: str):
    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise _missing_slat_error(exc)

    features = np.asarray(features, dtype=float)
    coords = np.asarray(coords, dtype=float)
    adata = ad.AnnData(X=features)
    adata.obs_names = [f"{prefix}_{i:05d}" for i in range(features.shape[0])]
    adata.var_names = [f"feature_{j:05d}" for j in range(features.shape[1])]
    adata.obsm["spatial"] = coords[:, :2]
    return adata


def build_slat_transition_kernel(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    k_neighbors: int = 20,
    hidden_dim: int = 2048,
    n_layers: int = 1,
    epochs: int = 6,
    mlp_hidden: int = 256,
    alpha: float = 0.01,
    temperature: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, dict[str, object]]:
    try:
        import torch
        from scSLAT.model.loaddata import load_anndatas
        from scSLAT.model.preprocess import Cal_Spatial_Net
        from scSLAT.model.utils import run_SLAT
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise _missing_slat_error(exc)

    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {source.shape} and {target.shape}.")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"Feature dimensions differ: {source.shape[1]} != {target.shape[1]}.")
    if source.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("SLAT requires at least two source and target units.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_adata = _make_adata(source, source_coords, "source")
    target_adata = _make_adata(target, target_coords, "target")
    source_k = min(max(1, int(k_neighbors)), source_adata.n_obs - 1)
    target_k = min(max(1, int(k_neighbors)), target_adata.n_obs - 1)
    Cal_Spatial_Net(source_adata, k_cutoff=source_k, model="KNN", verbose=False)
    Cal_Spatial_Net(target_adata, k_cutoff=target_k, model="KNN", verbose=False)

    graph_data = load_anndatas(
        [source_adata, target_adata],
        feature="raw",
        self_loop=True,
        check_order=False,
    )
    features = [graph_data[0].x, graph_data[1].x]
    edges = [graph_data[0].edge_index, graph_data[1].edge_index]
    embd0, embd1, run_time = run_SLAT(
        features=features,
        edges=edges,
        epochs=int(epochs),
        LGCN_layer=int(n_layers),
        mlp_hidden=int(mlp_hidden),
        hidden_size=int(hidden_dim),
        alpha=float(alpha),
    )
    source_embedding = embd0.detach().cpu().numpy()
    target_embedding = embd1.detach().cpu().numpy()
    p = build_cosine_transition_kernel(
        source_embedding=source_embedding,
        target_embedding=target_embedding,
        temperature=temperature,
    )
    metadata = {
        "source_embedding": source_embedding,
        "target_embedding": target_embedding,
        "hard_match_index": np.argmax(p, axis=1),
        "run_time_seconds": float(run_time),
        "reference_api": "scSLAT.model.utils.run_SLAT",
        "k_neighbors": int(k_neighbors),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "temperature": float(temperature),
    }
    return p, metadata
