#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.coarse_frontends import (  # noqa: E402
    COARSE_FRONTEND_REGISTRY,
    CoarseFrontendRequest,
    prepare_coarse_input,
)
from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WYT FeatureAlign-DeltaEI coarse graining from spot H5AD and CCI inputs."
    )
    parser.add_argument("--method", choices=sorted(COARSE_FRONTEND_REGISTRY), required=True)
    parser.add_argument("--h5ad-t", type=Path, required=True)
    parser.add_argument("--h5ad-tp", type=Path, required=True)
    parser.add_argument("--cci-t", type=Path, required=True)
    parser.add_argument("--cci-tp", type=Path, required=True)
    parser.add_argument("--cci-index-t", type=Path, default=None)
    parser.add_argument("--cci-index-tp", type=Path, default=None)
    parser.add_argument("--grn-t", type=Path, default=None)
    parser.add_argument("--grn-tp", type=Path, default=None)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--cci-min", type=float, default=0.0)
    parser.add_argument("--regsim-knn-k", type=int, default=50)
    parser.add_argument("--regsim-weight", type=float, default=0.2)
    parser.add_argument("--network-svd-dim", type=int, default=32)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--pij-temperature", type=float, default=1.0)
    parser.add_argument("--grn-topk-targets", type=int, default=50)
    parser.add_argument("--grn-state-dim", type=int, default=64)
    parser.add_argument("--grn-projection-seed", type=int, default=20260713)
    parser.add_argument("--grn-knn-k", type=int, default=50)
    parser.add_argument("--grn-graph-weight", type=float, default=0.2)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mid-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--macro-layers", type=int, default=2)
    parser.add_argument("--knn-k", type=int, default=30)
    parser.add_argument("--local-dims", type=int, default=2)
    parser.add_argument(
        "--local-graph-mode",
        choices=["legacy_features", "coords", "all_features"],
        default="legacy_features",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--align-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-ei", type=float, default=1.0)
    parser.add_argument("--lambda-var", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--lambda-sharp", type=float, default=0.02)
    parser.add_argument("--lambda-proto", type=float, default=0.2)
    parser.add_argument("--lambda-min-usage", type=float, default=10.0)
    parser.add_argument("--lambda-max-usage", type=float, default=10.0)
    parser.add_argument("--embedding-target-std", type=float, default=0.05)
    parser.add_argument("--prototype-max-cosine", type=float, default=0.2)
    parser.add_argument("--min-usage-frac", type=float, default=0.01)
    parser.add_argument("--max-usage-frac", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    frontend_request = CoarseFrontendRequest(
        h5ad_t=args.h5ad_t,
        h5ad_tp=args.h5ad_tp,
        cci_t=args.cci_t,
        cci_tp=args.cci_tp,
        cci_index_t=args.cci_index_t,
        cci_index_tp=args.cci_index_tp,
        cci_min=args.cci_min,
        regsim_knn_k=args.regsim_knn_k,
        regsim_weight=args.regsim_weight,
        network_svd_dim=args.network_svd_dim,
        mid_dim=args.mid_dim,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        seed=args.seed,
        pij_temperature=args.pij_temperature,
        grn_t=args.grn_t,
        grn_tp=args.grn_tp,
        grn_topk_targets=args.grn_topk_targets,
        grn_state_dim=args.grn_state_dim,
        grn_projection_seed=args.grn_projection_seed,
        grn_knn_k=args.grn_knn_k,
        grn_graph_weight=args.grn_graph_weight,
    )
    prepared = prepare_coarse_input(args.method, frontend_request)
    config = WYTDeltaEIConfig(
        k=args.k,
        out_dir=args.out_dir,
        hidden_dim=args.hidden_dim,
        mid_dim=args.mid_dim,
        gnn_layers=args.gnn_layers,
        macro_layers=args.macro_layers,
        knn_k=args.knn_k,
        local_dims=args.local_dims,
        local_graph_mode=args.local_graph_mode,
        temperature=args.temperature,
        align_temperature=args.align_temperature,
        epochs=args.epochs,
        lr=args.lr,
        lambda_align=args.lambda_align,
        lambda_ei=args.lambda_ei,
        lambda_var=args.lambda_var,
        lambda_local=args.lambda_local,
        lambda_sharp=args.lambda_sharp,
        lambda_proto=args.lambda_proto,
        lambda_min_usage=args.lambda_min_usage,
        lambda_max_usage=args.lambda_max_usage,
        embedding_target_std=args.embedding_target_std,
        prototype_max_cosine=args.prototype_max_cosine,
        min_usage_frac=args.min_usage_frac,
        max_usage_frac=args.max_usage_frac,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )
    result = train_deltaei(prepared, config)
    print(
        f"completed method={args.method} K={args.k} "
        f"best_epoch={result.best_epoch} delta_EI={result.final_delta_ei:.6f} "
        f"out={result.out_dir}"
    )


if __name__ == "__main__":
    main()
