"""Run Louvain K150/K40 for the explicit source-to-target Plan 1 pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from ..common.manifest import load_source_target
from ..common.spot_reference import build_spot_reference, common_spot_reference_dir
from ..paths import DEFAULT_OUTPUT_ROOT
from ..louvain_cg.assignment_adapter import one_hot_assignment
from ..louvain_cg.closure import summarize_closure
from ..louvain_cg.direct_macro_pij import build_direct_macro_pij
from ..louvain_cg.layer_factory import discover_layer
from ..louvain_cg.reporting import write_comparison, write_pair_report
from ..louvain_cg.louvain_bridge import run_layer_networks, run_louvain_layer


def _spot_paths(root: Path, sample: dict[str, object]) -> dict[str, Path]:
    sample_id = str(sample["sample_id"])
    return {"h5ad": Path(str(sample["prepared_h5ad"])), "cci": root / "cci" / f"{sample_id}_CCI_total.npz", "index": root / "cci" / f"{sample_id}_index.tsv", "grn": root / "grn" / sample_id / "grn_edges.csv"}


def _spot_request(plan1_root: Path, source: dict[str, object], target: dict[str, object], args: argparse.Namespace) -> object:
    from mignet_ce.coarse_frontends._common import CoarseFrontendRequest

    source_paths, target_paths = _spot_paths(plan1_root, source), _spot_paths(plan1_root, target)
    return CoarseFrontendRequest(h5ad_t=source_paths["h5ad"], h5ad_tp=target_paths["h5ad"], cci_t=source_paths["cci"], cci_tp=target_paths["cci"], cci_index_t=source_paths["index"], cci_index_tp=target_paths["index"], grn_t=source_paths["grn"], grn_tp=target_paths["grn"], nmf_components=args.nmf_components, nmf_max_iter=args.nmf_max_iter, seed=args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan1-output-root", "--prepared-root", dest="plan1_output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plan1_root = args.plan1_output_root
    source, target = load_source_target(plan1_root)
    sample_ids = [str(source["sample_id"]), str(target["sample_id"])]
    print(f"Plan 2 source: {sample_ids[0]}")
    print(f"Plan 2 target: {sample_ids[1]}")
    spot_dir = common_spot_reference_dir(args.output_root)
    spot = build_spot_reference(_spot_request(plan1_root, source, target, args), spot_dir)
    comparison_rows: list[dict[str, object]] = []
    for k in (150, 40):
        layer_root = args.output_root / "louvain_cg" / f"k{k}"
        provenance: dict[str, object] = {"spot_reference": str(spot_dir), "spot_reference_cached": spot["cached"]}
        provenance["louvain"] = run_louvain_layer(spot_root=plan1_root / "prepared", spot_cci_root=plan1_root / "cci", layer_root=layer_root, k=k, sample_names=sample_ids)
        artifacts = discover_layer(layer_root, k, sample_ids)
        macro_ids = [artifact.h5ad.stem for artifact in artifacts.values()]
        provenance["networks"] = run_layer_networks(python=args.python, layer_name=f"louvain_k{k}", domains_root=layer_root / "domains", cci_root=layer_root / "cci", grn_root=layer_root / "grn", sample_names=macro_ids)
        src_artifact, tgt_artifact = artifacts[sample_ids[0]], artifacts[sample_ids[1]]
        src_macro, tgt_macro = src_artifact.h5ad.stem, tgt_artifact.h5ad.stem
        direct = build_direct_macro_pij(h5ad_t=src_artifact.h5ad, h5ad_tp=tgt_artifact.h5ad, cci_t=layer_root / "cci" / f"{src_macro}_CCI_total.npz", cci_tp=layer_root / "cci" / f"{tgt_macro}_CCI_total.npz", index_t=layer_root / "cci" / f"{src_macro}_index.tsv", index_tp=layer_root / "cci" / f"{tgt_macro}_index.tsv", grn_t=layer_root / "grn" / src_macro / "grn_edges.csv", grn_tp=layer_root / "grn" / tgt_macro / "grn_edges.csv", nmf_components=args.nmf_components, nmf_max_iter=args.nmf_max_iter, seed=args.seed)
        source_assignment, source_domains = one_hot_assignment(src_artifact.spot_domain_map, list(spot["source_units"]), list(direct["source_units"]))
        target_assignment, target_domains = one_hot_assignment(tgt_artifact.spot_domain_map, list(spot["target_units"]), list(direct["target_units"]))
        closure = summarize_closure(np.asarray(spot["pij"]), source_assignment, target_assignment, np.asarray(direct["pij"]))
        closure_summary = {key: value for key, value in closure.items() if key != "q_induced"}
        payload = {"source_sample": source["sample_id"], "target_sample": target["sample_id"], "k": k, "EI_spot": spot["micro_ei"], "EI_macro_direct": direct["ei"], "delta_EI_direct": float(direct["ei"]) - float(spot["micro_ei"]), "source_domains": source_domains, "target_domains": target_domains, "closure": closure_summary, "direct_metadata": direct["metadata"]}
        np.save(layer_root / "PIJ_macro_direct.npy", direct["pij"])
        np.save(layer_root / "PIJ_macro_induced.npy", closure["q_induced"])
        write_pair_report(layer_root / "causal_emergence_summary.json", payload)
        write_pair_report(layer_root / "dynamical_closure_summary.json", closure_summary)
        write_pair_report(layer_root / "provenance.json", provenance)
        row = {"k": k, "source_sample": source["sample_id"], "target_sample": target["sample_id"], "EI_spot": payload["EI_spot"], "EI_macro_direct": payload["EI_macro_direct"], "delta_EI_direct": payload["delta_EI_direct"], "closure_quality": closure_summary["closure_quality"], "direct_induced_q_js": closure_summary["direct_induced_q_js"]}
        write_comparison(layer_root / "louvain_comparison.csv", [row])
        comparison_rows.append(row)
    write_comparison(args.output_root / "louvain_cg" / "louvain_comparison.csv", comparison_rows)


if __name__ == "__main__":
    main()
