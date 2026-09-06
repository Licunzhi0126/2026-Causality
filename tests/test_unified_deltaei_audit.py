from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mignet_ce.visualization.downstream.config import MAPPING_COMPLETE, MAPPING_MATURITY
from mignet_ce.visualization.downstream.deltaei_contract import audit_full_cache_root
from mignet_ce.visualization.downstream.dynamic_closure.analysis import effective_information
from mignet_ce.visualization.downstream.mappings import OPTIMIZED_METHOD_BY_MAPPING


def _build_six_cache_fixture(tmp_path):
    pairs = ("11.5->12.5", "12.5->13.5", "13.5->14.5")
    micro = np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float32)
    macro = np.full((40, 40), 0.25 / 39.0, dtype=np.float32)
    np.fill_diagonal(macro, 0.75)
    soft = np.zeros((2, 40), dtype=np.float32)
    soft[0, :2] = (0.85, 0.15)
    soft[1, :2] = (0.12, 0.88)
    micro_ei = effective_information(micro)
    macro_ei = effective_information(macro)
    rows = []
    for mapping in (MAPPING_COMPLETE, MAPPING_MATURITY):
        method = OPTIMIZED_METHOD_BY_MAPPING[mapping]
        for pair in pairs:
            root = tmp_path / "optimized_coarse" / method / pair.replace("->", "_to_")
            root.mkdir(parents=True)
            np.save(root / "PIJ_micro_train.npy", micro)
            np.save(root / "PIJ_macro_train.npy", macro)
            np.save(root / "S_t.npy", soft)
            np.save(root / "S_tp.npy", soft)
            (root / "config.json").write_text(
                json.dumps({"k": 40, "epochs": 1500}), encoding="utf-8"
            )
            (root / "feature_manifest.json").write_text(
                json.dumps({"provenance": {"nmf_max_iter_used": 300}}),
                encoding="utf-8",
            )
            (root / "downstream_full_manifest.json").write_text(
                json.dumps(
                    {
                        "cache_protocol": "full_model_space_v2",
                        "model_state_contract": "full_soft_k",
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "K": 40,
                        "best_epoch": 1200,
                        "hardK_t": 2,
                        "hardK_tp": 2,
                        "Keff_t": 2.4,
                        "Keff_tp": 2.3,
                        "EI_micro_fixed": micro_ei,
                        "EI_macro_best_checkpoint": macro_ei,
                        "delta_EI_best_checkpoint": macro_ei - micro_ei,
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    "mapping": mapping,
                    "time_pair": pair,
                    "delta_EI_matched_spot": macro_ei - micro_ei,
                }
            )
    metrics = tmp_path / "01_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics, index=False)
    return metrics


def test_six_cache_audit_passes_for_exact_full_model_contract(tmp_path) -> None:
    metrics = _build_six_cache_fixture(tmp_path)
    table = audit_full_cache_root(tmp_path, metrics)
    assert len(table) == 6
    assert table["passed"].all()
    assert (table["epochs"] == 1500).all()
    assert (table["K"] == 40).all()
    assert (table["nmf_max_iter_used"] == 300).all()
    assert (table["cache_protocol"] == "full_model_space_v2").all()


def test_six_cache_audit_detects_downstream_deltaei_mismatch(tmp_path) -> None:
    metrics = _build_six_cache_fixture(tmp_path)
    frame = pd.read_csv(metrics)
    frame.loc[0, "delta_EI_matched_spot"] += 0.1
    frame.to_csv(metrics, index=False)
    table = audit_full_cache_root(tmp_path, metrics)
    assert not bool(table["passed"].all())
    assert table["downstream_delta_EI_abs_error"].max() > 0.09
