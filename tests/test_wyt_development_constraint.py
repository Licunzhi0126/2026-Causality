from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from scripts.build_maturity_proxy_from_h5ad import (
    StageInput,
    build_maturity_proxy,
)
from wyt_deltaei_coarse_grain.development import developmental_loss, load_maturity_csv
from wyt_deltaei_coarse_grain.trainer import WYTDeltaEIConfig, train_deltaei


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _macro_pij(inputs: MacroPijInputs):
    return torch.softmax(inputs.z_macro_t @ inputs.z_macro_tp.T, dim=1)


def _prepared_with_maturity() -> PreparedCoarseInput:
    rng = np.random.default_rng(22)
    count_t, count_tp = 6, 7
    micro_t = rng.normal(size=(count_t, 4)).astype(np.float32)
    micro_tp = rng.normal(size=(count_tp, 4)).astype(np.float32)
    logits = micro_t @ micro_tp.T
    micro_pij = np.exp(logits - logits.max(axis=1, keepdims=True))
    micro_pij /= micro_pij.sum(axis=1, keepdims=True)
    return PreparedCoarseInput(
        method="synthetic_maturity",
        unit_ids_t=[f"t{index}" for index in range(count_t)],
        unit_ids_tp=[f"q{index}" for index in range(count_tp)],
        network_t=sp.eye(count_t, format="csr"),
        network_tp=sp.eye(count_tp, format="csr"),
        encoder_features_t=np.random.default_rng(1).normal(size=(count_t, 5)).astype(np.float32),
        encoder_features_tp=np.random.default_rng(2).normal(size=(count_tp, 5)).astype(np.float32),
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij,
        micro_ei=0.1,
        macro_pij_builder=_macro_pij,
        maturity_t=np.linspace(0.0, 1.0, count_t, dtype=np.float32),
        maturity_tp=np.linspace(0.0, 1.0, count_tp, dtype=np.float32),
    )


def test_load_maturity_csv_aligns_by_unit_id_order(tmp_path) -> None:
    path = tmp_path / "maturity.csv"
    _write_csv(
        path,
        [
            {"spot_id": "b", "maturity": 0.9, "confidence": 2.0},
            {"spot_id": "a", "maturity": 0.1, "confidence": 1.0},
            {"spot_id": "c", "maturity": 0.5, "confidence": 3.0},
        ],
    )
    aligned = load_maturity_csv(
        path,
        ["a", "b", "c"],
        maturity_column="maturity",
        confidence_column="confidence",
        normalization="none",
    )
    assert aligned.values.tolist() == pytest.approx([0.1, 0.9, 0.5])
    assert aligned.confidence.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_load_maturity_csv_rejects_duplicate_and_mismatched_ids(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    _write_csv(
        duplicate,
        [
            {"spot_id": "a", "maturity": 0.1},
            {"spot_id": "a", "maturity": 0.2},
            {"spot_id": "b", "maturity": 0.3},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate maturity IDs"):
        load_maturity_csv(duplicate, ["a", "b"], maturity_column="maturity")

    mismatch = tmp_path / "mismatch.csv"
    _write_csv(
        mismatch,
        [
            {"spot_id": "a", "maturity": 0.1},
            {"spot_id": "x", "maturity": 0.2},
        ],
    )
    with pytest.raises(ValueError, match="missing.*unknown"):
        load_maturity_csv(mismatch, ["a", "b"], maturity_column="maturity")


def test_developmental_loss_penalizes_maturity_mixing_and_has_gradients() -> None:
    maturity = torch.tensor([0.0, 0.1, 0.9, 1.0], dtype=torch.float32)
    contiguous_logits = torch.tensor(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]],
        requires_grad=True,
    )
    mixed_logits = torch.tensor(
        [[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]],
        requires_grad=True,
    )
    contiguous_loss = developmental_loss(torch.softmax(contiguous_logits, dim=1), maturity)
    mixed_loss = developmental_loss(torch.softmax(mixed_logits, dim=1), maturity)
    assert mixed_loss.item() > contiguous_loss.item()
    mixed_loss.backward()
    assert mixed_logits.grad is not None
    assert torch.isfinite(mixed_logits.grad).all()


def test_train_deltaei_records_maturity_metrics_and_artifacts(tmp_path) -> None:
    result = train_deltaei(
        _prepared_with_maturity(),
        WYTDeltaEIConfig(
            k=3,
            out_dir=tmp_path,
            hidden_dim=8,
            mid_dim=4,
            epochs=2,
            knn_k=2,
            lambda_dev=0.05,
            log_every=1,
        ),
    )
    assert result.best_epoch in {1, 2}
    assert "L_dev" in (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["development_constraint_enabled"] is True
    assert summary["lambda_dev"] == pytest.approx(0.05)
    assert (tmp_path / "maturity_t.npy").exists()
    assert (tmp_path / "maturity_tp.npy").exists()


def test_build_maturity_proxy_outputs_stage_csvs_and_semantic_metadata(tmp_path) -> None:
    stages: list[StageInput] = []
    for index, label in enumerate(("12.5", "13.5")):
        units = [f"{label}_a", f"{label}_b", f"{label}_c"]
        values = np.asarray(
            [[1.0, 0.2, 0.4], [0.4, 1.2, 0.3], [0.2, 0.5, 1.4]],
            dtype=np.float32,
        ) + float(index)
        h5ad = tmp_path / f"{label}.h5ad"
        adata = ad.AnnData(
            X=values,
            obs=pd.DataFrame(index=pd.Index(units, name="spot_id")),
            var=pd.DataFrame(index=pd.Index(["g1", "g2", "g3"], name="gene")),
        )
        adata.layers["count"] = values
        adata.write_h5ad(h5ad)
        stages.append(
            StageInput(
                label=label,
                h5ad=h5ad,
                output_csv=tmp_path / f"{label}_maturity.csv",
            )
        )

    metadata_path = tmp_path / "maturity_metadata.json"
    build_maturity_proxy(
        stages,
        output_metadata=metadata_path,
        n_components=2,
        within_stage_weight=0.15,
        mode="stage_aware",
        seed=42,
    )

    expected_columns = {
        "spot_id",
        "maturity",
        "pseudotime",
        "sr",
        "potency_score",
        "stage_label",
        "maturity_source",
    }
    for stage in stages:
        table = pd.read_csv(stage.output_csv)
        assert set(table.columns) == expected_columns
        assert len(table) == 3
        assert table["maturity"].between(0.0, 1.0).all()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["semantic_status"] == "proxy_not_externally_validated_pseudotime"
    assert "within timepoint" in metadata["training_semantics_note"]
