from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp


LIB_DIR = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from cci_edge_lr_exporter import (
    OUTPUT_COLUMNS,
    SUPPORTED_LAYERS,
    ExportOptions,
    discover_export_jobs,
    export_one_job,
    iter_csr_chunks,
    run_export_jobs,
)


def _write_sample(
    cci_root: Path,
    layer: str = "seurat_k40",
    sample: str = "seurat_heart_11.5",
    bad_shape: bool = False,
) -> None:
    layer_root = cci_root / layer
    lr_dir = layer_root / f"{sample}_COMMOT_by_LR"
    lr_dir.mkdir(parents=True)
    pd.DataFrame({"domain_id": ["domain_001", "domain_002", "domain_003"]}).to_csv(
        layer_root / f"{sample}_index.tsv",
        sep="\t",
        index=False,
    )
    first = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.5],
                [0.75, 0.0, 0.0],
            ]
        )
    )
    second = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.25, 0.0],
            ]
        )
    )
    if bad_shape:
        first = first[:2, :2]
    files = [
        ("Vegfa-Kdr", "Vegfa", "Kdr", "VEGF", first),
        ("Fgf2-Fgfr1", "Fgf2", "Fgfr1", np.nan, second),
    ]
    manifest_rows = []
    total = sp.csr_matrix((3, 3), dtype=float)
    for chunk_id, (lr_key, ligand, receptor, pathway, matrix) in enumerate(files):
        filename = f"{sample}_COMMOT_LR_{lr_key}.npz"
        sp.save_npz(lr_dir / filename, matrix)
        manifest_rows.append(
            {
                "chunk_id": chunk_id,
                "lr_key": lr_key,
                "ligand": ligand,
                "receptor": receptor,
                "pathway": pathway,
                "obsp_key": f"commot-test-{lr_key}",
                "filename": filename,
                "nnz": matrix.nnz,
                "shape_0": matrix.shape[0],
                "shape_1": matrix.shape[1],
            }
        )
        if matrix.shape == (3, 3):
            total = total + matrix
    pd.DataFrame(manifest_rows).to_csv(
        layer_root / f"{sample}_COMMOT_lr_pairs.tsv",
        sep="\t",
        index=False,
    )
    sp.save_npz(layer_root / f"{sample}_CCI_total.npz", total)


def test_export_one_job_writes_directed_edge_lr_long_table(tmp_path: Path) -> None:
    cci_root = tmp_path / "cci"
    output_root = cci_root / "edge_lr_long"
    _write_sample(cci_root)
    jobs = discover_export_jobs(
        cci_root,
        output_root,
        layers=["seurat_k40"],
    )

    result = export_one_job(jobs[0], ExportOptions(chunk_rows=2))

    assert result.status == "completed"
    assert result.expected_rows == 5
    assert result.exported_rows == 5
    table = pq.read_table(jobs[0].output_path)
    assert tuple(table.column_names) == OUTPUT_COLUMNS
    frame = table.to_pandas()
    assert len(frame) == 5
    vegfa = frame.loc[frame["lr_key"] == "Vegfa-Kdr"].set_index(["sender", "receiver"])
    assert vegfa.loc[("domain_001", "domain_002"), "weight"] == pytest.approx(0.25)
    assert vegfa.loc[("domain_002", "domain_003"), "weight"] == pytest.approx(0.5)
    assert vegfa.loc[("domain_003", "domain_001"), "weight"] == pytest.approx(0.75)
    assert set(frame["layer"].astype(str)) == {"seurat_k40"}
    assert set(frame["organ"].astype(str)) == {"heart"}
    assert set(frame["stage"].astype(str)) == {"11.5"}
    assert frame.loc[frame["lr_key"] == "Fgf2-Fgfr1", "pathway"].isna().all()
    metadata = table.schema.metadata
    assert metadata[b"direction"] == b"matrix-row-is-sender;matrix-column-is-receiver"


def test_parallel_runner_emits_row_progress_for_each_sample(tmp_path: Path) -> None:
    cci_root = tmp_path / "cci"
    output_root = cci_root / "edge_lr_long"
    _write_sample(cci_root, sample="seurat_heart_11.5")
    _write_sample(cci_root, sample="seurat_brain_12.5")
    jobs = discover_export_jobs(cci_root, output_root, layers=["seurat_k40"])
    events = []

    results = run_export_jobs(
        jobs,
        ExportOptions(chunk_rows=2),
        workers=2,
        progress_callback=events.append,
    )

    assert {result.status for result in results} == {"completed"}
    assert {result.sample for result in results} == {
        "seurat_heart_11.5",
        "seurat_brain_12.5",
    }
    row_events = [event for event in events if event["type"] == "rows"]
    assert {event["sample"] for event in row_events} == {
        "seurat_heart_11.5",
        "seurat_brain_12.5",
    }
    final_rows = {}
    for event in row_events:
        final_rows[event["sample"]] = event["exported_rows"]
    assert final_rows == {"seurat_heart_11.5": 5, "seurat_brain_12.5": 5}


def test_discovery_rejects_inconsistent_manifest_shapes(tmp_path: Path) -> None:
    cci_root = tmp_path / "cci"
    output_root = cci_root / "edge_lr_long"
    _write_sample(cci_root, bad_shape=True)

    with pytest.raises(ValueError, match="inconsistent matrix shapes"):
        discover_export_jobs(cci_root, output_root, layers=["seurat_k40"])
    assert not list(output_root.rglob("*.parquet"))


def test_worker_error_preserves_partial_and_no_final_output(tmp_path: Path) -> None:
    cci_root = tmp_path / "cci"
    output_root = cci_root / "edge_lr_long"
    sample = "seurat_heart_11.5"
    _write_sample(cci_root, sample=sample)
    job = discover_export_jobs(cci_root, output_root, layers=["seurat_k40"])[0]
    first_matrix = job.lr_dir / f"{sample}_COMMOT_LR_Vegfa-Kdr.npz"
    sp.save_npz(first_matrix, sp.eye(2, format="csr"))

    result = export_one_job(job, ExportOptions(chunk_rows=2))

    assert result.status == "error"
    assert "has shape (2, 2)" in result.error
    assert not job.output_path.exists()
    assert result.partial_file
    assert Path(result.partial_file).exists()


def test_existing_output_is_not_overwritten_by_default(tmp_path: Path) -> None:
    cci_root = tmp_path / "cci"
    output_root = cci_root / "edge_lr_long"
    _write_sample(cci_root)
    job = discover_export_jobs(cci_root, output_root, layers=["seurat_k40"])[0]
    first = export_one_job(job, ExportOptions(chunk_rows=2))
    original_mtime = job.output_path.stat().st_mtime_ns
    second = export_one_job(job, ExportOptions(chunk_rows=2))

    assert first.status == "completed"
    assert second.status == "skipped_existing"
    assert second.exported_rows == 5
    assert job.output_path.stat().st_mtime_ns == original_mtime


def test_only_supported_layers_are_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported CCI layers"):
        discover_export_jobs(
            tmp_path,
            tmp_path / "out",
            layers=["louvain_k150"],
        )
    assert SUPPORTED_LAYERS == ("seurat_k150", "seurat_k40", "spot")


def test_strict_grid_reports_missing_organ_stage_samples(tmp_path: Path) -> None:
    cci_root = tmp_path / "cci"
    _write_sample(cci_root)

    with pytest.raises(FileNotFoundError, match="missing 11 expected organ-stage samples"):
        discover_export_jobs(
            cci_root,
            cci_root / "edge_lr_long",
            layers=["seurat_k40"],
            strict_grid=True,
        )


def test_csr_chunking_preserves_row_column_and_value_order() -> None:
    matrix = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
    )
    chunks = list(iter_csr_chunks(matrix, chunk_rows=2))
    rows = np.concatenate([chunk[0] for chunk in chunks])
    columns = np.concatenate([chunk[1] for chunk in chunks])
    values = np.concatenate([chunk[2] for chunk in chunks])

    assert rows.tolist() == [0, 0, 1, 2]
    assert columns.tolist() == [1, 2, 0, 1]
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
