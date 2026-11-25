from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy import sparse

from mignet_ce.downstream.post_perbutation.config import LEVELS, STRENGTHS, Task, tasks_for
from mignet_ce.downstream.post_perbutation.scheduler import GIB, MODULE, CardMemory, GpuAdmission, Scheduler, Slot, Work, device_tokens


def _factory(root: Path) -> None:
    rng = np.random.default_rng(31)
    genes = [f"gene{i}" for i in range(8)]
    for time, count in (("11.5", 10), ("12.5", 12)):
        stem = f"spot_heart_{time}"
        units = [f"spot_{i}" for i in range(count)]
        spot = root / "spot" / "heart"
        spot.mkdir(parents=True, exist_ok=True)
        with h5py.File(spot / f"{stem}.h5ad", "w") as handle:
            handle.create_dataset("obs/_index", data=np.asarray(units, dtype="S"))
            handle.create_dataset("var/_index", data=np.asarray(genes, dtype="S"))
            handle.create_dataset("obsm/spatial", data=rng.random((count, 2)))
            handle.create_dataset("X", data=rng.uniform(1., 6., (count, len(genes))))
        cci = root / "cci" / "spot"
        cci.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cci / f"{stem}_CCI_total.npz", sparse.csr_matrix(rng.uniform(.7, 2., (count, count))))
        (cci / f"{stem}_index.tsv").write_text("spot_id\n" + "\n".join(units) + "\n", encoding="utf-8")
        grn = root / "grn" / "spot" / stem
        grn.mkdir(parents=True)
        (grn / "grn_edges.csv").write_text("regulator,target,weight\n" + "\n".join(
            f"{gene},{genes[(i + 1) % len(genes)]},{i + 1}" for i, gene in enumerate(genes)), encoding="utf-8")
        dev = root / "developmental_features" / "spot"
        dev.mkdir(parents=True, exist_ok=True)
        (dev / f"heart_{time}_features.csv").write_text("unit_id,pseudotime\n" + "\n".join(
            f"{unit},{i / count}" for i, unit in enumerate(units)), encoding="utf-8")
        seurat = root / "seurat_k40" / "heart"
        seurat.mkdir(parents=True, exist_ok=True)
        (seurat / f"seurat_heart_{time}_spot_domain_map.csv").write_text("spot_id,domain_id\n" + "\n".join(
            f"{unit},{i % 2}" for i, unit in enumerate(units)), encoding="utf-8")


def test_dry_run_has_43_conditions_and_preserves_the_full_grid(tmp_path):
    _factory(tmp_path)
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, "-B", "-m", MODULE, "--data-root", str(tmp_path),
                             "--output-root", str(output), "--k", "2", "--dry-run"], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["requested_trainings"] == len(tasks_for("all")) == 43
    assert len(tasks_for("all", strengths=STRENGTHS, levels=LEVELS)) == 127
    assert report["existence_groups"] == 202 and report["gpu_budget_gib"] == 18
    assert report["max_workers_per_gpu"] == 6
    assert report["devices"] == ["cuda:0", "cuda:1"] and report["workers_per_gpu"] == 3
    assert report["source"]["units"] == 10 and report["target"]["units"] == 12
    assert not output.exists()


def test_device_mapping_and_solo_retry_drains_the_selected_card(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,2")
    assert device_tokens(["cuda:0", "cuda:1"]) == {"cuda:0": "4", "cuda:1": "2"}
    with pytest.raises(ValueError, match="exceed"):
        device_tokens(["cuda:0", "cuda:2"])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.root = tmp_path
    scheduler.last_family = "symmetry"
    scheduler.slots = [SimpleNamespace(device=device, work=None, stop=lambda: None)
                       for device in ("cuda:0", "cuda:1") for _ in range(3)]
    scheduler.slots[3].work = SimpleNamespace(retry=False)
    scheduler.ready = {name: deque() for name in ("p0", "cci", "symmetry")}
    task = Task("cci", "positive", 10, "P1")
    scheduler.oom_wait = deque([scheduler.make_work(task, {})])
    for topology in ("P2", "P3", "P4", "P5"):
        scheduler.ready["cci"].append(scheduler.make_work(Task("cci", "positive", 10, topology), {}))
    scheduler.start_work = lambda slot, work: setattr(slot, "work", work)
    scheduler.can_start = lambda slot, work: True
    scheduler.dispatch()
    assert scheduler.slots[0].work.retry
    assert all(slot.work is None for slot in scheduler.slots[1:3])
    assert all(slot.work is not None for slot in scheduler.slots[3:])
    assert len(scheduler.ready["cci"]) == 2


def test_two_launchers_share_card_admission_and_solo_retry(monkeypatch, tmp_path):
    from mignet_ce.downstream.post_perbutation import scheduler

    monkeypatch.setattr(scheduler.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(scheduler, "_nvidia_memory", lambda: {"GPU-A": ("0", CardMemory(0, 24 * GIB, {}))})
    devices = {"cuda:0": {"uuid": "GPU-A", "visible_device": "0"}}
    left, right = GpuAdmission(devices, 18, 3, 6), GpuAdmission(devices, 18, 3, 6)
    slots = [Slot(str(i), "cuda:0", "0", tmp_path / "log") for i in range(6)]
    item = Work(Task("cci", "positive", 10, "P1"), tmp_path, {})
    retry = Work(item.task, tmp_path, {}, retry=True)
    try:
        assert all(left.admit(slot, item) for slot in slots[:3])
        assert right.admit(slots[3], item)
        assert not right.admit(slots[4], item)
        assert not right.admit(slots[5], retry)
        left.finished(slots[0], "cci", {"status": "complete", "metrics": {"peak_cuda_reserved_bytes": 2 * GIB}})
        assert left.limit["cuda:0"] == 4
        for slot in slots[:3]:
            left.release(slot)
        right.release(slots[3])
        assert right.admit(slots[5], retry)
        assert not left.admit(slots[0], item)
    finally:
        for slot in slots:
            left.release(slot)


def test_cpu_cli_trains_and_resumes_both_experiment_families(tmp_path):
    pytest.importorskip("torch")
    _factory(tmp_path)
    output = tmp_path / "out"
    args = [sys.executable, "-B", "-m", MODULE, "--data-root", str(tmp_path), "--output-root", str(output),
            "--experiment", "all", "--devices", "cpu", "--workers-per-gpu", "2", "--cpu-workers", "2",
            "--cci-topologies", "P1", "--cci-strengths", "10", "--k", "10", "--epochs", "45",
            "--random-partitions", "2", "--symmetry-levels", "20"]
    with (tmp_path / "cli.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
    records = json.loads((output / "conditions.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, records
    assert len(records) == 15
    assert {"P0", "cci-P1-positive-10", "cci-P1-negative-10"}.issubset(records)
    for row in records.values():
        assert row["status"] == "complete"
        assert row["metrics"]["delta_ei_trained_bits"] == pytest.approx(
            row["metrics"]["macro_ei_trained_bits"] - row["metrics"]["micro_ei_bits"])
        assert (Path(row["directory"]) / "training" / "Q_induced.npy").is_file()
        if row["family"] in {"global", "local", "weak"}:
            assert row["baseline"] == records["P0"]["directory"]
            assert row["metrics"]["rho"] > 0
            assert "learned_local_distance" in row["metrics"]
    assert (output / "cci_perturbation.pdf").is_file()
    assert (output / "symmetry_perturbation.pdf").is_file()
    assert (output / "symmetry_existence.csv").is_file()
    for stem in ("cci_results_table", "symmetry_existence_table", "symmetry_perturbation_table"):
        for suffix in ("png", "pdf", "csv"):
            assert (output / f"{stem}.{suffix}").stat().st_size > 0
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert {name: value["complete"] for name, value in progress.items()} == {
        "P0": 1, "CCI": 2, "Symmetry-existence": 6, "Symmetry-perturbation": 12,
    }
    subprocess.run([*args, "--resume"], timeout=120, check=True, env={**os.environ, "PYTHONUTF8": "1"})
    assert all(len(list(Path(row["directory"]).parent.glob("attempt-*"))) == 1 for row in records.values())
    scoped = [("symmetry-existence" if value == "all" else value) for value in args]
    subprocess.run(scoped, timeout=120, check=True, env={**os.environ, "PYTHONUTF8": "1"})
    existence_scope = output / "symmetry_existence"
    scoped_state = json.loads((existence_scope / "conditions.json").read_text(encoding="utf-8"))
    assert scoped_state["P0"]["directory"] == records["P0"]["directory"]
    assert (existence_scope / "symmetry_existence_table.pdf").is_file()
    subprocess.run([sys.executable, "-B", "-m", MODULE, "--experiment", "report", "--output-root", str(output)],
                   timeout=120, check=True)
