from dataclasses import replace
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from test_shr_method_8 import prepared_input, single_cpu_thread
from mignet_ce.coarse_frontends import COARSE_METHOD_SPECS, prepare_coarse_input
from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import prepare as prepare_original
from mignet_ce.representations.coarse_input import MacroPijInputs
from scripts import run_wyt_deltaei_coarse_grain as cli
from scripts import run_multiscale_maturity_cci_grn as multiscale
from wyt_deltaei_coarse_grain.macro_builder import macro_matrix, pool_feature_blocks, pool_to_macro


def _cli_args(method, out_dir):
    return ["--method", method, "--h5ad-t", "t.h5ad", "--h5ad-tp", "tp.h5ad",
            "--cci-t", "t.npz", "--cci-tp", "tp.npz", "--grn-t", "t.csv", "--grn-tp", "tp.csv",
            "--maturity-t", "mt.csv", "--maturity-tp", "mp.csv", "--k", "3", "--out-dir", str(out_dir)]


@pytest.mark.parametrize("method", ["shr_method_8", "shr_method_10"])
def test_cli_builds_selected_config_and_runs_local_trainer(tmp_path, monkeypatch, method):
    import mignet_ce.coarse_frontends as frontends
    prepared = prepared_input(method)
    monkeypatch.setattr(frontends, "prepare_coarse_input", lambda selected, request: prepared)
    cli.main(_cli_args(method, tmp_path / method) + [
        "--epochs", "2", "--hidden-dim", "16", "--temperature", ".7", "--knn-k", "3",
        "--lambda-retain", ".4", "--log-every", "1",
    ])
    assert (tmp_path / method / "summary.json").is_file()
    assert not any("SHR.zip" in str(getattr(module, "__file__", "")) for module in sys.modules.values())
    args = cli.build_argparser().parse_args(_cli_args(method, tmp_path / "config"))
    config = cli.build_shr_config(args, .05)
    assert config.training_mode == method
    assert config.objective_version == COARSE_METHOD_SPECS[method].objective_version
    assert config.lambda_retain == .5
    if method == "shr_method_8":
        assert config.graph_projection_dim == 32
    else:
        assert config.attention_heads == 4


def _write_stage(tmp_path, label, count, seed):
    rng = np.random.default_rng(seed)
    units = [f"{label}_{i}" for i in range(count)]
    genes = [f"g{i}" for i in range(8)]
    h5ad = tmp_path / f"{label}.h5ad"
    with h5py.File(h5ad, "w") as handle:
        handle.create_group("obs").create_dataset("_index", data=np.array(units, dtype="S"))
        handle.create_group("var").create_dataset("_index", data=np.array(genes, dtype="S"))
        handle.create_dataset("X", data=rng.poisson(4, (count, len(genes))).astype(np.float32))
        handle.create_group("obsm").create_dataset("spatial", data=rng.normal(size=(count, 2)))
    cci = tmp_path / f"{label}_CCI_total.npz"
    sp.save_npz(cci, sp.csr_matrix(rng.random((count, count)).astype(np.float32)))
    index = tmp_path / f"{label}_index.tsv"
    pd.DataFrame({"spot_id": units}).to_csv(index, sep="\t", index=False)
    grn = tmp_path / f"{label}_grn.csv"
    pd.DataFrame({"regulator": genes, "target": genes[1:] + genes[:1],
                  "weight": rng.random(len(genes)) + .1}).to_csv(grn, index=False)
    maturity = tmp_path / f"{label}_maturity.csv"
    pd.DataFrame({"spot_id": units, "maturity": np.linspace(0., 1., count)}).iloc[::-1].to_csv(maturity, index=False)
    return h5ad, cci, index, grn, maturity


def test_real_file_formats_share_input_protocol_and_differentiable_macro_builder(tmp_path):
    source = _write_stage(tmp_path, "source", 18, 5)
    target = _write_stage(tmp_path, "target", 21, 6)
    request = CoarseFrontendRequest(
        h5ad_t=source[0], h5ad_tp=target[0], cci_t=source[1], cci_tp=target[1],
        cci_index_t=source[2], cci_index_tp=target[2], grn_t=source[3], grn_tp=target[3],
        maturity_t=source[4], maturity_tp=target[4],
        grn_feature_method="legacy_nmf", cci_feature_method="laplacian_hks", alpha_cci=.20,
        mid_dim=8, nmf_max_iter=3,
    )
    original = prepare_original(request)
    for method in ("shr_method_8", "shr_method_10"):
        prepared = prepare_coarse_input(method, request)
        assert prepared.method == method
        for key in ("encoder_features_t", "encoder_features_tp", "micro_features_t", "micro_pij", "maturity_t"):
            np.testing.assert_array_equal(getattr(prepared, key), getattr(original, key))
        np.testing.assert_allclose(prepared.maturity_t, np.linspace(0., 1., 18), atol=1e-6)
        assert prepared.provenance["coarse_method"] == method
        left_logits, right_logits = torch.randn(18, 3, requires_grad=True), torch.randn(21, 3, requires_grad=True)
        left, right = left_logits.softmax(1), right_logits.softmax(1)
        blocks_t = {key: torch.tensor(values) for key, values in prepared.feature_blocks_t.items()}
        blocks_tp = {key: torch.tensor(values) for key, values in prepared.feature_blocks_tp.items()}
        macro = prepared.macro_pij_builder(MacroPijInputs(
            pool_to_macro(torch.tensor(prepared.micro_features_t), left),
            pool_to_macro(torch.tensor(prepared.micro_features_tp), right),
            macro_matrix(torch.tensor(prepared.network_t.toarray()), left),
            macro_matrix(torch.tensor(prepared.network_tp.toarray()), right),
            pool_feature_blocks(blocks_t, left), pool_feature_blocks(blocks_tp, right),
        ))
        assert macro.shape == (3, 3) and torch.isfinite(macro).all()
        torch.testing.assert_close(macro.sum(1), torch.ones(3))
        macro[:, 0].sum().backward()
        assert torch.isfinite(left_logits.grad).all() and torch.isfinite(right_logits.grad).all()


def test_multiscale_method_identity_and_command_paths(tmp_path):
    from test_multiscale_maturity_orchestrator import _prepared
    source, target = [_prepared(tmp_path, "spot", time, 50) for time in ("11.5", "12.5")]
    for method in ("shr_method_8", "shr_method_10"):
        assert method in multiscale.SUPPORTED_METHODS
        specs = multiscale.build_run_specs([source, target], method=method, out_root=tmp_path / "runs",
                                            scales=["spot"], pairs=[("11.5", "12.5")],
                                            k_by_scale={"spot": [40]}, seeds=[42])
        command = multiscale.runner_command(specs[0], ["--epochs", "2"])
        assert command[command.index("--method") + 1] == method
        assert specs[0].run_dir == tmp_path / "runs" / method / "spot" / "K40" / "11.5_to_12.5"
        assert specs[0].frontend == method and specs[0].training_mode == method
