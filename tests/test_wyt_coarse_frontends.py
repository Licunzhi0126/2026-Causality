from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from mignet_ce.coarse_frontends.registry import COARSE_FRONTEND_REGISTRY
from mignet_ce.coarse_frontends.method_specs import (
    DYNAMIC_CLOSURE_TWO_STAGE,
    LEGACY_SINGLE_STAGE, CQ_ONLY,
    FORMAL_TWO_STAGE_PROTOCOL, get_coarse_method_spec, training_mode_for_method,
)
from mignet_ce.representations.coarse_input import PreparedCoarseInput


def test_coarse_frontend_registry_has_exact_planned_methods() -> None:
    assert set(COARSE_FRONTEND_REGISTRY) == {
        "formal_two_stage",
        "complete_combined_coarse",
        "complete_combined_coarse_maturity_cci_grn",
        "maturity_cci_grn_two_stage",
        "two_stage_noKeff", "two_stage_noKeff_HardEI", "single_stage_cq_only",
    }


def test_prepared_coarse_input_rejects_nonstochastic_pij() -> None:
    prepared = PreparedCoarseInput(
        method="synthetic",
        unit_ids_t=["a", "b"],
        unit_ids_tp=["c", "d"],
        network_t=sp.eye(2, format="csr"),
        network_tp=sp.eye(2, format="csr"),
        encoder_features_t=np.ones((2, 2)),
        encoder_features_tp=np.ones((2, 2)),
        micro_features_t=np.ones((2, 2)),
        micro_features_tp=np.ones((2, 2)),
        micro_pij=np.ones((2, 2)),
        micro_ei=0.0,
        macro_pij_builder=lambda _: None,
    )
    with pytest.raises(ValueError, match="row-stochastic"):
        prepared.validate()


def test_training_modes_are_method_isolated() -> None:
    assert training_mode_for_method("formal_two_stage") == DYNAMIC_CLOSURE_TWO_STAGE
    assert get_coarse_method_spec("formal_two_stage").objective_version == FORMAL_TWO_STAGE_PROTOCOL == "formal_two_stage_main"
    assert training_mode_for_method("complete_combined_coarse") == LEGACY_SINGLE_STAGE
    assert (
        training_mode_for_method("complete_combined_coarse_maturity_cci_grn")
        == LEGACY_SINGLE_STAGE
    )
    assert training_mode_for_method("maturity_cci_grn_two_stage") == DYNAMIC_CLOSURE_TWO_STAGE
    for method in ("two_stage_noKeff", "two_stage_noKeff_HardEI"):
        assert training_mode_for_method(method) == DYNAMIC_CLOSURE_TWO_STAGE
    assert training_mode_for_method("single_stage_cq_only") == CQ_ONLY


@pytest.mark.parametrize("method", ["shr_method_8", "shr_method_10"])
def test_retired_methods_are_rejected_by_metadata_frontend_and_dispatch(tmp_path, method):
    from types import SimpleNamespace
    from mignet_ce.coarse_frontends import get_coarse_method_spec, prepare_coarse_input
    from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei

    with pytest.raises(ValueError, match="retired"):
        get_coarse_method_spec(method)
    with pytest.raises(ValueError, match="Unknown coarse frontend"):
        prepare_coarse_input(method, SimpleNamespace(grn_feature_method="legacy",
                                                    cci_feature_method="shared_nmf", alpha_cci=.01))
    with pytest.raises(ValueError, match="retired"):
        train_deltaei(SimpleNamespace(method=method), WYTDeltaEIConfig(k=2, out_dir=tmp_path / "run"))
    assert not (tmp_path / "run").exists()
