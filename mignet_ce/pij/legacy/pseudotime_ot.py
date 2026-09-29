from __future__ import annotations

from typing import Sequence

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.legacy._developmental_ot_components import (
    build_pseudotime_cost,
    developmental_metadata,
    make_developmental_table_loader,
)
from mignet_ce.pij.legacy._ot_common import run_ot_pij_method
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels


class PseudotimeOTPijMethod:
    name = "pseudotime_ot"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        table = make_developmental_table_loader(context, cfg)

        def component_builder(
            source_features: np.ndarray,
            target_features: np.ndarray,
            source_coords: np.ndarray | None,
            target_coords: np.ndarray | None,
            space: str,
            t0: int,
            t1: int,
        ):
            source_table = table(space, t0)
            target_table = table(space, t1)
            cost = build_pseudotime_cost(source_table, target_table, self.name)
            metadata = developmental_metadata(source_table, target_table, ["pseudotime"], cfg)
            return {"pseudotime": cost}, {"pseudotime": 1.0}, metadata

        return run_ot_pij_method(
            context=context,
            cfg=cfg,
            pairs=pairs,
            method_name=self.name,
            component_builder=component_builder,
        )
