from __future__ import annotations

from typing import Sequence

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels


class LaplacianPijMethod:
    name = "laplacian"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        from mignet_ce.representations.laplacian import build_laplacian_result

        return (
            build_laplacian_result(
                context=context,
                n_components=cfg.laplacian_components,
                normalized=cfg.laplacian_normalized,
            ),
            None,
        )
