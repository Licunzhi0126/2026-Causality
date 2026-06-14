from __future__ import annotations

from typing import Sequence

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels


class JointNMFPijMethod:
    name = "joint_nmf"

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        from mignet_ce.representations.joint_nmf import build_joint_nmf_result

        return (
            build_joint_nmf_result(
                context=context,
                n_components=cfg.nmf_components,
                max_iter=cfg.nmf_max_iter,
                seed=cfg.nmf_seed,
            ),
            None,
        )
