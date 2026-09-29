from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


TimePair = Tuple[int, int]
PairFeatures = Dict[TimePair, Tuple[np.ndarray, np.ndarray]]

if TYPE_CHECKING:
    from mignet_ce.config import TemporalRunConfig
    from mignet_ce.networks.base import NetworkContext


@dataclass
class MethodResult:
    lower_features: List[np.ndarray]
    upper_features: List[np.ndarray]
    lower_coords: Optional[List[np.ndarray]] = None
    upper_coords: Optional[List[np.ndarray]] = None
    method_metadata: dict[str, object] = field(default_factory=dict)
    pairwise_lower_features: Optional[PairFeatures] = None
    pairwise_upper_features: Optional[PairFeatures] = None


@dataclass
class TransitionKernels:
    p_lower: Dict[TimePair, np.ndarray] = field(default_factory=dict)
    p_upper: Dict[TimePair, np.ndarray] = field(default_factory=dict)
    kernel_metadata: dict[str, object] = field(default_factory=dict)
    kernel_diagnostics: dict[str, Dict[TimePair, dict[str, np.ndarray]]] = field(
        default_factory=lambda: {"lower": {}, "upper": {}}
    )


class PijMethod(Protocol):
    name: str

    def run(
        self,
        context: "NetworkContext",
        cfg: "TemporalRunConfig",
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        ...
