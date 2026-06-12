from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


TimePair = Tuple[int, int]
PairFeatures = Dict[TimePair, Tuple[np.ndarray, np.ndarray]]


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
