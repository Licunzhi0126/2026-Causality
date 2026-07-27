from __future__ import annotations

"""Typed boundary between method-specific preparation and DeltaEI training."""

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import scipy.sparse as sp


MacroPijBuilder = Callable[["MacroPijInputs"], Any]


@dataclass(frozen=True)
class MacroPijInputs:
    """Differentiable macro-level inputs assembled during one training step."""

    z_macro_t: Any
    z_macro_tp: Any
    network_macro_t: Any
    network_macro_tp: Any
    feature_blocks_t: Mapping[str, Any] = field(default_factory=dict)
    feature_blocks_tp: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedCoarseInput:
    """Validated, method-neutral input consumed by the shared coarse-grain core."""

    method: str
    unit_ids_t: list[str]
    unit_ids_tp: list[str]
    network_t: sp.csr_matrix
    network_tp: sp.csr_matrix
    encoder_features_t: np.ndarray
    encoder_features_tp: np.ndarray
    micro_features_t: np.ndarray
    micro_features_tp: np.ndarray
    micro_pij: np.ndarray
    micro_ei: float
    macro_pij_builder: MacroPijBuilder
    feature_blocks_t: Mapping[str, np.ndarray] = field(default_factory=dict)
    feature_blocks_tp: Mapping[str, np.ndarray] = field(default_factory=dict)
    coords_t: np.ndarray | None = None
    coords_tp: np.ndarray | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        n_t = len(self.unit_ids_t)
        n_tp = len(self.unit_ids_tp)
        if not self.method:
            raise ValueError("PreparedCoarseInput.method cannot be empty.")
        if n_t == 0 or n_tp == 0:
            raise ValueError("PreparedCoarseInput requires non-empty unit IDs at both times.")
        if len(set(self.unit_ids_t)) != n_t or len(set(self.unit_ids_tp)) != n_tp:
            raise ValueError("PreparedCoarseInput unit IDs must be unique within each time.")
        if self.network_t.shape != (n_t, n_t) or self.network_tp.shape != (n_tp, n_tp):
            raise ValueError(
                "Network shapes must match unit counts; "
                f"got {self.network_t.shape}, {self.network_tp.shape} for {n_t}, {n_tp}."
            )
        for name, values, expected_rows in (
            ("encoder_features_t", self.encoder_features_t, n_t),
            ("encoder_features_tp", self.encoder_features_tp, n_tp),
            ("micro_features_t", self.micro_features_t, n_t),
            ("micro_features_tp", self.micro_features_tp, n_tp),
        ):
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] == 0:
                raise ValueError(
                    f"{name} must have shape ({expected_rows}, d>0); got {array.shape}."
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values.")
        pij = np.asarray(self.micro_pij)
        if pij.shape != (n_t, n_tp):
            raise ValueError(f"micro_pij must have shape {(n_t, n_tp)}; got {pij.shape}.")
        if not np.isfinite(pij).all() or np.any(pij < 0.0):
            raise ValueError("micro_pij must contain finite non-negative values.")
        if not np.allclose(pij.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("micro_pij must be row-stochastic.")
        if not np.isfinite(float(self.micro_ei)):
            raise ValueError("micro_ei must be finite.")
        if set(self.feature_blocks_t) != set(self.feature_blocks_tp):
            raise ValueError("Feature-block keys must match across time.")
        for key in self.feature_blocks_t:
            source = np.asarray(self.feature_blocks_t[key])
            target = np.asarray(self.feature_blocks_tp[key])
            if source.ndim != 2 or target.ndim != 2:
                raise ValueError(f"Feature block {key!r} must be 2D.")
            if source.shape[0] != n_t or target.shape[0] != n_tp:
                raise ValueError(f"Feature block {key!r} row counts do not match units.")
            if source.shape[1] != target.shape[1]:
                raise ValueError(f"Feature block {key!r} dimensions differ across time.")
        for name, coords, expected_rows in (
            ("coords_t", self.coords_t, n_t),
            ("coords_tp", self.coords_tp, n_tp),
        ):
            if coords is not None:
                array = np.asarray(coords)
                if array.ndim != 2 or array.shape[0] != expected_rows:
                    raise ValueError(f"{name} row count does not match units.")

    def manifest(self) -> dict[str, object]:
        return {
            "method": self.method,
            "unit_count_t": len(self.unit_ids_t),
            "unit_count_tp": len(self.unit_ids_tp),
            "network_shape_t": list(self.network_t.shape),
            "network_shape_tp": list(self.network_tp.shape),
            "encoder_feature_shape_t": list(self.encoder_features_t.shape),
            "encoder_feature_shape_tp": list(self.encoder_features_tp.shape),
            "micro_feature_shape_t": list(self.micro_features_t.shape),
            "micro_feature_shape_tp": list(self.micro_features_tp.shape),
            "micro_pij_shape": list(self.micro_pij.shape),
            "micro_ei": float(self.micro_ei),
            "feature_blocks": {
                key: {
                    "shape_t": list(np.asarray(self.feature_blocks_t[key]).shape),
                    "shape_tp": list(np.asarray(self.feature_blocks_tp[key]).shape),
                }
                for key in self.feature_blocks_t
            },
            "coords_available_t": self.coords_t is not None,
            "coords_available_tp": self.coords_tp is not None,
            "provenance": dict(self.provenance),
        }
