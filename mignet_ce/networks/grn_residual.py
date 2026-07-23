from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PearsonResidualConfig:
    """Configuration for the fixed, no-cross-time GRN residual transform."""

    theta: float = 1.0
    positive_only: bool = True
    eps: float = 1e-8


def positive_pearson_residual(
    expression: np.ndarray,
    *,
    config: PearsonResidualConfig = PearsonResidualConfig(),
) -> np.ndarray:
    """Transform one layer at one time point without using paired-time data.

    Parameters
    ----------
    expression:
        Nonnegative unit-by-gene matrix for exactly one layer and one time.
    config:
        Fixed transform settings. The default theta=1 is the frozen
        development setting used in the experiment.

    Notes
    -----
    This function never receives CCI, PIJ, EI, a target-time matrix, or a
    source-target matching. Therefore it cannot use those quantities.
    """

    values = np.maximum(
        np.nan_to_num(
            np.asarray(expression, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )
    if values.ndim != 2:
        raise ValueError(f"expression must be 2D; got shape {values.shape}.")
    if config.theta <= 0.0:
        raise ValueError("theta must be positive.")
    if config.eps <= 0.0:
        raise ValueError("eps must be positive.")

    total = float(values.sum())
    if total <= config.eps:
        return np.zeros_like(values)

    row_mass = values.sum(axis=1, keepdims=True)
    column_mass = values.sum(axis=0, keepdims=True)
    expected = (row_mass @ column_mass) / total
    denominator = np.sqrt(
        expected + (expected * expected) / float(config.theta) + config.eps
    )
    residual = np.nan_to_num(
        (values - expected) / denominator,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if config.positive_only:
        residual = np.maximum(residual, 0.0)
    return residual
