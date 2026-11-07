"""Differentiable CQ-only objective terms and effective-signal selection."""
import math

import torch


def cq_safe(retained: torch.Tensor, available: torch.Tensor, signal_threshold: float) -> torch.Tensor:
    return retained / available.clamp_min(float(signal_threshold))


def available_information_floor(available: torch.Tensor, signal_threshold: float) -> torch.Tensor:
    threshold = available.new_tensor(float(signal_threshold)).clamp_min(1e-12)
    return torch.relu((threshold - available) / threshold).pow(2)


def retained_information_floor(retained: torch.Tensor, reference: float | None) -> torch.Tensor:
    if reference is None:
        return retained.new_zeros(())
    denominator = retained.new_tensor(float(reference)).clamp_min(1e-12)
    return torch.relu((denominator - retained) / denominator).pow(2)


def cq_checkpoint_eligible(*, available: float, retained: float, cq: float,
                           signal_threshold: float, keff_t: float, keff_tp: float) -> bool:
    tolerance = 1e-8
    return bool(
        all(math.isfinite(x) for x in (available, retained, cq, keff_t, keff_tp))
        and available >= signal_threshold - tolerance
        and retained >= -tolerance and cq >= -tolerance
    )


def cq_low_signal_checkpoint_eligible(*, available: float, retained: float, cq: float,
                                      signal_threshold: float, keff_t: float, keff_tp: float) -> bool:
    """Allow finite positive-information candidates below the strict signal floor.

    Keff remains diagnostic. Ranking uses the existing cq_safe training score,
    so a vanishing denominator cannot inflate the fallback score.
    """
    tolerance = 1e-8
    return bool(
        all(math.isfinite(x) for x in (available, retained, cq, keff_t, keff_tp))
        and tolerance < available < signal_threshold - tolerance
        and retained >= -tolerance and cq >= -tolerance
    )


def cq_only_loss(values: dict, config, signal_threshold: float) -> tuple[torch.Tensor, dict]:
    cq = cq_safe(values["I_retained"], values["I_available"], signal_threshold)
    available_floor = available_information_floor(values["I_available"], signal_threshold)
    retained_floor = retained_information_floor(values["I_retained"], config.retained_floor_abs)
    common = sum(getattr(config, "lambda_" + key) * values[key]
                 for key in ("align", "var", "local", "sharp", "dev", "proto"))
    loss = (common - config.lambda_cq * cq
            + config.lambda_within * values["within"] + config.lambda_inter * values["inter"]
            + config.lambda_closure * values["closure"]
            + config.lambda_available_floor * available_floor
            + config.lambda_retained_floor_abs * retained_floor)
    return loss, {"CQ": cq, "L_available_floor": available_floor, "L_retained_floor_abs": retained_floor}
