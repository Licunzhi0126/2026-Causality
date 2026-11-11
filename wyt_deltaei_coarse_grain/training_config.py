"""One configuration resolver for training and saved-run reuse audits."""
from dataclasses import fields

from mignet_ce.coarse_frontends.method_specs import (
    CQ_ONLY, DYNAMIC_CLOSURE_TWO_STAGE, NO_KEFF_EI_RETAIN_RATIOS,
    get_coarse_method_spec,
)


def build_training_config(args):
    from wyt_deltaei_coarse_grain.trainer import WYTDeltaEIConfig
    from wyt_deltaei_coarse_grain.two_stage_trainer import WYTTwoStageDeltaEIConfig
    from wyt_deltaei_coarse_grain.two_stage_trainer import WYTFormalTwoStageConfig
    from wyt_deltaei_coarse_grain.cq_only_trainer import WYTCQOnlyConfig

    spec = get_coarse_method_spec(args.method)
    config_type = (WYTCQOnlyConfig if spec.training_mode == CQ_ONLY else
                   WYTTwoStageDeltaEIConfig if spec.training_mode == DYNAMIC_CLOSURE_TWO_STAGE else
                   WYTDeltaEIConfig)
    if args.method == "formal_two_stage":
        config_type = WYTFormalTwoStageConfig
    values = {field.name: getattr(args, field.name) for field in fields(config_type)
              if getattr(args, field.name, None) is not None}
    values["lambda_dev"] = args.lambda_dev if args.lambda_dev is not None else spec.default_lambda_dev
    if args.method == "formal_two_stage":
        values.update(lambda_var=0.0, lambda_local=0.0, lambda_proto=0.0, lambda_sharp=0.0,
                      lambda_dead_usage=0.0, lambda_within=0.0, lambda_inter=0.0,
                      report_hard_occupancy=True)
    if args.method in NO_KEFF_EI_RETAIN_RATIOS:
        values.update(keff_policy="iterative_no_floor")
        for name, value in (("lambda_keff", 0.0), ("lambda_dead_usage", 0.0),
                            ("ei_retain_ratio", NO_KEFF_EI_RETAIN_RATIOS[args.method])):
            values.setdefault(name, value)
        if values["lambda_keff"] != 0 or values["lambda_dead_usage"] != 0 or any(
            values.get(name) is not None for name in ("keff_min", "checkpoint_keff_min")
        ):
            raise ValueError("No-Keff methods require zero Keff/dead-usage weights and no Keff floors.")
    if spec.training_mode == CQ_ONLY:
        unsupported = (
            "stage1_ratio", "ei_retain_ratio", "lambda_ei_constraint", "lambda_retain_norm",
            "retain_floor_ratio", "lambda_retain_floor", "low_signal_pair_weight",
            "low_signal_threshold_bits", "checkpoint_retain_ratio",
        )
        for name in unsupported:
            if getattr(args, name, None) is not None:
                raise ValueError(f"CQ-only does not support --{name.replace('_', '-')}")
        if values.get("lambda_keff", 0) != 0 or values.get("lambda_dead_usage", 0) != 0 or any(
            values.get(name) is not None for name in ("keff_min", "checkpoint_keff_min")
        ):
            raise ValueError("CQ-only requires zero Keff/dead-usage weights and no Keff floors.")
    return config_type(**values)
