from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLSrCostMixCosSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_Sr_costmix_cos_sot"
    component_keys = ("L", "Sr")
    vector_metric = "cosine"
