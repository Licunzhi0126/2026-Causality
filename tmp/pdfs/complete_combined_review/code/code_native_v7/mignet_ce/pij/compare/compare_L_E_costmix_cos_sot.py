from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLECostMixCosSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_E_costmix_cos_sot"
    component_keys = ("L", "E")
    vector_metric = "cosine"
