from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLESrCostMixCosSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_E_Sr_costmix_cos_sot"
    component_keys = ("L", "E", "Sr")
    vector_metric = "cosine"
