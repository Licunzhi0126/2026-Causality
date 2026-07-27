from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareESrCostMixCosSotPijMethod(CompareCostFusionSotBase):
    name = "compare_E_Sr_costmix_cos_sot"
    component_keys = ("E", "Sr")
    vector_metric = "cosine"
