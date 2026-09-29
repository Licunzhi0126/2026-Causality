from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLSrCostMixEucSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_Sr_costmix_euc_sot"
    component_keys = ("L", "Sr")
    vector_metric = "euclidean"
