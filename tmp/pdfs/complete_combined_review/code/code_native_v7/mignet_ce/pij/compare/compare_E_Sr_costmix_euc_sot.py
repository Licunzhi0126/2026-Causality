from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareESrCostMixEucSotPijMethod(CompareCostFusionSotBase):
    name = "compare_E_Sr_costmix_euc_sot"
    component_keys = ("E", "Sr")
    vector_metric = "euclidean"
