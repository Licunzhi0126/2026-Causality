from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLESrCostMixEucSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_E_Sr_costmix_euc_sot"
    component_keys = ("L", "E", "Sr")
    vector_metric = "euclidean"
