from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLECostMixEucSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_E_costmix_euc_sot"
    component_keys = ("L", "E")
    vector_metric = "euclidean"
