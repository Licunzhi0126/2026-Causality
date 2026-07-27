from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareLEucSotPijMethod(CompareCostFusionSotBase):
    name = "compare_L_euc_sot"
    component_keys = ("L",)
    vector_metric = "euclidean"
