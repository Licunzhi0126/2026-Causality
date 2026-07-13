from mignet_ce.pij.compare.cost_fusion import CompareCostFusionSotBase


class CompareEEucSotPijMethod(CompareCostFusionSotBase):
    name = "compare_E_euc_sot"
    component_keys = ("E",)
    vector_metric = "euclidean"
