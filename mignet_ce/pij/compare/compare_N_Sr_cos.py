from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNSrCosPijMethod(ComparePijMethodBase):
    name = "compare_N_Sr_cos"
    feature_keys = ("N", "Sr")
    pij_key = "cos"
