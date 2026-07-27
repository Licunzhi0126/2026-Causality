from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareSrCosPijMethod(ComparePijMethodBase):
    name = "compare_Sr_cos"
    feature_keys = ("Sr",)
    pij_key = "cos"
