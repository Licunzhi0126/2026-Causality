from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNCosPijMethod(ComparePijMethodBase):
    name = "compare_N_cos"
    feature_keys = ("N",)
    pij_key = "cos"
