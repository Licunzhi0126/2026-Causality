from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNLCosPijMethod(ComparePijMethodBase):
    name = "compare_N_L_cos"
    feature_keys = ("N", "L")
    pij_key = "cos"
