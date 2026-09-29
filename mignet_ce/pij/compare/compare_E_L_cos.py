from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareELCosPijMethod(ComparePijMethodBase):
    name = "compare_E_L_cos"
    feature_keys = ("E", "L")
    pij_key = "cos"
