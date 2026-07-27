from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareECosPijMethod(ComparePijMethodBase):
    name = "compare_E_cos"
    feature_keys = ("E",)
    pij_key = "cos"
