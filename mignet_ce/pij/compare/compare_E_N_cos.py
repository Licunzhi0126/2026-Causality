from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareENCosPijMethod(ComparePijMethodBase):
    name = "compare_E_N_cos"
    feature_keys = ("E", "N")
    pij_key = "cos"
