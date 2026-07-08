from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareLCosPijMethod(ComparePijMethodBase):
    name = "compare_L_cos"
    feature_keys = ("L",)
    pij_key = "cos"
