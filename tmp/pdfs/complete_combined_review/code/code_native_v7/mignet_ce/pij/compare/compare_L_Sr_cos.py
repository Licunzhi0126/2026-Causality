from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareLSrCosPijMethod(ComparePijMethodBase):
    name = "compare_L_Sr_cos"
    feature_keys = ("L", "Sr")
    pij_key = "cos"
