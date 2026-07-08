from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareESrCosPijMethod(ComparePijMethodBase):
    name = "compare_E_Sr_cos"
    feature_keys = ("E", "Sr")
    pij_key = "cos"
