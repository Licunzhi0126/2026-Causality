from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareESotPijMethod(ComparePijMethodBase):
    name = "compare_E_sot"
    feature_keys = ("E",)
    pij_key = "sot"
