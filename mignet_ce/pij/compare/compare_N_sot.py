from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNSotPijMethod(ComparePijMethodBase):
    name = "compare_N_sot"
    feature_keys = ("N",)
    pij_key = "sot"
