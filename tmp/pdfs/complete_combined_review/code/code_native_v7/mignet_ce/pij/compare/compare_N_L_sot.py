from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNLSotPijMethod(ComparePijMethodBase):
    name = "compare_N_L_sot"
    feature_keys = ("N", "L")
    pij_key = "sot"
