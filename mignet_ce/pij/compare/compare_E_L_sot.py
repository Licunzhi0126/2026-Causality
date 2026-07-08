from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareELSotPijMethod(ComparePijMethodBase):
    name = "compare_E_L_sot"
    feature_keys = ("E", "L")
    pij_key = "sot"
