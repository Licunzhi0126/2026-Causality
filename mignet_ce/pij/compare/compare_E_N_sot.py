from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareENSotPijMethod(ComparePijMethodBase):
    name = "compare_E_N_sot"
    feature_keys = ("E", "N")
    pij_key = "sot"
