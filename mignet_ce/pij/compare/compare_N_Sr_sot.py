from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNSrSotPijMethod(ComparePijMethodBase):
    name = "compare_N_Sr_sot"
    feature_keys = ("N", "Sr")
    pij_key = "sot"
