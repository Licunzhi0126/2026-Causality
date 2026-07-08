from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareSrSotPijMethod(ComparePijMethodBase):
    name = "compare_Sr_sot"
    feature_keys = ("Sr",)
    pij_key = "sot"
