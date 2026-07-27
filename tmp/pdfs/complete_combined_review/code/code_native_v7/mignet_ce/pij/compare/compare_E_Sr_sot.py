from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareESrSotPijMethod(ComparePijMethodBase):
    name = "compare_E_Sr_sot"
    feature_keys = ("E", "Sr")
    pij_key = "sot"
