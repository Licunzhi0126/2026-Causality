from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNSrKlPijMethod(ComparePijMethodBase):
    name = "compare_N_Sr_kl"
    feature_keys = ("N", "Sr")
    pij_key = "kl"
