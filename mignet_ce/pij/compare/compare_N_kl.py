from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNKlPijMethod(ComparePijMethodBase):
    name = "compare_N_kl"
    feature_keys = ("N",)
    pij_key = "kl"
