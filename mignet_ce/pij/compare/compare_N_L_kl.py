from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareNLKlPijMethod(ComparePijMethodBase):
    name = "compare_N_L_kl"
    feature_keys = ("N", "L")
    pij_key = "kl"
