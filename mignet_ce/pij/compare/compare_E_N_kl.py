from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareENKlPijMethod(ComparePijMethodBase):
    name = "compare_E_N_kl"
    feature_keys = ("E", "N")
    pij_key = "kl"
