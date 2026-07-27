from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareELKlPijMethod(ComparePijMethodBase):
    name = "compare_E_L_kl"
    feature_keys = ("E", "L")
    pij_key = "kl"
