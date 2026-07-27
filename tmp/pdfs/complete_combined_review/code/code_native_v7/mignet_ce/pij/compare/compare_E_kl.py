from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareEKlPijMethod(ComparePijMethodBase):
    name = "compare_E_kl"
    feature_keys = ("E",)
    pij_key = "kl"
