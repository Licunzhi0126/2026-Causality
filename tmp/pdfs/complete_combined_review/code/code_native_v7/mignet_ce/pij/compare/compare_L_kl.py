from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareLKlPijMethod(ComparePijMethodBase):
    name = "compare_L_kl"
    feature_keys = ("L",)
    pij_key = "kl"
