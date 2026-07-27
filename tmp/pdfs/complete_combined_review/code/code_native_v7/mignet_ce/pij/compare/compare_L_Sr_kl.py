from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareLSrKlPijMethod(ComparePijMethodBase):
    name = "compare_L_Sr_kl"
    feature_keys = ("L", "Sr")
    pij_key = "kl"
