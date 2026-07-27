from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareSrKlPijMethod(ComparePijMethodBase):
    name = "compare_Sr_kl"
    feature_keys = ("Sr",)
    pij_key = "kl"
