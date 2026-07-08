from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareESrKlPijMethod(ComparePijMethodBase):
    name = "compare_E_Sr_kl"
    feature_keys = ("E", "Sr")
    pij_key = "kl"
