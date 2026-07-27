from __future__ import annotations

from mignet_ce.pij.compare.common import ComparePijMethodBase


class CompareLSotPijMethod(ComparePijMethodBase):
    name = "compare_L_sot"
    feature_keys = ("L",)
    pij_key = "sot"
