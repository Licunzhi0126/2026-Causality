from __future__ import annotations

import matplotlib as mpl

from mignet_ce.downstream.analysis.config import (
    MAPPING_COMPLETE,
    MAPPING_K150,
    MAPPING_K40,
    MAPPING_MATURITY,
    MAPPING_TWO_STAGE,
)
from mignet_ce.downstream.analysis.mappings import COLORS, MARKERS
from mignet_ce.downstream.analysis.visualization.style import set_publication_style


def test_original_mapping_visual_language_is_preserved_and_two_stage_is_distinct() -> None:
    assert COLORS[MAPPING_K150] == "#2F74B8"
    assert COLORS[MAPPING_K40] == "#D45959"
    assert COLORS[MAPPING_COMPLETE] == "#4D9A73"
    assert COLORS[MAPPING_MATURITY] == "#8B5FBF"
    assert MARKERS[MAPPING_K150] == "o"
    assert MARKERS[MAPPING_K40] == "s"
    assert MARKERS[MAPPING_COMPLETE] == "D"
    assert MARKERS[MAPPING_MATURITY] == "^"
    assert COLORS[MAPPING_TWO_STAGE] not in {
        COLORS[MAPPING_K150],
        COLORS[MAPPING_K40],
        COLORS[MAPPING_COMPLETE],
        COLORS[MAPPING_MATURITY],
    }
    assert MARKERS[MAPPING_TWO_STAGE] == "P"


def test_publication_export_style_is_preserved() -> None:
    set_publication_style()
    assert mpl.rcParams["figure.dpi"] == 140
    assert mpl.rcParams["savefig.dpi"] == 300
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
