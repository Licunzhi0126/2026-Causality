from __future__ import annotations

import sys
from pathlib import Path

DATA_FACTORY_LIB = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(DATA_FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(DATA_FACTORY_LIB))

from factory_common import parse_sample_stem  # noqa: E402
from layer_specs import get_domain_layer_spec  # noqa: E402


def test_spatial_domain_layer_specs_are_registered() -> None:
    expected = {
        "spatial_domain_k40": ("spatial_domain", "exact_k", 40, "spatialDomain40"),
        "spatial_domain_k150": ("spatial_domain", "exact_k", 150, "spatialDomain150"),
        "spatial_domain_less_than5": ("spatial_domain", "less_than_5", None, "spatialDomainLessThan5"),
    }
    for layer, (family, mode, k, prefix) in expected.items():
        spec = get_domain_layer_spec(layer)
        assert spec.family == family
        assert spec.mode == mode
        assert spec.k == k
        assert spec.sample_prefix == prefix


def test_parse_sample_stem_accepts_spatial_domain_prefixes() -> None:
    assert parse_sample_stem("spatialDomain40_heart_11.5") == ("heart", "11.5")
    assert parse_sample_stem("spatialDomain150_brain_12.5") == ("brain", "12.5")
    assert parse_sample_stem("spatialDomainLessThan5_lung_14.5") == ("lung", "14.5")
