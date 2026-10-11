from __future__ import annotations

from mignet_ce.downstream.perturbation.validation import validate_maturity


def test_validate_maturity_accepts_official_feature_columns(tmp_path) -> None:
    maturity = tmp_path / "features.csv"
    maturity.write_text("unit_id,pseudotime\na,0.1\nb,0.2\n", encoding="utf-8")
    validate_maturity(maturity, ["a", "b"])
