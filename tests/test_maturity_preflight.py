from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mignet_ce.downstream.analysis import preparation


def _cfg(root: Path):
    return SimpleNamespace(
        developmental_feature_root=root,
        data_root=root / "data",
        organ="heart",
    )


def test_maturity_schema_is_rejected_before_training(tmp_path, monkeypatch) -> None:
    source = tmp_path / "spot" / "heart_11.5_features.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"unit_id": ["a", "b"], "other": [0.1, 0.2]}).to_csv(source, index=False)
    monkeypatch.setattr(preparation, "read_index", lambda _path: ["a", "b"])
    monkeypatch.setattr(preparation, "cci_index_path", lambda *_args: tmp_path / "index.txt")
    with pytest.raises(ValueError, match="pseudotime"):
        preparation._validated_maturity_source(_cfg(tmp_path), "11.5")


def test_maturity_schema_requires_complete_unique_finite_ids(tmp_path, monkeypatch) -> None:
    source = tmp_path / "spot" / "heart_11.5_features.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"unit_id": ["a", "b"], "pseudotime": [0.1, 0.8]}).to_csv(source, index=False)
    monkeypatch.setattr(preparation, "read_index", lambda _path: ["a", "b"])
    monkeypatch.setattr(preparation, "cci_index_path", lambda *_args: tmp_path / "index.txt")
    kind, resolved, id_column, value_column = preparation._validated_maturity_source(
        _cfg(tmp_path), "11.5"
    )
    assert kind == "developmental_features"
    assert resolved == source
    assert (id_column, value_column) == ("unit_id", "pseudotime")
