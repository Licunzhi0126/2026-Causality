"""Read explicit source/target roles from an existing manifest."""

from __future__ import annotations

import json
from pathlib import Path


def load_source_target(prepared_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Return the one source and one target sample from an existing manifest."""
    manifest = Path(prepared_root) / "metadata" / "dataset_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("samples", [])
    source = [row for row in rows if row.get("role") == "source"]
    target = [row for row in rows if row.get("role") == "target"]
    if len(source) != 1 or len(target) != 1:
        raise ValueError(
            f"Expected exactly one source and one target in {manifest}; "
            f"found source={len(source)}, target={len(target)}. "
            "Provide an existing manifest with explicit source and target roles."
        )
    return source[0], target[0]
