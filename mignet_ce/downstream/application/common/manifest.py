"""Read the explicit source/target roles written by Plan 1."""

from __future__ import annotations

import json
from pathlib import Path


def load_source_target(plan1_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Return the one source and one target sample from a Plan 1 manifest."""
    manifest = Path(plan1_root) / "metadata" / "dataset_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("samples", [])
    source = [row for row in rows if row.get("role") == "source"]
    target = [row for row in rows if row.get("role") == "target"]
    if len(source) != 1 or len(target) != 1:
        raise ValueError(
            f"Expected exactly one source and one target in {manifest}; "
            f"found source={len(source)}, target={len(target)}. "
            "Re-run Plan 1 with --source-sample and --target-sample."
        )
    return source[0], target[0]
