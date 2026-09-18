from __future__ import annotations

ASSET_NAMES = ("table1", "table2", "figure1", "table3", "figure2")


def normalize_asset_names(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not values or "all" in values:
        return ASSET_NAMES
    unknown = sorted(set(values) - set(ASSET_NAMES))
    if unknown:
        raise ValueError(f"Unknown asset names {unknown}; choose from {list(ASSET_NAMES)} or 'all'.")
    requested = set(values)
    return tuple(name for name in ASSET_NAMES if name in requested)
