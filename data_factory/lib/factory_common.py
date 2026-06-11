from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


RAW_E1S1_ROOT = Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1")
FACTORY_OUTPUT_ROOT = Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory")
COMMOT_REFERENCE_DIR = Path("/home/jovyan/work/2026 Causality/COMMOT/reference/COMMOT")

STAGES: Tuple[str, ...] = ("11.5", "12.5", "13.5", "14.5")
ORGANS: Tuple[str, ...] = ("heart", "brain", "lung")
ORGAN_LABELS: Dict[str, Tuple[str, ...]] = {
    "heart": ("Heart",),
    "brain": ("Brain",),
    "lung": ("Lung", "Lung primordium"),
}

AUXILIARY_H5AD_SUFFIXES: Tuple[str, ...] = (
    "_spots_with_domain.h5ad",
    "_COMMOT.h5ad",
)

SAMPLE_RE = re.compile(
    r"^(?P<prefix>spot|organ|seurat|louvain150|louvain1100|louvainLessThan5)_(?P<organ>[A-Za-z]+)_(?P<stage>\d+(?:\.\d+)?)$"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def raw_stage_path(input_root: Path, stage: str) -> Path:
    return input_root / f"E{stage}_E1S1.MOSTA.h5ad"


def normalize_organ(value: str) -> str:
    organ = str(value).strip().lower()
    if organ not in ORGAN_LABELS:
        raise ValueError(f"Unsupported organ: {value!r}. Expected one of {sorted(ORGAN_LABELS)}.")
    return organ


def parse_sample_stem(stem: str, parent_name: Optional[str] = None) -> Tuple[str, str]:
    match = SAMPLE_RE.match(stem)
    if match:
        return match.group("organ").lower(), match.group("stage")
    if parent_name and parent_name.lower() in ORGAN_LABELS:
        stage_match = re.search(r"(\d+(?:\.\d+)?)", stem)
        if stage_match:
            return parent_name.lower(), stage_match.group(1)
    raise ValueError(f"Cannot parse organ/stage from sample stem: {stem!r}")


def iter_h5ad_files(input_root: Path, exclude_auxiliary: bool = True) -> Iterator[Path]:
    suffixes = AUXILIARY_H5AD_SUFFIXES if exclude_auxiliary else tuple()
    for path in sorted(input_root.rglob("*.h5ad")):
        if not path.is_file():
            continue
        if suffixes and any(path.name.endswith(suffix) for suffix in suffixes):
            continue
        yield path


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if not fieldnames:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    new_rows = [dict(row) for row in rows]
    if not path.exists() or path.stat().st_size == 0:
        if not fieldnames:
            keys: List[str] = []
            for row in new_rows:
                for key in row.keys():
                    if key not in keys:
                        keys.append(key)
            fieldnames = keys
        write_csv(path, new_rows, fieldnames=fieldnames)
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = [key for key in (reader.fieldnames or []) if key is not None]
        existing_rows: List[dict] = []
        extra_keys: List[str] = []
        for old_row in reader:
            extras = old_row.pop(None, None)
            if extras:
                for idx, value in enumerate(extras, start=1):
                    key = f"extra_{idx}"
                    if key not in extra_keys:
                        extra_keys.append(key)
                    old_row[key] = value
            existing_rows.append(old_row)

    if not fieldnames:
        keys = list(existing_fieldnames)
        for key in extra_keys:
            if key not in keys:
                keys.append(key)
        for row in new_rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys

    write_csv(path, existing_rows + new_rows, fieldnames=fieldnames)


def print_table(rows: Iterable[dict], columns: Sequence[str]) -> None:
    rows = list(rows)
    if not rows:
        print("(no rows)")
        return
    widths = {
        col: max(len(str(col)), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    print("  ".join(str(col).ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
