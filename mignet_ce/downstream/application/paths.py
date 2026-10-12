"""Input discovery and sample metadata conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


APPLICATION_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = APPLICATION_ROOT / "input_data"
DEFAULT_OUTPUT_ROOT = APPLICATION_ROOT / "output"

_SAMPLE_RE = re.compile(
    r"^(?P<gsm>GSM\d+)_V10A20-(?P<individual>0(?:72|71)-[BCD]1)$"
)
_TIMEPOINT_BY_PREFIX = {"072": "d7", "071": "d21"}


@dataclass(frozen=True)
class SamplePaths:
    """One H5AD input and its biological-cluster mapping."""

    sample_id: str
    timepoint: str
    individual_id: str
    h5ad_path: Path
    cluster_map_path: Path


def parse_sample_id(sample_id: str) -> tuple[str, str]:
    """Return the documented timepoint and individual ID for a sample stem."""
    match = _SAMPLE_RE.fullmatch(sample_id)
    if match is None:
        raise ValueError(
            f"Unsupported sample name {sample_id!r}; expected GSM<id>_V10A20-071/072-[BCD]1."
        )
    individual_id = match.group("individual")
    return _TIMEPOINT_BY_PREFIX[individual_id[:3]], individual_id


def discover_samples(input_root: Path, sample_names: tuple[str, ...] = ()) -> list[SamplePaths]:
    """Discover matching H5AD/cluster-map pairs in a flat input directory."""
    input_root = Path(input_root)
    requested = set(sample_names)
    samples: list[SamplePaths] = []
    for h5ad_path in sorted(input_root.glob("*.h5ad")):
        sample_id = h5ad_path.stem
        if requested and sample_id not in requested:
            continue
        timepoint, individual_id = parse_sample_id(sample_id)
        cluster_map_path = h5ad_path.with_name(f"{sample_id}_cluster_mapping.csv")
        if not cluster_map_path.is_file():
            raise FileNotFoundError(f"Cluster map is missing for {sample_id}: {cluster_map_path}")
        samples.append(SamplePaths(sample_id, timepoint, individual_id, h5ad_path, cluster_map_path))
    if requested:
        found = {sample.sample_id for sample in samples}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Requested sample H5AD files were not found: {missing}")
    if not samples:
        raise FileNotFoundError(f"No supported H5AD files found in {input_root}")
    return samples

