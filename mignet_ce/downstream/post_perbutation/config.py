from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import re


DATA_ROOT = Path("/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory")
TOPOLOGIES = ("P1", "P2", "P3", "P4", "P5")
FAMILIES = ("global", "local", "weak")
VARIANTS = ("enhance", "keep", "break", "random")
STRENGTHS = tuple(range(10, 100, 10))
LEVELS = (20, 50, 80)
DEFAULT_STRENGTHS = (10, 50, 90)
DEFAULT_LEVELS = (50,)
PIPELINES = {"cci": "cci", "symmetry-existence": "symmetry_existence", "symmetry-perturbation": "symmetry_perturbation"}
PROTOCOL = "post_perbutation_v1"


@dataclass(frozen=True)
class Settings:
    organ: str = "heart"
    source_time: str = "11.5"
    target_time: str = "12.5"
    k: int = 40
    epochs: int = 1500
    train_seed: int = 42
    perturbation_seed: int = 42
    backbone_fraction: float = 0.10
    region_node_fraction: float = 0.40
    region_edge_fraction: float = 0.40
    mirror_fraction: float = 1.0
    random_partitions: int = 100
    pairs_per_cluster: int = 256

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.organ):
            raise ValueError("Invalid organ name")
        if any(not re.fullmatch(r"\d+(?:\.\d+)?", time) for time in (self.source_time, self.target_time)):
            raise ValueError("Time points must be numeric")
        if float(self.source_time) >= float(self.target_time):
            raise ValueError("Source time must precede target time")
        if self.k < 2 or self.epochs < 2 or min(self.train_seed, self.perturbation_seed) < 0:
            raise ValueError("Invalid K, epoch count or seed")
        fractions = (self.backbone_fraction, self.region_node_fraction, self.region_edge_fraction, self.mirror_fraction)
        if not all(0 < fraction <= 1 for fraction in fractions):
            raise ValueError("CCI fractions must be in (0, 1]")
        if min(self.random_partitions, self.pairs_per_cluster) < 1:
            raise ValueError("Random partition and pair counts must be positive")


@dataclass(frozen=True)
class Task:
    family: str
    variant: str = ""
    level: int = 0
    topology: str = ""

    @property
    def id(self) -> str:
        if self.family == "p0":
            return "P0"
        if self.family == "cci":
            return f"cci-{self.topology}-{self.variant}-{self.level:02d}"
        if self.family == "existence":
            return f"existence-{self.level:04d}"
        return f"sym-{self.family}-{self.variant}-{self.level:02d}"


def tasks_for(experiment: str, topologies=TOPOLOGIES, strengths=DEFAULT_STRENGTHS,
              families=FAMILIES, levels=DEFAULT_LEVELS) -> tuple[Task, ...]:
    tasks = [Task("p0")]
    if experiment in {"all", "cci"}:
        tasks.extend(Task("cci", variant, strength, topology)
                     for topology in topologies for strength in strengths
                     for variant in ("positive", "negative"))
    if experiment in {"all", "symmetry-perturbation"}:
        tasks.extend(Task(family, variant, level) for family in families
                     for level in levels for variant in VARIANTS)
    return tuple(tasks)


def default_cpu_workers() -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(32, available // 4))


def seed_for(seed: int, *parts: object) -> int:
    payload = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(sha256(payload).digest()[:8], "little")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
