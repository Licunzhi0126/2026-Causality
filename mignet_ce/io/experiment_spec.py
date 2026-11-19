"""Validated experiment metadata shared by the factory and Formal Core.

Profiles select samples, never change the model or information mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class ExperimentSpec:
    dataset_id: str
    species: str
    organs: tuple[str, ...]
    time_points: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    organ_labels: dict[str, str]
    stage_labels: dict[str, str]
    raw_files: dict[str, str]
    gene_column: str = "Gene"
    output_layout: str = "flat"
    min_cell_pct: float = .01
    distance_threshold: float = 200.

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.dataset_id):
            raise ValueError("Invalid dataset_id")
        if not self.organs or len(set(self.organs)) != len(self.organs) or any(
                not re.fullmatch(r"[a-z]+", x) for x in self.organs):
            raise ValueError("Select unique lowercase scope identifiers")
        if len(self.time_points) < 2 or len(set(self.time_points)) != len(self.time_points):
            raise ValueError("Select at least two unique stages")
        if any(not re.fullmatch(r"\d+(?:\.\d+)?", x) for x in self.time_points):
            raise ValueError("Initial protocol requires numeric stage IDs")
        values = list(map(float, self.time_points))
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("Stages must be in increasing developmental order")
        if not self.pairs or len(set(self.pairs)) != len(self.pairs):
            raise ValueError("Select unique time pairs")
        for a, b in self.pairs:
            if a not in self.time_points or b not in self.time_points or float(a) >= float(b):
                raise ValueError(f"Invalid forward time pair {a}->{b}")
        if set(self.organ_labels) != set(self.organs) or set(self.stage_labels) != set(self.time_points):
            raise ValueError("Labels must cover exactly the selected scopes/stages")
        if self.raw_files and set(self.raw_files) != set(self.time_points):
            raise ValueError("raw_files must cover exactly the selected stages")
        for value in self.raw_files.values():
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError("Raw inputs must be relative paths inside data-root")
        if self.output_layout not in ("flat", "nested"):
            raise ValueError("output_layout must be flat or nested")
        if not 0 < self.min_cell_pct <= 1 or not self.distance_threshold > 0:
            raise ValueError("Invalid CCI filter or distance threshold")

    def to_dict(self):
        from dataclasses import asdict
        return json.loads(json.dumps(asdict(self)))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def pair_labels(self):
        return tuple(f"{a}->{b}" for a, b in self.pairs)

    def metadata(self):
        return {"profile_sha256": self.fingerprint, "profile": self.to_dict()}


def load_experiment(path: Path | str | None) -> ExperimentSpec | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("organs", "time_points"):
        data[key] = tuple(map(str, data[key]))
    data["pairs"] = tuple(tuple(map(str, pair)) for pair in data["pairs"])
    spec = ExperimentSpec(**data)
    spec.validate()
    return spec


def bind_experiment(args):
    """Resolve CLI defaults per invocation; no mutable module-wide profile state."""
    spec = load_experiment(getattr(args, "experiment_config", None))
    args.experiment = spec
    allowed = spec.organs if spec else ("lung", "brain", "heart")
    if getattr(args, "organs", None) is None:
        args.organs = list(allowed)
    if not args.organs or len(set(args.organs)) != len(args.organs) or not set(args.organs) <= set(allowed):
        raise ValueError("Scopes must belong to the selected experiment")
    if hasattr(args, "figure_pair"):
        args.figure_pair = args.figure_pair or (spec.pair_labels[1] if spec and len(spec.pairs) > 1
                                                else spec.pair_labels[0] if spec else "12.5->13.5")
        if args.figure_pair not in (spec.pair_labels if spec else ("11.5->12.5", "12.5->13.5", "11.5->13.5")):
            raise ValueError("Figure pair must belong to the selected experiment")
    if hasattr(args, "figure_organ"):
        args.figure_organ = args.figure_organ or (spec.organs[0] if spec else "heart")
    return args


def validate_scopes(organs, experiment=None):
    organs = tuple(organs)
    allowed = experiment.organs if experiment else ("lung", "brain", "heart")
    if not organs or len(set(organs)) != len(organs) or not set(organs) <= set(allowed):
        raise ValueError("Select unique supported organs/scopes")
    return organs


def guard_output_profile(root, experiment, *, write=False):
    """Reject cross-profile reuse before any training, backup, or output mutation."""
    if experiment is None:
        return
    root = Path(root)
    binding = root / "experiment_profile.json"
    if binding.is_file():
        if json.loads(binding.read_text(encoding="utf-8")) != experiment.metadata():
            raise ValueError("Output experiment profile mismatch; select a new output-root")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Nonempty output-root has no experiment binding; select a new output-root")
    elif write:
        root.mkdir(parents=True, exist_ok=True)
        with binding.open("x", encoding="utf-8") as stream:
            json.dump(experiment.metadata(), stream, indent=2)
            stream.write("\n")
