from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MOUSE_EMBRYO_DATA_ROOT",
        "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory",
    )
)
DEFAULT_WORK_ROOT = Path(os.environ.get("CAUSALITY_WORK_ROOT", "/home/jovyan/work/2026 Causality"))
DEFAULT_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "output" / "mignet_vertical"


@dataclass(frozen=True)
class LayerSpec:
    name: str
    sample_prefix: str | Tuple[str, ...]
    unit_kind: str = "domain"

    @property
    def sample_prefixes(self) -> Tuple[str, ...]:
        if isinstance(self.sample_prefix, tuple):
            return self.sample_prefix
        return (self.sample_prefix,)

    def sample_stem(self, organ: str, stage: str) -> str:
        return f"{self.sample_prefixes[0]}_{organ}_{stage}"

    def candidate_sample_stems(self, organ: str, stage: str) -> Tuple[str, ...]:
        return tuple(f"{prefix}_{organ}_{stage}" for prefix in self.sample_prefixes)


@dataclass(frozen=True)
class VerticalPairSpec:
    lower_layer: str
    upper_layer: str

    @classmethod
    def parse(cls, value: str) -> "VerticalPairSpec":
        if ":" in value:
            lower, upper = value.split(":", 1)
        elif "->" in value:
            lower, upper = value.split("->", 1)
        else:
            raise ValueError(f"Cannot parse level pair {value!r}; use lower:upper.")
        return cls(lower.strip(), upper.strip())

    def label(self) -> str:
        return f"{self.lower_layer}_to_{self.upper_layer}"


LAYER_SPECS: Dict[str, LayerSpec] = {
    "spot": LayerSpec(name="spot", sample_prefix="spot", unit_kind="spot"),
    "seurat_less_than5": LayerSpec(name="seurat_less_than5", sample_prefix="seuratLessThan5"),
    "seurat_k150": LayerSpec(name="seurat_k150", sample_prefix="seurat150"),
    "seurat_k40": LayerSpec(name="seurat_k40", sample_prefix=("seurat", "seurat40")),
    "louvain_k40": LayerSpec(name="louvain_k40", sample_prefix="louvain40"),
    "louvain_k150": LayerSpec(name="louvain_k150", sample_prefix="louvain150"),
    "louvain_less_than5": LayerSpec(name="louvain_less_than5", sample_prefix="louvainLessThan5"),
}

DEFAULT_LEVEL_ORDER: Tuple[str, ...] = ("spot", "louvain_less_than5", "louvain_k150", "seurat_k40")
DEFAULT_LEVEL_PAIRS: Tuple[VerticalPairSpec, ...] = (
    VerticalPairSpec("spot", "louvain_less_than5"),
    VerticalPairSpec("louvain_less_than5", "louvain_k150"),
    VerticalPairSpec("louvain_k150", "seurat_k40"),
)
PAIR_PRESETS: Dict[str, Tuple[VerticalPairSpec, ...]] = {
    "legacy_mixed_adjacent": DEFAULT_LEVEL_PAIRS,
    "louvain_adjacent": (
        VerticalPairSpec("spot", "louvain_less_than5"),
        VerticalPairSpec("louvain_less_than5", "louvain_k150"),
        VerticalPairSpec("louvain_k150", "louvain_k40"),
    ),
    "louvain_all": (
        VerticalPairSpec("spot", "louvain_less_than5"),
        VerticalPairSpec("louvain_less_than5", "louvain_k150"),
        VerticalPairSpec("louvain_k150", "louvain_k40"),
        VerticalPairSpec("spot", "louvain_k150"),
        VerticalPairSpec("spot", "louvain_k40"),
        VerticalPairSpec("louvain_less_than5", "louvain_k40"),
    ),
    "seurat_adjacent": (
        VerticalPairSpec("spot", "seurat_less_than5"),
        VerticalPairSpec("seurat_less_than5", "seurat_k150"),
        VerticalPairSpec("seurat_k150", "seurat_k40"),
    ),
    "seurat_all": (
        VerticalPairSpec("spot", "seurat_less_than5"),
        VerticalPairSpec("seurat_less_than5", "seurat_k150"),
        VerticalPairSpec("seurat_k150", "seurat_k40"),
        VerticalPairSpec("spot", "seurat_k150"),
        VerticalPairSpec("spot", "seurat_k40"),
        VerticalPairSpec("seurat_less_than5", "seurat_k40"),
    ),
}

PIJ_METHODS = {"joint_nmf", "laplacian", "3dot", "slat"}
EMBEDDING_METHODS = {"joint_nmf", "laplacian"}


@dataclass
class TemporalRunConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    organs: Sequence[str] = ("heart", "brain", "lung")
    time_points: Sequence[str] = ("11.5", "12.5")
    level_pairs: Sequence[VerticalPairSpec] = field(default_factory=lambda: list(DEFAULT_LEVEL_PAIRS))
    expr_threshold: float = 0.0
    cci_min: float = 0.0
    top_k_targets_per_regulator: int = 20
    require_target_expression_for_inter: bool = True
    nmf_components: int = 5
    nmf_max_iter: int = 300
    nmf_seed: int = 42
    embedding_method: str = "joint_nmf"
    pij_method: str | None = None
    export_pij: bool = False
    pij_feature_components: int | None = 30
    pij_temperature: float = 1.0
    ot_epsilon: float = 0.05
    ot_gamma: float = 1.0
    ot_max_iter: int = 100
    ot_sim_k: int = 10
    ot_dist_k: int = 50
    slat_k_neighbors: int = 20
    slat_hidden_dim: int = 2048
    slat_mlp_hidden: int = 256
    slat_layers: int = 1
    slat_epochs: int = 6
    slat_alpha: float = 0.01
    slat_temperature: float = 0.1
    slat_seed: int = 42
    laplacian_components: int = 5
    laplacian_normalized: bool = True
    kraskov_k: int = 3
    feature_log1p: bool = True
    export_features: bool = True
    export_graphs: bool = False
    export_pij_topk: int = 10

    def normalized_pairs(self) -> List[VerticalPairSpec]:
        return [pair if isinstance(pair, VerticalPairSpec) else VerticalPairSpec.parse(str(pair)) for pair in self.level_pairs]

    def effective_pij_method(self) -> str:
        return self.pij_method or self.embedding_method

    def validate(self) -> None:
        if self.embedding_method not in EMBEDDING_METHODS:
            raise ValueError(f"embedding_method must be one of {sorted(EMBEDDING_METHODS)}.")
        method = self.effective_pij_method()
        if method not in PIJ_METHODS:
            raise ValueError(f"pij_method must be one of {sorted(PIJ_METHODS)}.")
        if self.pij_temperature <= 0:
            raise ValueError("pij_temperature must be positive.")
        if self.ot_epsilon <= 0:
            raise ValueError("ot_epsilon must be positive.")
        if self.ot_gamma <= 0:
            raise ValueError("ot_gamma must be positive.")
        if self.slat_temperature <= 0:
            raise ValueError("slat_temperature must be positive.")
        if self.laplacian_components <= 0:
            raise ValueError("laplacian_components must be positive.")
        for pair in self.normalized_pairs():
            for layer in (pair.lower_layer, pair.upper_layer):
                if layer not in LAYER_SPECS:
                    raise ValueError(f"Unsupported layer {layer!r}. Expected one of {sorted(LAYER_SPECS)}.")
