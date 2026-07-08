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
DEFAULT_ABLATION_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "output" / "mignet_vertical_ablation"


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
    "spatial_domain_less_than5": LayerSpec(name="spatial_domain_less_than5", sample_prefix="spatialDomainLessThan5"),
    "spatial_domain_k150": LayerSpec(name="spatial_domain_k150", sample_prefix="spatialDomain150"),
    "spatial_domain_k40": LayerSpec(name="spatial_domain_k40", sample_prefix="spatialDomain40"),
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
    "spatial_domain_adjacent": (
        VerticalPairSpec("spot", "spatial_domain_less_than5"),
        VerticalPairSpec("spatial_domain_less_than5", "spatial_domain_k150"),
        VerticalPairSpec("spatial_domain_k150", "spatial_domain_k40"),
    ),
    "spatial_domain_all": (
        VerticalPairSpec("spot", "spatial_domain_less_than5"),
        VerticalPairSpec("spatial_domain_less_than5", "spatial_domain_k150"),
        VerticalPairSpec("spatial_domain_k150", "spatial_domain_k40"),
        VerticalPairSpec("spot", "spatial_domain_k150"),
        VerticalPairSpec("spot", "spatial_domain_k40"),
        VerticalPairSpec("spatial_domain_less_than5", "spatial_domain_k40"),
    ),
}

PIJ_METHODS = {
    "joint_nmf",
    "laplacian",
    "3dot",
    "slat",
    "expr_ot",
    "pure_expression_ot",
    "energy_ot",
    "energy_entropy_ot",
    "pseudotime_ot",
    "sr_ot",
    "spatial_ot",
    "sr_spatial_ot",
    "pseudotime_spatial_ot",
    "sr_expression_ot",
    "pseudotime_expression_ot",
    "expr_pseudotime_sr_ot",
    "expr_pseudotime_sr_spatial_ot",
    "expr_pseudotime_sr_energy_ot",
    "expr_pseudotime_sr_energy_spatial_ot",
    "velocity_ot",
    "development_ot",
}
PIJ_METHOD_PRESETS = {
    "core": ("joint_nmf", "laplacian", "3dot", "slat"),
    "ot_basic": ("expr_ot", "energy_entropy_ot"),
    "pure_expression": ("pure_expression_ot",),
    "development_scalar": ("pseudotime_ot", "sr_ot"),
    "development_all": ("pseudotime_ot", "sr_ot", "velocity_ot", "development_ot"),
    "ot_ablation_v2": (
        "sr_ot",
        "pseudotime_ot",
        "spatial_ot",
        "sr_spatial_ot",
        "pseudotime_spatial_ot",
        "sr_expression_ot",
        "pseudotime_expression_ot",
    ),
    "ot_ablation_v3": (
        "energy_ot",
        "expr_pseudotime_sr_ot",
        "expr_pseudotime_sr_spatial_ot",
        "expr_pseudotime_sr_energy_ot",
    ),
    "ot_ablation_v4": (
        "expr_pseudotime_sr_energy_ot",
        "expr_pseudotime_sr_energy_spatial_ot",
    ),
    "development_ot_ablation": (
        "sr_ot",
        "pseudotime_ot",
        "spatial_ot",
        "sr_spatial_ot",
        "pseudotime_spatial_ot",
        "sr_expression_ot",
        "pseudotime_expression_ot",
    ),
    "all": (
        "joint_nmf",
        "laplacian",
        "3dot",
        "slat",
        "expr_ot",
        "pure_expression_ot",
        "energy_ot",
        "energy_entropy_ot",
        "pseudotime_ot",
        "sr_ot",
        "spatial_ot",
        "sr_spatial_ot",
        "pseudotime_spatial_ot",
        "sr_expression_ot",
        "pseudotime_expression_ot",
        "expr_pseudotime_sr_ot",
        "expr_pseudotime_sr_spatial_ot",
        "expr_pseudotime_sr_energy_ot",
        "expr_pseudotime_sr_energy_spatial_ot",
        "velocity_ot",
        "development_ot",
    ),
}
EMBEDDING_METHODS = {"joint_nmf", "laplacian"}
NETWORK_METHODS = {
    "legacy_mixed_grn_cci",
    "legacy_inter_cci_only",
    "legacy_inter_additive_grn_cci",
    "clean_grn_cci_mix",
    "clean_grn_cci_expr_mix",
    "clean_expression_cci_mix",
    "unit_specific_clean_grn_cci_mix",
    "cross_cell_multilayer",
    "expression_only",
}
DEVELOPMENT_PIJ_METHODS = {
    "pseudotime_ot",
    "sr_ot",
    "sr_spatial_ot",
    "pseudotime_spatial_ot",
    "sr_expression_ot",
    "pseudotime_expression_ot",
    "expr_pseudotime_sr_ot",
    "expr_pseudotime_sr_spatial_ot",
    "expr_pseudotime_sr_energy_ot",
    "expr_pseudotime_sr_energy_spatial_ot",
    "velocity_ot",
    "development_ot",
}


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
    network_method: str = "legacy_mixed_grn_cci"
    native_intra_use_expression_mask: bool = True
    native_cci_inter_use_expression_mask: bool = False
    grn_expression_weight_mode: str = "geometric_mean"
    grn_expression_transform: str = "log1p_minmax"
    grn_expression_weight_floor: float = 0.0
    unit_grn_fallback: str = "sample_grn_expression_weighted"
    cross_cell_lr_use_grn_gate: bool = False
    cross_cell_top_k_edges: int = 1000
    export_pij: bool = False
    pij_archive_root: Path | None = None
    export_pair_artifacts: bool = False
    development_feature_root: Path | None = None
    pij_feature_aggregation: str = "mean"
    pij_missing_feature_policy: str = "impute_mean"
    pij_feature_components: int | None = 30
    pij_temperature: float = 1.0
    pij_expr_weight: float = 1.0
    pij_spatial_weight: float = 0.2
    pij_graph_energy_weight: float = 0.2
    pij_pseudotime_weight: float = 0.5
    pij_sr_weight: float = 0.5
    pij_potency_weight: float = 0.5
    pij_velocity_weight: float = 0.5
    pij_backward_pseudotime_weight: float = 0.0
    pij_reverse_potency_weight: float = 0.0
    pij_entropy_epsilon: float = 0.05
    pij_use_unbalanced_ot: bool = False
    pij_unbalanced_mass: float = 1.0
    pij_cost_metric: str = "cosine"
    pure_expression_normalize: bool = True
    pure_expression_log1p: bool = True
    pure_expression_scale_factor: float = 10000.0
    pure_expression_max_genes: int | None = 2000
    pure_expression_gene_selection: str = "variance"
    pure_expression_pca_components: int | None = None
    pure_expression_scaler: str = "standard"
    pure_expression_cosine_eps: float = 1e-8
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
    export_raw_native_features: bool = False
    export_feature_diagnostics: bool = False
    export_pij_topk: int = 0
    max_workers: int = 1
    progress: bool = False
    progress_log: Path | None = None
    progress_refresh_interval: float = 0.5
    pij_device: str = "cpu"
    pij_gpu_dtype: str = "float32"
    pij_gpu_min_entries: int = 5_000_000
    pij_gpu_fallback_cpu: bool = True

    def normalized_pairs(self) -> List[VerticalPairSpec]:
        return [pair if isinstance(pair, VerticalPairSpec) else VerticalPairSpec.parse(str(pair)) for pair in self.level_pairs]

    def effective_pij_method(self) -> str:
        return self.pij_method or self.embedding_method

    def effective_pij_archive_root(self) -> Path:
        return Path(self.pij_archive_root) if self.pij_archive_root is not None else Path(self.data_root) / "pij"

    def validate(self) -> None:
        if self.embedding_method not in EMBEDDING_METHODS:
            raise ValueError(f"embedding_method must be one of {sorted(EMBEDDING_METHODS)}.")
        method = self.effective_pij_method()
        if method not in PIJ_METHODS:
            raise ValueError(f"pij_method must be one of {sorted(PIJ_METHODS)}.")
        if method in DEVELOPMENT_PIJ_METHODS and self.development_feature_root is None:
            raise ValueError(f"{method} requires development_feature_root.")
        if self.network_method not in NETWORK_METHODS:
            raise ValueError(f"network_method must be one of {sorted(NETWORK_METHODS)}.")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1.")
        if self.progress_refresh_interval <= 0:
            raise ValueError("progress_refresh_interval must be positive.")
        if self.pij_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("pij_device must be one of: cpu, cuda, auto")
        if self.pij_gpu_dtype not in {"float32", "float64"}:
            raise ValueError("pij_gpu_dtype must be one of: float32, float64")
        if self.pij_gpu_min_entries < 0:
            raise ValueError("pij_gpu_min_entries must be non-negative")
        if not isinstance(self.native_intra_use_expression_mask, bool):
            raise ValueError("native_intra_use_expression_mask must be bool.")
        if not isinstance(self.native_cci_inter_use_expression_mask, bool):
            raise ValueError("native_cci_inter_use_expression_mask must be bool.")
        if self.grn_expression_weight_mode not in {"none", "geometric_mean", "product", "min"}:
            raise ValueError(
                "grn_expression_weight_mode must be one of ['none', 'geometric_mean', 'product', 'min']."
            )
        if self.grn_expression_transform not in {"log1p_minmax", "log1p_zscore", "none"}:
            raise ValueError(
                "grn_expression_transform must be one of ['log1p_minmax', 'log1p_zscore', 'none']."
            )
        if self.grn_expression_weight_floor < 0:
            raise ValueError("grn_expression_weight_floor must be nonnegative.")
        if self.unit_grn_fallback not in {
            "error",
            "sample_grn_masked",
            "sample_grn_expression_weighted",
            "skip_unit_intra",
        }:
            raise ValueError(
                "unit_grn_fallback must be one of "
                "['error', 'sample_grn_masked', 'sample_grn_expression_weighted', 'skip_unit_intra']."
            )
        if not isinstance(self.cross_cell_lr_use_grn_gate, bool):
            raise ValueError("cross_cell_lr_use_grn_gate must be bool.")
        if self.cross_cell_top_k_edges <= 0:
            raise ValueError("cross_cell_top_k_edges must be positive.")
        if self.pij_temperature <= 0:
            raise ValueError("pij_temperature must be positive.")
        if self.pij_entropy_epsilon <= 0:
            raise ValueError("pij_entropy_epsilon must be positive.")
        if self.pij_unbalanced_mass <= 0:
            raise ValueError("pij_unbalanced_mass must be positive.")
        if self.pij_cost_metric not in {"cosine", "euclidean"}:
            raise ValueError("pij_cost_metric must be one of ['cosine', 'euclidean'].")
        if method == "pure_expression_ot" and self.network_method != "expression_only":
            raise ValueError("pure_expression_ot requires network_method='expression_only'.")
        if self.pure_expression_scale_factor <= 0:
            raise ValueError("pure_expression_scale_factor must be positive.")
        if self.pure_expression_max_genes is not None and self.pure_expression_max_genes <= 0:
            raise ValueError("pure_expression_max_genes must be positive when provided.")
        if self.pure_expression_gene_selection not in {"variance", "all"}:
            raise ValueError("pure_expression_gene_selection must be one of ['variance', 'all'].")
        if self.pure_expression_pca_components is not None and self.pure_expression_pca_components <= 0:
            raise ValueError("pure_expression_pca_components must be positive when provided.")
        if self.pure_expression_scaler not in {"standard", "minmax", "none"}:
            raise ValueError("pure_expression_scaler must be one of ['standard', 'minmax', 'none'].")
        if self.pure_expression_cosine_eps <= 0:
            raise ValueError("pure_expression_cosine_eps must be positive.")
        if self.pij_feature_aggregation not in {"mean", "median"}:
            raise ValueError("pij_feature_aggregation must be one of ['mean', 'median'].")
        if self.pij_missing_feature_policy not in {"error", "impute_mean", "ignore"}:
            raise ValueError("pij_missing_feature_policy must be one of ['error', 'impute_mean', 'ignore'].")
        for field_name in (
            "pij_expr_weight",
            "pij_spatial_weight",
            "pij_graph_energy_weight",
            "pij_pseudotime_weight",
            "pij_sr_weight",
            "pij_potency_weight",
            "pij_velocity_weight",
            "pij_backward_pseudotime_weight",
            "pij_reverse_potency_weight",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be nonnegative.")
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
