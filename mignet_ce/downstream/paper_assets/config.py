from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_TIME_POINTS = ("11.5", "12.5", "13.5")
DEFAULT_TIME_PAIRS = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
OPTIMAL_TIME_PAIRS = ("11.5->12.5", "11.5->13.5", "12.5->13.5")
INPUT_SCALES = ("spot", "seurat_k150", "seurat_k40")
OPTIMAL_K_BY_SCALE = {"spot": 40, "seurat_k150": 40, "seurat_k40": 10}
DEFAULT_FIGURE_PAIR = "12.5->13.5"

# Display name -> registered pij method in the current project.
# These names match the paper-facing ablation labels used in the requested Table 1.
DEFAULT_PIJ_ABLATIONS = {
    "full_NG_KLot": ("light_cci_grn", "NG_KLot"),
    "NG_BlockKL": ("light_cci_grn", "compare_N_kl"),
    "N_KL": ("light_cci", "compare_N_kl"),
    "N_SOT": ("light_cci", "compare_N_sot"),
    "N_Cosine": ("light_cci", "compare_N_cos"),
    "Laplacian_KL": ("light_cci", "compare_L_kl"),
    "Expression_KL": ("light_cci", "compare_E_kl"),
    "PureExpression_OT": ("expression_only", "pure_expression_ot"),
}

DEFAULT_LEVELS = {
    "Spot -> K150": ("spot", "seurat_k150"),
    "K150 -> K40": ("seurat_k150", "seurat_k40"),
    "Spot -> K40": ("spot", "seurat_k40"),
}
K10_LEVELS = {
    "K40 -> K10": ("seurat_k40", "seurat_k10"),
    "K150 -> K10": ("seurat_k150", "seurat_k10"),
    "Spot -> K10": ("spot", "seurat_k10"),
}

DEFAULT_COARSE_METHODS = (
    "complete_combined_coarse",
    "complete_combined_coarse_maturity_cci",
    "complete_combined_coarse_maturity_cci_grn",
)


@dataclass(frozen=True)
class AssetConfig:
    """Configuration for publication asset generation.

    The asset package is a read-only consumer of already-computed results. It never
    reruns the scientific pipelines. Outputs are written outside the source tree.
    """

    vertical_ablation_root: Path
    coarse_root: Path
    data_root: Path
    output_root: Path
    slice_root: Path | None = None
    organ: str = "heart"
    network_method: str = "light_cci_grn"
    primary_pij_method: str = "NG_KLot"
    time_points: Sequence[str] = DEFAULT_TIME_POINTS
    time_pairs: Sequence[str] = DEFAULT_TIME_PAIRS
    figure_pair: str = DEFAULT_FIGURE_PAIR
    pij_ablations: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_PIJ_ABLATIONS))
    levels: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_LEVELS))
    coarse_methods: Sequence[str] = DEFAULT_COARSE_METHODS
    primary_coarse_method: str = "complete_combined_coarse_maturity_cci_grn"
    coarse_scale: str = "spot"
    coarse_k: int | None = 64
    coarse_seed: int | None = 42
    cluster_count_policy: str = "adjacent_chain"
    strict: bool = True
    dpi: int = 300
    ei_metrics_path: Path | None = None
    seurat_cg_metrics_path: Path | None = None
    k10_ablation_root: Path | None = None
    multiscale_roots: Sequence[Path] = ()
    optimal_k_by_scale: Mapping[str, int] = field(default_factory=lambda: dict(OPTIMAL_K_BY_SCALE))
    optimal_time_pairs: Sequence[str] = OPTIMAL_TIME_PAIRS

    def normalized(self) -> "AssetConfig":
        return AssetConfig(
            vertical_ablation_root=Path(self.vertical_ablation_root).expanduser().resolve(),
            coarse_root=Path(self.coarse_root).expanduser().resolve(),
            data_root=Path(self.data_root).expanduser().resolve(),
            output_root=Path(self.output_root).expanduser().resolve(),
            slice_root=(Path(self.slice_root).expanduser().resolve() if self.slice_root else None),
            organ=str(self.organ),
            network_method=str(self.network_method),
            primary_pij_method=str(self.primary_pij_method),
            time_points=tuple(map(str, self.time_points)),
            time_pairs=tuple(map(str, self.time_pairs)),
            figure_pair=str(self.figure_pair),
            pij_ablations=dict(self.pij_ablations),
            levels=dict(self.levels),
            coarse_methods=tuple(map(str, self.coarse_methods)),
            primary_coarse_method=str(self.primary_coarse_method),
            coarse_scale=str(self.coarse_scale),
            coarse_k=(int(self.coarse_k) if self.coarse_k is not None else None),
            coarse_seed=(int(self.coarse_seed) if self.coarse_seed is not None else None),
            cluster_count_policy=str(self.cluster_count_policy),
            strict=bool(self.strict),
            dpi=int(self.dpi),
            ei_metrics_path=(Path(self.ei_metrics_path).expanduser().resolve() if self.ei_metrics_path else None),
            seurat_cg_metrics_path=(Path(self.seurat_cg_metrics_path).expanduser().resolve() if self.seurat_cg_metrics_path else None),
            k10_ablation_root=(Path(self.k10_ablation_root).expanduser().resolve() if self.k10_ablation_root else None),
            multiscale_roots=tuple(Path(root).expanduser().resolve() for root in self.multiscale_roots),
            optimal_k_by_scale=dict(self.optimal_k_by_scale),
            optimal_time_pairs=tuple(map(str, self.optimal_time_pairs)),
        )

    def validate(self) -> None:
        if len(tuple(self.time_points)) != 3:
            raise ValueError("Paper assets are configured for exactly 11.5/12.5/13.5 by default.")
        expected = set(self.time_pairs)
        if self.figure_pair not in expected:
            raise ValueError(f"figure_pair={self.figure_pair!r} must be one of {sorted(expected)}")
        if self.cluster_count_policy != "adjacent_chain":
            raise ValueError("Only cluster_count_policy='adjacent_chain' is currently supported.")
        if self.primary_coarse_method not in set(self.coarse_methods):
            raise ValueError("primary_coarse_method must be listed in coarse_methods.")
        if self.dpi < 72:
            raise ValueError("dpi must be >= 72")
        if set(self.optimal_k_by_scale) != set(INPUT_SCALES):
            raise ValueError(f"optimal_k_by_scale must have exactly {INPUT_SCALES}")
        if len(set(self.optimal_time_pairs)) != len(self.optimal_time_pairs):
            raise ValueError("optimal_time_pairs contains duplicates")
