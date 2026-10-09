"""Frozen paper-ablation experiment matrix. NO automatic model selection here.

The experiments were decided from the exploratory Round-11/12 sandbox pilot.
This module deliberately does not import mignet_ce.pij.ablation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

TIME_POINTS = ('11.5', '12.5', '13.5')
TIME_PAIRS = ('11.5->12.5', '12.5->13.5', '11.5->13.5')
HIERARCHIES = (
    ('Spot -> K150', 'spot', 'seurat_k150'),
    ('K150 -> K40', 'seurat_k150', 'seurat_k40'),
    ('K40 -> K10', 'seurat_k40', 'seurat_k10'),
    ('Spot -> K40', 'spot', 'seurat_k40'),
    ('K150 -> K10', 'seurat_k150', 'seurat_k10'),
    ('Spot -> K10', 'spot', 'seurat_k10'),
)
FEATURES = ('NMF', 'Laplacian')
T1_CONFIGS = tuple((g, c) for g in FEATURES for c in FEATURES)
# Locked choices from the exploratory Round 12; NEVER recompute choices from formal results.
FIXED_GRN = 'NMF'
FIXED_CCI = 'Laplacian'
FIXED_PIJ = 'KL+OT'
GRN_PROVIDER = {'NMF': 'legacy_nmf', 'Laplacian': 'legacy_lap_hks'}
CCI_PROVIDER = {'NMF': 'N', 'Laplacian': 'L'}

@dataclass(frozen=True)
class FormalAblationConfig:
    data_root: Path
    output_root: Path
    organ: str = 'heart'
    alpha_cci: float = 0.20
    beta: float = 0.05
    temperature: float = 0.8
    grn_residual_lambda: float = 0.15
    nmf_components: int = 5
    nmf_max_iter: int = 300
    nmf_seed: int = 42
    laplacian_components: int = 5
    dpi: int = 250
    include_k10: bool = False  # K10 unavailable in current uploaded data; opt in when present.

    def validate(self) -> None:
        if abs(self.alpha_cci - 0.20) > 1e-12:
            raise ValueError('Table 1/2/3 protocol fixes alpha_cci=0.20.')
        if abs(self.beta - 0.05) > 1e-12 or abs(self.temperature - 0.8) > 1e-12:
            raise ValueError('Frozen protocol fixes KL beta=0.05 and transition tau=0.8.')
        if abs(self.grn_residual_lambda - 0.15) > 1e-12:
            raise ValueError('Frozen protocol fixes GRN residual lambda=0.15.')
        if min(self.nmf_components, self.nmf_max_iter, self.laplacian_components, self.dpi) <= 0:
            raise ValueError('Components, iteration counts and DPI must be positive.')
        if not self.organ:
            raise ValueError('An organ is required.')

    def selected_hierarchies(self) -> tuple[tuple[str, str, str], ...]:
        return HIERARCHIES if self.include_k10 else tuple(h for h in HIERARCHIES if 'seurat_k10' not in h)

    def public_contract(self) -> dict:
        result = asdict(self)
        result['data_root'] = str(self.data_root.resolve())
        result.pop('output_root')  # moving output location must not invalidate scientific cache
        result.update(time_points=list(TIME_POINTS), time_pairs=list(TIME_PAIRS),
                      hierarchies=[list(item) for item in self.selected_hierarchies()],
                      table1_feature_grid=[list(item) for item in T1_CONFIGS],
                      table2_fixed_feature=[FIXED_GRN, FIXED_CCI],
                      table3_fixed=[FIXED_GRN, FIXED_CCI, FIXED_PIJ],
                      cost_fusion='(1-alpha)*GRN_raw_KL+alpha*CCI_raw_KL',
                      feature_engine='production_low_level_pairwise_zscore',
                      grn_provider=GRN_PROVIDER, cci_provider=CCI_PROVIDER,
                      explicit_no_selection=True)
        return result
