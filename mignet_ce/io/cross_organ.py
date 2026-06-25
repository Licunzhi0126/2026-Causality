from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from mignet_ce.io.loaders import LayerPaths


@dataclass(frozen=True)
class CrossOrganLayerSpec:
    layer: str
    sample_prefix: str

    def sample_stem(self, stage: str) -> str:
        return f"{self.sample_prefix}_all_organs_{stage}"


CROSS_ORGAN_LAYER_SPECS: Dict[str, CrossOrganLayerSpec] = {
    "seurat_k40": CrossOrganLayerSpec("seurat_k40", "seurat_k40"),
    "louvain_k150": CrossOrganLayerSpec("louvain_k150", "louvain_k150"),
}


class CrossOrganDataResolver:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def paths(self, layer: str, stage: str) -> LayerPaths:
        try:
            spec = CROSS_ORGAN_LAYER_SPECS[layer]
        except KeyError as exc:
            raise ValueError(f"Unsupported cross-organ layer {layer!r}. Expected one of {sorted(CROSS_ORGAN_LAYER_SPECS)}.") from exc
        sample = spec.sample_stem(str(stage))
        return LayerPaths(
            layer=layer,
            organ="all_organs",
            stage=str(stage),
            sample_stem=sample,
            candidate_sample_stems=[sample],
            h5ad=self.data_root / "cross_organ" / layer / f"{sample}.h5ad",
            grn_edges=self.data_root / "grn" / "cross_organ" / layer / sample / "grn_edges.csv",
            cci_total=self.data_root / "cci" / "cross_organ" / layer / f"{sample}_CCI_total.npz",
            cci_manifest=self.data_root / "cci" / "cross_organ" / layer / f"{sample}_COMMOT_lr_pairs.tsv",
            cci_index=self.data_root / "cci" / "cross_organ" / layer / f"{sample}_index.tsv",
            cci_lr_dir=self.data_root / "cci" / "cross_organ" / layer / f"{sample}_COMMOT_by_LR",
            spot_domain_map=self.data_root / "cross_organ" / layer / f"{sample}_spot_domain_map.csv",
            unit_grn_edges=self.data_root / "grn_unit_specific" / "cross_organ" / layer / sample / "unit_grn_edges.csv",
        )
