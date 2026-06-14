from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DomainLayerSpec:
    name: str
    family: str
    output_name: str
    sample_prefix: str
    domain_manifest: str
    grn_manifest: str
    cci_manifest: str
    mode: str
    k: Optional[int]
    unit_kind: str = "domain"
    max_spots_per_domain: int = 4


DOMAIN_LAYER_SPECS: Dict[str, DomainLayerSpec] = {
    "louvain_less_than5": DomainLayerSpec(
        name="louvain_less_than5",
        family="louvain",
        output_name="louvain_less_than5",
        sample_prefix="louvainLessThan5",
        domain_manifest="domain_manifest_louvain_less_than5.csv",
        grn_manifest="grn_manifest_louvain_less_than5.csv",
        cci_manifest="cci_manifest_louvain_less_than5.csv",
        mode="less_than_5",
        k=None,
    ),
    "louvain_k40": DomainLayerSpec(
        name="louvain_k40",
        family="louvain",
        output_name="louvain_k40",
        sample_prefix="louvain40",
        domain_manifest="domain_manifest_louvain_k40.csv",
        grn_manifest="grn_manifest_louvain_k40.csv",
        cci_manifest="cci_manifest_louvain_k40.csv",
        mode="exact_k",
        k=40,
    ),
    "louvain_k150": DomainLayerSpec(
        name="louvain_k150",
        family="louvain",
        output_name="louvain_k150",
        sample_prefix="louvain150",
        domain_manifest="domain_manifest_louvain_k150.csv",
        grn_manifest="grn_manifest_louvain_k150.csv",
        cci_manifest="cci_manifest_louvain_k150.csv",
        mode="exact_k",
        k=150,
    ),
    "louvain_k1100": DomainLayerSpec(
        name="louvain_k1100",
        family="louvain",
        output_name="louvain_k1100",
        sample_prefix="louvain1100",
        domain_manifest="domain_manifest_louvain_k1100.csv",
        grn_manifest="grn_manifest_louvain_k1100.csv",
        cci_manifest="cci_manifest_louvain_k1100.csv",
        mode="exact_k",
        k=1100,
    ),
    "spatial_domain_less_than5": DomainLayerSpec(
        name="spatial_domain_less_than5",
        family="spatial_domain",
        output_name="spatial_domain_less_than5",
        sample_prefix="spatialDomainLessThan5",
        domain_manifest="domain_manifest_spatial_domain_less_than5.csv",
        grn_manifest="grn_manifest_spatial_domain_less_than5.csv",
        cci_manifest="cci_manifest_spatial_domain_less_than5.csv",
        mode="less_than_5",
        k=None,
    ),
    "spatial_domain_k40": DomainLayerSpec(
        name="spatial_domain_k40",
        family="spatial_domain",
        output_name="spatial_domain_k40",
        sample_prefix="spatialDomain40",
        domain_manifest="domain_manifest_spatial_domain_k40.csv",
        grn_manifest="grn_manifest_spatial_domain_k40.csv",
        cci_manifest="cci_manifest_spatial_domain_k40.csv",
        mode="exact_k",
        k=40,
    ),
    "spatial_domain_k150": DomainLayerSpec(
        name="spatial_domain_k150",
        family="spatial_domain",
        output_name="spatial_domain_k150",
        sample_prefix="spatialDomain150",
        domain_manifest="domain_manifest_spatial_domain_k150.csv",
        grn_manifest="grn_manifest_spatial_domain_k150.csv",
        cci_manifest="cci_manifest_spatial_domain_k150.csv",
        mode="exact_k",
        k=150,
    ),
    "seurat_less_than5": DomainLayerSpec(
        name="seurat_less_than5",
        family="seurat",
        output_name="seurat_less_than5",
        sample_prefix="seuratLessThan5",
        domain_manifest="domain_manifest_seurat_less_than5.csv",
        grn_manifest="grn_manifest_seurat_less_than5.csv",
        cci_manifest="cci_manifest_seurat_less_than5.csv",
        mode="less_than_5",
        k=None,
    ),
    "seurat_k150": DomainLayerSpec(
        name="seurat_k150",
        family="seurat",
        output_name="seurat_k150",
        sample_prefix="seurat150",
        domain_manifest="domain_manifest_seurat_k150.csv",
        grn_manifest="grn_manifest_seurat_k150.csv",
        cci_manifest="cci_manifest_seurat_k150.csv",
        mode="exact_k",
        k=150,
    ),
    "seurat_k40": DomainLayerSpec(
        name="seurat_k40",
        family="seurat",
        output_name="seurat_k40",
        sample_prefix="seurat",
        domain_manifest="domain_manifest_seurat_k40.csv",
        grn_manifest="grn_manifest_seurat_k40.csv",
        cci_manifest="cci_manifest_seurat_k40.csv",
        mode="exact_k",
        k=40,
    ),
}


def get_domain_layer_spec(layer: str) -> DomainLayerSpec:
    try:
        return DOMAIN_LAYER_SPECS[layer]
    except KeyError as exc:
        raise ValueError(f"Unsupported domain layer {layer!r}. Expected one of {sorted(DOMAIN_LAYER_SPECS)}.") from exc
