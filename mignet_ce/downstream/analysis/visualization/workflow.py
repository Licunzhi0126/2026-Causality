from __future__ import annotations

"""Render figures exclusively from persisted downstream analysis tables."""

import hashlib
import json
from pathlib import Path

from ..workflow import UNIFIED_TABLE_FILES, load_unified_analysis_tables
from .determinism_degeneracy.plots import plot_unified_ei_overview
from .dynamic_closure.plots import (
    plot_cross_representation_consistency,
    plot_dynamical_closure_three_panels,
)
from .fate_path.plots import plot_unified_fate_paths
from .grn_cci.plots import plot_unified_mechanism
from .null_model.plots import plot_unified_random_null
from .spatial.plots import (
    plot_unified_effective_spatial,
    plot_unified_spatial_state_ei,
)


UNIFIED_FIGURE_FILES = {
    "ei": "fig01_causal_emergence_decomposition.png",
    "spatial_ei": "fig02_state_level_ei_spatial.png",
    "null": "fig03_matched_random_null.png",
    "consistency": "fig04_cross_representation_consistency.png",
    "effective": "fig05_effective_states_spatial.png",
    "mechanism": "fig06_grn_cci_mechanism.png",
    "fate": "fig07_macro_fate_paths.png",
    "closure": "fig08_dynamical_closure_three_panels.png",
}


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_unified_downstream_figures(results_dir: Path) -> dict[str, object]:
    """Render the figure suite without recomputing or mutating analysis outputs."""

    root = Path(results_dir).resolve()
    tables_dir = root / "tables"
    table_paths = {
        name: tables_dir / filename for name, filename in UNIFIED_TABLE_FILES.items()
    }
    before = {name: _checksum(path) for name, path in table_paths.items()}
    tables = load_unified_analysis_tables(root)
    figures_dir = root / "figures"
    paths = {name: figures_dir / filename for name, filename in UNIFIED_FIGURE_FILES.items()}

    plot_unified_ei_overview(tables["metrics"], paths["ei"])
    plot_unified_spatial_state_ei(tables["spatial_spots"], paths["spatial_ei"])
    plot_unified_random_null(tables["null_distribution"], paths["null"])
    plot_cross_representation_consistency(tables["consistency"], paths["consistency"])
    plot_unified_effective_spatial(tables["effective"], tables["spatial"], paths["effective"])
    plot_unified_mechanism(tables["mechanism"], paths["mechanism"])
    plot_unified_fate_paths(tables["fate"], paths["fate"])
    plot_dynamical_closure_three_panels(tables["closure"], tables["metrics"], paths["closure"])

    after = {name: _checksum(path) for name, path in table_paths.items()}
    if before != after:
        raise RuntimeError("Visualization modified one or more persisted analysis tables")
    pngs = [paths[name] for name in UNIFIED_FIGURE_FILES]
    missing = [path for path in (*pngs, *(path.with_suffix(".pdf") for path in pngs)) if not path.exists()]
    if missing:
        raise RuntimeError(f"Visualization did not produce expected files: {missing}")
    manifest = {
        "workflow": "read_only_downstream_visualization",
        "analysis_table_sha256": before,
        "figures_png": [str(path.relative_to(root)) for path in pngs],
        "figures_pdf": [str(path.with_suffix(".pdf").relative_to(root)) for path in pngs],
        "analysis_tables_unchanged": True,
    }
    manifest_path = root / "audit" / "visualization_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_dir": root,
        "figures_dir": figures_dir,
        "figure_count": len(pngs),
        "manifest": manifest_path,
    }
