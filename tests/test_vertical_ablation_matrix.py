from __future__ import annotations

import pandas as pd

from mignet_ce.config import TemporalRunConfig
from mignet_ce.pipelines.vertical_ablation import VerticalAblationPipeline


def test_vertical_ablation_writes_two_by_four_manifest(tmp_path, monkeypatch) -> None:
    class FakeVerticalPipeline:
        def __init__(self, cfg: TemporalRunConfig):
            self.cfg = cfg

        def run(self) -> pd.DataFrame:
            self.cfg.output_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "organ": "heart",
                        "lower_layer": "spot",
                        "upper_layer": "louvain_less_than5",
                        "network_method": self.cfg.network_method,
                        "pij_method": self.cfg.effective_pij_method(),
                        "status": "written",
                    }
                ]
            ).to_csv(self.cfg.output_root / "run_summary.csv", index=False)
            return pd.DataFrame(
                [
                    {
                        "network_method": self.cfg.network_method,
                        "pij_method": self.cfg.effective_pij_method(),
                        "organ": "heart",
                        "lower_layer": "spot",
                        "upper_layer": "louvain_less_than5",
                        "time_pair": "11.5->12.5",
                        "EI_gain": 0.0,
                        "DI": 0.0,
                        "TE": 0.0,
                    }
                ]
            )

    monkeypatch.setattr("mignet_ce.pipelines.vertical_ablation.VerticalMIGNetPipeline", FakeVerticalPipeline)
    base_cfg = TemporalRunConfig(output_root=tmp_path / "unused", organs=["heart"])
    pipeline = VerticalAblationPipeline(
        base_cfg=base_cfg,
        network_methods=["legacy_mixed_grn_cci", "cross_cell_multilayer"],
        pij_methods=["joint_nmf", "laplacian", "3dot", "slat"],
        output_root=tmp_path / "ablation",
    )

    metrics = pipeline.run()
    manifest = pd.read_csv(tmp_path / "ablation" / "ablation_manifest.csv")
    all_metrics = pd.read_csv(tmp_path / "ablation" / "all_metrics.csv")

    assert len(manifest) == 8
    assert len(metrics) == 8
    assert len(all_metrics) == 8
    assert set(all_metrics["network_method"]) == {"legacy_mixed_grn_cci", "cross_cell_multilayer"}
    assert set(all_metrics["pij_method"]) == {"joint_nmf", "laplacian", "3dot", "slat"}
