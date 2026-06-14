from __future__ import annotations

import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from mignet_ce.config import DEFAULT_ABLATION_OUTPUT_ROOT, NETWORK_METHODS, PIJ_METHODS, TemporalRunConfig
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline, _json_default


class VerticalAblationPipeline:
    def __init__(
        self,
        base_cfg: TemporalRunConfig,
        network_methods: Sequence[str] | None = None,
        pij_methods: Sequence[str] | None = None,
        output_root: Path | None = None,
        fail_fast: bool = False,
    ):
        self.base_cfg = base_cfg
        self.network_methods = list(network_methods or sorted(NETWORK_METHODS))
        self.pij_methods = list(pij_methods or ["joint_nmf", "laplacian", "3dot", "slat"])
        self.output_root = Path(output_root or DEFAULT_ABLATION_OUTPUT_ROOT)
        self.fail_fast = bool(fail_fast)
        self._validate_methods()

    def _validate_methods(self) -> None:
        unknown_networks = set(self.network_methods) - set(NETWORK_METHODS)
        if unknown_networks:
            raise ValueError(f"Unsupported network_methods {sorted(unknown_networks)}.")
        unknown_pij = set(self.pij_methods) - set(PIJ_METHODS)
        if unknown_pij:
            raise ValueError(f"Unsupported pij_methods {sorted(unknown_pij)}.")

    def run(self) -> pd.DataFrame:
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_rows: List[dict[str, object]] = []
        metrics_tables: List[pd.DataFrame] = []
        run_summary_tables: List[pd.DataFrame] = []

        with (self.output_root / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_cfg": asdict(self.base_cfg),
                    "network_methods": self.network_methods,
                    "pij_methods": self.pij_methods,
                    "output_root": str(self.output_root),
                    "fail_fast": self.fail_fast,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        for network_method in self.network_methods:
            for pij_method in self.pij_methods:
                combo_root = self.output_root / f"network={network_method}" / f"pij={pij_method}"
                combo_root.mkdir(parents=True, exist_ok=True)
                embedding_method = pij_method if pij_method in {"joint_nmf", "laplacian"} else "joint_nmf"
                cfg = replace(
                    self.base_cfg,
                    output_root=combo_root,
                    network_method=network_method,
                    pij_method=pij_method,
                    embedding_method=embedding_method,
                )
                try:
                    metrics = VerticalMIGNetPipeline(cfg).run()
                    if not metrics.empty and "status" not in metrics.columns:
                        metrics["status"] = "written"
                    metrics_tables.append(metrics)
                    combo_status = "written" if not metrics.empty else "empty"
                    summary_path = combo_root / "run_summary.csv"
                    if summary_path.exists():
                        summary = pd.read_csv(summary_path)
                        summary["network_method"] = network_method
                        summary["pij_method"] = pij_method
                        leading = ["network_method", "pij_method"]
                        summary = summary.loc[:, leading + [col for col in summary.columns if col not in leading]]
                        run_summary_tables.append(summary)
                        statuses = set(summary.get("status", pd.Series(dtype=str)).astype(str))
                        if "error" in statuses and not metrics.empty:
                            combo_status = "partial"
                        elif "error" in statuses:
                            combo_status = "error"
                    manifest_rows.append(
                        {
                            "network_method": network_method,
                            "pij_method": pij_method,
                            "output_root": str(combo_root),
                            "status": combo_status,
                            "metrics_rows": int(len(metrics)),
                        }
                    )
                except Exception as exc:
                    manifest_rows.append(
                        {
                            "network_method": network_method,
                            "pij_method": pij_method,
                            "output_root": str(combo_root),
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=8),
                            "metrics_rows": 0,
                        }
                    )
                    if self.fail_fast:
                        self._write_outputs(manifest_rows, metrics_tables, run_summary_tables)
                        raise

        return self._write_outputs(manifest_rows, metrics_tables, run_summary_tables)

    def _write_outputs(
        self,
        manifest_rows: Sequence[dict[str, object]],
        metrics_tables: Sequence[pd.DataFrame],
        run_summary_tables: Sequence[pd.DataFrame],
    ) -> pd.DataFrame:
        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(self.output_root / "ablation_manifest.csv", index=False)

        all_metrics = pd.concat([table for table in metrics_tables if not table.empty], ignore_index=True) if metrics_tables else self._empty_all_metrics()
        if all_metrics.empty:
            all_metrics = self._empty_all_metrics()
        all_metrics.to_csv(self.output_root / "all_metrics.csv", index=False)

        all_summary = pd.concat(run_summary_tables, ignore_index=True) if run_summary_tables else pd.DataFrame()
        all_summary.to_csv(self.output_root / "all_run_summary.csv", index=False)
        self._write_pivots(all_metrics)
        return all_metrics

    @staticmethod
    def _empty_all_metrics() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "network_method",
                "pij_method",
                "organ",
                "lower_layer",
                "upper_layer",
                "time_pair",
                "EI_lower",
                "EI_upper",
                "EI_gain",
                "TE_raw",
                "TE",
                "DI_raw",
                "DI",
                "status",
            ]
        )

    def _write_pivots(self, all_metrics: pd.DataFrame) -> None:
        if all_metrics.empty:
            for metric in ("EI_gain", "DI", "TE"):
                pd.DataFrame().to_csv(self.output_root / f"metric_pivot_{metric}.csv")
            return
        for metric in ("EI_gain", "DI", "TE"):
            if metric not in all_metrics.columns:
                pd.DataFrame().to_csv(self.output_root / f"metric_pivot_{metric}.csv")
                continue
            pivot = all_metrics.pivot_table(
                index="network_method",
                columns="pij_method",
                values=metric,
                aggfunc="mean",
            )
            pivot.to_csv(self.output_root / f"metric_pivot_{metric}.csv")
