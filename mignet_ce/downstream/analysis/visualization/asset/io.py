from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


_TIME_PAIR_RE = re.compile(r"(?P<t>\d+(?:\.\d+)?)_(?:to|TO)_(?P<tp>\d+(?:\.\d+)?)")


def normalize_time_pair(value: object) -> str:
    text = str(value).strip().replace("→", "->")
    if "->" in text:
        left, right = [part.strip() for part in text.split("->", 1)]
        return f"{left}->{right}"
    if "_to_" in text:
        left, right = [part.strip() for part in text.split("_to_", 1)]
        return f"{left}->{right}"
    if ":" in text:
        left, right = [part.strip() for part in text.split(":", 1)]
        return f"{left}->{right}"
    return text


def _json(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_vertical_metrics(root: Path) -> pd.DataFrame:
    """Load the vertical-ablation result table without rerunning PIJ methods."""

    root = Path(root)
    candidates = [root / "all_metrics.csv", root / "metrics.csv"]
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if path.exists():
            frames = [pd.read_csv(path)]
            break
    if not frames:
        for path in sorted(root.glob("network=*/pij=*/metrics.csv")):
            frame = pd.read_csv(path)
            network = path.parent.parent.name.removeprefix("network=")
            pij = path.parent.name.removeprefix("pij=")
            if "network_method" not in frame.columns:
                frame["network_method"] = network
            if "pij_method" not in frame.columns:
                frame["pij_method"] = pij
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No vertical-ablation metrics found under {root}. Expected all_metrics.csv "
            "or network=*/pij=*/metrics.csv."
        )
    metrics = pd.concat(frames, ignore_index=True)
    rename = {}
    if "EI_local" in metrics.columns and "EI_lower" not in metrics.columns:
        rename["EI_local"] = "EI_lower"
    if "EI_global" in metrics.columns and "EI_upper" not in metrics.columns:
        rename["EI_global"] = "EI_upper"
    metrics = metrics.rename(columns=rename)
    if "EI_gain" not in metrics.columns and {"EI_lower", "EI_upper"}.issubset(metrics.columns):
        metrics["EI_gain"] = pd.to_numeric(metrics["EI_upper"], errors="coerce") - pd.to_numeric(
            metrics["EI_lower"], errors="coerce"
        )
    required = {"pij_method", "organ", "lower_layer", "upper_layer", "time_pair", "EI_gain"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Vertical metrics are missing columns: {sorted(missing)}")
    for column in ("network_method", "pij_method", "organ", "lower_layer", "upper_layer"):
        if column in metrics.columns:
            metrics[column] = metrics[column].astype(str)
    metrics["time_pair"] = metrics["time_pair"].map(normalize_time_pair)
    metrics["EI_gain"] = pd.to_numeric(metrics["EI_gain"], errors="coerce")
    if "status" in metrics.columns:
        good = ~metrics["status"].astype(str).str.lower().isin({"error", "failed", "failure"})
        metrics = metrics.loc[good].copy()
    return metrics


def find_domain_map(data_root: Path, layer: str, organ: str, stage: str) -> Path:
    root = Path(data_root)
    layer_aliases = {
        "seurat_k150": ("seurat_k150", "k150"),
        "seurat_k40": ("seurat_k40", "k40"),
    }
    aliases = layer_aliases.get(layer, (layer,))
    matches: list[Path] = []
    for alias in aliases:
        base = root / alias / organ
        if base.exists():
            matches.extend(sorted(base.glob(f"*_{organ}_{stage}_spot_domain_map.csv")))
    if not matches:
        # Support passing the layer folder itself as data_root.
        matches.extend(sorted(root.glob(f"**/*_{organ}_{stage}_spot_domain_map.csv")))
        if layer == "seurat_k150":
            matches = [p for p in matches if "150" in p.name or "k150" in str(p.parent).lower()]
        elif layer == "seurat_k40":
            matches = [p for p in matches if "150" not in p.name and "k150" not in str(p.parent).lower()]
    if not matches:
        raise FileNotFoundError(f"No domain map found for layer={layer}, organ={organ}, stage={stage} under {root}")
    return matches[0]


def load_domain_map(data_root: Path, layer: str, organ: str, stage: str) -> pd.DataFrame:
    path = find_domain_map(data_root, layer, organ, stage)
    frame = pd.read_csv(path)
    required = {"spot_id", "domain_id", "x", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["domain_id"] = frame["domain_id"].astype(str)
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    return frame.dropna(subset=["x", "y"])


def load_spot_coordinates(data_root: Path, organ: str, stage: str) -> pd.DataFrame:
    """Get spot coordinates from either K40 or K150 domain maps.

    Both domain maps are defined on the original heart spots, so this avoids loading a
    very large spot h5ad just for x/y coordinates.
    """

    for layer in ("seurat_k40", "seurat_k150"):
        try:
            frame = load_domain_map(data_root, layer, organ, stage)
            return frame[["spot_id", "x", "y"]].drop_duplicates("spot_id")
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not resolve spot coordinates for {organ} E{stage}")


def _decode(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def load_full_slice(slice_root: Path | None, stage: str, organ: str) -> pd.DataFrame | None:
    """Load full-slice x/y and organ annotation when the original MOSTA h5ad exists."""

    if slice_root is None:
        return None
    root = Path(slice_root)
    candidates = [
        root / f"E{stage}_E1S1.MOSTA.h5ad",
        root / f"E{stage}_E1S1.h5ad",
        root / f"E{stage}.h5ad",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    with h5py.File(path, "r") as handle:
        if "obs" not in handle or "obsm" not in handle or "spatial" not in handle["obsm"]:
            raise ValueError(f"{path} does not contain obs and obsm/spatial")
        obs = handle["obs"]
        spatial = np.asarray(handle["obsm"]["spatial"][:], dtype=float)
        if "cell_name" in obs:
            spot_ids = _decode(obs["cell_name"][:])
        elif "_index" in obs:
            spot_ids = _decode(obs["_index"][:])
        else:
            spot_ids = [str(i) for i in range(len(spatial))]
        annotation = [""] * len(spatial)
        if "annotation" in obs:
            item = obs["annotation"]
            categories_ref = item.attrs.get("categories")
            if categories_ref is not None:
                categories = _decode(handle[categories_ref][:])
                codes = np.asarray(item[:], dtype=int)
                annotation = [categories[c] if 0 <= c < len(categories) else "" for c in codes]
            else:
                annotation = _decode(item[:])
    frame = pd.DataFrame({"spot_id": spot_ids, "x": spatial[:, 0], "y": spatial[:, 1], "annotation": annotation})
    frame["is_target_organ"] = frame["annotation"].astype(str).str.lower().eq(organ.lower())
    return frame


def _path_time_pair(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = _TIME_PAIR_RE.fullmatch(part)
        if match:
            return f"{match.group('t')}->{match.group('tp')}"
    return None


def _path_scale(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in {"spot", "seurat_k150", "seurat_k40"}:
            return part
    return None


def discover_coarse_runs(coarse_root: Path, methods: Iterable[str]) -> pd.DataFrame:
    """Discover coarse-graining run summaries recursively.

    This works with both the multiscale wrapper layout and direct single-run folders.
    The summary JSON is authoritative for the method and best-checkpoint metrics.
    """

    root = Path(coarse_root)
    allowed = set(map(str, methods))
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            summary = _json(summary_path)
        except Exception as exc:
            warnings.warn(f"Skipping unreadable {summary_path}: {exc}", RuntimeWarning)
            continue
        method = str(summary.get("method", ""))
        if method not in allowed:
            continue
        run_dir = summary_path.parent
        time_pair = _path_time_pair(run_dir)
        scale = _path_scale(run_dir)
        context_path = run_dir / "experiment_context.json"
        if context_path.exists():
            context = _json(context_path)
            source = str(context.get("source_time", ""))
            target = str(context.get("target_time", ""))
            if source and target:
                time_pair = f"{source}->{target}"
            scale = str(context.get("input_scale", scale or "")) or scale
        seed = None
        for part in run_dir.parts:
            if part.startswith("seed_"):
                try:
                    seed = int(part.removeprefix("seed_"))
                except ValueError:
                    pass
        row: dict[str, object] = {
            "method": method,
            "time_pair": normalize_time_pair(time_pair) if time_pair else None,
            "scale": scale,
            "seed": seed,
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
        }
        for key in (
            "best_epoch",
            "EI_micro_fixed",
            "EI_macro_best_checkpoint",
            "delta_EI_best_checkpoint",
            "hardK_t",
            "hardK_tp",
            "Keff_t",
            "Keff_tp",
            "K",
        ):
            row[key] = summary.get(key)

        # Paper-table contract: best_epoch comes from summary.json, while DeltaEI and
        # hard-K are read from the matching metrics.csv row whenever available.
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists() and summary.get("best_epoch") is not None:
            try:
                history = pd.read_csv(metrics_path)
                epoch = int(summary["best_epoch"])
                match = history[pd.to_numeric(history.get("epoch"), errors="coerce") == epoch]
                if len(match) == 1:
                    best = match.iloc[0]
                    row["metrics_path"] = str(metrics_path)
                    row["delta_EI_paper"] = best.get("delta_EI", summary.get("delta_EI_best_checkpoint"))
                    row["hardK_t_paper"] = best.get("hardK_t", summary.get("hardK_t"))
                    row["hardK_tp_paper"] = best.get("hardK_tp", summary.get("hardK_tp"))
                elif len(match) > 1:
                    raise ValueError(f"Multiple metrics rows at best_epoch={epoch}")
            except Exception as exc:
                warnings.warn(f"Could not resolve best-epoch metrics for {run_dir}: {exc}", RuntimeWarning)
        row.setdefault("delta_EI_paper", summary.get("delta_EI_best_checkpoint"))
        row.setdefault("hardK_t_paper", summary.get("hardK_t"))
        row.setdefault("hardK_tp_paper", summary.get("hardK_tp"))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(
            f"No summary.json files for methods {sorted(allowed)} were found under {root}."
        )
    frame["time_pair"] = frame["time_pair"].map(normalize_time_pair)
    for column in ("delta_EI_best_checkpoint", "delta_EI_paper", "hardK_t", "hardK_tp", "hardK_t_paper", "hardK_tp_paper", "best_epoch", "K", "seed"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def select_coarse_run(
    runs: pd.DataFrame,
    method: str,
    scale: str,
    time_pair: str,
    *,
    k: int | None = 40,
    seed: int | None = 42,
    strict: bool = True,
) -> pd.Series:
    target_pair = normalize_time_pair(time_pair)
    subset = runs[
        (runs["method"].astype(str) == str(method))
        & (runs["scale"].astype(str) == str(scale))
        & (runs["time_pair"].astype(str) == target_pair)
    ].copy()
    if k is not None and "K" in subset.columns:
        subset = subset[pd.to_numeric(subset["K"], errors="coerce") == int(k)]
    if seed is not None and "seed" in subset.columns and subset["seed"].notna().any():
        subset = subset[pd.to_numeric(subset["seed"], errors="coerce") == int(seed)]
    if subset.empty:
        raise KeyError(
            f"Missing coarse run for method={method}, scale={scale}, time_pair={target_pair}, "
            f"K={k}, seed={seed}"
        )
    if len(subset) > 1:
        detail = subset[[c for c in ("run_dir", "K", "seed", "best_epoch") if c in subset.columns]].to_dict(orient="records")
        if strict:
            raise ValueError(f"Ambiguous coarse runs for {method} {target_pair}: {detail}")
        subset = subset.sort_values("run_dir")
    return subset.iloc[0]


def load_assignments(run_dir: Path, side: str) -> pd.DataFrame:
    if side not in {"t", "tp"}:
        raise ValueError("side must be 't' or 'tp'")
    path = Path(run_dir) / f"assignments_{side}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"spot_id", "hard_cluster"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["hard_cluster"] = frame["hard_cluster"].astype(str)
    return frame
