#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_METHODS = (
    "compare_N_kl",
    "compare_NG_kl_splitbeta_v1",
    "compare_Ncomp_Gcos_v2",
    "compare_Nshape_Gcos_v3",
)
DEFAULT_TIME_PAIRS = ("11.5->12.5", "12.5->13.5", "13.5->14.5")
DEFAULT_LEVEL_PAIRS = (
    "spot:seurat_k150",
    "spot:seurat_k40",
    "seurat_k150:seurat_k40",
)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = list(map(str, frame.columns))

    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.8g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _level_pair(frame: pd.DataFrame) -> pd.Series:
    return frame["lower_layer"].astype(str) + ":" + frame["upper_layer"].astype(str)


def discover_metrics(runs_root: Path) -> list[Path]:
    return sorted(
        path
        for path in runs_root.rglob("metrics.csv")
        if path.parent.name.startswith("time=") and any(part.startswith("method=") for part in path.parts)
    )


def load_metrics(runs_root: Path) -> pd.DataFrame:
    paths = discover_metrics(runs_root)
    if not paths:
        raise FileNotFoundError(f"No run-level metrics.csv files found under {runs_root}")
    tables: list[pd.DataFrame] = []
    for path in paths:
        table = pd.read_csv(path)
        table["source_metrics_path"] = str(path.resolve())
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    required = {
        "network_method",
        "pij_method",
        "organ",
        "lower_layer",
        "upper_layer",
        "time_pair",
        "EI_lower",
        "EI_upper",
        "EI_gain",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics are missing required columns: {sorted(missing)}")
    metrics["level_pair"] = _level_pair(metrics)
    key = ["pij_method", "organ", "level_pair", "time_pair"]
    duplicates = metrics.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate benchmark cells found:\n{metrics.loc[duplicates, key].to_string(index=False)}")
    for column in ("EI_lower", "EI_upper", "EI_gain"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return metrics.sort_values(key).reset_index(drop=True)


def validate_matrix(
    metrics: pd.DataFrame,
    *,
    methods: Iterable[str],
    time_pairs: Iterable[str],
    level_pairs: Iterable[str],
) -> None:
    expected_methods = list(map(str, methods))
    expected_times = list(map(str, time_pairs))
    expected_levels = list(map(str, level_pairs))
    expected = {
        (method, time_pair, level_pair)
        for method in expected_methods
        for time_pair in expected_times
        for level_pair in expected_levels
    }
    actual = set(
        metrics.loc[:, ["pij_method", "time_pair", "level_pair"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"Benchmark matrix mismatch. Missing={missing}; unexpected={unexpected}")


def summarize_methods(
    metrics: pd.DataFrame,
    *,
    target_positive_ratio: float,
    target_mean: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in metrics.groupby("pij_method", sort=True):
        gains = group["EI_gain"].to_numpy(dtype=float)
        finite = gains[np.isfinite(gains)]
        positive_count = int(np.count_nonzero(finite > 0.0))
        count = int(finite.size)
        ratio = float(positive_count / count) if count else float("nan")
        mean = float(np.mean(finite)) if count else float("nan")
        rows.append(
            {
                "pij_method": method,
                "count": count,
                "mean_EI_gain": mean,
                "median_EI_gain": float(np.median(finite)) if count else float("nan"),
                "std_EI_gain": float(np.std(finite)) if count else float("nan"),
                "min_EI_gain": float(np.min(finite)) if count else float("nan"),
                "max_EI_gain": float(np.max(finite)) if count else float("nan"),
                "positive_count": positive_count,
                "positive_ratio": ratio,
                "mean_EI_lower": float(group["EI_lower"].mean()),
                "mean_EI_upper": float(group["EI_upper"].mean()),
                "passes_positive_ratio": bool(np.isfinite(ratio) and ratio >= target_positive_ratio),
                "passes_mean_EI_gain": bool(np.isfinite(mean) and mean > target_mean),
                "passes_target": bool(
                    np.isfinite(ratio) and ratio >= target_positive_ratio and np.isfinite(mean) and mean > target_mean
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_target", "positive_ratio", "mean_EI_gain"], ascending=[False, False, False]
    )


def summarize_strata(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimensions in (("pij_method", "level_pair"), ("pij_method", "time_pair")):
        for key, group in metrics.groupby(list(dimensions), sort=True):
            values = group["EI_gain"].to_numpy(dtype=float)
            key_values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(dimensions, key_values))
            row.update(
                {
                    "stratum": dimensions[-1],
                    "count": int(values.size),
                    "mean_EI_gain": float(np.mean(values)),
                    "median_EI_gain": float(np.median(values)),
                    "positive_count": int(np.count_nonzero(values > 0.0)),
                    "positive_ratio": float(np.mean(values > 0.0)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def load_feature_diagnostics(runs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(runs_root.rglob("feature_block_summary.json")):
        if not path.parent.name.startswith("side="):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        nmf = metadata.get("nmf", {})
        blocks = payload.get("blocks", {})
        source_zero = [float(value.get("source_zero_row_fraction", 0.0)) for value in blocks.values()]
        target_zero = [float(value.get("target_zero_row_fraction", 0.0)) for value in blocks.values()]
        rows.append(
            {
                "entry_method": path.parts[next(i for i, part in enumerate(path.parts) if part.startswith("method="))][7:],
                "layer_pair": metadata.get("layer"),
                "time_pair": metadata.get("pair"),
                "side": metadata.get("side"),
                "layer": metadata.get("layer"),
                "nmf_model_type": nmf.get("model_type"),
                "nmf_source_reconstruction_error": nmf.get("source_reconstruction_error"),
                "nmf_target_reconstruction_error": nmf.get("target_reconstruction_error"),
                "nmf_nonfinite": bool(nmf.get("nonfinite", False)),
                "nmf_zero_column_count": int(sum(int(value) for value in nmf.get("zero_columns", {}).values())),
                "max_source_zero_row_fraction": max(source_zero, default=0.0),
                "max_target_zero_row_fraction": max(target_zero, default=0.0),
                "source_feature_path": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def sparse_entropy_decomposition(
    matrix: sparse.spmatrix,
    *,
    eps: float = 1.0e-12,
) -> dict[str, float | int]:
    """Recompute H(J), H(J|I), and EI without densifying the full Pij matrix."""
    probability = sparse.csr_matrix(matrix, dtype=float, copy=True)
    if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] == 0:
        raise ValueError(f"Pij must be a non-empty 2D matrix; got shape={probability.shape}.")
    if probability.data.size and (not np.isfinite(probability.data).all() or np.any(probability.data < 0.0)):
        raise ValueError("Pij contains non-finite or negative entries.")

    n_source, n_target = map(int, probability.shape)
    row_sums = np.asarray(probability.sum(axis=1)).ravel()
    zero_rows = row_sums == 0.0
    nonzero_rows = ~zero_rows
    scale = np.zeros_like(row_sums, dtype=float)
    scale[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    probability = sparse.diags(scale, format="csr") @ probability

    entropy_terms = -probability.data * np.log2(probability.data + eps)
    row_ids = np.repeat(np.arange(n_source, dtype=np.int64), np.diff(probability.indptr))
    row_entropy = np.bincount(row_ids, weights=entropy_terms, minlength=n_source).astype(float)

    zero_count = int(np.count_nonzero(zero_rows))
    if zero_count:
        uniform_value = 1.0 / n_target
        uniform_entropy = -float(n_target) * uniform_value * np.log2(uniform_value + eps)
        row_entropy[zero_rows] = uniform_entropy

    marginal = np.asarray(probability.sum(axis=0)).ravel()
    if zero_count:
        marginal += zero_count / n_target
    marginal /= n_source
    h_j = -float(np.sum(marginal * np.log2(marginal + eps)))
    h_j_given_i = float(np.mean(row_entropy))
    max_entropy = float(np.log2(n_target))
    return {
        "n_source": n_source,
        "n_target": n_target,
        "nnz": int(probability.nnz),
        "zero_row_count": zero_count,
        "H_J": h_j,
        "H_J_given_I": h_j_given_i,
        "EI_recomputed": h_j - h_j_given_i,
        "normalized_H_J": h_j / max_entropy if max_entropy > 0.0 else 0.0,
        "mean_row_entropy": h_j_given_i,
        "median_row_entropy": float(np.median(row_entropy)),
        "p05_row_entropy": float(np.quantile(row_entropy, 0.05)),
        "p95_row_entropy": float(np.quantile(row_entropy, 0.95)),
        "mean_effective_row_support": float(np.mean(np.exp2(row_entropy))),
        "max_probability": float(probability.data.max(initial=0.0)),
    }


def _single_path_value(path: Path, prefix: str) -> str:
    values = {part[len(prefix) :] for part in path.parts if part.startswith(prefix)}
    if len(values) != 1:
        raise ValueError(f"Expected one unique {prefix!r} path component in {path}; found {sorted(values)}")
    return values.pop()


def load_entropy_diagnostics(runs_root: Path, *, level_pairs: Iterable[str]) -> pd.DataFrame:
    pair_lookup = {
        level_pair.replace(":", "_to_"): level_pair
        for level_pair in map(str, level_pairs)
    }
    rows: list[dict[str, object]] = []
    for path in sorted(runs_root.rglob("pij_row_normalized_sparse.npz")):
        pair_key = _single_path_value(path, "pair=")
        if pair_key not in pair_lookup:
            raise ValueError(f"Unexpected level-pair artifact {pair_key!r}: {path}")
        time_key = _single_path_value(path, "time=")
        row: dict[str, object] = {
            "pij_method": _single_path_value(path, "method="),
            "organ": _single_path_value(path, "organ="),
            "level_pair": pair_lookup[pair_key],
            "time_pair": time_key.replace("_to_", "->"),
            "side": _single_path_value(path, "side="),
            "pij_path": str(path.resolve()),
        }
        row.update(sparse_entropy_decomposition(sparse.load_npz(path)))
        rows.append(row)
    return pd.DataFrame(rows)


def validate_entropy_matrix(
    entropy: pd.DataFrame,
    *,
    methods: Iterable[str],
    time_pairs: Iterable[str],
    level_pairs: Iterable[str],
) -> None:
    expected = {
        (str(method), str(time_pair), str(level_pair), side)
        for method in methods
        for time_pair in time_pairs
        for level_pair in level_pairs
        for side in ("lower", "upper")
    }
    if entropy.empty:
        raise ValueError("No pair-level pij_row_normalized_sparse.npz artifacts were found.")
    key = ["pij_method", "time_pair", "level_pair", "side"]
    duplicates = entropy.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate entropy cells found:\n{entropy.loc[duplicates, key].to_string(index=False)}")
    actual = set(entropy.loc[:, key].astype(str).itertuples(index=False, name=None))
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"Entropy matrix mismatch. Missing={missing}; unexpected={unexpected}")


def compare_entropy_with_metrics(metrics: pd.DataFrame, entropy: pd.DataFrame) -> pd.DataFrame:
    if entropy.empty:
        return pd.DataFrame()
    index = ["pij_method", "organ", "time_pair", "level_pair"]
    values = [
        "H_J",
        "H_J_given_I",
        "EI_recomputed",
        "normalized_H_J",
        "mean_row_entropy",
        "median_row_entropy",
        "p05_row_entropy",
        "p95_row_entropy",
        "mean_effective_row_support",
        "n_source",
        "n_target",
        "nnz",
        "zero_row_count",
        "max_probability",
    ]
    wide = entropy.pivot(index=index, columns="side", values=values)
    wide.columns = [f"{value}_{side}" for value, side in wide.columns]
    wide = wide.reset_index()
    metric_columns = index + ["EI_lower", "EI_upper", "EI_gain"]
    comparison = metrics.loc[:, metric_columns].merge(wide, on=index, how="left", validate="one_to_one")
    comparison["EI_gain_recomputed"] = comparison["EI_recomputed_upper"] - comparison["EI_recomputed_lower"]
    comparison["EI_lower_discrepancy"] = comparison["EI_lower"] - comparison["EI_recomputed_lower"]
    comparison["EI_upper_discrepancy"] = comparison["EI_upper"] - comparison["EI_recomputed_upper"]
    comparison["EI_gain_discrepancy"] = comparison["EI_gain"] - comparison["EI_gain_recomputed"]
    return comparison


def build_report(
    *,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    entropy_comparison: pd.DataFrame,
    target_positive_ratio: float,
    target_mean: float,
    split_name: str,
) -> str:
    lines = [
        f"# LightCCI feature benchmark: {split_name}",
        "",
        f"Target gate: positive ratio >= {target_positive_ratio:.1%}, mean deltaEI > {target_mean:.6g} bit.",
        "",
        "This split is evaluated with fixed recipes. Pairwise NMF uses the source and target of the current pair "
        "and is therefore transductive; no third time point or upper-layer label is used in the fit.",
        "",
        "## Method summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Negative cells",
        "",
    ]
    negative = metrics.loc[
        metrics["EI_gain"] <= 0.0,
        ["pij_method", "time_pair", "level_pair", "EI_lower", "EI_upper", "EI_gain"],
    ]
    lines.append(dataframe_to_markdown(negative) if not negative.empty else "No non-positive deltaEI cells.")
    lines.extend(["", "## Diagnostics", ""])
    if diagnostics.empty:
        lines.append("No feature-version pair diagnostics were found (expected for the frozen legacy baseline).")
    else:
        compact = diagnostics.groupby("entry_method", sort=True).agg(
            diagnostic_rows=("entry_method", "size"),
            max_nmf_source_error=("nmf_source_reconstruction_error", "max"),
            max_nmf_target_error=("nmf_target_reconstruction_error", "max"),
            any_nmf_nonfinite=("nmf_nonfinite", "max"),
            max_zero_column_count=("nmf_zero_column_count", "max"),
            max_source_zero_row_fraction=("max_source_zero_row_fraction", "max"),
            max_target_zero_row_fraction=("max_target_zero_row_fraction", "max"),
        )
        lines.append(dataframe_to_markdown(compact.reset_index()))
    lines.extend(["", "## Entropy decomposition", ""])
    if entropy_comparison.empty:
        lines.append("No exported pair-level Pij matrices were found, so entropy was not recomputed.")
    else:
        entropy_summary = entropy_comparison.groupby("pij_method", sort=True).agg(
            mean_H_J_lower=("H_J_lower", "mean"),
            mean_H_J_given_I_lower=("H_J_given_I_lower", "mean"),
            mean_H_J_upper=("H_J_upper", "mean"),
            mean_H_J_given_I_upper=("H_J_given_I_upper", "mean"),
            mean_normalized_H_J_lower=("normalized_H_J_lower", "mean"),
            mean_normalized_H_J_upper=("normalized_H_J_upper", "mean"),
            max_abs_EI_discrepancy=("EI_gain_discrepancy", lambda values: float(np.max(np.abs(values)))),
            max_zero_rows_lower=("zero_row_count_lower", "max"),
            max_zero_rows_upper=("zero_row_count_upper", "max"),
        )
        lines.append(dataframe_to_markdown(entropy_summary.reset_index()))
    lines.append("")
    return "\n".join(lines)


def analyze_benchmark(
    *,
    runs_root: Path,
    output_dir: Path,
    methods: Iterable[str],
    time_pairs: Iterable[str],
    level_pairs: Iterable[str],
    target_positive_ratio: float,
    target_mean: float,
    split_name: str,
    require_entropy_artifacts: bool = False,
) -> dict[str, object]:
    runs_root = Path(runs_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty comparison directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(runs_root)
    validate_matrix(metrics, methods=methods, time_pairs=time_pairs, level_pairs=level_pairs)
    summary = summarize_methods(
        metrics,
        target_positive_ratio=target_positive_ratio,
        target_mean=target_mean,
    )
    strata = summarize_strata(metrics)
    diagnostics = load_feature_diagnostics(runs_root)
    entropy = load_entropy_diagnostics(runs_root, level_pairs=level_pairs)
    if require_entropy_artifacts:
        validate_entropy_matrix(
            entropy,
            methods=methods,
            time_pairs=time_pairs,
            level_pairs=level_pairs,
        )
    entropy_comparison = compare_entropy_with_metrics(metrics, entropy)

    metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    strata.to_csv(output_dir / "stratified_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "diagnostics_summary.csv", index=False)
    entropy.to_csv(output_dir / "entropy_by_side.csv", index=False)
    entropy_comparison.to_csv(output_dir / "entropy_comparison.csv", index=False)
    gate = {
        "split": split_name,
        "target_positive_ratio": float(target_positive_ratio),
        "target_mean_EI_gain": float(target_mean),
        "strictly_greater_mean": True,
        "methods": summary.to_dict(orient="records"),
        "all_methods_pass": bool(summary["passes_target"].all()),
        "any_method_passes": bool(summary["passes_target"].any()),
    }
    (output_dir / "target_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(
        summary=summary,
        metrics=metrics,
        diagnostics=diagnostics,
        entropy_comparison=entropy_comparison,
        target_positive_ratio=target_positive_ratio,
        target_mean=target_mean,
        split_name=split_name,
    )
    (output_dir / "lightcci_feature_benchmark.md").write_text(report, encoding="utf-8")
    return gate


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a locked LightCCI feature-version benchmark matrix.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--time-pairs", nargs="+", default=list(DEFAULT_TIME_PAIRS))
    parser.add_argument("--level-pairs", nargs="+", default=list(DEFAULT_LEVEL_PAIRS))
    parser.add_argument("--target-positive-ratio", type=float, default=0.90)
    parser.add_argument("--target-mean", type=float, default=2.0)
    parser.add_argument("--split-name", default="development_adjacent_pairs")
    parser.add_argument(
        "--require-entropy-artifacts",
        action="store_true",
        help="Require one exported sparse Pij for every method/time/level/side benchmark cell.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if not 0.0 <= args.target_positive_ratio <= 1.0:
        raise ValueError("--target-positive-ratio must be between 0 and 1.")
    gate = analyze_benchmark(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        methods=list(map(str, args.methods)),
        time_pairs=list(map(str, args.time_pairs)),
        level_pairs=list(map(str, args.level_pairs)),
        target_positive_ratio=float(args.target_positive_ratio),
        target_mean=float(args.target_mean),
        split_name=str(args.split_name),
        require_entropy_artifacts=bool(args.require_entropy_artifacts),
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
