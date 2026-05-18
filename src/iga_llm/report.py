from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import aggregate_predictions, calibration_bins, format_metric, mean_std_ci
from .utils import ensure_dir, json_dump, read_jsonl


def _rows_from_prediction_files(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def aggregate_per_seed(rows: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("model_id", "unknown")),
                str(row.get("benchmark", "unknown")),
                str(row.get("method", "unknown")),
                int(row.get("seed", -1)),
                str(row.get("evaluation_mode", "unknown")),
            )
        ].append(row)
    out_rows = []
    for (model_id, benchmark, method, seed, evaluation_mode), group in sorted(groups.items()):
        metrics = aggregate_predictions(group)
        metrics.update({"model_id": model_id, "benchmark": benchmark, "method": method, "seed": seed, "evaluation_mode": evaluation_mode})
        out_rows.append(metrics)
    return pd.DataFrame(out_rows)


def aggregate_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return per_seed
    key_cols = ["model_id", "benchmark", "method", "evaluation_mode"]
    metric_cols = [c for c in per_seed.columns if c not in key_cols + ["seed"]]
    rows: list[dict[str, Any]] = []
    for keys, group in per_seed.groupby(key_cols, dropna=False):
        row = dict(zip(key_cols, keys))
        row["num_seeds"] = int(group["seed"].nunique())
        row["n_total"] = float(group.get("n", pd.Series(dtype=float)).sum())
        for col in metric_cols:
            if col == "n":
                continue
            vals = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy(dtype=float)
            stats = mean_std_ci(vals)
            row[col] = stats["mean"]
            row[f"{col}_std_across_seeds"] = stats["std"]
            row[f"{col}_ci_low_across_seeds"] = stats["ci_low"]
            row[f"{col}_ci_high_across_seeds"] = stats["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key_cols).reset_index(drop=True)


def build_calibration_bins(rows: list[dict[str, Any]], n_bins: int = 15) -> pd.DataFrame:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("correct") is None or row.get("confidence") is None:
            continue
        groups[(str(row.get("model_id", "unknown")), str(row.get("benchmark", "unknown")), str(row.get("method", "unknown")), int(row.get("seed", -1)))].append(row)
    out: list[dict[str, Any]] = []
    for (model_id, benchmark, method, seed), group in groups.items():
        bins = calibration_bins([bool(r["correct"]) for r in group], [float(r["confidence"]) for r in group], n_bins=n_bins)
        for b in bins:
            b.update({"model_id": model_id, "benchmark": benchmark, "method": method, "seed": seed})
            out.append(b)
    return pd.DataFrame(out)


def paired_deltas(rows: list[dict[str, Any]], baseline: str = "vanilla_mc") -> pd.DataFrame:
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("correct") is None:
            continue
        key = (str(row.get("model_id")), str(row.get("benchmark")), str(row.get("id")), int(row.get("seed", -1)))
        by_key.setdefault(key, {})[str(row.get("method"))] = row
    out = []
    for (model_id, benchmark, ex_id, seed), methods in by_key.items():
        if baseline not in methods:
            continue
        base_correct = float(bool(methods[baseline].get("correct")))
        for method, row in methods.items():
            if method == baseline:
                continue
            out.append(
                {
                    "model_id": model_id,
                    "benchmark": benchmark,
                    "id": ex_id,
                    "seed": seed,
                    "baseline": baseline,
                    "method": method,
                    "baseline_correct": base_correct,
                    "method_correct": float(bool(row.get("correct"))),
                    "delta_correct": float(bool(row.get("correct"))) - base_correct,
                }
            )
    return pd.DataFrame(out)


def summarize_paired_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return delta_df
    rows = []
    for keys, group in delta_df.groupby(["model_id", "benchmark", "baseline", "method"], dropna=False):
        vals = group["delta_correct"].to_numpy(dtype=float)
        stats = mean_std_ci(vals)
        row = dict(zip(["model_id", "benchmark", "baseline", "method"], keys))
        row.update({"mean_delta_accuracy": stats["mean"], "std_delta_accuracy": stats["std"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "n_pairs": len(vals)})
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_mean_std(row: pd.Series, metric: str) -> str:
    mean = row.get(metric)
    std = row.get(f"{metric}_std_across_seeds")
    if pd.isna(mean):
        return "--"
    if std is not None and not pd.isna(std):
        return f"{format_metric(mean)} $\\pm$ {format_metric(std)}"
    return format_metric(mean)



def _latex_escape(text: Any) -> str:
    return str(text).replace("_", r"\_")


def write_latex_main_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = ["model_id", "benchmark", "method", "accuracy", "hallucination_rate", "ece", "brier", "answer_rate", "refusal_rate", "mean_latency_s", "num_seeds"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    main_methods = [
        "vanilla_mc",
        "temperature_mc",
        "self_consistency_mc",
        "semantic_entropy_mc",
        "iga_mc",
        "iga_v2_lowrank_mc",
        "iga_v3_risk_sparse_mc",
        "iga_v3_entropy_sparse_mc",
        "iga_v3_hybrid_sparse_mc",
        "iga_v3_risk_mc",
    ]
    compact = df[df["method"].isin(main_methods)] if not df.empty else df
    if compact.empty:
        compact = df
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main factuality and calibration results. FA and answer rate are higher-is-better; hallucination rate, ECE, Brier, refusal rate, and latency are lower-is-better. Values are mean $\pm$ std across seeds when multiple seeds are available.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lllcccccc}",
        r"\toprule",
        r"Model & Benchmark & Method & FA$\uparrow$ & HR$\downarrow$ & ECE$\downarrow$ & Brier$\downarrow$ & Ans.$\uparrow$ & Lat. s$\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in compact.sort_values(["model_id", "benchmark", "method"]).iterrows():
        model_short = str(row["model_id"]).replace("meta-llama/", "").replace("mistralai/", "")
        method = _latex_escape(row["method"])
        lines.append(
            f"{_latex_escape(model_short)} & {_latex_escape(row['benchmark'])} & {method} & "
            f"{_fmt_mean_std(row, 'accuracy')} & {_fmt_mean_std(row, 'hallucination_rate')} & "
            f"{_fmt_mean_std(row, 'ece')} & {_fmt_mean_std(row, 'brier')} & "
            f"{_fmt_mean_std(row, 'answer_rate')} & {_fmt_mean_std(row, 'mean_latency_s')} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_ablation_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ab = df[df["method"].str.startswith("ablate_", na=False)].copy() if not df.empty and "method" in df else df
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{IGA ablations. Values are averaged across benchmark examples and seeds.}",
        r"\label{tab:ablations}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Benchmark & Ablation & FA$\uparrow$ & HR$\downarrow$ & ECE$\downarrow$ & Lat. s$\downarrow$ \\",
        r"\midrule",
    ]
    if ab.empty:
        lines.append(r"-- & -- & -- & -- & -- & -- \\")
    else:
        for _, row in ab.sort_values(["benchmark", "method"]).iterrows():
            method = _latex_escape(row["method"])
            lines.append(
                f"{_latex_escape(row['benchmark'])} & {method} & "
                f"{_fmt_mean_std(row, 'accuracy')} & {_fmt_mean_std(row, 'hallucination_rate')} & "
                f"{_fmt_mean_std(row, 'ece')} & {_fmt_mean_std(row, 'mean_latency_s')} \\\\" 
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_efficiency_table(latency_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Generation latency and memory. Overhead is relative to vanilla for the same model, seed, and prompt length.}",
        r"\label{tab:efficiency}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Method & Prompt toks & Lat. s$\downarrow$ & Tok/s$\uparrow$ & Mem MB$\downarrow$ \\",
        r"\midrule",
    ]
    if latency_df.empty:
        lines.append(r"-- & -- & -- & -- & -- & -- \\")
    else:
        for _, row in latency_df.sort_values(["model_id", "prompt_tokens", "method"]).iterrows():
            model_short = str(row.get("model_id", "unknown")).replace("meta-llama/", "").replace("mistralai/", "")
            lines.append(
                f"{_latex_escape(model_short)} & {_latex_escape(row.get('method'))} & {int(row.get('prompt_tokens', 0))} & "
                f"{format_metric(row.get('latency_s'))} & {format_metric(row.get('tokens_per_second'))} & {format_metric(row.get('peak_memory_mb'))} \\\\" 
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_latency(paths: list[str]) -> pd.DataFrame:
    rows = _rows_from_prediction_files(paths) if paths else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    key_cols = ["model_id", "method", "prompt_tokens", "max_new_tokens", "seed"]
    numeric = ["latency_s", "tokens_per_second", "peak_memory_mb", "generated_tokens"]
    out = df.groupby(key_cols, dropna=False)[numeric].mean().reset_index()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate IGA JSONL prediction files into CSV, LaTeX, calibration, and audit summaries.")
    parser.add_argument("--predictions", nargs="*", default=[])
    parser.add_argument("--latency", nargs="*", default=[])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rows = _rows_from_prediction_files(args.predictions) if args.predictions else []
    per_seed = aggregate_per_seed(rows)
    across = aggregate_across_seeds(per_seed)
    per_seed.to_csv(out_dir / "summary_per_seed.csv", index=False)
    across.to_csv(out_dir / "summary_by_model_benchmark_method.csv", index=False)
    bins = build_calibration_bins(rows, n_bins=args.n_bins)
    bins.to_csv(out_dir / "calibration_bins.csv", index=False)
    deltas = paired_deltas(rows)
    deltas.to_csv(out_dir / "paired_deltas_vs_vanilla.csv", index=False)
    summarize_paired_deltas(deltas).to_csv(out_dir / "paired_delta_summary.csv", index=False)
    write_latex_main_table(across, out_dir / "main_results_table.tex")
    write_latex_ablation_table(across, out_dir / "ablation_table.tex")
    latency_df = aggregate_latency(args.latency)
    latency_df.to_csv(out_dir / "latency_summary.csv", index=False)
    write_latex_efficiency_table(latency_df, out_dir / "efficiency_table.tex")
    report = {
        "prediction_files": args.predictions,
        "latency_files": args.latency,
        "n_prediction_rows": len(rows),
        "n_groups": int(len(across)),
        "outputs": [
            "summary_per_seed.csv",
            "summary_by_model_benchmark_method.csv",
            "calibration_bins.csv",
            "paired_deltas_vs_vanilla.csv",
            "paired_delta_summary.csv",
            "main_results_table.tex",
            "ablation_table.tex",
            "latency_summary.csv",
            "efficiency_table.tex",
        ],
    }
    json_dump(out_dir / "REPORT.json", report)
    print(across.to_string(index=False) if not across.empty else "No prediction rows found.")


if __name__ == "__main__":
    main()
