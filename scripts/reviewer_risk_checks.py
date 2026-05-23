#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def ece(conf, corr, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    corr = np.asarray(corr, dtype=float)
    out = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            out += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return float(out)


def aurc(conf, corr):
    conf = np.asarray(conf, dtype=float)
    corr = np.asarray(corr, dtype=float)
    order = np.argsort(-conf)
    c = corr[order]
    risks = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(risks.mean())


def acc_at(conf, corr, coverage):
    conf = np.asarray(conf, dtype=float)
    corr = np.asarray(corr, dtype=float)
    order = np.argsort(-conf)
    k = max(1, int(round(coverage * len(order))))
    return float(corr[order[:k]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    rows = []
    for raw in args.predictions:
        path = Path(raw)
        for r in read_jsonl(path):
            method = r.get("method")
            rows.append({
                "path": str(path),
                "id": r.get("id"),
                "benchmark": r.get("benchmark"),
                "method": method,
                "seed": int(r.get("seed", 0)),
                "correct": int(bool(r.get("correct"))),
                "confidence": float(r.get("confidence", 0.0)),
            })
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method, g in df.groupby("method"):
        conf = g["confidence"].to_numpy()
        corr = g["correct"].to_numpy()
        row = {"method": method, "n": len(g), "accuracy": corr.mean(), f"ece{args.bins}": ece(conf, corr, args.bins), "aurc": aurc(conf, corr)}
        for cov in [0.50, 0.70, 0.90, 1.00]:
            row[f"acc_at_{int(cov*100)}"] = acc_at(conf, corr, cov)
            row[f"risk_at_{int(cov*100)}"] = 1.0 - row[f"acc_at_{int(cov*100)}"]
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values(["aurc", "accuracy"], ascending=[True, False])
    summary.to_csv(out_dir / "selective_prediction_summary.csv", index=False)
    nohal = df[df["benchmark"] != "HaluEval-QA"]
    rows2 = []
    for method, g in nohal.groupby("method"):
        conf = g["confidence"].to_numpy()
        corr = g["correct"].to_numpy()
        rows2.append({
            "method": method,
            "n": len(g),
            "accuracy_no_halueval": corr.mean(),
            f"ece{args.bins}_no_halueval": ece(conf, corr, args.bins),
            "aurc_no_halueval": aurc(conf, corr),
        })
    pd.DataFrame(rows2).sort_values("accuracy_no_halueval", ascending=False).to_csv(out_dir / "no_halueval_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nWrote:", out_dir)


if __name__ == "__main__":
    main()
