#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load(paths, force_method=None):
    rows = []
    for path in paths:
        path = Path(path)
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "id": r.get("id"),
                    "benchmark": r.get("benchmark"),
                    "seed": int(r.get("seed", 0)),
                    "method": force_method or r.get("method"),
                    "correct": int(bool(r.get("correct"))),
                    "confidence": float(r.get("confidence", 0.0)),
                })
    return pd.DataFrame(rows)


def ece(conf, corr, bins=10):
    conf = np.asarray(conf)
    corr = np.asarray(corr)
    n = len(conf)
    val = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (conf >= lo) & ((conf <= hi) if b == bins - 1 else (conf < hi))
        if mask.any():
            val += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return float(val)


def aurc(conf, corr):
    order = np.argsort(-np.asarray(conf))
    corr = np.asarray(corr)[order]
    return float((1.0 - np.cumsum(corr) / np.arange(1, len(corr)+1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iga", nargs="+", required=True)
    ap.add_argument("--lora", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    iga = load(args.iga)
    iga["method"] = "iga"
    lora = load(args.lora, force_method="lora")
    merged = iga.merge(lora, on=["seed", "benchmark", "id"], suffixes=("_iga", "_lora"))
    rng = np.random.default_rng(13)
    n = len(merged)
    def point(sample):
        return {
            "acc_diff_iga_minus_lora": sample.correct_iga.mean() - sample.correct_lora.mean(),
            "ece_diff_iga_minus_lora": ece(sample.confidence_iga, sample.correct_iga) - ece(sample.confidence_lora, sample.correct_lora),
            "aurc_diff_iga_minus_lora": aurc(sample.confidence_iga, sample.correct_iga) - aurc(sample.confidence_lora, sample.correct_lora),
        }
    point_est = point(merged)
    boots = {k: [] for k in point_est}
    for _ in range(args.boot):
        idx = rng.integers(0, n, n)
        sample = merged.iloc[idx]
        p = point(sample)
        for k, v in p.items():
            boots[k].append(v)
    rows = []
    for k, vals in boots.items():
        lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
        rows.append({"metric": k, "point": point_est[k], "boot_p2_5": lo, "boot_median": med, "boot_p97_5": hi})
    result = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
