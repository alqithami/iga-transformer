from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

_REFUSAL_PATTERNS = [
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bunknown\b",
    r"\bunsure\b",
    r"\bi am not sure\b",
    r"\bunable to verify\b",
    r"\bno reliable information\b",
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_refusal(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in _REFUSAL_PATTERNS)


def expected_calibration_error(correct: list[bool] | np.ndarray, confidence: list[float] | np.ndarray, n_bins: int = 15) -> float:
    correct_arr = np.asarray(correct, dtype=float)
    conf_arr = np.asarray(confidence, dtype=float)
    if correct_arr.size == 0:
        return float("nan")
    conf_arr = np.clip(conf_arr, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        if mask.any():
            acc = correct_arr[mask].mean()
            conf = conf_arr[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def calibration_bins(correct: list[bool] | np.ndarray, confidence: list[float] | np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    correct_arr = np.asarray(correct, dtype=float)
    conf_arr = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    n = max(1, len(correct_arr))
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = ((conf_arr >= lo) & (conf_arr <= hi)) if hi == 1.0 else ((conf_arr >= lo) & (conf_arr < hi))
        rows.append(
            {
                "bin": float(idx),
                "lo": float(lo),
                "hi": float(hi),
                "n": float(mask.sum()),
                "fraction": float(mask.sum() / n),
                "accuracy": float(correct_arr[mask].mean()) if mask.any() else float("nan"),
                "confidence": float(conf_arr[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return rows


def brier_score(correct: list[bool] | np.ndarray, confidence: list[float] | np.ndarray) -> float:
    correct_arr = np.asarray(correct, dtype=float)
    conf_arr = np.asarray(confidence, dtype=float)
    if correct_arr.size == 0:
        return float("nan")
    return float(np.mean((conf_arr - correct_arr) ** 2))


def bootstrap_ci(values: list[float] | np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def mean_std_ci(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    # t≈1.96 is acceptable for summary; raw per-seed CSV is also emitted.
    half = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "n": float(arr.size)}


def majority_vote(labels: list[Any]) -> Any:
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def aggregate_predictions(rows: list[dict[str, Any]], n_bins: int = 15) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(rows))}
    if not rows:
        return out
    choice_rows = [r for r in rows if r.get("correct") is not None and r.get("confidence") is not None]
    if choice_rows:
        correct = [bool(r.get("correct")) for r in choice_rows]
        conf = [float(r.get("confidence", 0.0)) for r in choice_rows]
        out["accuracy"] = float(np.mean(correct))
        out["hallucination_rate"] = 1.0 - out["accuracy"]
        lo, hi = bootstrap_ci([float(x) for x in correct], seed=0)
        out["accuracy_ci_low_bootstrap_examples"] = lo
        out["accuracy_ci_high_bootstrap_examples"] = hi
        out["ece"] = expected_calibration_error(correct, conf, n_bins=n_bins)
        out["brier"] = brier_score(correct, conf)
        out["parse_success_rate"] = float(np.mean([bool(r.get("parse_success", True)) for r in choice_rows if r.get("parse_success") is not None])) if any(r.get("parse_success") is not None for r in choice_rows) else float("nan")
    lengths = [r.get("generated_tokens") for r in rows if r.get("generated_tokens") is not None]
    if lengths:
        out["mean_generated_tokens"] = float(np.mean(lengths))
    refusals = [bool(r.get("is_refusal", False)) for r in rows if "is_refusal" in r]
    if refusals:
        out["refusal_rate"] = float(np.mean(refusals))
        out["answer_rate"] = 1.0 - out["refusal_rate"]
    else:
        out["refusal_rate"] = 0.0
        out["answer_rate"] = 1.0
    latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
    tokens = [float(r.get("generated_tokens", 0.0)) for r in rows if r.get("latency_s")]
    if latencies:
        out["mean_latency_s"] = float(np.mean(latencies))
        denom = sum(latencies)
        out["tokens_per_second"] = float(sum(tokens) / denom) if denom > 0 else float("nan")
    for key in ["iga_gamma_mean", "iga_gamma_max", "iga_gate_mean", "iga_gate_max", "iga_pattern_mean", "iga_active_heads"]:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if values:
            out[key] = float(np.mean(values))
    errors = [r for r in rows if r.get("error")]
    out["error_rate"] = float(len(errors) / len(rows))
    return out


def format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
    except Exception:
        return "--"
    if math.isnan(v):
        return "--"
    return f"{v:.{digits}f}"
