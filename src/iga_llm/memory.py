from __future__ import annotations

import os
import resource

import torch


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return float(torch.mps.current_allocated_memory() / (1024**2))
        except Exception:
            pass
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes. Heuristic: huge values are bytes.
    if rss_kb > 10_000_000:
        return float(rss_kb / (1024**2))
    return float(rss_kb / 1024)
