#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

def is_bad(x):
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))

bad_files = []
for arg in sys.argv[1:]:
    p = Path(arg)
    rows = 0
    bad_nll = 0
    bad_prob = 0
    pred_counts = {}
    model_ids = set()
    methods = set()
    benches = set()

    with p.open() as f:
        for line in f:
            rows += 1
            r = json.loads(line)
            model_ids.add(r.get("model_id"))
            methods.add(r.get("method"))
            benches.add(r.get("benchmark"))
            pred_counts[r.get("pred_choice")] = pred_counts.get(r.get("pred_choice"), 0) + 1

            nlls = r.get("choice_nlls") or []
            probs = r.get("choice_probs") or []
            if any(is_bad(x) for x in nlls):
                bad_nll += 1
            if any(is_bad(x) for x in probs):
                bad_prob += 1

    status = "OK"
    if bad_nll or bad_prob:
        status = "BAD"
        bad_files.append(str(p))

    print(
        f"{status:3s} rows={rows:4d} bad_nll={bad_nll:4d} bad_prob={bad_prob:4d} "
        f"pred_counts={pred_counts} model={model_ids} method={methods} bench={benches} file={p}"
    )

if bad_files:
    print("\nFiles with NaN/Inf values:")
    for p in bad_files:
        print(" ", p)
    raise SystemExit(1)
