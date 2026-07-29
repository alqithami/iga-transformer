#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

OUT="${OUT:-results/qwen2_5_7b_three_seed_compare}"
export OUT

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import subprocess
import sys

expected_rows = {"truthfulqa": 149, "fever": 300, "halueval": 300}
selected = []

for seed in (1, 2, 3):
    candidates = sorted(Path("results").glob(f"qwen2_5_7b_lowrank_g2_matched_seed{seed}_*"), reverse=True)
    complete = []
    for root in candidates:
        files = sorted((root / "predictions").glob("*.jsonl"))
        if len(files) != 15:
            continue
        valid = True
        for path in files:
            benchmark = next((name for name in expected_rows if name in path.name.lower()), None)
            if benchmark is None or sum(1 for _ in path.open()) != expected_rows[benchmark]:
                valid = False
                break
        if valid:
            complete.append((root, files))
    if not complete:
        raise SystemExit(f"No complete 15-file Qwen result directory found for seed {seed}.")
    root, files = complete[0]
    print(f"Using seed {seed}: {root}")
    selected.extend(files)

out = Path(os.environ["OUT"])
aggregate = out / "aggregate"
aggregate.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [sys.executable, "-m", "iga_llm.report", "--out_dir", str(aggregate), "--predictions", *map(str, selected)],
    check=True,
)
print(f"Wrote {aggregate}")
PY
