#!/usr/bin/env bash
set -euo pipefail
cd ~/IGA_GPU
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import subprocess, sys
out = Path('results/qwen2_5_7b_three_seed_compare')
agg = out / 'aggregate'
agg.mkdir(parents=True, exist_ok=True)
files = []
for root in sorted(Path('results').glob('qwen2_5_7b_lowrank_g2_matched_seed*')):
    files.extend(sorted((root / 'predictions').glob('*.jsonl')))
seen=set(); files=[f for f in files if not (str(f) in seen or seen.add(str(f)))]
print(f'Using {len(files)} Qwen prediction files')
subprocess.run([sys.executable, '-m', 'iga_llm.report', '--out_dir', str(agg), '--predictions', *map(str, files)], check=True)
print('Wrote', agg)
PY
