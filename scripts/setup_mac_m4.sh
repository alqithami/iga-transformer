#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-mac.txt
python -m pip install -e .
cat > .env.mac_m4 <<'ENV'
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
ENV
python - <<'PY'
import torch
print('torch', torch.__version__)
print('mps_available', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
import iga_llm
print('iga_llm', iga_llm.__version__)
PY
printf '\nSetup complete. Next:\n  source .venv/bin/activate\n  source .env.mac_m4\n  bash scripts/run_smoke.sh\n'
