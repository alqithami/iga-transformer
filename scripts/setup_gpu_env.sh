#!/usr/bin/env bash
set -euo pipefail
cd "${1:-$HOME/IGA_GPU}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -U transformers datasets accelerate safetensors sentencepiece protobuf tokenizers tqdm numpy pandas scikit-learn pyyaml psutil 'huggingface_hub[cli]' peft
python -m pip install -e .
python - <<'PY'
import torch, transformers, datasets, accelerate, pandas, sklearn, yaml
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('bf16_supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
PY
