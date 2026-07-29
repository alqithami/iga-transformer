#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -m compileall -q src/iga_llm scripts

for script in scripts/*.sh; do
  bash -n "$script"
done

python - <<'PY'
from pathlib import Path
import yaml
paths = sorted(Path("configs").glob("*.yaml"))
for path in paths:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Configuration is not a mapping: {path}")
print(f"Validated {len(paths)} YAML configurations.")
PY

if grep -RInE '/Users/[^/]+/|/root/|/home/[^/]+/' \
  README.md docs configs src pyproject.toml requirements_gpu.txt CITATION.cff 2>/dev/null; then
  echo "Machine-specific path found." >&2
  exit 1
fi

tracked_model_state="$(git ls-files '*.pt' '*.pth' '*.bin' '*.safetensors' || true)"
if [[ -n "$tracked_model_state" ]]; then
  echo "Generated model state is tracked:" >&2
  printf '%s\n' "$tracked_model_state" >&2
  exit 1
fi

echo "Static checks passed."
