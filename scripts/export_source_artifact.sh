#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash scripts/static_check.sh

OUT="${1:-iga_source_artifact.zip}"
if [[ "$OUT" != /* ]]; then
  OUT="$REPO_ROOT/$OUT"
fi
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/iga_source_artifact"
mkdir -p "$DEST"

cp README.md LICENSE CITATION.cff pyproject.toml requirements_gpu.txt "$DEST/"
cp -R configs docs scripts src "$DEST/"

find "$DEST" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$DEST" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

python - "$DEST" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
rows = []
for path in sorted(p for p in root.rglob('*') if p.is_file()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append({'path': str(path.relative_to(root)), 'sha256': digest, 'bytes': path.stat().st_size})
(root / 'MANIFEST.json').write_text(json.dumps({'files': rows}, indent=2) + '\n')
PY

rm -f "$OUT"
(
  cd "$STAGE"
  zip -qr "$OUT" iga_source_artifact
)
sha256sum "$OUT"
echo "Wrote $OUT"
