#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a local clone of https://github.com/alqithami/iga-transformer
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(pwd)}"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "ERROR: run this from a git clone root or pass the clone path as the first argument."
  exit 1
fi

rsync -av --delete "$SRC_DIR/src/" "$REPO_DIR/src/"
rsync -av --delete "$SRC_DIR/configs/" "$REPO_DIR/configs/"
rsync -av --delete "$SRC_DIR/scripts/" "$REPO_DIR/scripts/"
rsync -av "$SRC_DIR/docs/" "$REPO_DIR/docs/"
cp "$SRC_DIR/README.md" "$REPO_DIR/README.md"
cp "$SRC_DIR/pyproject.toml" "$REPO_DIR/pyproject.toml"
cp "$SRC_DIR/requirements_gpu.txt" "$REPO_DIR/requirements_gpu.txt"
cp "$SRC_DIR/.gitignore" "$REPO_DIR/.gitignore"
cp "$SRC_DIR/CITATION.cff" "$REPO_DIR/CITATION.cff"

cd "$REPO_DIR"
python -m compileall src/iga_llm

echo

echo "Updated files copied. Review with:"
echo "  git status"
echo "  git diff --stat"
echo "  git diff"
echo

echo "Recommended commit:"
echo "  git add README.md pyproject.toml requirements_gpu.txt .gitignore CITATION.cff docs configs scripts src"
echo "  git commit -m 'Update IGA reproducibility software artifact'"
