#!/usr/bin/env bash
# Shared shell helpers for experiment scripts.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/results/paper_data}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'Required file not found: %s\n' "$path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf 'Required directory not found: %s\n' "$path" >&2
    exit 1
  fi
}

require_paper_data() {
  require_file "$DATA_ROOT/data/calibration_train_mix.jsonl"
  require_file "$DATA_ROOT/data/calibration_dev_mix.jsonl"
  require_file "$DATA_ROOT/data/truthfulqa_eval.jsonl"
  require_file "$DATA_ROOT/data/fever_eval.jsonl"
  require_file "$DATA_ROOT/data/halueval_eval.jsonl"
}

expected_rows() {
  case "$1" in
    truthfulqa) printf '149\n' ;;
    fever|halueval) printf '300\n' ;;
    *) printf '0\n' ;;
  esac
}
