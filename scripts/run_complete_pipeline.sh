#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Compatibility wrapper around the reviewer-grade matrix runner.
# Use environment variables CONFIGS, SEEDS, LIMIT_TRAIN, LIMIT_DEV, LIMIT_EVAL, RUN_ABLATIONS, SKIP_GENERATION, SKIP_LATENCY.
bash scripts/run_full_mac_m4max.sh "$@"
