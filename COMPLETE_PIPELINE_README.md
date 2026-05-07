# Complete pipeline guide

This file is retained for compatibility. The authoritative docs are:

- `README.md` for setup and commands
- `RUNBOOK.md` for reviewer-audit mapping
- `STATUS.md` for what changed and remaining external dependencies

Use:

```bash
cd code
bash scripts/setup_mac_m4.sh
source .venv/bin/activate
source .env.mac_m4
bash scripts/run_smoke.sh
CONFIGS="configs/llama3_8b_iga.yaml configs/mistral_7b_iga.yaml" SEEDS="1 2 3" LIMIT_TRAIN=1000 LIMIT_DEV=300 LIMIT_EVAL=300 RUN_ABLATIONS=1 bash scripts/run_full_mac_m4max.sh
```
