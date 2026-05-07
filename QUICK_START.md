# Quick start

```bash
cd code
bash scripts/setup_mac_m4.sh
source .venv/bin/activate
source .env.mac_m4
bash scripts/run_smoke.sh
```

First serious run:

```bash
CONFIGS="configs/mistral_7b_iga.yaml" SEEDS="1" LIMIT_TRAIN=300 LIMIT_DEV=100 LIMIT_EVAL=100 RUN_ABLATIONS=1 SKIP_GENERATION=1 bash scripts/run_full_mac_m4max.sh
```

Full two-model run:

```bash
bash scripts/run_two_model_paper_matrix.sh
```

See `README.md` and `RUNBOOK.md` for the complete reviewer audit workflow.
