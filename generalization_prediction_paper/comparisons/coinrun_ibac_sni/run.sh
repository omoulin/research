#!/usr/bin/env bash
# Run the IBAC-SNI comparison on GPU, logging to results/<variant>.log.
#   bash run.sh                 # main arm: IBAC-SNI (lambda=0.5, beta=1e-4), 10 x 2M steps
#   bash run.sh ibac baseline   # extra arms, run one after another
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available')"
mkdir -p results
for v in "${@:-ibac-sni}"; do
    "$PY" -u run_ibac_sni.py --variant "$v" --device cuda 2>&1 | tee -a "results/$v.log"
done
"$PY" compare.py
