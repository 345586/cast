#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?usage: run_train.sh DATA_ROOT OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?usage: run_train.sh DATA_ROOT OUTPUT_ROOT}"
SEED="${3:-45}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m cast.train --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --trial "seed-${SEED}" --seed "$SEED"
