#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?usage: run_eval.sh DATA_ROOT CHECKPOINT OUTPUT_ROOT SPLIT}"
CHECKPOINT="${2:?usage: run_eval.sh DATA_ROOT CHECKPOINT OUTPUT_ROOT SPLIT}"
OUTPUT_ROOT="${3:?usage: run_eval.sh DATA_ROOT CHECKPOINT OUTPUT_ROOT SPLIT}"
SPLIT="${4:-test}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m cast.evaluate --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --output-root "$OUTPUT_ROOT" --split "$SPLIT"
