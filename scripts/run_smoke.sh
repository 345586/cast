#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?usage: run_smoke.sh DATA_ROOT OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?usage: run_smoke.sh DATA_ROOT OUTPUT_ROOT}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m cast.train --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --trial smoke --epochs 1 --patience 1 --smoke
