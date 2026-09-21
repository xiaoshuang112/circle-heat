#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p runs/v6
.venv/bin/python -u train.py \
  --labels data/labels-v5.jsonl \
  --output runs/v6 \
  --epochs 100 \
  --batch-size 16 \
  --heatmap-sigma 1.0 \
  "$@" 2>&1 | tee runs/v6/train.log
