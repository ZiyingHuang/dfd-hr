#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid RUN_ID: %s\n' "$RUN_ID" >&2
    exit 1
fi
OUTPUT_GROUP="${OUTPUT_GROUP:-dfbench}"
if [[ ! "$OUTPUT_GROUP" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid OUTPUT_GROUP: %s\n' "$OUTPUT_GROUP" >&2
    exit 1
fi
OUTPUT_DIR="$PROJECT_ROOT/outputs/$OUTPUT_GROUP/train/$RUN_ID"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/training${PYTHONPATH:+:$PYTHONPATH}"

python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --detector_path ./training/config/detector/dfd_hr.yaml \
    --train_dataset FaceForensics++ \
    --test_dataset Celeb-DF-v2 \
    --output_dir "$OUTPUT_DIR" \
    --run_id "$RUN_ID" \
    --output_group "$OUTPUT_GROUP" \
    --ddp
