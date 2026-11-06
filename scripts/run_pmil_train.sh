#!/usr/bin/env bash
# Train DFD-HR + PMIL, then (by default) evaluate the best checkpoint on the 6 target test sets.
#
# Main experiment (fast, frozen official DFD-HR + trainable PMIL branch):
#   RUN_ID=pmil_main bash scripts/run_pmil_train.sh
# Ablation (any detector yaml):
#   DETECTOR=training/config/detector/pmil/abl_free_queries.yaml RUN_ID=abl_free bash scripts/run_pmil_train.sh
#
# Env: DETECTOR, RUN_ID, NPROC (default 4), CUDA_VISIBLE_DEVICES, MASTER_PORT,
#      AUTO_EVAL (default 1), EVAL_GPUS (default 0,1,2,3), OUTPUT_GROUP (default pmil)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

DETECTOR="${DETECTOR:-./training/config/detector/dfd_hr_pmil.yaml}"
RUN_ID="${RUN_ID:-pmil_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29517}"
OUTPUT_GROUP="${OUTPUT_GROUP:-pmil}"
AUTO_EVAL="${AUTO_EVAL:-1}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid RUN_ID: %s\n' "$RUN_ID" >&2; exit 1
fi
[[ -f "$DETECTOR" ]] || { printf 'Missing detector config: %s\n' "$DETECTOR" >&2; exit 1; }
[[ -f training/config/part_template_32.npy ]] || {
    printf 'Missing training/config/part_template_32.npy -- run scripts/build_part_template.py first\n' >&2; exit 1; }

OUTPUT_DIR="$PROJECT_ROOT/outputs/$OUTPUT_GROUP/train/$RUN_ID"
if [[ -e "$OUTPUT_DIR/training.log" ]]; then
    printf 'Run already exists: %s\n' "$OUTPUT_DIR" >&2; exit 1
fi
mkdir -p "$OUTPUT_DIR"
cp "$DETECTOR" "$OUTPUT_DIR/detector_config.yaml"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HUB_DISABLE_XET=1 \
       HF_HUB_DISABLE_TELEMETRY=1 NO_ALBUMENTATIONS_UPDATE=1 \
       PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/training${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # less fragmentation

# Same protocol as scripts/run_dfd_hr.sh: train on FF++ c23, select the checkpoint on Celeb-DF-v2.
python3 -m torch.distributed.launch \
    --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    scripts/train.py \
    --detector_path "$DETECTOR" \
    --train_dataset FaceForensics++ \
    --test_dataset Celeb-DF-v2 \
    --output_dir "$OUTPUT_DIR" \
    --run_id "$RUN_ID" \
    --output_group "$OUTPUT_GROUP" \
    --no-save_feat \
    --ddp

CKPT="$OUTPUT_DIR/test/avg/ckpt_best.pth"
[[ -f "$CKPT" ]] || CKPT="$OUTPUT_DIR/test/Celeb-DF-v2/ckpt_best.pth"
printf 'best checkpoint: %s\n' "$CKPT"

if [[ "$AUTO_EVAL" == "1" ]]; then
    CHECKPOINT="$CKPT" DETECTOR="$DETECTOR" RUN_ID="${RUN_ID}_eval" \
    OUTPUT_GROUP="$OUTPUT_GROUP" GPUS="${EVAL_GPUS:-0,1,2,3}" \
        bash scripts/run_pmil_eval.sh
fi
