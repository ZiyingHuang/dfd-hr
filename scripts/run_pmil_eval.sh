#!/usr/bin/env bash
# Evaluate a checkpoint on the 6 target test sets with the paper protocol (32 frames, video AUC):
#   Celeb-DF-v2, FaceShifter, DeepFakeDetection, DFDC, DFDCP  <- /data/hjr/dfbench/rgb (JSON manifests)
#   WDF                                                         <- /data/hjr/deepfake/testset (folder labels)
#
# Usage:
#   CHECKPOINT=outputs/pmil/train/<RUN>/test/avg/ckpt_best.pth \
#   DETECTOR=training/config/detector/dfd_hr_pmil.yaml RUN_ID=<RUN>_eval bash scripts/run_pmil_eval.sh
# Baseline with the identical protocol:
#   CHECKPOINT=checkpoints/ckpt_best.pth DETECTOR=training/config/detector/dfd_hr.yaml \
#   RUN_ID=baseline_6set bash scripts/run_pmil_eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/ckpt_best.pth}"
DETECTOR="${DETECTOR:-$PROJECT_ROOT/training/config/detector/dfd_hr_pmil.yaml}"
TEST_CONFIG="${TEST_CONFIG:-$PROJECT_ROOT/training/config/test_config.yaml}"
WDF_ROOT="${WDF_ROOT:-/data/hjr/deepfake/testset}"
OUTPUT_GROUP="${OUTPUT_GROUP:-pmil}"
GPUS="${GPUS:-0,1,2,3}"
RUN_ID="${RUN_ID:-eval_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid RUN_ID: %s\n' "$RUN_ID" >&2; exit 1
fi
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [[ "${#GPU_LIST[@]}" -lt 4 ]]; then
    printf 'GPUS needs 4 ids (got %s); repeat ids to share a GPU, e.g. GPUS=0,0,1,1\n' "$GPUS" >&2; exit 1
fi
for f in "$CHECKPOINT" "$DETECTOR" "$TEST_CONFIG"; do
    [[ -f "$f" ]] || { printf 'Missing file: %s\n' "$f" >&2; exit 1; }
done
[[ -d "$WDF_ROOT/WDF" ]] || { printf 'Missing WDF folder: %s/WDF\n' "$WDF_ROOT" >&2; exit 1; }

RESULT_DIR="$PROJECT_ROOT/outputs/$OUTPUT_GROUP/eval/$RUN_ID"
mkdir -p "$RESULT_DIR"
if [[ -e "$RESULT_DIR/${OUTPUT_GROUP}_${RUN_ID}.json" ]]; then
    printf 'Output already exists for RUN_ID: %s\n' "$RUN_ID" >&2; exit 1
fi
WORK_DIR="$RESULT_DIR/workers"
mkdir -p "$WORK_DIR"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HUB_DISABLE_XET=1 \
       HF_HUB_DISABLE_TELEMETRY=1 NO_ALBUMENTATIONS_UPDATE=1 \
       PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/training"

json_eval() {  # gpu name datasets...
    local gpu="$1" name="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/test.py \
        --detector_path "$DETECTOR" --test_config "$TEST_CONFIG" \
        --test_dataset "$@" --weights_path "$CHECKPOINT" \
        --output_dir "$WORK_DIR" --run_id "$RUN_ID" --source_type json --quiet \
        > "$WORK_DIR/${name}_gpu${gpu}.log" 2>&1
}
wdf_eval() {  # gpu
    local gpu="$1"
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/evaluate_deepfake_testset.py \
        --root "$WDF_ROOT" --datasets WDF --checkpoint "$CHECKPOINT" \
        --detector-path "$DETECTOR" --test-config "$TEST_CONFIG" \
        --output-dir "$WORK_DIR" --run-id "$RUN_ID" --frame-num 32 --batch-size 32 \
        --source-type folder --quiet \
        > "$WORK_DIR/wdf_gpu${gpu}.log" 2>&1
}

printf 'checkpoint=%s\ndetector=%s\nresults=%s\n' "$CHECKPOINT" "$DETECTOR" "$RESULT_DIR"
pids=()
json_eval "${GPU_LIST[0]}" cdf_fsh Celeb-DF-v2 FaceShifter & pids+=($!)
json_eval "${GPU_LIST[1]}" dfd DeepFakeDetection & pids+=($!)
json_eval "${GPU_LIST[2]}" dfdc DFDC & pids+=($!)
( json_eval "${GPU_LIST[3]}" dfdcp DFDCP && wdf_eval "${GPU_LIST[3]}" ) & pids+=($!)

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || { printf 'worker failed: pid=%s (see %s)\n' "$pid" "$WORK_DIR" >&2; status=1; }
done

set +e
python3 scripts/finalize_evaluation.py \
    --input-dir "$WORK_DIR" --output-dir "$RESULT_DIR" \
    --namespace "$OUTPUT_GROUP" --run-id "$RUN_ID" --worker-log-dir "$WORK_DIR" \
    --datasets Celeb-DF-v2 FaceShifter DeepFakeDetection DFDC DFDCP WDF
final_status=$?
set -e
python3 scripts/summarize_video_auc.py "$RESULT_DIR/${OUTPUT_GROUP}_${RUN_ID}.json" || true
[[ "$status" -ne 0 ]] && final_status=1
exit "$final_status"
