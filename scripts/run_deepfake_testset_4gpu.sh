#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

SOURCE_ROOT="/data/hjr/deepfake/testset"
CHECKPOINT="$PROJECT_ROOT/checkpoints/ckpt_best.pth"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid RUN_ID: %s\n' "$RUN_ID" >&2
    exit 1
fi
OUTPUT_DIR="$PROJECT_ROOT/outputs/deepfake/eval/$RUN_ID"

if [[ ! -d "$SOURCE_ROOT" ]]; then
    printf 'Missing read-only source dataset: %s\n' "$SOURCE_ROOT" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    printf 'Missing checkpoint: %s\n' "$CHECKPOINT" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
if [[ -e "$OUTPUT_DIR/deepfake_${RUN_ID}.json" || -e "$OUTPUT_DIR/deepfake_${RUN_ID}.log" ]]; then
    printf 'Output already exists for RUN_ID: %s\n' "$RUN_ID" >&2
    exit 1
fi
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dfdhr-deepfake-${RUN_ID}.XXXXXX")"
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT
trap 'exit 130' INT TERM

COMMON_ENV=(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
    "HF_HUB_DISABLE_XET=1"
    "HF_HUB_DISABLE_TELEMETRY=1"
    "NO_ALBUMENTATIONS_UPDATE=1"
    "PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/training"
)

run_worker() {
    local gpu="$1"
    local log_name="$2"
    shift 2
    local log_file="$TEMP_DIR/${RUN_ID}_${log_name}_gpu${gpu}.log"
    printf 'GPU %s -> %s (%s)\n' "$gpu" "$log_name" "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
        python3 scripts/evaluate_deepfake_testset.py \
        --root "$SOURCE_ROOT" \
        --datasets "$@" \
        --checkpoint "$CHECKPOINT" \
        --detector-path ./training/config/detector/dfd_hr.yaml \
        --test-config ./training/config/test_config.yaml \
        --output-dir "$TEMP_DIR" \
        --run-id "$RUN_ID" \
        --frame-num 32 \
        --batch-size 32 \
        --quiet \
        --source-type folder \
        > "$log_file" 2>&1 &
    LAST_PID=$!
}

pids=()
run_worker 0 cdf_dfd CDF DFD
pids+=("$LAST_PID")
run_worker 1 dfdc_dfdcp DFDC DFDCP
pids+=("$LAST_PID")
run_worker 2 dfr_ffiw DFR FFIW
pids+=("$LAST_PID")
run_worker 3 wdf WDF
pids+=("$LAST_PID")

status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        printf 'completed: pid=%s\n' "$pid"
    else
        printf 'failed: pid=%s\n' "$pid" >&2
        status=1
    fi
done

set +e
env "${COMMON_ENV[@]}" python3 scripts/finalize_evaluation.py \
    --input-dir "$TEMP_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --namespace deepfake \
    --run-id "$RUN_ID" \
    --worker-log-dir "$TEMP_DIR" \
    --datasets CDF DFD DFDC DFDCP DFR FFIW WDF
final_status=$?
set -e
if [[ "$status" -ne 0 ]]; then
    final_status=1
fi
exit "$final_status"
