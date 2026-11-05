#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"

CHECKPOINT="$PROJECT_ROOT/checkpoints/ckpt_best.pth"
DETECTOR="$PROJECT_ROOT/training/config/detector/dfd_hr.yaml"
TEST_CONFIG="${TEST_CONFIG:-$PROJECT_ROOT/training/config/test_config.yaml}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid RUN_ID: %s\n' "$RUN_ID" >&2
    exit 1
fi
RESULT_DIR="$PROJECT_ROOT/outputs/dfbench/eval/$RUN_ID"

if [[ ! -f "$CHECKPOINT" ]]; then
    printf 'Missing checkpoint: %s\n' "$CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$DETECTOR" ]]; then
    printf 'Missing detector config: %s\n' "$DETECTOR" >&2
    exit 1
fi
if [[ ! -f "$TEST_CONFIG" ]]; then
    printf 'Missing test config: %s\n' "$TEST_CONFIG" >&2
    exit 1
fi

python3 - <<'PYCONFIG'
from pathlib import Path
import yaml
root = Path.cwd()
for name in ("train_config.yaml", "test_config.yaml"):
    with (root / "training/config" / name).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["frame_num"]["train"] == 8, name
    assert config["frame_num"]["test"] == 32, name
print("Verified paper frame protocol: train=8, test=32")
PYCONFIG

mkdir -p "$RESULT_DIR"
if [[ -e "$RESULT_DIR/dfbench_${RUN_ID}.json" || -e "$RESULT_DIR/dfbench_${RUN_ID}.log" ]]; then
    printf 'Output already exists for RUN_ID: %s\n' "$RUN_ID" >&2
    exit 1
fi
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dfdhr-dfbench-${RUN_ID}.XXXXXX")"
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

run_eval() {
    local gpu="$1"
    local log_name="$2"
    shift 2
    local log_file="$TEMP_DIR/${RUN_ID}_${log_name}_gpu${gpu}.log"

    printf 'GPU %s -> %s (%s)\n' "$gpu" "$log_name" "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
        python3 scripts/test.py \
        --detector_path ./training/config/detector/dfd_hr.yaml \
        --test_config "$TEST_CONFIG" \
        --test_dataset "$@" \
         --weights_path ./checkpoints/ckpt_best.pth \
         --output_dir "$TEMP_DIR" \
         --run_id "$RUN_ID" \
         --source_type json \
         --quiet \
         > "$log_file" 2>&1 &
    LAST_PID=$!
}

pids=()
run_eval 0 celebd_f_v1_v2 Celeb-DF-v1 Celeb-DF-v2
pids+=("$LAST_PID")
run_eval 1 deepfakedetection_faceshifter DeepFakeDetection FaceShifter
pids+=("$LAST_PID")
run_eval 2 dfdc_uadfv DFDC UADFV
pids+=("$LAST_PID")
run_eval 3 dfdcp DFDCP
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

run_fallback() {
    local gpu="$1"
    local log_name="$2"
    local dataset="$3"
    CUDA_VISIBLE_DEVICES="$gpu" env "${COMMON_ENV[@]}" \
        python3 scripts/evaluate_deepfake_testset.py \
        --root /data/hjr/deepfake/testset \
        --datasets "$dataset" \
        --checkpoint "$CHECKPOINT" \
        --detector-path ./training/config/detector/dfd_hr.yaml \
        --test-config ./training/config/test_config.yaml \
        --output-dir "$TEMP_DIR" \
        --run-id "$RUN_ID" \
        --frame-num 32 \
        --batch-size 32 \
        --source-type folder \
        --quiet \
        > "$TEMP_DIR/${RUN_ID}_${log_name}_gpu${gpu}.log" 2>&1 &
    LAST_FALLBACK_PID=$!
}

set +e
fallback_pids=()
run_fallback 1 dfr DFR
fallback_pids+=("$LAST_FALLBACK_PID")
run_fallback 2 wdf WDF
fallback_pids+=("$LAST_FALLBACK_PID")
run_fallback 3 ffiw FFIW
fallback_pids+=("$LAST_FALLBACK_PID")
for pid in "${fallback_pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
set -e

set +e
env "${COMMON_ENV[@]}" python3 scripts/finalize_evaluation.py \
    --input-dir "$TEMP_DIR" \
    --output-dir "$RESULT_DIR" \
    --namespace dfbench \
    --run-id "$RUN_ID" \
    --worker-log-dir "$TEMP_DIR" \
    --datasets Celeb-DF-v1 Celeb-DF-v2 DeepFakeDetection FaceShifter DFDC DFDCP UADFV DFR WDF FFIW
final_status=$?
set -e
if [[ "$status" -ne 0 ]]; then
    final_status=1
fi
exit "$final_status"
