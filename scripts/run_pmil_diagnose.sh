#!/usr/bin/env bash
# Dump per-frame logits of every branch for FF++ (calibration only) + the 6 test sets, then analyze.
# Every dataset is split into one shard per GPU, so all GPUs finish at about the same time.
#   CHECKPOINT=outputs/pmil/train/pmil_v2/test/avg/ckpt_best.pth \
#   DETECTOR=training/config/detector/dfd_hr_pmil_v2.yaml bash scripts/run_pmil_diagnose.sh
# Env: GPUS (default 0,1,2,3), BATCH (frames per forward, default 64; each frame = 5 ViT-L crops),
#      WORKERS (data-loading processes per GPU, default 8), RUN_ID,
#      REFERENCE (eval JSON of the same checkpoint; the orig branch must reproduce it)
# Output: outputs/pmil/diagnose/<RUN_ID>/diagnosis.md
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
DETECTOR="${DETECTOR:-training/config/detector/dfd_hr_pmil.yaml}"
RUN_ID="${RUN_ID:-$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")}"
GPUS="${GPUS:-0,1,2,3}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
IFS=',' read -r -a G <<< "$GPUS"
N="${#G[@]}"
OUT="outputs/pmil/diagnose/$RUN_ID"
mkdir -p "$OUT"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python NO_ALBUMENTATIONS_UPDATE=1 \
       PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/training:$PROJECT_ROOT/scripts"

# name:frames  (FF++ only provides calibration statistics, 8 frames are enough)
JOBS=(FaceForensics++:8 Celeb-DF-v2:32 FaceShifter:32 DeepFakeDetection:32 DFDC:32 DFDCP:32 WDF:32)

worker() {  # shard_index gpu
    local k="$1" gpu="$2" job name frames
    for job in "${JOBS[@]}"; do
        name="${job%%:*}"; frames="${job##*:}"
        CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/diagnose_pmil.py dump --checkpoint "$CHECKPOINT" \
            --detector "$DETECTOR" --dataset "$name" --frame_num "$frames" \
            --batch_size "$BATCH" --workers "$WORKERS" --shard "$k" --num_shards "$N" \
            --out_dir "$OUT" > "$OUT/${name}.s${k}of${N}.log" 2>&1 || return 1
    done
}
printf 'diagnose %s -> %s (%s shards, batch %s)\n' "$CHECKPOINT" "$OUT" "$N" "$BATCH"
pids=()
for k in "${!G[@]}"; do worker "$k" "${G[$k]}" & pids+=($!); done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
if [[ "$status" -ne 0 ]]; then
    printf 'a dump failed; last lines of the logs:\n' >&2
    tail -n 5 "$OUT"/*.log >&2
fi
python3 scripts/diagnose_pmil.py analyze --out_dir "$OUT" ${REFERENCE:+--reference "$REFERENCE"}
exit "$status"
