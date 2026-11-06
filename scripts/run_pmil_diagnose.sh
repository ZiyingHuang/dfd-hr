#!/usr/bin/env bash
# Dump per-frame logits of every branch for FF++ (calibration only) + the 6 test sets, then analyze.
#   CHECKPOINT=outputs/pmil/train/pmil_main/test/avg/ckpt_best.pth bash scripts/run_pmil_diagnose.sh
# Output: outputs/pmil/diagnose/<RUN_ID>/diagnosis.md
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$PROJECT_ROOT"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
DETECTOR="${DETECTOR:-training/config/detector/dfd_hr_pmil.yaml}"
RUN_ID="${RUN_ID:-$(basename "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")")}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a G <<< "$GPUS"
OUT="outputs/pmil/diagnose/$RUN_ID"
mkdir -p "$OUT"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python NO_ALBUMENTATIONS_UPDATE=1 \
       PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/training"
run() {  # gpu dataset frames
    CUDA_VISIBLE_DEVICES="$1" python3 scripts/diagnose_pmil.py dump --checkpoint "$CHECKPOINT" \
        --detector "$DETECTOR" --dataset "$2" --frame_num "$3" --out_dir "$OUT" > "$OUT/$2.log" 2>&1
}
printf 'diagnose %s -> %s\n' "$CHECKPOINT" "$OUT"
pids=()
( run "${G[0]}" FaceForensics++ 8 && run "${G[0]}" Celeb-DF-v2 32 ) & pids+=($!)
( run "${G[1]}" DeepFakeDetection 32 && run "${G[1]}" FaceShifter 32 ) & pids+=($!)
run "${G[2]}" DFDC 32 & pids+=($!)
( run "${G[3]}" DFDCP 32 && run "${G[3]}" WDF 32 ) & pids+=($!)
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[[ "$status" -ne 0 ]] && { printf 'a dump failed, see %s/*.log\n' "$OUT" >&2; tail -n 5 "$OUT"/*.log >&2; }
python3 scripts/diagnose_pmil.py analyze --out_dir "$OUT"
exit "$status"
