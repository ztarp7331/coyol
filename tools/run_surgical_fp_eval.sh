#!/usr/bin/env bash
# Dual-run surgical FP ranking path on cars878.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SCRATCH="${SCRATCH:-/tmp/surgical_fp_eval}"
mkdir -p "$SCRATCH"
BENCH="${BENCH:-./build/det_bench}"
COMMON=(
  --architecture kshira
  --manifest datasets/prepared/cars/train/manifest.txt
  --eval-manifest datasets/prepared/cars/valid/manifest.txt
  --width 160 --height 160 --classes 5 --features 8
  --max-detections 6 --precision f32
  --learning-rate 0.004
)

run_one() {
  local tag="$1"
  local epochs="$2"
  local thr="${3:-0.05}"
  local t0 t1
  echo "=== ${tag} epochs=${epochs} thr=${thr} ==="
  t0=$(date +%s)
  "$BENCH" "${COMMON[@]}" --epochs "$epochs" --threshold "$thr" \
    >"$SCRATCH/${tag}.txt" 2>&1
  t1=$(date +%s)
  echo "wall_s=$((t1 - t0))" | tee -a "$SCRATCH/${tag}.txt"
  echo "----- ${tag} -----"
  # Key lines only
  grep -E 'calibrate thr=0\.(05|20|25)|calibrated_threshold|samples=|eval_tp|arena_budget|wall_s' \
    "$SCRATCH/${tag}.txt" || true
}

run_one f32_ep8_a 8 0.05
run_one f32_ep8_b 8 0.05
run_one f32_ep10_a 10 0.05
run_one f32_ep10_b 10 0.05
run_one f32_ep12_a 12 0.05

echo "DONE scratch=$SCRATCH"
