#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/build"
cmake --build . -j"$(nproc)"
./kshira_tests
./kshira_learning_tests
./det_tests
SCRATCH=/tmp/surgical_fp_eval2
mkdir -p "$SCRATCH"
BENCH=./det_bench
for tag in a b; do
  echo "=== ep8 ${tag} ==="
  t0=$(date +%s)
  "$BENCH" \
    --architecture kshira \
    --manifest ../datasets/prepared/cars/train/manifest.txt \
    --eval-manifest ../datasets/prepared/cars/valid/manifest.txt \
    --width 160 --height 160 --classes 5 --features 8 \
    --max-detections 6 --precision f32 --learning-rate 0.004 \
    --epochs 8 --threshold 0.05 \
    >"$SCRATCH/ep8_${tag}.txt" 2>&1
  t1=$(date +%s)
  echo "wall_s=$((t1 - t0))"
  grep -E 'calibrate thr=0.050|calibrate thr=0.200|calibrate thr=0.250|eval_tp|calibrated_threshold|arena_budget' \
    "$SCRATCH/ep8_${tag}.txt" || true
done
echo DONE
