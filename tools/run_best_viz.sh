#!/usr/bin/env bash
# Best surgical FP path: ep8 lr=0.004 (dual F1 0.0355 TP20), full train+viz.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/viz_all results/viz_thr05
VIZ=./build/viz_detect

echo "=== UNIT TESTS ==="
./build/kshira_tests
./build/kshira_learning_tests
./build/det_tests

echo "=== FULL TRAIN (ep8 best recipe) + SAVE + ALL VALID thr=0.25 ==="
"$VIZ" \
  --train-manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --out results/detections_all.txt \
  --save results/kshira_cars.bin \
  --all \
  --epochs 8 \
  --lr 0.004 \
  --threshold 0.25 \
  --features 8 \
  --max-det 6 \
  --precision f32

echo "=== DRAW thr=0.25 ==="
python3 tools/draw_detections.py \
  --report results/detections_all.txt \
  --images-root datasets/prepared/cars/valid \
  --out-dir results/viz_all

echo "=== RELOAD thr=0.05 (operating point) ==="
"$VIZ" \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --load results/kshira_cars.bin \
  --out results/detections_thr05.txt \
  --all \
  --threshold 0.05 \
  --max-det 6

python3 tools/draw_detections.py \
  --report results/detections_thr05.txt \
  --images-root datasets/prepared/cars/valid \
  --out-dir results/viz_thr05

echo "=== SUMMARY ==="
ls -la results/kshira_cars.bin results/detections_all.txt results/detections_thr05.txt
echo "viz_all:" "$(find results/viz_all -name 'sample_*.png' | wc -l)"
echo "viz_thr05:" "$(find results/viz_thr05 -name 'sample_*.png' | wc -l)"
echo -n "GT lines thr025: "; grep -c '^GT ' results/detections_all.txt || true
echo -n "PRED lines thr025: "; grep -c '^PRED ' results/detections_all.txt || true
echo -n "GT lines thr05: "; grep -c '^GT ' results/detections_thr05.txt || true
echo -n "PRED lines thr05: "; grep -c '^PRED ' results/detections_thr05.txt || true
echo DONE
