# C-OLOY / KSHIRA handoff

**Snapshot date:** 2026-08-02  
**Repository:** `https://github.com/ztarp7331/coyol.git`  
**Checkout:** `master` (working tree advanced beyond `0d7fb57`)  
**Previous remote HEAD:** `0d7fb57` (`Expose detector resource profiles`)

This document is the continuation point for research and implementation.
Metrics are scoped exactly; they are not product-qualification claims unless
stated as such.

## 1. Original objective

Dataset-neutral YOLO-capability replacement in ISO C:

- single public detector lifecycle (not a dataset-specific trainer);
- train from random weights; stream arbitrary image/box data;
- save/load/resume/validate; direct boxes, classes, confidence, multi-scale,
  fixed top-K NMS-free deploy;
- F32 / real INT8 / real INT4;
- caller-owned hot path; low-clock CPU first, then FPGA;
- pursue sub-second learning on ~5k-image workloads, report misses honestly;
- eventual ~1 W board qualification (not measured yet).

**KSHIRA is an architecture profile of `det_model`, not a second framework.**
Product evidence goes through `det_bench` / `det_*` with `--architecture kshira`.

## 2. Product surface

```
det_context_create -> det_model_build -> det_train ->
det_predict / det_evaluate -> det_save / det_load
```

`det_model_spec.architecture` selects CDET or KSHIRA. KSHIRA owns a bounded
arena and coexists with other models without shared-state aliasing.

RAD encoder (compact):  
`image -> 3x3 stem -> depthwise dilations (1,2,4) -> average-project -> top-K head`

- Average branch fusion (branch-select experiment **reverted**).
- Softmax class CE; background/empty objectness steps via adapter.
- Inference: skip untrained P4/P5 scales; class-aware IoU suppress (0.5) on
  fixed top-K list (duplicate FP reduction without full NMS heap).

### Research innovations landed (gap-driven, 2026-08-02)

Literature gaps vs TinyML/YOLO edge practice that we closed in-code:

| Gap in prior KSHIRA | Research source idea | Implementation |
|---|---|---|
| Single-cell positives only | FCOS/ATSS center sampling | 3×3 in-box neighbor positives (head-only, class at center only) |
| Random background negatives | Hard-negative mining | Probe 8 outside cells, train top objectness cells |
| Flat mid scores flood top-K | Varifocal / quality ranking | Deploy score = `quality² × class` |
| Soft objectness gradients | Focal loss | Positive focusing `(1−p)²`, negative `p³` style |
| Multi-epoch collapse | LR schedules | Inverse-time epoch LR: `lr/(1+0.35·epoch)` |
| L2 box residuals explode | Smooth-L1 | Clamped smooth-L1 on box distances |

Honest result: **FP flood is substantially down** (preds ~1100→~500–770); TP not yet
competitive. This is compositional systems research progress, not SOTA accuracy.

## 3. Source map

| Area | Location |
|---|---|
| Public API | `include/det.h` |
| KSHIRA adapter | `src/det_kshira.inc` |
| RAD kernels | `src/kshira_rad.c` |
| Raw PNM stream | `src/det_io.c` |
| Product bench | `tools/bench.c` (`--epochs`, auto threshold calibrate on eval manifest) |
| YOLO convert | `tools/yolo_to_manifest.py` |
| CSV boxes convert | `tools/csv_boxes_to_manifest.py` |
| Class filter | `tools/filter_manifest_class.py` |
| Viz dump + draw | `tools/viz_detect.c`, `tools/draw_detections.py` |
| Tests | `tests/test_det.c`, `tests/test_kshira.c`, `tests/test_kshira_learning.c` |

## 4. Real datasets (Kaggle CLI)

Downloaded with `python -m kaggle` (Windows host; not synthetic patterns):

| Dataset | Role | Prepared path |
|---|---|---|
| `abdallahwagih/cars-detection` | primary 5-class | `datasets/prepared/cars/` (878/250/126) |
| `sshikamaru/car-object-detection` | extra 1-class cars (CSV boxes) | `datasets/prepared/car_od/` (284/71 labeled) |
| merged 1-class stream | cars class-2 only + car_od | `datasets/prepared/cars_merged/` (652/161) |

`datasets/` is gitignored. Rebuild:

```sh
python -m kaggle datasets download -d abdallahwagih/cars-detection \
  -p datasets/kaggle/cars-detection --unzip
python3 tools/yolo_to_manifest.py \
  --source 'datasets/kaggle/cars-detection/Cars Detection' \
  --output datasets/prepared/cars

python -m kaggle datasets download -d sshikamaru/car-object-detection \
  -p datasets/kaggle/car-object-detection --unzip
python3 tools/csv_boxes_to_manifest.py \
  --images datasets/kaggle/car-object-detection/data/training_images \
  --csv 'datasets/kaggle/car-object-detection/data/train_solution_bounding_boxes (1).csv' \
  --output datasets/prepared/car_od --class-id 0
```

## 5. Measured multi-epoch + calibration results

All through integrated `det_bench --architecture kshira`, raw manifests,
160×160, features=8, max_det=8, arena 256 KiB. Auto score grid calibration
runs when `--eval-manifest` is set (F1-max threshold).

### 5-class cars (`abdallahwagih`, train 878 / valid 250)

| Config | epochs | lr | pred | TP | precision | recall | mean IoU | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F32 pre-innovation | 1 | 0.01 | 1098 | 6 | 0.0055 | 0.0132 | 0.564 | after IoU-suppress only |
| F32 pre-innovation | 5 | 0.003 | 1292 | 14 | 0.0108 | 0.0308 | 0.632 | prior multi-epoch best |
| F32 **post-innovation** | 1 | 0.01 | **516** | 0 | 0.0000 | 0.0000 | 0.000 | hard-neg + quality² cut FP hard |
| F32 **post-innovation** | **5** | **0.005** | **770** | **8** | **0.0104** | **0.0176** | **0.553** | fewer FP; TP mid; loss≈0.97 |
| F32 | 10 | 0.01 | 330 | 2 | 0.0061 | 0.0044 | 0.529 | collapsed (LR too high) |
| INT8 | 5 | 0.003 | 1324 | 0 | 0.0000 | 0.0000 | 0.000 | multi-epoch INT8 failed class/IoU match |

Best 5-epoch F32 run:

- `train_plus_decode_ms≈14570` (~3.0 s/epoch with decode)
- `updates=17080`, `loss≈3.26`
- calibrated threshold flat across 0.05–0.30 (scores still poorly separated)
- arena `260641/262144` PASS; infer ≪ 33 ms PASS
- `train_under_1s=FAIL` (raw decode included)

### ~5k real-image exposures (honest)

Merged 1-class stream: **652 unique** train images × **8 epochs** = **5216
exposures** (not 5k unique images).

| Config | unique | exposures | precision | recall | mean IoU (on TP) | TP |
|---|---:|---:|---:|---:|---:|---:|
| F32 1-class 8ep lr0.005 | 652 | 5216 | 0.0024 | 0.0057 | **0.7498** | 2 |

When a box matches, localization IoU is strong (~0.75), but objectness almost
never fires a true positive. Multi-epoch alone does **not** yet solve the
quality gate.

### FP flood mitigations (still in tree)

1. Skip untrained multi-scale heads at predict.
2. Class-aware IoU 0.5 suppress on top-K.
3. 2× outside-box background objectness steps (cell center not in any GT).
4. Stronger positive objectness loss weight (2×).
5. Viz outputs: `results/viz/sample_*.png` (green=GT, red=PRED).

Net effect vs pre-fix: predictions **2000 → ~1100** on 1-epoch 5-class;
multi-epoch 5×F32 further raises TP 6→14 without flooding back to 2000.

## 6. Tests / build

WSL gcc 15.2 / cmake 3.31:

- Debug + Release CTest green (`det_tests`, `kshira_tests`,
  `kshira_learning_tests` public-API two-class smoke).
- Learning test no longer dumps internal RAD features or hard-fails a fake
  four-class synthetic gate.

## 7. Commands

```sh
cmake -S . -B build-loop -DCMAKE_BUILD_TYPE=Release
cmake --build build-loop -j2
ctest --test-dir build-loop --output-on-failure

# Best measured multi-epoch real run
./build-loop/det_bench --architecture kshira --precision f32 \
  --classes 5 --features 8 --arena-kib 256 --max-detections 8 \
  --learning-rate 0.003 --epochs 5 --width 160 --height 160 \
  --manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt

# ~5k exposures (652 unique × 8)
./build-loop/det_bench --architecture kshira --precision f32 \
  --classes 1 --features 8 --arena-kib 256 --max-detections 8 \
  --learning-rate 0.005 --epochs 8 --width 160 --height 160 \
  --manifest datasets/prepared/cars_merged/train/manifest.txt \
  --eval-manifest datasets/prepared/cars_merged/valid/manifest.txt

# Visual boxes
./build-loop/viz_detect --epochs 5 --threshold 0.12 \
  --train-manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --out results/detections.txt --preview 12
python3 tools/draw_detections.py --report results/detections.txt \
  --images-root datasets/prepared/cars/valid --out-dir results/viz
```

## 8. Immediate next gates

1. **Objectness schedule:** decay LR after epoch 2; reduce background weight
   mid-training; hard-negative mine high-score empty cells.
2. **True 5k unique images** from a larger Kaggle export (current merged set is
   652 unique; 5216 is multi-epoch exposure count).
3. **INT8 multi-epoch parity** — currently collapses on this path; compare to
   F32 after LR schedule fix before claiming quantized learning.
4. Packed INT4 deployment export independent of FP32 masters.
5. Low-clock CPU profile → FPGA → board power near 1 W.
6. Keep `PLAN.md` / `KSHIRA_DESIGN.md` updated with measured results only.

## 9. Non-claims

Still **not** verified:

- competitive real-image accuracy or COCO mAP;
- sub-second raw 5k-unique training;
- multi-epoch INT8/INT4 accuracy parity;
- packed deployment without FP32 masters;
- low-clock / FPGA / 1 W board power;
- KSHIRA `GLOBAL_BP`;
- confidence calibration that separates TP/FP by score (grid is flat today).

Best honest statement: the integrated framework trains and evaluates on real
Kaggle streams; multi-epoch F32 with lower LR modestly improves TP/IoU; FP
flood is reduced but accuracy remains far from a YOLO replacement.
