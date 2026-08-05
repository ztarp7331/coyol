# C-OLOY / KSHIRA handoff (reviewer snapshot)

**Snapshot date:** 2026-08-05 (surgical FP path locked)  
**Repository:** `https://github.com/ztarp7331/coyol.git`  
**Design contract:** `PLAN_UPDATED.md` (contrast-modulated object detection under 256 KiB)  
**Accepted next-work review:** `docs/reviewer_response.md` (**Fast-Convergence Path**, anonymous 2026-08-05)  
**Product goal (reframed):** approach nano-YOLO accuracy with lower CPU **time-to-accuracy**, peak train memory, and runtime infrastructure — not arena size alone.  
**Prior remote baseline note:** metrics in §5 supersede the 2026-08-02 table

This document is the continuation point for research review and implementation.
Metrics are scoped exactly; they are not COCO/product-qualification claims unless
stated as such.

### Honest 1–2 week outlook (YOLO-close?)

**No — not “close to YOLO” accuracy in 1–2 weeks** if that means nano-YOLO /
COCO-style mAP, clean score separation, or product-looking boxes.

| Horizon | Realistic outcome on *this* stack + cars878-scale data |
|---|---|
| **1–2 weeks of focused work** | Maybe recover/beat closed F1 **0.043 / TP 26**; improve thr≥0.20 TP tail; cut FP somewhat; better viz. **Not** YOLO-class. |
| **What blocks YOLO-close** | ~878 unique trains, 160×160 mono, f8 head (~2.6k params), mid-band ranking still unsolved (scores glued), no ImageNet pretrain, no large multi-scale capacity |
| **What *can* move in 1–2 weeks** | Surgical ranking + in-domain data + score histograms + harder pos lift; INT8 expanded already shows F1 can exceed 0.043 with more unique images |
| **What would take longer** | Order-of-magnitude F1/AP (0.3–0.5+), clean P/R curves, multi-dataset generalization — needs more data, capacity, and possibly pretrain — weeks→months, not a sprint alone |

**Claim boundary remains:** YOLO-like **lifecycle** (train/predict/save/quant in
ISO C, arena-bounded). Accuracy goal is **approach** nano-YOLO under those
constraints — not claim parity after a ranking tweak.

---

## 1. Objective (what “done” means)

Dataset-neutral YOLO-capability **workflow** replacement in ISO C:

- single public detector lifecycle (not a dataset-specific trainer);
- train from random weights; stream arbitrary image/box data;
- save/load/resume/validate; direct boxes, classes, confidence;
- multi-scale heads (ODT path); fixed top-K + class-aware suppress;
- F32 / real INT8 / real INT4;
- caller-owned arena (~256 KiB compact KSHIRA);
- low-clock CPU first, FPGA later;
- pursue fast training on real data; report misses honestly.

**KSHIRA is an architecture profile of `det_model`, not a second framework.**  
Product evidence goes through `det_bench` / `det_*` with `--architecture kshira`.

**Claim boundary:** YOLO-like **lifecycle**, not COCO-scale mAP parity.

---

## 2. Product surface

```
det_context_create → det_model_build → det_train →
det_predict / det_evaluate → det_save / det_load
```

`det_model_spec.architecture` selects CDET or KSHIRA. KSHIRA owns a bounded
arena and coexists with other models without shared-state aliasing.

### RAD / KSHIRA forward path (measured baseline)

The **§5 dual-run numbers** were produced under the hybrid head with separate
objectness and class logits:

```
image
  → 3×3 stem (stride 4 → 40×40 @ 160)
  → depthwise dilated branches d∈{1,2,4}  [sequential workspace: one branch map]
  → mean-fuse branches
  → pointwise mixer (project_weights C×C)
  → local contrast C = 0.10 · log1p(κ)
  → hybrid head input H = [semantic C-ch, contrast]
  → box distances + objectness + class logits
  → score = sharpen(objectness, T=0.55) × max_class
  → fixed top-K + class-aware IoU suppress (0.40) + cross-class near-dup kill
```

### Fast-Convergence Path (latest review; Phase 1 in progress)

Latest `docs/reviewer_response.md` reframes success as **time-to-accuracy** on
CPU and ranks work as:

**Phase 1 — establish accuracy (current):** quality-class head → score
histograms → neighbour class-quality → two-level HNM → bounded ranking → more
unique / hard-background data.

**Phase 2 — fast convergence:** training-only dense auxiliary assignment,
fixed/adaptive gradient budgets, progressive resolution, measure time-to-F1.

**Phase 3 — architecture:** two-stage stem, shared P4, contrast ablate/gate.

**Phase 4 — C engine:** SIMD, fusion, thread pool, INT8 train transition.

**Central novelty target:** adaptive gradient-budgeted dense detection
(dense quality eval + deterministic tile-local backward in an arena-bounded
ISO C runtime).

**Phase 1 code status (surgical FP locked 2026-08-05):** quality-class layout
(4+K), identity mixer, quality bias −2.0, max-quality predict + T=0.42 sharpen,
VFL class-quality train/background/multiscale, two-level HNM + spatial diversity,
**surgical FP background** (argmax-class head only, mid-band LR 1.35×, easy-BG
score-gate), bounded ranking pairs (**epoch≥2**, surgical neg class), milder LR
decay 0.20. Dual-run F32 QC **F1 0.0355 / TP 20** (§5.1c). Ranking mid-band
**still unsolved** (thr 0.05–0.25 same pred count on locked model).

---

## 3. Source map

| Area | Location |
|---|---|
| Public API | `include/det.h` |
| KSHIRA adapter | `src/det_kshira.inc` |
| RAD kernels (train/infer) | `src/kshira_rad.c`, `src/kshira_rad_internal.h` |
| State export (KRAD v2) | `src/kshira_rad_state.c` |
| Session / phase / quant / sparse | `src/kshira_session.c`, `phase`, `quant`, `sparse` |
| Raw PNM stream | `src/det_io.c` |
| Product bench | `tools/bench.c` |
| Dataset tools | `tools/yolo_to_manifest.py`, `csv_boxes_to_manifest.py`, `filter_manifest_class.py`, `expand_cars_stream.py` |
| Viz | `tools/viz_detect.c`, `tools/draw_detections.py`, `tools/run_best_viz.sh` |
| Tests | `tests/test_det.c`, `test_kshira.c`, `test_kshira_learning.c` |
| Design contract | `PLAN_UPDATED.md` |
| Metrics handoff | `docs/HANDOFF.md` (this file) |
| Design notes | `docs/KSHIRA_DESIGN.md` |
| Research critique | `docs/RESEARCH_CRITIQUE.md` |
| External ranking review | `docs/reviewer_response.md` |

---

## 4. Real datasets (Kaggle)

| Dataset | Role | Prepared path |
|---|---|---|
| `abdallahwagih/cars-detection` | primary **5-class** | `datasets/prepared/cars/` **878 / 250 / 126** |
| `sshikamaru/car-object-detection` | extra 1-class | `datasets/prepared/car_od/` |
| Phase D expand | cars + car_od (class→2) | `datasets/prepared/cars_plus_od/` **1162 / 250** |
| Phase D hard-BG | cars + empty BG | `datasets/prepared/cars_plus_bg/` **949 / 250** |
| Phase D 1-class | car-only expanded | `datasets/prepared/cars_1c_expanded/` |

Build expanded streams (no architecture change):

```sh
python3 tools/expand_cars_stream.py
```
| merged 1-class | cars class-2 + car_od | `datasets/prepared/cars_merged/` (~652 / 161) |

`datasets/` is gitignored. Rebuild (Windows host: `python -m kaggle`):

```sh
python -m kaggle datasets download -d abdallahwagih/cars-detection \
  -p datasets/kaggle/cars-detection --unzip
python3 tools/yolo_to_manifest.py \
  --source 'datasets/kaggle/cars-detection/Cars Detection' \
  --output datasets/prepared/cars
```

**5k unique one-pass:** **not claimed.** Only **878 unique** train images on the
primary set. Multi-epoch exposures (878×8 = 7024) are **not** 5k unique images.

---

## 5. Best measured results (2026-08-05)

All through integrated `det_bench --architecture kshira`, raw manifests,
160×160, auto F1-max threshold calibrate when `--eval-manifest` is set.

### 5.0 Multi-objective note

Success is **not** “fits 256 KiB” alone. Report **accuracy**, wall-clock
**time-to-metrics**, **peak arena HWM**, portable **ISO C** kernels, and
**quant** smoke together. A path that is complete but lower F1 is documented
honestly; it does **not** replace the closed peak until it beats it dual-run.

### 5.1 Closed peak F32 baseline (objectness × class — still the accuracy bar)

**Dual-run identical** under pre–quality-class scoring:

```bash
./build-loop/det_bench --architecture kshira \
  --manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --epochs 8 --learning-rate 0.004 \
  --width 160 --height 160 --classes 5 \
  --features 8 --max-detections 6 --threshold 0.05 --precision f32
```

| Metric | Value |
|---|---:|
| precision | **0.0344** |
| recall | **0.0573** |
| **F1** | **0.0430** |
| TP / FP / FN | **26** / 730 / 428 |
| predictions | 756 |
| mean IoU (on TP) | **0.6178** |
| AP50 | 0.0012 |
| loss | 1.061 |
| train_e2e | ~35 s (~4.4 s/epoch) |
| infer | ~0.6 ms |
| arena HWM / capacity | **158721 / 262144 PASS** |

### 5.1b Quality-class + Phase D data (honest multi-objective)

**Recipe (no arch churn):** quality-class head, hist-match VFL, HNM 1→2→3, late
ranking (**epoch≥4** so INT8 ep5 last epoch participates), f8, max_det=6.
`kshira_rad_train_rank_pair` keeps **separate pos/neg feature scales** for INT8
QAS (fixed skeptic bug). **Eval always** `cars/valid` (250) for 5-class fairness.

#### F32 dual-run (identical) — pre-surgical QC

| Train stream | unique | F1 | TP | TP@0.20 | train_e2e | wall |
|---|---:|---:|---:|---:|---:|---:|
| cars878 (control) | 878 | 0.0281 | 16 | 16 | ~28 s | ~40 s |
| cars_plus_od | 1162 | 0.0037 | 2 | 2 | ~40 s | ~50 s |
| cars_plus_bg | 949 | 0.0027 | 2 | 2 | ~29–33 s | ~40–45 s |
| Best prior QC (session) | 878 | **0.0350** | **20** | **20** | ~27–30 s | — |
| Closed peak §5.1 | 878 | **0.0430** | **26** | ~2 | ~35 s | — |

**F32 finding:** naive unique-image expansion (domain-shifted car_od / hard-BG)
**hurt** F32 on this recipe. Pre-surgical best F32 QC ~0.035 / TP 20 on cars878 —
**below** closed F1/TP peak.

### 5.1c Surgical FP ranking path (locked current QC F32 bar)

**Intent:** more epochs at similar wall-clock by only tweaking FP-causing head
weights (winning class row + mild mid-band boost), not denser full-network BG.

**Recipe:** quality-class 4+K; surgical `kshira_rad_train_background_step`
(argmax class only; skip max_p&lt;0.05; mid-band LR×1.35); surgical
`kshira_rad_train_rank_pair` (neg = winning class only; ranking **epoch≥2**);
HNM 1→2→3 + late enc FULL; LR `lr/(1+0.20·epoch)`; deploy score
`σ(logit_q / 0.42)`; f8; max_det=6; ep8; lr=0.004; F32.

```bash
./build/det_bench --architecture kshira \
  --manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --epochs 8 --learning-rate 0.004 \
  --width 160 --height 160 --classes 5 \
  --features 8 --max-detections 6 --threshold 0.05 --precision f32
```

| Metric | Dual A | Dual B |
|---|---:|---:|
| **F1** | **0.0355** | **0.0355** |
| TP / FP | **20** / 654 | **20** / 654 |
| P / R | 0.0297 / 0.0441 | identical |
| mIoU (TP) | ~0.59 | identical |
| train_e2e | ~62–65 s | ~62 s |
| arena | **158601 PASS** | PASS |

**vs last restore (0.0281 / TP 16):** F1 **+26% relative**, TP **+4**.  
**vs best prior QC (0.0350 / TP 20):** essentially **tie** (0.0355).  
**vs closed peak (0.0430 / TP 26):** still short.  
**ep10:** F1 0.0313 / TP 18 — **worse** than ep8; more epochs alone not free win.

**Ranking honesty:** thr 0.05 / 0.20 / 0.25 all same pred=674 / TP=20 on dual
calibrate — mid-band scores still glued; surgical path recovered F1/TP, did
**not** open a clean high-thr operating point.

**Viz (full train + all 250 valid):** `tools/run_best_viz.sh` →
`results/kshira_cars.bin`, `results/viz_all/` (thr 0.25), `results/viz_thr05/`.
Green=GT, red=PRED. PRED count thr0.25 = thr0.05 = **674** (same glued band).

#### INT8 dual-run (identical, ep5, lr=0.003, ranking active on epoch 4)

**Bug fixes before this table:** (1) `kshira_rad_train_rank_pair` used separate
`pos_feature_scale` / `neg_feature_scale` for INT8 QAS; (2) ranking gate lowered
from `epoch>=5` (never ran on ep5) to **`epoch>=4`** (last epoch of ep5).

| Train stream | F1 | TP | P | R | train_e2e | arena |
|---|---:|---:|---:|---:|---:|---|
| cars878 | 0.0411 | 16 | 0.0494 | 0.0352 | ~49–53 s | PASS |
| **cars_plus_od** | **0.0539** | **18** | **0.0841** | 0.0396 | ~70–72 s | **158601 PASS** |

**INT8 finding (ranking active):** with **1162 unique** trains, INT8 dual-run
F1 **0.0539** **exceeds** closed F32 F1 **0.0430**, with higher precision.
Prior without-ranking INT8 dual was F1 0.0476; ranking participation improved
F1 further. TP **18** still below closed TP **26**.

| Multi-objective axis | Phase D status |
|---|---|
| Detection accuracy | F32 peak still §5.1; INT8+expanded F1 beats 0.043 |
| CPU time-to-accuracy | F32 ~28–40 s/8ep; INT8 ~50–68 s/5ep |
| Peak train memory | **158601 / 262144 PASS** all streams |
| Portable C kernels | unchanged ISO C |
| Quant deploy/adapt | INT8 dual non-collapsed; expanded helps F1/P |

Hard-BG-only and 1-class expanded streams did **not** help F32. Further gains need
**in-domain** unique 5-class images (not off-domain remap alone).

### 5.2 Score calibration grid (non-flat)

| thr | pred | TP | F1 | notes |
|---:|---:|---:|---:|---|
| 0.05–0.15 | 756 | 26 | 0.0430 | F1-max operating point |
| 0.18 | 610 | 18 | 0.0338 | count changes |
| 0.20 | 16 | 2 | 0.0085 | **P≈0.125** high-precision band |
| ≥0.25 | 0 | 0 | 0 | — |

**Reviewer signal:** almost no true detections enter thr≥0.20. Ranking gap.

### 5.3 vs documented prior baseline (5-class, multi-epoch)

| Config | ep | max_det | TP | F1 | P | R | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior best (2026-08-02) | 5 | 8 | 18 | 0.0268 | 0.0202 | 0.0396 | 0.621 |
| **Current best** | 8 | 6 | **26** | **0.0430** | **0.0344** | **0.0573** | 0.618 |

Δ: F1 **+60%**, TP **+44%**, precision **+71%**.

### 5.4 INT8 smoke (decoded detections)

```bash
# 5 ep, lr=0.003, max_det=6
precision≈0.018  recall≈0.026  F1≈0.022  TP=12  mIoU≈0.61
arena PASS; scores separate across thresholds; non-collapsed
```

---

## 6. PLAN_UPDATED mechanisms shipped

| Mechanism | Status | Notes |
|---|---|---|
| Pointwise channel mixer | **shipped** | `project_weights` C×C after branch mean |
| Contrast-modulated hybrid head | **shipped** | head_in = C+1; C = 0.10·log1p(κ); radius 2 |
| Task-aligned assignment | **partial** | centre-prior multi-positive + IoU-aware VFL targets; full staged top-k drop **not** used (hurt early TP) |
| Balanced focal / VFL objectness | **shipped** | VFL on positives; hard-neg p^γ; bias init −2 (baseline) |
| Block-local sparse train | **shipped** | 13×13 dependency tile; centre FULL encoder; transactional deltas |
| Sequential branch memory | **shipped** | one branch workspace; HWM 261k→**159k** |
| Multi-scale heads | **present** | ODT path; PRE co-supervision **ablated** (regressed F1) |
| INT8 / INT4 path | **shipped** | real quant forward + QAS; state format **v2** |
| Quality-class head (4+K) | **shipped** | train/infer/background/multiscale; dual-run §5.1b–c |
| Surgical FP background | **shipped** | argmax-class head only; mid-band LR; easy-BG gate |
| Bounded ranking loss | **shipped** | surgical neg class; det train **epoch≥2** |
| Two-level HNM | **shipped** | diverse head-only + late encoder FULL |
| Identity mixer + bias ~−2.0 | **shipped** | reset init |

---

## 7. Ablations (do not re-land without new evidence)

| Change | Outcome |
|---|---|
| features=12 (8–12 ep) | TP collapsed 2–6; underfit on 878 unique images |
| Multi-scale PRE co-supervision | F1 0.043→0.031; broke update-count tests |
| Strong FCOS centerness on score | cut TPs |
| max_det=5 | F1 ≈0.043, no gain vs 6 |
| Aggressive staged assignment (contrast×IoU top-k drop) | early TP collapse |

---

## 8. How to build / test / bench

```bash
# WSL recommended
cmake -S . -B build-loop -DCMAKE_BUILD_TYPE=Release
cmake --build build-loop -j$(nproc)
cd build-loop && ctest --output-on-failure

# Best real-data recipe (baseline bar)
./build-loop/det_bench --architecture kshira \
  --manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --epochs 8 --learning-rate 0.004 \
  --features 8 --max-detections 6 --precision f32
```

Tests of note:

- `test_hybrid_contrast_learning` — hybrid head + non-stale parameter change  
- `test_kshira_resource_profiles` — f12 fits 256 KiB; f16 needs larger arena  
- `test_kshira_learning` — public-API synthetic multi-epoch smoke  

**Note:** if the working tree is mid quality-class migration, expect train/tests
to fail until VFL class-quality and HNM are finished. Prefer a clean dual-run
only after `ctest` is green.

---

## 9. Reviewer checklist

1. **Accuracy bar:** Is F1 0.043 / TP 26 on 878 unique images acceptable to continue
   architecture work, or is unique-image expansion mandatory first?  
2. **Memory:** Sequential branch path frees ~100 KiB at same accuracy. Canonical?  
3. **Ranking diagnosis:** Agree that thr≥0.20 empty TP tail is the main scientific
   failure (not box regression)?  
4. **Next architecture:** Accept quality-class head + ranking + two-level HNM
   (`reviewer_response.md`) over f12 / deeper box head?  
5. **Assignment:** Full PLAN_UPDATED staged top-k was softened for stability;
   restore only with instrumented ablations.  
6. **Claims:** Docs must not imply COCO/YOLO mAP or 5k-unique one-pass qualification.  
7. **Train time:** ~4.4 s/epoch CPU; sub-second train is a stretch goal, not a gate
   over accuracy.  
8. **max_detections=6:** Research vs deploy default? Audit recall ceiling from GT
   counts per image (`reviewer_response.md` §9).  
9. **WIP honesty:** Surgical FP is dual-run §5.1c (F1 0.0355); closed peak 0.043
   remains accuracy bar until beaten dual-run.  
10. **YOLO-close in 1–2 weeks?** No (see top outlook). Sprint target = beat 0.043
    + thr≥0.20 TP, not nano-YOLO mAP.

---

## 10. Recommended next work (priority order)

Aligned with latest `docs/reviewer_response.md` (do **not** widen channels first):

### Phase 1 — establish accuracy (now; surgical path locked)

1. ~~Finish quality-class / HNM / ranking wiring~~ **done** (§5.1c).  
2. **Score-distribution instrumentation** (GT vs FP histograms) — still the
   diagnostic that proves ranking is moving.  
3. **Open thr≥0.20 TP tail** without crushing TP@0.05 (asymmetric ranking /
   pos-only late phase / stop crushing shared head row).  
4. Dual-run: **beat F1 0.043 / TP 26** on cars878 F32.  
5. **In-domain** unique 5-class images (not off-domain remap alone). INT8 +
   cars_plus_od already shows F1 can exceed 0.043 with more unique data.  
6. Avoid denser architecture churn that previously collapsed TP (f12, dual-sup,
   H-flip, aggressive mid-band 1.75×).

### Phase 2 — fast convergence

7. Training-only dense auxiliary assignment (deploy path unchanged).  
8. Fixed then adaptive **gradient budgets** (surgical FP is a first step).  
9. Progressive resolution / encoder unlock; report **time-to-F1 / time-to-AP50**.

### Phase 3 — architecture

11. Two-stage information-preserving stem (sequential arena).  
12. Shared P4 (shared weights + small scale bias).  
13. Contrast ablate; learnable gate only if useful.

### Phase 4 — C engine

14. SIMD-blocked kernels, fusion, thread pool, INT8 train transition.

**Provisional gates** (development only; then nano-YOLO parity race):

| Metric | Current | Next gate |
|---|---:|---:|
| TP | 26 | ≥40 without inflating pred count |
| FP/TP | ~28:1 | &lt;15:1 |
| F1 | 0.043 | ≥0.07 |
| TP at thr 0.20 | ~2 | ≥8 |
| Matched IoU | 0.618 | remain ≥0.60 |
| Arena | ~159 KiB | remain &lt;220 KiB (compact profile) |
| Later | — | within ~10% relative AP50 of nano-YOLO; 2–3× faster time-to-that-AP |

---

## 11. Related docs

| Doc | Use when |
|---|---|
| `docs/README.md` | Index + 30-minute review path |
| `docs/reviewer_response.md` | Full ranking diagnosis and experiment order |
| `docs/RESEARCH_CRITIQUE.md` | Scientific gaps and attack surface |
| `docs/KSHIRA_DESIGN.md` | Module contracts, arena, quant, phases |
| `PLAN_UPDATED.md` | Authoritative design for contrast / assignment / tiles |
