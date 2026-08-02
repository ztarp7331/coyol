# Research critique & innovation guide

**Audience:** you (and future collaborators) doing independent research on this codebase  
**Purpose:** one place to understand what exists, what is weak, what is over-claimed, and where to innovate next  
**Date:** 2026-08-02  
**Repo:** C-OLOY / KSHIRA (`det_*` framework + KSHIRA architecture profile)

This is **not** a marketing doc. Prefer it when you want to attack the design
scientifically. Pair it with the other documents listed below.

---

## 0. Which document to open for what

| Document | Role | Use it when… |
|---|---|---|
| **`docs/RESEARCH_CRITIQUE.md` (this file)** | Flaws, gaps, open questions, experiment ideas | You want to innovate or break assumptions |
| **`docs/HANDOFF.md`** | Snapshot of measured numbers + how to resume | You need latest benchmarks, commands, dataset paths |
| **`docs/KSHIRA_DESIGN.md`** | Implementation contract (arena, quant, RAD, phases) | You need module APIs and what was already built |
| **`PLAN.md`** | Original goal / success contract / milestones | You need the product thesis and long-term gates |
| **`include/det.h`** | Public API truth | You change training/eval/IO contracts |
| **`src/kshira_rad.c` + `src/det_kshira.inc`** | Actual learning behavior | You change loss, assignment, objectness, multi-scale |

**Start here for research:** this file → then `HANDOFF.md` metrics → then the C files above.

---

## 1. Thesis of the project (what we claim to be doing)

Build a **dataset-neutral object detector framework in ISO C** that:

1. trains from random weights (not transfer-learning only);
2. streams arbitrary image/box data via callbacks / manifests;
3. runs F32 / real INT8 / real INT4;
4. uses a **caller-owned arena** (~256 KiB class for compact KSHIRA);
5. targets **low-clock CPU then FPGA**, not CUDA-first;
6. borrows ideas from **YOLO26-style NMS-free deploy** + **TinyML on-device train** (QAS, sparse updates);
7. keeps **KSHIRA as a profile of `det_model`**, not a second product.

If your research does not serve this thesis, you are writing a different project.

---

## 2. Architecture in one page (mental model)

```
Public lifecycle (product surface)
  det_context_create → det_model_build(architecture=CDET|KSHIRA)
  → det_train → det_predict / det_evaluate → det_save / det_load

KSHIRA path (det_kshira.inc)
  session + sparse mask + phase PRE→TRAIN→ODT
  → kshira_rad_* kernels

RAD forward (compact)
  image
    → 3×3 stem (stride 4 map)
    → depthwise 3×3 dilations d∈{1,2,4}
    → project = average of branches
    → head: 4 distances + objectness + class logits
    → score = quality² × softmax(class)   [current]
    → fixed top-K, then class-aware IoU suppress 0.5
```

**Training (LOCAL_FAST):** mostly **one primary map cell per box** (box center),
plus recent **3×3 center-sampling neighbors** (objectness/box only), plus
**hard-negative objectness** on outside cells.

**Not trained end-to-end like PyTorch YOLO:** no autograd graph; explicit
scalar schedules; encoder updates are single-target straight-through.

---

## 3. Known flaws & scientific weaknesses (attack these)

These are real. They are the best places to innovate.

### 3.1 Learning / accuracy

| ID | Flaw | Why it matters | Where in code |
|---|---|---|---|
| A1 | **Single-primary assignment** | Only the center cell gets full class+encoder update; multi-object / off-center objects under-supervised | `kshira_rad_train_step`, neighbors head-only |
| A2 | **No true one-to-many / one-to-one dual head** | YOLO26 trains dual assignment; we approximate with top-K + suppress | predict path only one head |
| A3 | **Class imbalance** | Cars dataset dominated by class “Car”; other classes rarely win | dataset + CE without reweight |
| A4 | **Objectness still poorly calibrated** | Score grid is flat; threshold search does not separate TP/FP | `score = q²·class`, eval calibrate in `bench.c` |
| A5 | **Quality² can kill recall** | Suppresses mid-confidence true peaks; 1-ep TP can go to 0 | `kshira_rad_predict` |
| A6 | **Multi-epoch can collapse** | High LR or aggressive hard-neg drives objectness toward zero | `det_kshira_train` LR decay + HNM |
| A7 | **INT8 multi-epoch parity failed** | Quantized train path does not yet track F32 learning quality | quant path in `kshira_rad.c` |
| A8 | **Eval is class-aware greedy IoU@0.5** | Wrong class ⇒ no TP even if box is right; understates localization-only progress | `det_eval.inc` |
| A9 | **Synthetic domain ≠ real photos** | Early IoU gains on curricula did not transfer to Kaggle cars | `kshira_domain` vs raw manifests |
| A10 | **No multi-scale training by default** | P4/P5 heads optional ODT; LOCAL_FAST is P3-centric | multiscale path |

### 3.2 Systems / representation

| ID | Flaw | Why it matters | Where |
|---|---|---|---|
| S1 | **Packed INT4 deploy still pending** | Runtime can run INT4 math, but checkpoint still FP32-master | serialize path |
| S2 | **GLOBAL_BP not on KSHIRA** | No conventional full-backprop reference for KSHIRA | `det_kshira` rejects GLOBAL_BP |
| S3 | **Arena headroom is tight** | 8-feat/160² ~260 KiB/256 KiB; 80-class needs larger arena | resource profiles |
| S4 | **Raw train includes decode** | Sub-second gate fails partly due to PGM I/O, not only math | `det_io.c` + bench timing |
| S5 | **No FPGA / power measurement** | 1 W goal is aspirational only | none yet |
| S6 | **NMS-free is incomplete** | Still post-hoc IoU suppress on top-K; not pure one-to-one learned uniqueness | `suppress_duplicate_detections` |

### 3.3 Methodology / claims (research integrity)

| ID | Flaw | Why it matters |
|---|---|---|
| M1 | **“5k images” often means exposures or synthetic** | 652 unique × 8 epochs ≠ 5k unique real images |
| M2 | **Synthetic mAP is not COCO** | Pattern generators overstate learning |
| M3 | **Individual ideas are not novel** | NMS-free, QAS, sparse, center sampling exist in literature; novelty is **composition in pure C** |
| M4 | **Accuracy too low for detector SOTA papers** | Reviewers will reject “YOLO replacement” without VOC/COCO-scale evidence |
| M5 | **Timing host-dependent** | WSL OneDrive paths dominate decode time |

### 3.4 Concrete failure modes you have already observed

1. **Top-K flood:** many near-identical boxes ~same score (mitigated, not solved).  
2. **Wrong class, right blob:** ambulance/truck predicted as Car.  
3. **Corner bias:** collapsed models emit one junk box in image corner.  
4. **1-ep after hard-neg:** preds drop (good) but TP can drop to 0 (bad).  
5. **10-ep high LR:** objectness collapse.

Use these as regression tests when you change losses.

---

## 4. What is actually solid (do not break without reason)

- Dataset-neutral `det_dataset` + PNM manifest adapter (real Kaggle path works).
- Integrated KSHIRA via `DET_ARCH_KSHIRA` (single framework).
- Caller-owned arena + high-water accounting.
- Deterministic save/load with CRC (integrated checkpoints).
- Explicit phase machine PRE → TRAIN → ODT.
- Real INT8/INT4 forward arithmetic (even if learning parity is weak).
- Measured honesty in `HANDOFF.md` (failures recorded).

Innovate **on top of** these contracts; don’t invent a parallel API.

---

## 5. Literature map (for your own reading)

Read these with the flaws table open:

| Topic | Representative work | Link to our flaw |
|---|---|---|
| NMS-free / E2E detect | YOLO26, DETR lineage | A2, S6 |
| Tiny on-device train | MIT “On-Device Training Under 256KB” (QAS, sparse) | S3, A7 |
| Center / multi-positive assign | FCOS, ATSS, YOLO label assign | A1 |
| Quality / focal ranking | Focal Loss, Varifocal, QFL | A4, A5 |
| Hard negatives | Classic HNM, OHEM | A6 |
| TinyML detect on MCU | MCUNet detect notes | S5, mostly inference |
| C training frameworks | Darknet (C+CUDA) | contrast: we are no-GPU pure C |

**Research angle that is still defensible:**  
*“ISO-C co-designed train+detect under a hard arena with real INT4/INT8 and NMS-free deploy for FPGA/low-clock.”*  
Not: *“beats YOLO accuracy.”*

---

## 6. Metrics that matter (and how to lie less)

### Always report

- **unique images** vs **exposures** (epochs × unique)  
- **raw_manifest** vs **synthetic**  
- precision, recall, mean IoU **on TPs**, AP50, mAP50:95  
- prediction count (FP pressure)  
- train_plus_decode_ms **and** pure train if separable  
- arena high-water / capacity  
- precision mode (F32/INT8/INT4)

### Prefer for research ablations

- class-agnostic localization recall (IoU@0.5 ignore class) — *not implemented; easy experiment*  
- per-class AP  
- score histograms TP vs FP  
- objectness map visualization  
- LR / epoch curves of mean objectness on empty cells

### Do not claim

- sub-second raw 5k until true unique-image disk workload passes  
- 1 W until board power is measured  
- INT4 deploy until packed export exists independent of FP32 masters  

---

## 7. Innovation backlog (ranked for research payoff)

### Tier 1 — likely accuracy wins (start here)

1. **Assignment upgrade**  
   - Train all cells with center inside GT (radius schedule), not just 3×3.  
   - Soft objectness target = Gaussian from box center (not hard 0/1).  
   - Flaw: A1.

2. **Class rebalancing**  
   - Inverse-frequency CE weight or rare-class oversample in stream.  
   - Flaw: A3.

3. **Score / objectness recalibration**  
   - Replace `q²` with learnable or `q·(α+(1−α)q)·class`.  
   - Temperature on objectness; score histogram matching.  
   - Flaw: A4, A5.

4. **Localization-aware loss**  
   - Add (1−IoU) term on decoded box vs GT at assigned cell.  
   - Flaw: A8 understates pure localization progress.

5. **Two-phase train**  
   - Phase A: localization+objectness only (1-class).  
   - Phase B: freeze stem, train multi-class head.  
   - Flaw: A3, A6.

### Tier 2 — systems / speed

6. **Decode off the critical path**  
   - Pre-decode PGM→float cache for timing pure train.  
   - Flaw: S4, M5.

7. **INT8 parity**  
   - Calibrate more; freeze BN-free scales; compare cell-level logits F32 vs INT8.  
   - Flaw: A7.

8. **Packed INT4 export**  
   - Separate deploy blob from FP32 master.  
   - Flaw: S1.

### Tier 3 — product gates (hard)

9. True ≥5k **unique** real images.  
10. VOC or COCO-subset mAP baseline.  
11. FPGA lower + board power.

---

## 8. Suggested self-driven research protocol

For any change you invent:

1. **State hypothesis** (1 sentence) linked to a flaw ID (e.g. A1).  
2. **Minimal patch** (one mechanism).  
3. **Ablation table** on the same cars valid set:

   | variant | ep | lr | pred | TP | P | R | mean IoU | train_ms |

4. **Keep a control** (current `master` recipe: 5 ep, lr 0.005, F32 KSHIRA).  
5. **Viz 8 images** (`viz_detect` + `draw_detections.py`) — metrics lie; eyes don’t.  
6. **Log failure modes** (collapse, corner box, class mode collapse).  
7. **Update this file** §3 or §7 if you disprove or close a flaw.

### Baseline command (control)

```sh
./build-loop/det_bench --architecture kshira --precision f32 \
  --classes 5 --features 8 --arena-kib 256 --max-detections 8 \
  --learning-rate 0.005 --epochs 5 --width 160 --height 160 \
  --manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt
```

### Viz command

```sh
./build-loop/viz_detect --epochs 5 --threshold 0.12 \
  --train-manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --out results/detections.txt --preview 12
python3 tools/draw_detections.py \
  --report results/detections.txt \
  --images-root datasets/prepared/cars/valid \
  --out-dir results/viz
```

---

## 9. Code hotspots for experiments

| If you change… | Touch… | Watch… |
|---|---|---|
| Assignment / multi-positive | `src/kshira_rad.c` `kshira_rad_train_step` | train time, recall |
| Objectness / focal | `rad_train_positive_at_cell`, `train_background_step` | collapse, score hist |
| Hard-neg | `src/det_kshira.inc` HNM loop | speed (probes cost) |
| Score formula | `kshira_rad_predict` | precision/recall trade |
| LR schedule | `det_kshira_train` epoch_lr | multi-epoch stability |
| Eval matching | `src/det_eval.inc` | class-agnostic experiments |
| Dataset balance | new sampler / manifest tool | per-class AP |
| Memory | arena sizes, feature_channels | high-water fail-closed |

---

## 10. “Is this research innovation?” cheat sheet

| Claim level | Fair today? |
|---|---|
| Systems co-design prototype (C train+detect+quant+arena) | **Yes** |
| Composition of known TinyML + E2E-detect ideas | **Yes** |
| Novel loss/architecture with SOTA accuracy | **No** |
| YOLO replacement on real multi-class photos | **No** |
| FPGA / 1 W edge product | **No** |

Your research job is to move a row from No → Yes with evidence, without breaking the solid contracts in §4.

---

## 11. Open questions (write papers around these)

1. Can single-map dilated RAD match multi-scale YOLO accuracy under 256–512 KiB?  
2. What is the minimal positive-cell set for stable objectness on real photos?  
3. How to train INT4/INT8 online without FP32 master and without collapse?  
4. Can learned one-to-one uniqueness replace IoU suppress entirely?  
5. Does decode-free streaming + arena-only scratch enable true sub-second 5k unique?  
6. What FPGA schedule maps the static RAD graph to ~1 W at useful FPS?

---

## 12. Revision log

| Date | Note |
|---|---|
| 2026-08-02 | Initial critique after real Kaggle multi-epoch work + center-sampling/HNM/focal/quality² landing |
| 2026-08-03 | Tier-1: soft quality score `q(0.5+0.5q)c`, mild sqrt-IF CE (warm-up 64), light IoU box scale; F32 5ep TP 12 / R 0.026 / IoU 0.63; INT8 5ep TP 6 (was 0) |

When you close a flaw or invent a mechanism, append a row here and update §3/§7.

---

*Primary research reference for innovation. Metrics snapshot: `docs/HANDOFF.md`. Design contracts: `docs/KSHIRA_DESIGN.md`. Product goal: `PLAN.md`.*
