# Research critique & innovation guide

**Audience:** reviewers and collaborators  
**Purpose:** flaws, gaps, over-claims, and highest-leverage next experiments  
**Date:** 2026-08-05  
**Repo:** C-OLOY / KSHIRA  

This is **not** a marketing doc. Pair with:

| Document | Role |
|---|---|
| `docs/HANDOFF.md` | Latest metrics, commands, datasets |
| `docs/KSHIRA_DESIGN.md` | What the code contracts actually are |
| `docs/reviewer_response.md` | Latest Fast-Convergence Path (quality-class + budgeted sparse train) |
| `PLAN_UPDATED.md` | Design intent for contrast / assignment / tiles |

---

## 1. Thesis (judge work against this)

Build a **dataset-neutral object detector in ISO C** that:

1. trains from random weights;  
2. streams arbitrary image/box data;  
3. runs F32 / real INT8 / real INT4;  
4. uses a **caller-owned ~256 KiB** arena for the compact profile;  
5. targets **CPU then FPGA**, not CUDA-first;  
6. provides YOLO-like **lifecycle**, not COCO mAP parity;  
7. keeps **KSHIRA as a `det_model` profile**, not a second product.

If a change does not serve this thesis, it is a different project.

---

## 2. Where we are (honest)

### 2.1 Progress that is real

- Full public lifecycle on **real** Kaggle cars (not synthetic-only success).  
- PLAN_UPDATED core pieces in production path: **mixer, contrast hybrid head,
  13×13 tiles, VFL objectness, HNM, sequential branch memory**.  
- Closed dual-run F32 (pre-QC scoring): **F1 0.0430, TP 26, mIoU 0.618**.  
- Surgical FP QC dual-run F32: **F1 0.0355, TP 20** (locked; dual-identical).  
- INT8 + expanded unique (1162): F1 **0.0539** dual (beats closed F1; TP 18).  
- Arena HWM **~159 KiB** after sequential branches (was ~261 KiB).  
- Surgical FP: argmax-class head-only BG + ranking epoch≥2 (unit-tested).

### 2.2 What we are **not**

- Not competitive with YOLO / RT-DETR / COCO mAP.  
- Not 5k-unique one-pass qualified (**878 unique** train images only).  
- Not sub-second train (~4–8 s/epoch on this host).  
- Not board-power or FPGA measured.  
- Not “ranking solved”: thr 0.05–0.25 still same pred count on surgical model.  
- Not YOLO-close in a 1–2 week sprint (see §2.3).

### 2.3 1–2 week realism (ideating harder is not enough)

| Achievable in 1–2 weeks | Unrealistic in 1–2 weeks |
|---|---|
| Beat F1 0.043 / TP 26 dual-run F32 | nano-YOLO / COCO mAP parity |
| Open thr≥0.20 TP tail a little | Clean high-precision curves (P≫0.5 at useful R) |
| More in-domain unique images + INT8 | 10× F1 without data/capacity |
| Histograms proving ranking moves | Product-looking boxes on hard scenes |

**Why:** capacity (f8, ~2.6k params), data (878 mono 160×160), and shared head
row mean FP and TP fight the same weights. Surgical sparse updates recover F1;
they do not invent features YOLO gets from pretrain + millions of boxes.

---

## 3. Primary scientific failure modes

### 3.1 Precision bottleneck (dominant)

- ~730 false positives vs 26 true positives at F1-max thr.  
- Localization on true matches is **already useful** (mIoU ~0.62).  
- **Conclusion:** objectness/ranking and FP suppression, not box regression,
  are the main accuracy bottleneck.

### 3.2 High-confidence tail is empty

- thr=0.20 reaches **precision ~0.125** but only **~2 TP**.  
- Almost all detections live in a low mid-score band.  
- **Conclusion:** training does not push true centres far enough above
  background in score space (ranking / quality assignment gap).

### 3.3 Score product pathology (external review)

Measured baseline multiplies imperfect objectness and class scores:

```
S = sharpen(σ(objectness)) × max_k softmax(class)_k
```

Problems called out in `reviewer_response.md`:

- objectness and class trained for different objectives then multiplied;  
- softmax always places mass somewhere on background;  
- product compresses the useful score range (e.g. 0.35×0.50 = 0.175).  

**Proposed fix:** quality-class head with score = max_k σ(z_k) and IoU-aware
targets (VarifocalNet-style IoU-aware classification).

### 3.4 Data ceiling vs capacity

- **features=12** fits 256 KiB but **collapsed TP** (2–6) on 878 unique images.  
- **Conclusion:** more channels without more unique data underfits; do not
  treat capacity as free accuracy.

### 3.5 Assignment vs PLAN_UPDATED

- Full staged contrast×IoU top-k drop **hurt early TP** and was softened.  
- Current: centre + in-box neighbours with centre prior weights.  
- **Gap:** true task-aligned assignment (TOOD/ATSS-style) not fully realized;
  restore only after ranking is repaired and curves are instrumented.

### 3.6 Multi-scale

- P4/P5 heads exist for ODT; PRE multi-scale co-supervision **regressed F1**.  
- Deploy skips untrained scales (good for FP control).  
- **Gap:** scale-aware learning without flooding mid scores.

---

## 4. Over-claims to reject in review

| Claim | Reality |
|---|---|
| “YOLO accuracy” | Lifecycle yes; mAP no |
| “5k image qualification” | 878 unique; multi-epoch ≠ unique |
| “Sub-second training achieved” | ~35 s for 8 epochs |
| “Contrast solves objectness” | Helped system; FP flood remains |
| “features=12 always better” | Measured **worse** on this data |
| “Quality-class already proven” | WIP; dual-run baseline still O×P head |

---

## 5. Architecture attack surface (good research questions)

1. **Why does VFL + bias−2 not fill the thr≥0.20 TP tail?**  
   Instrument score histograms for GT cells vs FP cells after each epoch.

2. **Does quality-class scoring (max σ) restore the high-confidence TP tail?**  
   Report TP@0.20, FP/TP, and score histograms before/after.

3. **Is contrast a useful head feature or mostly noise after scale 0.10?**  
   Ablate contrast channel to zero at train and eval; report ΔF1/TP.

4. **Assignment:** restore staged a_t/b_t only after epoch-wise TP/FP curves
   are instrumented (not as a single big-bang).

5. **Sequential branch path:** prove encoder gradient parity vs three-map path
   (finite difference or golden run) for f8 and f12.

6. **Data:** hold architecture fixed; 2× unique images — does F1 scale?

7. **NMS:** soft-NMS score decay vs hard suppress 0.40 (only after ranking
   improves — Soft-NMS must not paper over weak scores).

8. **max_detections=6:** compute R_max from GT counts; raise research cap if
   recall is ceiling-limited.

---

## 6. Recommended experiment ladder

**Do not jump to capacity first.** Order matches `reviewer_response.md`
(Phase A–E condensed):

### Phase A — measurement ceilings

1. Audit `max_detections=6` vs GT objects per image.  
2. Score histograms (GT vs FP) every epoch.  
3. Letterbox vs square stretch (if time).

### Phase 1 — establish accuracy (highest leverage now)

4. **Quality-class head** (4+K, IoU targets, score = max σ).  
5. Score histograms (GT vs FP).  
6. Neighbour class-quality; prior bias sweep (−2.5/−3.5/−4.5).  
7. Two-level HNM + spatial diversity.  
8. Bounded pairwise ranking.  
9. More unique data + hard backgrounds.  
10. Dual-run gate: F1≥0.07, FP/TP&lt;15, TP@0.20≥8.

### Phase 2 — fast convergence (novelty core)

11. Training-only dense auxiliary assignment (deploy unchanged).  
12. Fixed then adaptive **gradient budgets** (dense forward / sparse tile backward).  
13. Progressive resolution and encoder unlocking.  
14. Report **time-to-F1** and **time-to-AP50**, not only s/epoch.

### Phase 3 — architecture

15. Two-stage stem; shared P4; contrast ablate/gate.  
16. Do **not** f12 until data + ranking succeed.

### Phase 4 — C engine

17. SIMD, fusion, thread pool, INT8 train transition.  
18. Later: nano-YOLO fair race (same split/CPU/time, random init both).

Stop criterion for architecture churn: if three consecutive ablations do not
move F1 by ≥0.005 dual-run, switch to data.

---

## 7. Literature anchors used (not exhaustive)

| Idea | Source family | Use in tree |
|---|---|---|
| Centre sampling | FCOS / ATSS | neighbour positives |
| Hard-negative mining | classic / RetinaNet | background steps |
| Quality-aware objectness | VarifocalNet | VFL on objectness (baseline) |
| IoU-aware classification | VarifocalNet | quality-class path (WIP) |
| Ranking / AP-oriented | AP-Loss, RankDetNet, RS Loss | bounded ranking + surgical FP (shipped; mid-band still weak) |
| Task-aligned assign | TOOD | partial (softened) |
| Centerness ranking | FCOS | tried; **ablated** (cut TP) |
| Sequential RF fusion | PLAN_UPDATED memory schedule | branch workspace |
| Sparse FP head updates | OHEM / hard-neg only | surgical argmax-class BG (shipped) |

---

## 8. Reviewer questions (explicit)

1. Is **F1 0.043 / TP 26** on 878 unique images sufficient to continue
   architecture research, or is data expansion mandatory first?  
2. Do you accept the **ranking diagnosis** (empty thr≥0.20 TP tail) as primary?  
3. Confirm **quality-class → ranking → HNM → data** over f12 / deeper box head?  
4. Should **max_detections=6** be the product default or only a research knob?  
5. Accept sequential branch fusion as the **canonical** memory path?  
6. State format after head-width change — bump version or width-check v2?  
7. What accuracy gate would you set before FPGA/power work?  
   (External provisional: F1≥0.07, TP@0.20≥8, FP/TP&lt;15, arena&lt;220 KiB.)

---

## 9. File pointers for deep review

| Concern | Open first |
|---|---|
| Loss / VFL / assignment | `src/kshira_rad.c` (`rad_train_positive_at_cell_ex`, `train_step`) |
| HNM / LR / multi-scale adapter | `src/det_kshira.inc` |
| Deploy score / NMS | `kshira_rad_predict`, `suppress_duplicate_detections` |
| Memory plan | `kshira_rad_build`, sequential branch block |
| Persistence | `src/kshira_rad_state.c` (version 2) |
| Metrics recipe | `tools/bench.c`, `docs/HANDOFF.md` §5 |
| Ranking diagnosis | `docs/reviewer_response.md` |

---

## 10. One-paragraph summary for a reviewer

KSHIRA Updated implements a novel ISO-C, arena-bounded detector with contrast-
modulated hybrid features and block-local training. Closed dual-run F32 peak is
F1 **0.043 / TP 26** (mIoU~0.62, arena ~159 KiB). Quality-class + surgical FP
ranking path dual-runs at F1 **0.0355 / TP 20** — recovers prior QC, does not
beat closed peak, and **does not** separate mid-band scores (thr flat). INT8 +
more unique images already exceeds closed F1. Dominant failure remains ranking /
FP on a small unique-image set. **1–2 weeks of further ideation can improve the
closed bar and ranking diagnostics; it will not deliver YOLO-class accuracy**
without substantially more data and capacity. Success metric remains multi-
objective time-to-accuracy under arena-bounded ISO C, not COCO mAP claims.
