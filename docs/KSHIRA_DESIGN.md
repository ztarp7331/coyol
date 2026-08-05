# KSHIRA implementation contract (reviewer edition)

**Date:** 2026-08-05  
**Authority:** `PLAN_UPDATED.md` for design intent; this file for what the tree actually implements.  
**Pair with:** `docs/HANDOFF.md` (metrics), `docs/RESEARCH_CRITIQUE.md` (gaps), `docs/reviewer_response.md` (next ranking path).

KSHIRA is an **architecture profile** of `det_model`. Callers use the same public
lifecycle as CDET; they select KSHIRA via `det_model_spec.architecture`.

**Multi-objective goal (not arena-only):** accuracy, CPU time-to-accuracy, peak
train memory, portable custom C kernels, quantized deploy/adapt — simultaneously.
See `docs/reviewer_response.md` (Fast-Convergence Path).

**Three layers of truth in this document:**

1. **Closed dual-run baseline** — F1 **0.0430** / TP **26** (pre quality-class).  
2. **Surgical QC dual-run (locked)** — F1 **0.0355** / TP **20** (quality-class +
   surgical FP / ranking); see `HANDOFF.md` §5.1c. Ranking mid-band still open.  
3. **Phase 2+** — budgeted dense-to-sparse train, data, engine — not yet dual-run
   as a new accuracy bar.

---

## 1. Public lifecycle

```
det_context_create(arena_bytes)
  → det_model_build(spec with DET_ARCH_KSHIRA)
  → det_train / det_predict / det_evaluate
  → det_save / det_load
  → det_model_destroy / det_context_destroy
```

Dataset-neutral: adapters (`det_manifest_*`, callback datasets) supply
`det_image` + `det_box[]`. No COCO/YOLO parsers inside kernels.

---

## 2. Module map

| Module | Responsibility |
|---|---|
| `kshira/core.h` + arena | Caller-owned bump allocation, high-water |
| `kshira/quant.h` | Symmetric INT4/INT8, pack, QAS gradient scale |
| `kshira/sparse.h` | Channel masks, update-mode memory plans |
| `kshira/phase.h` | PRE → TRAIN → ODT contracts |
| `kshira/session.h` | Arena + RAD + mask + phase composition |
| `kshira/rad.h` + `kshira_rad.c` | Stem, branches, mixer, contrast, head, train/infer |
| `kshira_rad_state.c` | Pointer-free export/import (**format version 2**) |
| `det_kshira.inc` | Adapter: train schedule, HNM, LR decay, multi-scale ODT |

---

## 3. RAD encoder (compact reference)

**Input:** 160×160, 1–4 channels (bench uses 1-channel PGM).  
**Map:** 40×40 at stride 4.  
**Features:** 8 default (12 fits 256 KiB after sequential branches; 16 needs ~384 KiB).

### 3.1 Forward graph (measured baseline)

1. **Stem** — 3×3, stride 4, ReLU.  
2. **Parallel depthwise dilations** — d ∈ {1,2,4}, same C (executed with **one**
   sequential branch workspace).  
3. **Mean fuse** — (B1+B2+B3)/3.  
4. **Pointwise mixer** — W∈ℝ^{C×C}, b∈ℝ^C, ReLU (**project_weights**).  
5. **Contrast** — neighbourhood r=2; κ=Σ_c (M−mean)²; C=0.10·log1p(κ).  
6. **Quality-class head** — input dim **C+1**; outputs = **4 box + K class-quality logits** (no separate objectness).

### 3.2 Memory-efficient branch schedule

- **One** full branch activation map is allocated; dilations run sequentially and
  accumulate into `fused` (PLAN_UPDATED memory-efficient schedule).  
- Local training recomputes branch cells on the stack for correct multi-dilation
  gradients (`depthwise_branch_cell`).  
- Measured: f8 profile high-water **~159 KiB** inside 256 KiB (was ~261 KiB).

### 3.3 Dependency tile

- Contrast radius 2 + max dilation 4 → **r_dep = 6** → **13×13** stem support.  
- Primary positive: FULL encoder update through the tile.  
- Neighbour positives: head-only (box + objectness / quality).

---

## 4. Training objectives (measured baseline)

### 4.1 Assignment

- Primary cell: ground-truth box centre on the P3 map.  
- Neighbours: box-adaptive radius (clamp 1–3), **in-box** cells only, LR scaled by
  polynomial centre prior.  
- Full PLAN_UPDATED staged contrast×IoU **top-k drop** was ablated (early TP collapse).  
- Soft IoU-aware targets remain on objectness (VFL).

### 4.2 Losses (current quality-class path)

| Head | Loss |
|---|---|
| Box | Smooth-L1 on (l,t,r,b) + (1−IoU) regularizer |
| Class-quality k* | **Varifocal** with IoU-aware target q (centre floor ≥0.80 after warm-up) |
| Other qualities / background | VFL negative (q=0); HNM on hard cells |
| Ranking | Bounded hinge via `kshira_rad_train_rank_pair` (train loop, epoch≥1) |

Historical closed peak (F1 0.043) used objectness VFL + softmax CE; see HANDOFF §5.1.

### 4.3 Hard-negative mining (`det_kshira.inc`) — baseline

- Outside-box cells only.  
- Epoch-staged budget (1→2→3); probe up to 16 cells; train hardest objectness.  
- Focal-style negative strength on background steps.

### 4.4 LR

- Inverse-time: `lr / (1 + 0.35 · epoch)` (aggressive decay; milder decay ablated).

### 4.5 Multi-scale

- Optional scale heads (P4/P5) for **ODT** (`multiscale_heads=1`).  
- PRE co-supervision of scale heads **ablated** (regressed real-car F1).  
- Inference skips untrained scale heads.

---

## 5. Inference / deploy scoring

### 5.1 Deploy scoring (quality-class — current)

```
Q_{x,k} = σ(z_{x,k})                    # independent class-quality logits
quality = max_k Q_{x,k}
score   = σ(logit(quality) / 0.55)      # mild monotone sharpen for top-K
```

No objectness × softmax product. Head width **4+K** (box + K qualities).

Post-process:

- insert fixed top-K by score;  
- class-aware IoU suppress **0.40**;  
- cross-class near-duplicate kill at higher IoU.

Positive targets: y_{k*} = IoU-aware quality; y_{k≠k*} = 0.  
Background: all y_k = 0.

### 5.2 Closed peak scoring (historical, §5.1 HANDOFF)

Previous dual-run peak used `sharpen(objectness) × max_class` (4+1+K head).

---

## 6. Phase 1 ranking path (shipped; surgical FP dual-run locked)

From `docs/reviewer_response.md` Fast-Convergence Phase 1 + surgical FP work.

| Item | Intent | Status |
|---|---|---|
| Quality-class head | 4+K outputs; VFL on class qualities | **shipped** (train/infer/background) |
| Score = max σ + T=0.42 sharpen | Deploy ranking | **shipped** |
| Surgical FP background | only winning-class head row; mid-band LR×1.35; easy-BG gate | **shipped** dual F1 0.0355 |
| Bounded ranking loss | hinge via `kshira_rad_train_rank_pair` (epoch≥2; surgical neg class; separate pos/neg QAS) | **shipped** |
| Two-level HNM | diverse head-only + top-1 encoder FULL late | **shipped** |
| Quality bias ~−2.0 | init prior | **shipped** |
| Identity mixer W=I+ε | preserve fused features early | **shipped** |
| Neighbour class-quality | soft IoU×centre on neighbours | **shipped** |
| LR decay 0.20 | more useful late epochs | **shipped** |
| Contrast gate | learnable / calibrated scale | **not started** |
| Gradient budgets / aux dense | Phase 2 novelty | partial (surgical = first sparse step) |

**Do not** claim accuracy beats F1 0.043 / TP 26 until dual-run exceeds it.
Surgical QC is **below** that bar; ranking thr-separation is still open.

---

## 7. Quantization

- Modes: F32, INT8, INT4 (W4A8 not on KSHIRA path).  
- Symmetric weight quant; activation scales via calibration / dynamic fallback.  
- QAS gradient scaling on quantized train.  
- Contrast remains non-negative feature; head sees float contrast in current path.  
- **State format v2** required (hybrid head width); v1 import rejected.  
- Ranking-path head width change (4+K vs 4+1+K) may require a **new state version**
  or a compatibility check on `outputs` before load — track in state work.

---

## 8. Persistence

- `kshira_rad_export_state` / `import_state` via `det_save` / `det_load`.  
- Magic `KRAD`, **version 2**.  
- Round-trip tested; predict outputs must match after import (padding-safe zero-init candidates).  
- After quality-class lands: verify v2 import rejects mismatched head width, or bump
  version with an explicit migration policy.

---

## 9. Phases

| Phase | Typical use |
|---|---|
| PRE | F32 full updates from random init |
| TRAIN | Quantized full / sparse after transition |
| ODT | Channel-masked head (and scale-head) adaptation |

`kshira_session` validates phase contracts before steps.

---

## 10. Resource profiles (measured)

| Profile | Arena | Result (approx HWM) |
|---|---|---|
| 5-class, f8, max_det 6 | 256 KiB | **~159 KiB** admitted |
| 5-class, f12, max_det 6 | 256 KiB | **~238 KiB** admitted |
| 5-class, f16, max_det 6 | 256 KiB | **reject** (needs ~384 KiB) |
| 80-class, f8, max_det 64 | 256 KiB | admitted after sequential branches |

**Policy:** extra free arena after sequential branches should prefer diagnostics,
hard-negative pools, and higher research `max_detections` — **not** immediate f12
(f12 collapsed TP on 878 unique images).

---

## 11. Contracts (hot path)

- **No** `malloc` inside train step / conv / assign / predict / NMS-like suppress.  
- Temporaries: arena-planned or fixed stack (e.g. feature vectors ≤ RAD_MAX_HEAD_IN).  
- Arena high-water reported on every `det_model_memory` / bench line.

---

## 12. What is intentionally incomplete vs PLAN_UPDATED

| Spec item | Implementation status |
|---|---|
| Full staged assignment schedule (a_t, b_t top-k) | Softened to centre-prior coverage |
| Piecewise-linear integer log1p for contrast deploy | F32 log1p reference; int approx optional later |
| Persistent full contrast map | On-demand per cell (saves arena) |
| Shared multi-scale affine g_s, b_s | Optional independent scale heads instead |
| Official 5k unique one-pass | Data not available; not claimed |
| Quality-class + ranking stack | Accepted next path; see §6 |

See `docs/RESEARCH_CRITIQUE.md` and `docs/reviewer_response.md` for scientific
priority of these gaps.
