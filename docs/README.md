# Documentation index (for reviewers)

**Snapshot date:** 2026-08-05  
**Repository:** `https://github.com/ztarp7331/coyol.git`  
**Claim boundary:** YOLO-like **lifecycle** and arena-bounded ISO C detection — **not** COCO mAP parity.  
**Reframed product goal (latest review):** fast-converging, dataset-neutral ISO C detector that approaches **nano-YOLO accuracy** with substantially lower CPU **time-to-accuracy**, peak train memory, and runtime infrastructure. “Lowest train time” = wall-clock to a fixed F1/AP, not only s/epoch.

| Document | Read when you need… |
|---|---|
| **[HANDOFF.md](HANDOFF.md)** | Latest metrics, exact bench commands, datasets, ablations, review checklist |
| **[KSHIRA_DESIGN.md](KSHIRA_DESIGN.md)** | What the tree implements: modules, forward graph, losses, memory, quant |
| **[RESEARCH_CRITIQUE.md](RESEARCH_CRITIQUE.md)** | Scientific gaps, over-claims, failure modes, experiment ladder |
| **[reviewer_response.md](reviewer_response.md)** | **Latest** anonymous review: Fast-Convergence Path + implementation phases |

**Design authority:** [`../PLAN_UPDATED.md`](../PLAN_UPDATED.md)  
**Public API truth:** [`../include/det.h`](../include/det.h)

---

### Suggested review order (30–45 min)

1. **This page** — claim boundary + best reproduce command  
2. **`HANDOFF.md` §5** (best metrics) + **§9–10** (checklist + next work)  
3. **`reviewer_response.md`** — reframed goal + Phase 1–4 priority (quality-class first)  
4. **`RESEARCH_CRITIQUE.md` §3–6** — failure modes + experiment ladder  
5. **`KSHIRA_DESIGN.md` §3–6** — architecture contract vs WIP quality-class  
6. Spot-check `src/kshira_rad.c`, `src/det_kshira.inc`, `src/kshira_rad_state.c`

---

### Best reproduce command (locked surgical QC)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/det_bench --architecture kshira \
  --manifest datasets/prepared/cars/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --epochs 8 --learning-rate 0.004 --features 8 --max-detections 6 --precision f32
```

| Metric | Closed F32 peak | Surgical QC F32 (locked) | Phase D INT8 + 1162 unique |
|---|---:|---:|---:|
| **F1** | **0.0430** | **0.0355** (dual) | **0.0539** (dual) |
| TP | **26** | **20** | 18 |
| unique train | 878 | 878 | **1162** (`cars_plus_od`) |
| train time | ~35 s / 8 ep F32 | ~62–65 s / 8 ep | ~70 s / 5 ep INT8 |
| Arena | ~159 KiB PASS | PASS | PASS |

**Surgical FP path:** only argmax-class head weights on hard BG; ranking epoch≥2;
mid-band mild boost. Recovers F1 vs restore 0.0281; **does not** open thr≥0.20
score separation (pred count still flat across thr). Full tables: `HANDOFF.md` §5.1c.

**Viz:** `bash tools/run_best_viz.sh` → `results/viz_all/` (250 PNGs).

**Phase D:** `python3 tools/expand_cars_stream.py`. F32 off-domain expansion
regressed; **INT8 dual on expanded data beats closed F1** with higher precision.
Closed **TP 26** remains the F32 recall bar. Full tables: `HANDOFF.md` §5.1b.

```bash
python3 tools/expand_cars_stream.py
./build/det_bench --architecture kshira \
  --manifest datasets/prepared/cars_plus_od/train/manifest.txt \
  --eval-manifest datasets/prepared/cars/valid/manifest.txt \
  --epochs 5 --learning-rate 0.003 --features 8 --max-detections 6 \
  --width 160 --height 160 --classes 5 --precision int8
```

### Can we hit YOLO-close in 1–2 weeks?

**No.** Sprint-realistic: beat F1 0.043 / TP 26 and improve ranking diagnostics.
YOLO-class accuracy needs far more unique data, capacity, and usually pretrain —
see `HANDOFF.md` top “Honest 1–2 week outlook”.

---

### One-paragraph status (honest)

KSHIRA is a `det_model` profile: ISO C, arena-bounded, multi-objective (accuracy,
time-to-accuracy, peak memory, portable kernels, quant)—not 256 KiB alone.
Quality-class + **surgical FP** ranking path is dual-run locked at F32 F1
**0.0355 / TP 20** (below closed F32 peak 0.043/26; above restore 0.0281/16).
Mid-band ranking remains unsolved (thr does not cleanly filter FPs). Phase D
INT8 on 1162-unique cars_plus_od reaches F1 **0.0539 / TP 18**. Next: open
thr≥0.20 TP tail + **in-domain** unique 5-class images; do not claim YOLO mAP
parity.
