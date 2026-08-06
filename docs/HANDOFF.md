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

**Phase 0/1 implementation added 2026-08-05:** `det_train_config` now accepts an
optional measured wall-clock budget and opt-in frozen-feature analytic quality-head
refresh. `tools/det_bench` exposes `--time-budget-ms` and `--analytic-readout`;
`det_eval_report` now emits score histograms for loose IoU-positive versus
non-matching candidates. The ridge readout uses caller-provided statistics, a
tested Cholesky solve, and objective-based rollback, so zero-valued new fields
preserve the legacy recipe.

Real cars878 validation with the new path and a 30,000 ms budget completed with
30,006.9 ms train-plus-decode and 30,011.3 ms end-to-end, arena high-water
158,601 / 262,144 bytes, F1 0.0254 (TP 16, FP 788, mIoU 0.6214, AP50 0.0007).

**2026-08-06 fast-convergence continuation:** the shared P3/P4/P5 research path
now has scale-aware assignment and epoch-3 auxiliary-scale unlocking. A small
time-budget controller was added in `include/kshira/budget.h` and
`src/kshira_budget.c`; `--adaptive-budget` makes its measured dense/mixed/residual
stages opt in, with a 1,000 ms finalization reserve. The established epoch
schedule remains the default after the adaptive ablation regressed seed 2.

The adaptive ablation completed within budget but was not promoted: seed 1 was
F1 0.0710 (36 TP / 524 FP, train-plus-decode 29.03 s), while seed 2 was F1
0.0032 (2 TP / 806 FP, 29.02 s). The default schedule on a fresh seed-2 run
was F1 0.0533 (32 TP / 714 FP, 29.03 s). These are internal cars878 metrics,
not nano-YOLO parity; the controller needs a residual/validation feedback loop
before it can be considered accuracy-improving.

An attempted fix for an uninitialized pyramid contrast slot was also reverted:
it produced zero true positives on the same real-data protocol. Full CTest
remains green after the controller module and opt-in CLI integration.
The analytic objective fell from 34,126.94 to 374.08, but the score histogram
remained concentrated in the 0.6–0.7 band; this is a working ablation, not a
nano-YOLO accuracy claim. The same timed run without readout produced F1 0.0025
(TP 2, FP 1,154), so the readout is useful relative to that short-budget control
but does not yet beat the locked 0.043 historical bar.

**Phase 2 cache/bootstrap ablations:** deterministic target-domain tight/context
crop bootstrap is implemented behind `--bootstrap` but failed its equal-time
gate on cars878 (F1 **0.0115**, TP 8, FP 932) and remains opt-in. The predecoded
manifest cache (`--predecode-cache`) is different: with the same 30,000 ms CPU
training budget and no analytic refresh it processed 7,406 real training samples
and reached validation F1 **0.0649**, TP 20, FP 142, precision 0.1235, recall
0.0441, AP50 0.0142, and mIoU 0.5957. Training was 29,010.5 ms; cache
preparation was 2,380.5 ms and is reported separately. This is the first new
real-data result above the locked F1 0.043 bar.

**Matched external baseline:** a pretrained YOLOv8n checkpoint (3,011,823
parameters) was adapted on the same 878/250 split with CPU, 160px inputs,
batch 16, cached images, and a 30-second training budget. Ultralytics reports
precision 0.0093, recall 0.4955, mAP50 **0.0892**, and mAP50:95 **0.0652** after
one completed epoch. This is a useful adaptation reference, not yet a shared
evaluator result: KSHIRA's F1/AP50 numbers above come from the C evaluator and
the two metric implementations still need to be unified before a parity claim.

The matching scratch YOLOv8n control reached precision 0.0123, recall 0.2589,
mAP50 **0.0320**, and mAP50:95 **0.0100** under the same conditions. Increasing
the existing KSHIRA RAD width from 8 to 32 channels was rejected: with a
1 MiB arena it processed only 1,805 samples and reached F1 0.0036 (AP50 0).
This confirms that the cached 8-channel path, not blind width expansion, is the
current compact KSHIRA profile.

The preparation-time crop variant was also gated on real data. Adding one
context crop per ground-truth box produced 2,554 samples; the predecoded
30-second run processed 8,453 samples but reached only F1 **0.0058** and AP50
0, so this bootstrap variant remains rejected. The YOLO teacher-manifest
exporter is wired for reproducible adaptation experiments. The initial
one-epoch teacher added zero non-overlapping boxes at confidence 0.05, 0.25,
or 0.50 and was rejected. A separately prepared, five-minute pretrained
YOLOv8n teacher reached YOLO mAP50 **0.3705** on the same validation split and
added **490** non-overlapping boxes to the 878-image training manifest at
confidence 0.25. One 30-second KSHIRA adaptation run on that real
teacher-augmented stream reached F1 **0.0516**, TP 22, FP 376, precision
0.0553, recall 0.0485, AP50 0.0052, and matched-box mean IoU 0.6258 on the
untouched 250-image validation manifest. However, the required three-seed
repeat was F1 **0.0516 / 0.0403 / 0.0277** (median 0.0403), so the historical
F1 0.043 bar was not beaten reproducibly. This remains a useful adaptation
signal, not a promoted nano-YOLO-equivalent result.

The current accuracy-profile candidate uses the same F8 legacy graph with
`max-detections=20` and the unweighted confidence-0.25 teacher stream. Three
30-second real-data runs reached F1 **0.0603 / 0.0732 / 0.0193** (median
**0.0603**), with AP50 **0.0161 / 0.0141 / 0.0013**. This clears the historical
F1 bar in the median but remains too variable and far below the pretrained
YOLO reference for promotion. `max-detections=64` and F16 were rejected as
accuracy improvements. Optional manifest confidence weights and reusable
`--init` checkpoints are implemented; confidence-weighted teacher targets
were rejected in one measured run (F1 0.0111).

**2026-08-06 fast-convergence continuation:** `src/kshira_pyramid.c` now
provides a learned P3/P4/P5 representation with an exact context-off pooled
fallback. The context profile unlocks P4/P5 positive auxiliary updates only
after the P3 warm-up (epoch 3) at a reduced rate. On the real cars split this
restored the staged context run to internal F1 **0.0430** (24 TP, 638 FP,
AP50 0.0010) after a prior context regression. An experimental 2x
context-only hard-negative rate reached internal F1 **0.0239**, common
mAP50 **0.0071**, and mAP50:95 **0.0021**. The normal-rate teacher control
reached internal F1 **0.0570** and common mAP50 **0.0068**. Neither is promoted:
both remain far below the matched YOLO mAP50 **0.0892** / mAP50:95 **0.0652**
reference and need repeated seeds.

The analytic readout now fits bounded logits rather than raw 1/0 targets;
raw targets made background converge toward a 0.5 sigmoid score. The corrected
readout remains opt-in and reached only F1 **0.0126** here. The teacher exporter
also accepts output paths outside the source tree by emitting absolute image
paths. Bootstrap remains opt-in after a teacher-plus-bootstrap run reached F1
**0.0108**.

The current scale-aware assignment sends each context-profile ground-truth box
to one P4/P5 level by area instead of supervising both levels. With the epoch-3
unlock, three real 30-second runs reached internal F1 **0.0648 / 0.0228 /
0.0264** (median **0.0264**) and common mAP50 **0.0073 / 0.0020 / 0.0022**
(median **0.0022**). This beats the historical F1 bar on one seed but is not
repeatable or close to the YOLO reference, so it remains a research profile,
not a promoted result.

The benchmark now supports `--init`, `--eval-only`, `--seed`,
`--max-train-samples`, `--freeze-encoder`, `--reset-schedule`, and
`--save-model`. Foundation checkpoints are evaluated separately from the
30-second adaptation window.
The 30-epoch teacher foundation itself reached F1 0.0321 before adaptation;
initialized adaptation was not yet reproducible across subsequent runs, so it
is not the admitted accuracy profile.

The common-evaluator gate is now implemented in
`tools/common_ultralytics_eval.py`. It performs Ultralytics-compatible
per-image one-to-one matching and calls the pinned Ultralytics
`ap_per_class` implementation for both KSHIRA reports and YOLO prediction
files. The native YOLO validation result is reproduced exactly at mAP50
**0.0892** and mAP50:95 **0.0652**. A saved 30-second KSHIRA teacher-stream
run (F8 legacy, seed 2, max detections 20) scored mAP50 **0.0039** and
mAP50:95 **0.0008** under the same evaluator. Therefore the comparison is now
metric-aligned, but KSHIRA is not yet nano-YOLO-like and the profile remains
unpromoted.

**2026-08-06 native Nano-mechanism continuation:** the canonical Ultralytics
YOLOv8n graph was vendored as a separate C inference library so its compact
mechanisms can be measured and selectively folded into KSHIRA: C2f feature
reuse, SPPF, PAN/FPN multi-scale features, DFL box decoding, class-aware NMS,
and an INT8/INT4 weight path. It is not treated as a second public framework;
the shared `det_model` training lifecycle remains the integration target.

The first native export exposed a real deployment bug rather than an accuracy
result: the exporter used an activation scale of 0.25 while preprocessing used
1/127, and postprocessing interpreted class logits as int8/32. The native
preprocess/postprocess contract now consumes the exported scale. On the real
250-image cars validation split, the calibrated native C graph at 160px,
confidence 0.25, and IoU 0.50 produced **F1 0.2622** (86 TP, 116 FP, 368 FN),
common-style AP50 **0.1334**, and mAP50:95 **0.0947**. This is materially above
the prior YOLOv8n 30-second reference in the same repository, but it is an
inference-only pretrained checkpoint result: it is not yet evidence that C can
train the whole graph in 30 seconds.

The native evaluator is `tools/eval_yolov8_native.py`; its result was verified
through `tools/yolov8_native_detect.c` on the real manifest, not a synthetic
mock. The next integration gate is to use this graph as an accuracy teacher
and train a bounded C-side KSHIRA adapter within the existing 30-second
budget, followed by F32/INT8/INT4 parity and checkpoint round-trip tests.

The first bounded adapter is now routed through `det_train` when a native
export is attached. It freezes the native graph and fits a small score affine
calibration against real box matches. The current Release run completed a
30,000 ms request in **27.743 s**, processed 81 real training images, and made
81 updates; the learned gain/bias were 0.9978/0.0036, so this calibration is
nearly identity. The native path can now be saved as a self-contained `CY8N`
profile checkpoint; a real-image smoke test produced an equivalent
checkpoint round-trip, and the serialized Release checkpoint reproduced the
common evaluator result below.

The native INT4 export path is also real but currently fails its parity gate:
the first eight-image smoke evaluation produced no true positives at the
default activation scale, and a second scale (0.5) also produced **0 TP**
with 12 false positives. INT4 is therefore kept as an explicit open
quantization task rather than being presented as a working Nano-equivalent
deployment mode.

A mixed W4A8 export is now available through
`tools/mix_yolo_quant.py`. It keeps the first 16 and final 18 convolutions in
INT8 and packs the remaining 29 body convolutions in INT4. On all 250 real
validation images this profile reached **56 TP / 56 FP / 398 FN**, F1
**0.1979**, AP50 **0.0858**, and mAP50:95 **0.0541**. Its weight blob is
1.98 MB versus 3.00 MB for the all-INT8 export. This is a useful configurable
compression profile, but it remains below the INT8 accuracy profile and is not
promoted as quantization parity.

The real Scratch gate was also rerun with KSHIRA's one-to-one and shared
multi-scale heads, adaptive/residual budgeting, and analytic readout. The
30-second controller finished early after processing all 878 training images:
the non-readout run reached AP50 **0.0012** with 60 TP and 2,860 FP, while the
analytic-readout run reached AP50 **0.0014** with 52 TP and 1,848 FP. The score
histogram remained collapsed, so this is evidence that Scratch still lacks a
useful learned representation; it is not a promoted Nano-like result.

The native path now also contains a standalone YOLO26n-style graph. The
Python reference was fine-tuned for the five-class cars data in roughly 34.5 s
of CPU training and saved as `runs/yolo26_30s_cpu_saved/weights/best.pt`;
`tools/export_yolo26_weights.py` exports its fused C3k2/C3k/C2PSA/SPPF/PSA
graph and the end-to-end `reg_max=1` one-to-one head (102 convolution records).
`yolov26_native_detect` executes the graph on real PGM images, and its C
postprocess follows YOLO26's two-stage top-anchor/top-class selection.
The exporter also emits raw FP32 sidecars for graph-parity diagnostics. The
public CY8N lifecycle now persists and reloads those sidecars when present;
the predictor selects the FP32 graph only under `--precision f32`, while the
INT8 lifecycle continues to use the compact quantized graph.

The first native INT8 measurements (AP50 **0.0183**, then **0.0423** with
per-stage scales) exposed two C graph mismatches: the checkpoint's SPPF block
has an enabled residual shortcut, and the attention Q/K/V tensors are
head-interleaved after PyTorch's reshape. Both are now implemented correctly
in the C graph. The corrected full validation result is INT8 AP50 **0.0703**
and mAP50:95 **0.0377**; the exact FP32 C reference reaches AP50 **0.0781**
and mAP50:95 **0.0458**. The Python reference remains approximately
**0.0963** mAP50 and **0.0596** mAP50:95. Quantization is now a bounded,
measured loss rather than an unverified topology mismatch, but this is still
not a Nano-like accuracy claim.

**End-to-end Adapt gate:** Release `det_bench` trained the frozen native
graph's C-side score adapter for **27,842 ms** (85 real training images and 85
updates), then evaluated all 250 validation images. The repository evaluator
reported 78 TP,
102 FP, 376 FN, AP50 **0.2642**, and mAP50:95 **0.1627**. The common
Ultralytics-compatible report path over the same serialized checkpoint reported
F1 **0.2403**, mAP50 **0.2800**, and mAP50:95 **0.2074**. This is the strongest
current real-data result, but it is explicitly **KSHIRA Adapt**: the native
YOLOv8n graph came from a pretrained 5-minute foundation checkpoint. It is not
the Scratch-mode 30-second result required for a full claim.

The same Release adapter benchmark was repeated with the mixed W4A8 export. It
completed in **27.588 s**, used 74 real training images and 74 updates, and
survived CY8N save/load. The C evaluator reported 58 TP, 64 FP, 396 FN, AP50
**0.2799**, and mAP50:95 **0.1598**. Its common evaluator report was F1
**0.2062**, mAP50 **0.2930**, and mAP50:95 **0.2048**, with a 2.03 MB
checkpoint. The native checkpoint reader now validates each convolution's
recorded bit width, so mixed-bit profiles are portable through the public
lifecycle.

The public KSHIRA lifecycle now also accepts the YOLO26 graph identifier and
serializes its per-stage activation scales, SiLU LUTs, and score-adapter
state. A clean Release build passed all three CTest suites, and the YOLO26
checkpoint round-trip smoke reproduced identical detections before and after
reload. The YOLO26 postprocess now ranks the selected candidates and applies
class-aware NMS before exposing them to training or evaluation; this fixes the
previous duplicate-box flood. The native adapter also learns a small,
scale-conditioned center/size correction from matched real boxes and persists
it in CY8N checkpoints. With the native maximum raised to 100 candidates, the
corrected 30-second Adapt run completed in **27.591 s** of end-to-end training
with **99 updates**, then evaluated all 250 validation images: **221 TP,
8,300 FP, 233 FN, AP50 0.1047, and mAP50:95 0.0502**. Mean matched IoU was
**0.7378**, inference averaged **312.1 ms/image** at 160px, the checkpoint was
**2.45 MB**, and the bounded arena remained within budget. The pre-NMS result
(AP50 0.0711, mAP50:95 0.0375) is superseded. A teacher cache containing
14,866 pseudo-boxes was also measured but rejected: it reached AP50 0.1014 and
mAP50:95 0.0492. Nano-like accuracy and low-millisecond inference remain open
gates; the native NMS/adapter path is a real improvement, not a completed
Nano-equivalence claim.

The same max-100 path through the public `--precision int8` lifecycle completed
in **27.754 s** and reproduced AP50 **0.1047** and mAP50:95 **0.0500**, with
the same checkpoint and arena bounds.

The next parity run corrected the C manifest adapter to use half-pixel
bilinear PNM resize, matching the Python preprocessing contract more closely.
With the FP32 sidecars enabled, the public YOLO26/KSHIRA Adapt profile completed
in **27.772 s**, processed 126 real training images, and reached **274 TP,
8,984 FP, 180 FN, AP50 0.1245, and mAP50:95 0.0691** on the 250-image
validation split. Mean matched IoU was **0.7540** and native inference was
**217.5 ms/image**. The corresponding INT8 run completed in **27.889 s** and
reached **AP50 0.1083 and mAP50:95 0.0516**. A fresh INT8 run using the
promoted native-adapter default `learning_rate=0.10` completed in **27.950 s**
and reached **AP50 0.1069 and mAP50:95 0.0515**, so the quantized route remains
within its measured parity band. Both results pass the CTest suite
and FP32 CY8N checkpoint round-trip; they are the current measured profiles,
but the remaining inference-latency and full Scratch-training gates keep the
Nano-equivalence claim open.

The saved FP32 checkpoint was independently scored through
`tools/common_ultralytics_eval.py` after `viz_detect` dumped all 100 allowed
native candidates per image: **precision 0.0335, recall 0.6980, F1 0.0628,
mAP50 0.1277, and mAP50:95 0.0716** over 250 images. The dump originally
under-counted four predictions because `viz_detect` retained the old 64-result
cap; that stale cap is now aligned with the native 100-candidate public path.

Two repeated native-adapter probes with an explicit learning rate of **0.10**
gave repository mAP50:95 values **0.0698 / 0.0696**; the saved run's common
evaluator result was **mAP50 0.1230 and mAP50:95 0.0726**. The benchmark now
uses `0.10` automatically for an attached native adapter unless the caller
passes `--learning-rate`, while Scratch and ordinary KSHIRA keep their prior
`0.01` default.

The three-seed native Adapt promotion audit at that default produced repository
mAP50:95 **0.0696 / 0.0707 / 0.0681** (median **0.0696**) with end-to-end
training times **27.97 / 27.94 / 27.95 s**. This is now a repeatable Adapt
profile, while it must remain labelled Adapt because its YOLO26 graph is
pretrained and the native C adapter learns only the deployment calibration and
box corrections.

An accuracy-oriented KSHIRA Scratch configuration (32 features, two-stage
stem, shared context-fusion pyramid, raw image cues, YOLO distance decode and
residual budgeting) was also measured on the same real split. It completed in
**29.14 s** but reached only **mAP50:95 0.0001**. This rejects the idea that
Scratch accuracy is fixed merely by widening the old RAD profile; assignment
and representation learning remain the next Scratch research gate.

The opt-in analytic-readout controller now reserves **3 s** of the requested
budget for its solver instead of leaving only the generic finalization reserve.
It accumulated **35,534** real candidate statistics and applied the solve in a
Scratch run, but that run still reached mAP50:95 **0.0000**. The solver is
therefore a functioning diagnostic and timing path, not yet a promoted Scratch
accuracy mechanism.

The optimized Release native kernel measures **0.39–0.40 s** per 160px image
on the current CPU, compared with about 1.50 s in the earlier debug build. A
plain non-MSVC CMake configure now defaults to Release so deployment timing is
not accidentally measured at `-O0`; SIMD/kernel fusion and sub-second hardware
qualification remain open.

The information-preserving space-to-depth stem is also implemented behind
`--stem space-to-depth`, with checkpoint/state versioning and round-trip tests.
Its equal-time cached 30-second cars run reached F1 0.0056 and AP50 0.0001,
so it remains an explicitly rejected ablation. The legacy stem remains the
default and keeps its original fast kernels.

The deployment-head separation required by the updated plan is now implemented
behind `--one-to-one-head`. Center assignments train a separate deployment head;
dense neighbors remain on the training head, and background/ranking/readout
updates are routed to the deployment head when enabled. At the time of this
experiment, the state format was KRAD v10 and the CDET wrapper was v17, with
round-trip coverage. A real cached
30-second cars run reached F1 0.0041, TP 4, FP 1,474, AP50 0.0001, and was
rejected as an accuracy profile. The flag is retained for further assignment
and end-to-end ablations; it is not the default and does not constitute a
YOLO26/NMS-free accuracy claim.

The shared P3/P4/P5 head profile is implemented behind `--shared-multiscale`.
It reuses one deployment head across pooled scales, keeps the independent scale
head storage available for training-only auxiliary work, and persists per-scale
gain/offset calibration in KRAD v8 / CDET v15 state. F32, INT8, and INT4
shared-profile parity is covered by the KSHIRA tests. Its first cached 30-second
cars run reached F1 0.0022 and AP50 0.0000, so it is also rejected as an
accuracy profile pending a stronger fused pyramid representation.

The learned wider-context pooled fusion ablation is now explicit behind
`--context-fusion` and is disabled by default. It is persisted in KRAD v10 / CDET
v17 at the time of that experiment. The first pooled-context implementation was rejected and replaced by a
trainable P4/P5 transform plus P5-to-P4 top-down addition. A new real-data
measurement is required before this profile can be promoted.

**2026-08-06 residual-budget continuation:** `kshira_rad_hard_negative_candidates`
now scans the complete P3 map once and returns sorted high-score residual cells;
the adapter filters GT-covered cells and retains the existing spatially diverse
budget. The mechanism is exposed through `--residual-budget` and is disabled by
default because its first real cars run reached F1 **0.0315** / AP50 **0.0051**,
below the legacy f16 control at F1 **0.0541** / AP50 **0.0157**. A separate raw
local-image-cue ablation (`--raw-features`) also regressed to F1 **0.0345** /
AP50 **0.0036**. Neither is promoted; both remain reproducible research
ablations. A clean f32 widening control reached F1 **0.0126** / AP50 **0.0004**
and reduced throughput to **85.9 images/s**, confirming that widening the
single-map legacy graph is not yet the planned balanced backbone.

**2026-08-06 two-stage stem continuation:** the learned stride-2 plus stride-2
stem is now available behind `--stem two-stage`, with learned backward updates
through both convolutions and F32/INT8/INT4 round-trip coverage. On the real
cars split, the first fixed-stem run reached F1 **0.0143** / AP50 **0.0001**;
after adding the pre-stem backward path, the saved checkpoint reached internal
F1 **0.0349** (18 TP, 560 FP), mean IoU **0.6175**, AP50 **0.0005**, and common
mAP50:95 **0.0001**. This is below the legacy f16 control at F1 **0.0541** /
AP50 **0.0157**, so the stem remains an opt-in rejected ablation. It does not
yet satisfy the plan's balanced-backbone gate.

The benchmark now accepts `--channels 1|3` and reports the input shape. The
default remains grayscale, while the model, manifest loader, synthetic smoke
path, and two-stage stem all execute a real three-channel path. The legacy
state formats are KRAD v11 and CDET v18, including the two-stage parameters and
persisted raw-feature configuration. The full WSL CTest suite remains green
(3/3 tests).

**2026-08-06 learned-fusion continuation:** the context profile now has
arena-resident cross-channel P3/P4/P5 lateral projections and a learned P5-to-P4
top-down matrix, identity-initialized so the old pooled behavior is the exact
starting point. The legacy profile does not allocate these tensors. Synthetic
F32/INT8/INT4 and context checkpoint round-trip tests pass, and the context
profile reports 10,684 parameter bytes at 16 channels. Its first real cached
cars run reached F1 **0.0179** / AP50 **0.0002** (10 TP, 652 FP), below the
legacy f16 control; it remains opt-in and rejected. The state formats for this
path are KRAD v12 and CDET v19.

The next stage increment adds a zero-initialized, trainable depthwise 3x3
residual refinement at each context-pyramid level. It is arena-resident and
included in the checkpoint/parity path; this is an implementation increment,
not an accuracy promotion. A full 30-second run with a single-threshold report
reached F1 **0.0264** / AP50 **0.0010** (12 TP, 688 FP), train-plus-decode
**29.03 s**, inference **59.8 ms**, and arena high-water **356,410 bytes**.
An earlier interrupted run's saved checkpoint evaluated at F1 **0.0441** /
AP50 **0.0074**, demonstrating instability rather than a promotion. The
refinement path remains rejected and the state formats are KRAD v13 and CDET
v20. The benchmark's explicit `--no-calibrate` option separates a declared
single-threshold evaluation from the much slower 11-threshold calibration.

**2026-08-06 deployment/assignment isolation:** context fusion now supports
`--p3-only-deploy`, retaining P4/P5 as training auxiliaries while emitting only
the stable P3 head. On the five-class cars validation split, the first
30-second cached run reached F1 **0.0542** (26 TP, 480 FP), AP50 **0.0213**,
train-plus-decode **29.01 s**, and inference **3.94 ms**. This restores the
legacy f16 accuracy range and passes the inference gate, but it is still far
below the common YOLO reference and is not promoted as Nano-like.

An opt-in YOLOv8-inspired task-aligned assignment path (`--quality-align`) was
also measured. It ranks in-box cells by class-score^0.5 × IoU^6, following the
local Ultralytics `TaskAlignedAssigner` contract, with a geometric warm-up.
The first warm-up run reached only F1 **0.0030** (2 TP, 866 FP) and processed
3,009 samples in the 30-second budget; the direct no-warm-up variant reached
F1 **0.0083**. The path is therefore rejected for now: KSHIRA's untrained box
head makes IoU^6 collapse, and the full-map ranking overhead reduces useful
training throughput. It remains opt-in for further redesign rather than
changing the established default.

The P3 deployment flag and assignment experiment are persisted in KRAD v15 /
CDET v22. Focused KSHIRA and detector tests remain green after the changes.

**2026-08-06 YOLO mechanism isolation:** the adjacent native C YOLOv8 graph was
verified as a portable inference reference; its unit test passes when the
compiler supplies the missing standard `<stddef.h>` include, but the graph is
explicitly inference-only and its exported activation scales are provisional.
Its useful mechanisms were therefore tested as KSHIRA ablations rather than
copied as a second runtime. An opt-in smooth positive box-distance
parameterization (`--yolo-distances`, softplus plus its matching derivative)
reduced the same cached context run to F1 **0.011** (6 TP, 562 FP), so it is
not promoted. The existing two-stage information-preserving stem likewise
processed only 1,355 samples and reached F1 **0.013**. A direct YOLO-like BCE
background gradient was also tested and reverted after reducing the measured
result; the established VFL background path remains the default. These
experiments confirm that isolated YOLO loss/stem substitutions do not yet
solve KSHIRA's score-separation gap.

The benchmark now reports an explicit `research_mode` selected by
`--research-mode scratch|adapt|full`, and KSHIRA checkpoints persist that mode
in the CDET wrapper. Teacher-assisted adaptation results can therefore no
longer be confused with scratch-training results.

## 3. Source map

| Area | Location |
|---|---|
| Public API | `include/det.h` |
| KSHIRA adapter | `src/det_kshira.inc` |
| RAD kernels (train/infer) | `src/kshira_rad.c`, `src/kshira_rad_internal.h` |
| State export (KRAD v15) | `src/kshira_rad_state.c` |
| Session / phase / quant / sparse | `src/kshira_session.c`, `phase`, `quant`, `sparse` |
| Raw PNM stream | `src/det_io.c` |
| Product bench | `tools/bench.c` |
| Dataset / native export tools | `tools/yolo_to_manifest.py`, `csv_boxes_to_manifest.py`, `filter_manifest_class.py`, `expand_cars_stream.py`, `prepare_yolo_manifest.py`, `prepare_crop_manifest.py`, `export_yolo_teacher_manifest.py`, `common_ultralytics_eval.py`, `eval_yolov8_native.py`, `mix_yolo_quant.py` |
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

## 12. Current continuation checkpoint (2026-08-06)

The C framework surface is now standalone in naming. The native graph runtime,
loader, checkpoint path, public API, CMake targets, and C tools no longer expose
the external reference model's name. The interoperation scripts and saved
research artifacts remain separate from the C runtime.

The restored source was regenerated and verified with:

- Release build: passed.
- CTest: 3/3 passed.
- Native lifecycle smoke: before=55 after=55 checkpoint_roundtrip=1.
- C/CMake source scan for the removed external name: clean.

A real 30-second CPU adaptation through the renamed API completed in
27.878 s end-to-end training and reached AP50=0.3004 and mAP50:95=0.1829
on the 250-image cars validation split. This is an Adapt result using an
imported graph and must not be described as Scratch training. The current
interface uses the native-graph-dir option.

The native adapter learning-rate sweep then promoted 0.02 as the default:
three real runs reached mAP50:95 0.1937 / 0.1954 / 0.1954 (median 0.1954),
with end-to-end training times 27.696 / 27.950 / 27.860 s. This is a
repeatable Adapt profile, not evidence of Scratch training or full objective
completion.

The same C adapter with INT8 graph execution completed in 27.732 s and reached
AP50=0.3076 and mAP50:95=0.1918, showing a 0.0036 absolute mAP50:95 drop from
the best FP32 run under this protocol.

INT4 execution completed in 28.010 s with a 1.55 MB parameter footprint, but
its mAP50:95 was 0.0452. INT4 is therefore supported and measurable, but is not
currently the promoted accuracy profile.

The best FP32 checkpoint was also passed through the repository's common
evaluator after a full C prediction dump at score threshold 0.05: precision
0.3136, recall 0.4257, F1 0.3222, mAP50 0.2545, and mAP50:95 0.1692 over all
250 validation images. This is the stronger cross-path comparison metric; the
internal C evaluator's 0.1954 should not be treated as interchangeable.

Current-source Scratch rechecks supersede the earlier predecode note above:
the legacy F8 predecoded run processed 60,718 samples and reached internal
mAP50:95=0.0026 after threshold calibration; without calibration it emitted no
validation predictions. Enabling quality-aligned assignment with dense-aux 16
processed 30,758 samples and emitted no validation predictions. These controls
are rejected on the current source, and the older 0.0649 Scratch number should
not be used as current evidence.

The C manifest cache now supports deterministic per-epoch shuffling through
--shuffle-cache. ASAN caught and the fix removed an initial cache-order bug;
release CTest is green afterward. On real Scratch data, shuffling improved the
internal F1 operating point to 0.0774 but left mAP50:95 at 0.0010, so it remains
opt-in. Applying the same option to the native Adapt path reached mAP50:95
0.1910, below the unshuffled 0.1954 median, and was not promoted as default.

The recent fixed-cue and naive dense-background Scratch ablations were rejected
(mAP50:95=0.0014 and 0.0000, respectively) and their experimental source
changes were reverted. Full Scratch parity remains an open research gate.

The native activation reservation was then instrumented with a C-side high-water
counter. A full 30-second adapter probe reached a 625,025-byte peak, so the
fixed arena was reduced from 64 MiB to 2 MiB. The real 30-second run completed
in 27.74 s with 86 samples; reported activation workspace fell to 2,329,120
bytes and the measured native arena peak remained 625,025 bytes. Release CTest
and lifecycle smoke remained green, and the existing ASAN det test binary
passed after the change. A final 0.5-second reserve experiment processed more
samples but scored mAP50:95=0.1924, below the promoted 2.5-second-reserve
median, so the original accuracy-oriented reserve remains default.

A standalone C teacher-manifest generator was also added. It runs the native
graph, preserves ground truth, skips same-class boxes with IoU at least 0.50,
and writes confidence-weighted pseudo boxes using absolute image paths so the
cache is portable. On the 878-line cars training manifest it added 90 boxes.
The resulting C-consumable cache was tested in a real 30-second Scratch run
(57,528 samples, 32.25 s including decode): it reached AP50=0.0004 and
mAP50:95=0.0000 with 8 TP and 366 FP. The cache path is therefore implemented
but rejected as an accuracy promotion; it does not establish Scratch parity.

Two current-source Scratch checks followed. The analytic readout path completed
the 30-second run in 29.13 s of training (52,045 samples), applied its solve,
but emitted zero validation detections; AP50 and mAP50:95 were both 0.0000.
The space-to-depth stem completed in 31.48 s end-to-end, processed 21,386
samples, and emitted six false positives with zero true positives. Both are
rejected. The legacy cached Scratch path remains the only reasonable research
baseline, while the native Adapt path remains the promoted accuracy profile.
A raw-candidate training experiment was then tested on the strongest current
single-output native graph. The trainer temporarily exposed every decoded
class candidate before final ranking, while keeping the graph weights frozen
and updating only the existing scalar adapters. The real 30-second CPU run
processed 62 images in 27.82 s end-to-end. On the full 250-image validation
split at the common 0.05 reporting threshold it reached precision=0.2264,
recall=0.3700, AP50=0.2848, and mAP50:95=0.1248. Since this was below the
established common-evaluator mAP50:95=0.1692, the raw-candidate path was
reverted and is not a promoted mechanism.

The next native adaptation refinement trains the three final per-scale class
head biases directly inside the owned native checkpoint blobs. It preserves
the frozen feature graph, uses fractional accumulator rounding for INT8 bias
updates, and updates the FP32 sidecar when present. With the promoted
learning_rate=0.02 recipe, three real 30-second runs on the cars training
manifest processed 64 / 67 / 68 images in 28.03 / 27.99 / 27.77 s. The common
250-image evaluator at threshold 0.05 reported mAP50:95=0.1907 / 0.1903 /
0.1909 and AP50=0.3074 / 0.3075 / 0.3075 (median mAP50:95=0.1907). The saved
seed-7 checkpoint retained the result under INT8 execution: mAP50:95=0.1907
and AP50=0.3074. This is now the promoted native Adapt refinement, but it is
still head-bias adaptation rather than full convolutional backpropagation or
Scratch training.

A center-based dense-cell extension was tested against that promoted path. It
used the raw C output tensors without another forward pass, but reduced the
30-second throughput to 57 images and produced 4,426 validation predictions
with 4,220 false positives; common mAP50:95 fell to 0.1567. The extension was
reverted. Dense supervision therefore remains an open research direction,
not a current default.

A frozen-backbone final-head-weight ablation was subsequently measured using
center positives and bounded hard negatives. It completed in 27.74 s and
reached AP50=0.3110 but mAP50:95=0.1899, slightly below the promoted
per-scale-bias median, with more false positives. The feature-capture and
weight-update path was reverted; final-head weights remain an open gate.

The promoted adapter was then extended with a class-and-scale box correction.
The C trainer learns four bounded values per class and output scale: center-x,
center-y, log-width and log-height. The values are stored in the native
checkpoint, and version-2 checkpoints are accepted by the generic C loader
while version-1 checkpoints remain readable. Three real 30-second cars runs
processed 75 / 77 / 89 images in 27.95 / 27.90 / 27.72 seconds. The common
250-image evaluator at threshold 0.05 reported mAP50:95=0.1927 / 0.1927 /
0.1923 and AP50=0.3081 / 0.3081 / 0.3093. The seed-7 checkpoint retained
mAP50:95=0.1927 and AP50=0.3081 under INT8 execution. This class-and-scale
correction is now the promoted native Adapt profile; full feature and weight
backpropagation and Scratch parity remain open gates.

A lower 0.30-IoU native matching threshold was also tested as a localization
ablation. Its seed-7 30-second run processed 83 images and raised AP50 to
0.3122 and recall to 0.4053, but the common evaluator produced 856 predictions
and mAP50:95 fell to 0.1854. The change was reverted; the 0.50-IoU matcher is
retained for the promoted profile.

The complete three-scale native graph was then exercised through the same C
adaptation path. Its FP32 reference forward initially failed because the
adapter reserved the compact single-scale arena; the required arena was
measured and set to 4 MiB for the three-scale path, with a 3,458,900-byte
high-water peak. The graph now passes standalone S8 and FP32 forward, C
checkpoint round-trip, and real 30-second adaptation (108 samples in 28.05 s).
However, its current exported score/box profile reached only AP50=0.1247 and
mAP50:95=0.0703 on the full 250-image evaluator, with no predictions at the
0.05 operating threshold after FP32 adaptation. The alternate quantized
calibration reached mAP50:95=0.0075 and produced 9,246 predictions. The
three-scale runtime is therefore implemented and measurable but is not an
accuracy promotion; the single-scale native Adapt profile remains promoted
while export calibration and trainable graph-weight adaptation remain open.

A dense final class-head weight update was then implemented in C as a direct
real-data experiment. The native forward retained the final class-head input
tiles, and the trainer applied bounded positive and hard-negative gradients to
the mutable INT8 head weights with fractional accumulators. The path completed
a real 30-second run in 27.93 seconds over 76 images, but its full 250-image
evaluation reached only mAP50:95=0.1601, below the promoted 0.1927 profile.
The experiment was rejected and its feature retention and weight-update code
was removed; no unvalidated graph-weight training remains in the promoted C
path.

A quality-aware class target was then tested in the same adapter: matched
detections retained a target of 0.5--1.0 based on IoU instead of a binary one.
The three real 30-second runs completed in 27.77--27.91 seconds and the full
evaluator reported mAP50:95=0.1958 / 0.1909 / 0.1921 for seeds 7 / 2 / 3,
with median 0.1921 versus the promoted 0.1927 median. The single-seed gain was
not repeatable, so the binary target was restored and further score-target
ablations were stopped.

An OpenMP convolution experiment was rejected after parity testing: repeated
single-image native inference was nondeterministic, so its apparent 93 ms/image
speedup and higher sample count were invalid. The OpenMP changes were removed;
the scalar kernel remains the verified deployment path until a race-free
parallel design is available.

After the reversion, the retained seed-7 class-and-scale checkpoint was
re-evaluated through the rebuilt scalar C path on all 250 validation images:
AP50=0.3081 and mAP50:95=0.1927, with 170 true positives and 572 false
positives. This is the current verified native Adapt baseline.

A class-conditioned geometry quality readout was then implemented as a
zero-initialized, persisted C head over candidate area, position, width and
height. It completed a real 30-second run in 27.93 seconds over 64 images, but
the full validation result fell to AP50=0.3027 and mAP50:95=0.1885. The
readout and its temporary checkpoint-version extension were removed; ranking
adaptation remains closed until a measured representation improvement is
available.

The base graph also gained a genuine FP32 forward and post-processing path
using the loaded full-precision sidecars, while the compact integer path
remains available for deployment. The route compiled, passed the existing
native tests, and completed a real 30-second seed-7 adaptation over 72 images
in 27.94 seconds. Full validation reached AP50=0.3081 and mAP50:95=0.1921,
slightly below the retained scalar integer baseline of 0.1927, so FP32 is
retained as a precision-correct runtime option but is not promoted as an
accuracy improvement. The native C implementation still requires a stronger
representation or trainable graph path to close the nano-like accuracy gate.
An eval-only run of the retained seed-7 checkpoint through this FP32 route
reproduced the integer baseline exactly at AP50=0.3081 and mAP50:95=0.1927,
so the small probe gap is attributable to training trajectory rather than
forward or post-processing parity.

A bounded FP32 final class-head weight experiment then used graph class-feature
tiles with center-cell positives and hard negatives. It completed 66 real
training samples in 27.75 seconds, but full validation fell to AP50=0.3081
and mAP50:95=0.1916. The feature retention and weight-update path was
removed; the retained profile still updates only the validated native
adaptation parameters.

A direct FP32 DFL regression-bias experiment then used the same native forward
pass and ground-truth center-cell geometry. It completed 64 real samples in
27.89 seconds, but full validation reached AP50=0.3080 and mAP50:95=0.1917,
below the retained 0.1927 baseline. The regression-bias path was removed; the
native adaptation profile remains unchanged.

A 128x128 INT8 native adaptation was measured because the stated budget does
not impose a pixel-resolution floor. It processed 95 real samples in 27.66
seconds, but full validation fell to AP50=0.2715 and mAP50:95=0.1532 versus
the 160x160 baseline. The lower-resolution profile was rejected; 160x160
remains the retained operating point.

Grid-level class supervision was then tested at the retained 160x160
resolution. The C trainer reused the native raw grid, assigned ground-truth
center cells to output scales, and updated existing class biases from hard
positives and negatives without another forward pass. It processed 62 real
samples in 27.69 seconds, but full validation reached AP50=0.3080 and
mAP50:95=0.1917, below the retained 0.1927 baseline. The raw-grid
instrumentation was removed and the deterministic validated adapter remains.

The proposed trainable KSHIRA combination was also measured directly: 32
features, space-preserving stem, shared multi-scale/context fusion, dense
auxiliary budget 8, quality alignment, adaptive/residual budgeting, 4-second
crop bootstrap, and analytic readout. With a 16 MiB arena it completed 179
real samples in 30.06 seconds, but full validation produced 7,532 predictions
with 7,512 false positives and mAP50:95=0.0000. The plan switches are real
and serializable, but this current scratch representation is not an accuracy
profile; the native Adapt baseline remains the strongest measured path.

Native class-aware NMS was measured on the retained checkpoint at IoU
thresholds 0.35, 0.45 and 0.55. Full validation mAP50:95 was 0.1923, 0.1927
and 0.1918 respectively; 0.45 remains the validated default.

The native C adaptation path now also has an opt-in target-domain crop phase.
It reuses the deterministic tight/context crop generator and updates only the
owned score, class/scale, and localization adapters while retaining the full
native graph. A real 30-second INT8 run with a 4-second crop budget processed
12 crop samples plus 65 full images in 27.85 seconds. On the 126-image test
split, the matched no-bootstrap control reached AP50=0.3368 and
mAP50:95=0.2242; the crop run reached AP50=0.3361 and mAP50:95=0.2254.
The seed-2 repeat reached mAP50:95=0.2246 versus 0.2251 for its no-bootstrap
control. The signal is not repeatable, so crop adaptation remains opt-in and
is rejected for promotion; the default path is unchanged.

The native int8 convolution audit found a stale correctness defect in the
legacy reference kernel: vertical padding was guarded, but horizontal padding
could index outside the input tensor. The corrected generic path and new
dense/depthwise 3x3 fast paths now agree exactly on real graph detections; a
focused C regression test covers the padded case. The fast path measured 3.12
images/s versus 2.16 images/s for the corrected reference in a 5-second real
adaptation probe. A fresh corrected-padding 30-second run processed 92 real
images in 27.89 seconds and reached AP50=0.3264 and mAP50:95=0.2070 on the
126-image test split. Previous checkpoints trained against the stale behavior
are not comparable evidence.

The native memory report was also corrected to expose the native graph arena
capacity instead of the unrelated KSHIRA arena. The 160x160 native profile
now reports a 2 MiB arena and passes its capacity check. A 192x192 real run
processed 63 images and reached mAP50:95=0.1948, so 160x160 remains preferred.
An explicit learning rate of 0.05 reached 0.2052 on the same corrected split,
below the 0.2070 result at 0.02, so the default learning rate remains 0.02.

The activation-scale sweep then found a stronger native starting profile. On a
fixed 20-image real screen, the 0.0125 export reached mAP50:95=0.1379 versus
0.0872 for 0.025; 0.00625 was not retained after its full adaptation fell to
0.2117. Two real 30-second 0.0125 adaptations processed 91 and 94 images in
27.64 and 27.86 seconds. On the authoritative 250-image validation manifest,
the saved seed-7 and seed-2 checkpoints reached mAP50:95=0.2259 and 0.2256,
with identical AP50=0.3617 and aggregate 208 TP / 606 FP / 246 FN. The same
seed-7 checkpoint through the FP32 sidecar path reproduced AP50=0.3617 and
mAP50:95=0.2259 exactly; native inference measured about 293 ms/image. This
0.0125 profile is now the strongest retained native Adapt candidate, but it is
still adaptation over a pretrained graph and does not close the Scratch gate.

The C loader now reconstructs missing FP32 parameter sidecars directly from the
integer graph package, so the product does not require a second runtime or a
framework-owned conversion step. A reload-specific arena sizing defect was
fixed: checkpoints that contain reconstructed FP32 parameters now expand the
native activation arena before their first forward. Release CTest and the
native lifecycle round-trip pass after this change.

The resulting base FP32 path was measured on the authoritative 250-image
validation manifest with the same 30-second real-data budget. Seed 7 processed
85 images in 28.00 seconds and reached AP50=0.3813 / mAP50:95=0.2583; seed 2
processed 91 images in 27.93 seconds and reached AP50=0.3813 /
mAP50:95=0.2557. This is the strongest current C-side result, but it remains
an adaptation over a loaded graph rather than full scratch training.

An opt-in feature-map class readout was also measured. Replacing the final
class projections collapsed validation precision; a conservative residual
blend improved that ablation but still reached only AP50=0.3303 /
mAP50:95=0.2103 on the 5-second screen. It remains experimental and is not
part of the default path. The next accuracy gate is still genuine trainable
graph weights or a stronger scratch representation under the same budget.

45. Candidate-level FP32 class-head adaptation (2026-08-07)

The next trainable-weight gate is now implemented in C as an opt-in native
adaptation path. It retains the existing native feature graph, but updates the
final per-scale class projection at the feature-map cell that emitted each
training candidate. Positive and unmatched candidates provide weighted binary
gradients; the step is normalized and deliberately small. The default API path
is unchanged, and the updated FP32 sidecars are already covered by the native
checkpoint format.

On the locked 250-image validation manifest, a real 30-second seed-7 run
processed 100 images in 27.96 s and reached AP50=0.3810 / mAP50:95=0.2590,
versus 0.2583 for the plain seed-7 FP32 control. The seed-2 repeat processed
114 images in 27.72 s and reached AP50=0.3822 / mAP50:95=0.2611, versus
0.2557 for its plain control. This is a small but repeatable improvement in
the two measured seeds and is retained as the strongest opt-in FP32 native
adaptation candidate. It is not yet the Scratch gate or full graph training;
INT8 parity and scratch training remain open.

The same candidate update is now available for INT8 deployment graphs. It uses
fractional per-weight accumulators and updates the owned signed-byte weights
only when a complete quantized step is available. Two real 30-second INT8
runs reached AP50=0.3617 and mAP50:95=0.2286 / 0.2287 on seeds 7 and 2,
respectively, versus the prior INT8 control near 0.2259 / 0.2256. The path is
retained as the strongest quantized adaptation candidate; it remains an
adaptation profile rather than scratch or full graph training.

46. Stop point and human-review package (2026-08-07)

Architecture experimentation is paused here for human review. The strongest
measured FP32 candidate is the opt-in candidate-level class-head adaptation:
AP50=0.3810 / mAP50:95=0.2590 at seed 7 and AP50=0.3822 /
mAP50:95=0.2611 at seed 2, with 27.72--27.96 seconds of real 30-second
training. The strongest measured INT8 candidate reaches AP50=0.3617 /
mAP50:95=0.2286--0.2287, with 27.79--27.86 seconds of training. These are
adaptation profiles over a loaded native graph, not scratch-trained models.

The INT8 checkpoint `runs/native_candidate_int8_roundtrip.bin` was reopened
through the C loader and rescored on all 250 validation images. It reproduced
the save-time result exactly: 814 predictions, 206 TP, 608 FP, 248 FN,
AP50=0.3589, and mAP50:95=0.2157. This is a serialization/round-trip check,
not a replacement for the strongest 30-second quality run.

The scratch path remains explicitly rejected for accuracy: its 30-second
real-data run processed 11,212 samples and measured about 1.18 ms inference,
but reached AP50=0.0014 / mAP50:95=0.0002 with 3,094 false positives. The
fast inference result is retained as evidence, not promoted as a detector.
Other failed representation, readout, regression, crop, resolution, and
threshold experiments remain documented above rather than being silently
deleted.

For visual inspection, `runs/review/fp32_detections.txt` and
`runs/review/int8_detections.txt` contain C-generated predictions for the
first 12 validation images. The corresponding PNGs in
`runs/review/fp32/` and `runs/review/int8/` show green ground truth and red
predictions. The repository was audited but no ambiguous source, dataset,
graph package, experiment, or result artifact was deleted; the worktree
contains active research changes and provenance is not yet sufficient to
classify the broad generated directories as unused.

47. Prediction-only visual review and available dataset sweep (2026-08-07)

The visualization helper now accepts `--pred-only` and omits ground-truth
overlays. The new review package is under
`runs/review/predictions_only/`; it contains six images per dataset for both
the retained FP32 reference checkpoint and the INT8 round-trip checkpoint.
The raw C reports remain beside the PNGs, while the PNGs show only model
predictions and confidence labels.

The workspace contains no cat or animal dataset. The available real-data
variants are all car-domain data: `cars`, `car_od`, `cars_1c_expanded`,
`cars_carclass`, `cars_expanded`, `cars_merged`, `cars_plus_bg`, and
`cars_plus_od`. All eight variants were rendered successfully in both
precisions. This is a visual domain-variation check, not evidence of
cross-category generalization; animal-category testing remains pending until
such a dataset is supplied or added.

48. Kaggle animal zero-shot visual review (2026-08-07)

The Kaggle CLI was available as `python -m kaggle`. Three small annotated
datasets were downloaded into `datasets/kaggle/` and converted into the C
manifest/PGM format under `datasets/prepared/kaggle_animals/`: 520
monkey/cat/dog images, 1,100 cat/dog images, and 27 pig images. The conversion
utility clamps annotation coordinates to image boundaries after the pig set
exposed one negative coordinate that the C manifest reader correctly rejected.

The retained FP32 and INT8 checkpoints were run without retraining on 12
images from each animal set. Prediction-only PNGs are under
`runs/review/predictions_only/kaggle_animals/`. The C visualizer uses generic
`class_N` labels for this review because the checkpoints were trained on five
vehicle output classes. These are zero-shot qualitative outputs, not animal
accuracy measurements; an animal-trained checkpoint is required before AP or
class-level generalization claims are valid.

49. Fresh per-dataset animal retraining (2026-08-07)

The animal test was rerun as fresh training for each dataset, using the
scratch C path with the dataset's own class list. This is the relevant test of
the short-training objective; the vehicle checkpoint was not reused as an
animal detector. A manifest-boundary issue discovered during the first runs
was fixed in `src/det_io.c`: resized annotation coordinates are now clamped to
the target image bounds. Before that fix, exact image-edge boxes could make
training fail validation. A separate attempt to adapt the loaded five-class
native graph to the animal domains became numerically invalid after only a
small number of samples, so it was retained as a documented failure and not
used for the results below.

All six runs used a real 30-second budget and saved native checkpoints. The
FP32 runs completed as follows: monkey/cat/dog processed 9,020 samples and
ended at loss 0.909726; cats/dogs processed 12,432 samples and ended at loss
0.797563; pigs processed 7,498 samples and ended at loss 0.679290. The INT8
runs processed 3,621, 5,009, and 3,766 samples, ending at losses 2.754136,
13.107421, and 1.665872 respectively. These are training-throughput and loss
measurements, not quality claims.

The full prediction-only reports were scored with the class-aware report
utility in `tools/eval_viz_report.py` because the existing full evaluator
failed late on the large raw prediction stream. Results are:

| Dataset | TP / FP | F1 | AP50 | mAP50:95 |
| --- | --- | ---: | ---: | ---: |
| monkey/cat/dog FP32 | 55 TP / 4,308 FP | 0.0190 | 0.0006 | 0.0001 |
| monkey/cat/dog INT8 | 135 TP / 601 FP | 0.1251 | 0.0282 | 0.0081 |
| cats/dogs FP32 | 258 TP / 7,550 FP | 0.0575 | 0.0077 | 0.0012 |
| cats/dogs INT8 | 0 TP / 14 FP | 0.0000 | 0.0000 | 0.0000 |
| pigs FP32 | 9 TP / 258 FP | 0.0602 | 0.0113 | 0.0023 |
| pigs INT8 | 0 TP / 0 FP | 0.0000 | 0.0000 | 0.0000 |

The semantic prediction-only images are under
`runs/review/predictions_only/kaggle_animals_retrained/`. Some individual
examples receive a plausible label, including cat and pig, but the dataset
metrics show that the models do not yet detect these categories reliably.
The current conclusion is therefore: the fast retraining pipeline works on
new class vocabularies, but the scratch model still needs a substantial
quality improvement before it can be described as a general-purpose detector.

50. Generalized scratch-training audit and retained path (2026-08-07)

The scratch path was audited using fresh, class-aware 80/20 splits rather than
training and evaluating on the same manifest. `tools/split_manifest.py` makes
these splits deterministically. The resulting train/validation counts were
416/104 for monkey/cat/dog, 880/220 for cats/dogs, and 22/5 for pigs. The pig
validation set is too small for a stable generalization estimate, so its
numbers must be treated as a smoke test.

The audit found four concrete training defects or inefficiencies. Manifest
training replayed class-grouped files in the same order each epoch unless the
entire pixel dataset was cached; residual hard-negative updates repeatedly
recomputed the same feature map; the reported IoU loss did not contribute a
box gradient; and the default dense auxiliary schedule spent too much of the
short budget on low-value updates. The retained corrections are lightweight
manifest-offset shuffling without a pixel cache, cached head-only residual
updates followed by one full encoder update, a bounded 0.25-weight
finite-difference IoU gradient, a dense auxiliary budget of two, and a larger
2/4/6 residual-negative schedule. All of these changes remain in the C
training path.

Short controlled runs showed why the changes were retained. On the grouped
cats/dogs manifest, shuffling improved AP50 from 0.0045 to 0.0210. Adding the
bounded IoU gradient raised the same short shuffled gate to AP50=0.0406 and
mAP50:95=0.0082. The gradient was not uniformly beneficial: the comparable
monkey/cat/dog gate moved from AP50=0.0108 to 0.0066. Feature width was also
non-monotonic: on the three-class holdout, 16 features reached AP50=0.0142 in
a short gate, while 8 and 24 features reached 0.0005 and 0.0051. This is why
the final three-class check uses 16 features while the smaller tasks use 8.

A positive-class softmax ranking loss was tested and fully rejected. On a
fixed-size holdout gate it reduced monkey/cat/dog AP50 from 0.0142 to 0.0005
and did not improve cats/dogs. Its code was removed, including the orphaned
helper, so this failed direction is documented but not carried in the product.

The retained path was then run for a real 30-second CPU budget on each
training split and evaluated only on the corresponding holdout:

| Dataset | Train samples | Inference | TP / FP / FN | AP50 | mAP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| monkey/cat/dog, F16 | 7,445 | 0.850 ms | 16 / 868 / 279 | 0.0030 | 0.0004 |
| cats/dogs, F8 | 9,599 | 0.468 ms | 45 / 1,006 / 200 | 0.0102 | 0.0018 |
| pigs, F8 | 11,318 | 0.512 ms | 1 / 52 / 7 | 0.0032 | 0.0006 |

These results verify the CPU budget, native checkpoint path, small memory
footprint, and sub-millisecond inference on this machine. They do not meet the
accuracy objective. The remaining dominant failure is poor ranking and
localization on unseen samples, with many false positives and low recall. The
current code is therefore a cleaner and faster experimental baseline, not a
completed general-purpose detector.
