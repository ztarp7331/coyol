# Anonymous review: Fast-Convergence Path (accepted)

**Role:** latest external diagnosis and architecture direction.  
**Project status docs:** `docs/HANDOFF.md`, `docs/KSHIRA_DESIGN.md`, `docs/RESEARCH_CRITIQUE.md`, `docs/README.md`.  
**Closed dual-run baseline (do not supersede until re-measured):** F1 **0.0430**, TP **26**, arena ~**159 KiB**.  
**Implementation status:** Phase 1 quality-class head is in progress; do not treat WIP as a new accuracy claim.

---

## Agreed—the goal should be reframed

KSHIRA should aim to become:

> **A fast-converging, dataset-neutral object detector written in ISO C that approaches nano-YOLO accuracy while requiring substantially less CPU training time, memory, and runtime infrastructure.**

The project should **not** optimize only for a 256 KiB arena. It must simultaneously improve:

1. Detection accuracy.
2. CPU **time-to-accuracy**.
3. Peak training memory.
4. Portability through custom C kernels.
5. Quantized deployment and adaptation.

“Lowest training time” must mean **wall-clock CPU time required to reach a specified AP/F1**, not merely seconds per epoch. A model that trains quickly but produces poor detections is not competitive.

## Where KSHIRA currently stands

The architectural foundation is useful: ISO C, caller-owned memory, random-weight training, quantization, local tile updates and CPU/FPGA-oriented deployment are already explicit project goals. 

But the closed baseline is still far from YOLO-like accuracy:

- F1: **0.0430**
- Precision: **0.0344**
- Recall: **0.0573**
- TP/FP/FN: **26 / 730 / 428**
- Matched-box IoU: **0.618**
- Training: approximately **35 seconds for eight epochs**
- Arena high-water: approximately **159 KiB** 

The encouraging point is that localization is not completely broken. The dominant issue is ranking: almost no true detections reach confidence 0.20, while hundreds of background predictions survive at lower thresholds. 

So the next version should not begin by increasing channels or adding a transformer. It should improve **how supervision and CPU computation are allocated**.

# Recommended architecture: KSHIRA Fast-Convergence Path

## 1. Complete the quality-class head first

Replace:

```text
box + objectness + softmax class
score = objectness × class_probability
```

with:

```text
box + K independent class-quality values
score = max(sigmoid(class_quality[k]))
target[class] = IoU(predicted_box, ground_truth)
target[other classes] = 0
```

This is already partially planned in the project and reduces the five-class head from ten outputs to nine. 

This must be finished before experimenting with more capacity because it directly addresses the current false-positive and compressed-score problem.

Required additions:

- Varifocal-style quality loss for every class.
- Background target of zero for every class.
- Quality supervision for the centre and one or two good neighbour cells.
- Initial quality bias sweep around −2.5, −3.5 and −4.5.
- Separate positive and negative loss normalization.
- Per-epoch positive/negative score histograms.

## 2. Introduce training-only dense supervision

Recent fast-converging detectors increasingly use **more supervision during training than during inference**.

YOLOv10 uses consistent dual assignments to combine one-to-many training signals with an end-to-end prediction path.  RT-DETRv3 adds training-only dense positive supervision without increasing inference cost.  DEIM reports faster convergence by increasing useful matches and controlling their quality. 

For KSHIRA, implement a much smaller C-friendly version:

### Deploy path

One quality-class prediction per cell, with fixed top-K output and suppression.

### Training-only auxiliary path

For each ground-truth object:

- One mandatory centre positive.
- One best-IoU neighbour.
- Optionally one scale-appropriate neighbour.
- Three to five head-only auxiliary positives.
- One primary positive receives full encoder-tile backward.

The auxiliary path should reuse the same features and preferably share most head weights. It disappears entirely during inference.

This should improve convergence without permanently increasing inference memory or computation.

## 3. Make the central novelty **budgeted dense-to-sparse training**

This is the strongest potential research contribution.

### Forward pass

Evaluate the inexpensive quality head densely over the 40×40 map.

### Backward pass

Do not backpropagate through every location. Allocate a fixed gradient budget per image:

```text
1 centre positive:       full 13×13 encoder tile
1 secondary positive:    full or partial encoder tile
8–16 hard negatives:     head-only updates
1–3 hardest negatives:   full encoder-tile updates
remaining cells:         no backward work
```

Add spatial suppression so all hard negatives do not come from the same textured area.

Then make the budget adaptive:

```text
If positive and negative score distributions overlap:
    spend more encoder updates on hard negatives
If ranking becomes clean:
    reduce negative tile updates
If localization stalls:
    allocate a second positive encoder tile
```

SparseProp demonstrates that genuinely sparse forward/backward computation can accelerate neural-network training on commodity CPUs, including training sparse models from scratch.  Tiny Training Engine similarly shows that sparse updates and compile-time graph pruning can make training practical within 256 KiB, although its demonstrated setting is primarily on-device transfer learning rather than full object detection from random initialization. 

### Potential novelty claim

A defensible paper direction would be:

> **Adaptive gradient-budgeted dense detection: dense candidate evaluation combined with deterministic, score-overlap-controlled tile-local backward propagation in an arena-bounded ISO-C training runtime.**

The individual ingredients are not entirely new. The possible novelty is their combination into a detector where:

- backward computation is explicitly budgeted;
- the budget changes according to measured ranking difficulty;
- only selected spatial dependency tiles update the encoder;
- training memory is statically bounded;
- the full system trains without Python or automatic differentiation.

A wider literature and patent search would still be required before claiming originality formally.

## 4. Preserve more visual information without widening the network

The present stride-4 stem immediately converts 160×160 into 40×40. That is extremely cheap but may destroy information before the learned encoder can use it.

Test a two-stage downsampling stem:

```text
160×160
→ inexpensive 3×3 stride-2 spatial stem
→ depthwise 3×3 stride-2 + pointwise mixer
→ 40×40
```

This retains an intermediate 80×80 representation but does not require keeping the complete map after downsampling. Execute and release it sequentially through the arena.

Recent ultra-lightweight detector work also emphasizes information-preserving downsampling and efficient feature aggregation rather than simply reducing channel counts. 

Keep the identity-initialized pointwise mixer. Do not widen to 12 channels yet—the project’s own f12 experiments reduced true positives. 

## 5. Use two shared scales, not three independent heads

For general YOLO-like behaviour, one 40×40 map will eventually limit performance across object sizes. But the existing independent P4/P5 co-supervision increased false positives and reduced F1.

A lighter solution is:

- P3: 40×40.
- P4: 20×20 created by pooled or depthwise-downsampled P3 features.
- Shared quality-class and box weights across both scales.
- One small learned scale bias or affine value per level.
- Initially permit full encoder backward only through P3.
- Train P4 head-only until ranking stabilizes.

This provides scale coverage without maintaining three separate detection heads.

## 6. Optimize specifically for CPU training

The custom C runtime should use the architecture’s fixed dimensions as an advantage:

- Static tensor shapes and precomputed offsets.
- Direct convolution for low-channel 3×3 and depthwise kernels instead of general `im2col`.
- Blocked channel layouts aligned to SIMD width.
- Fused depthwise, bias and activation loops where mathematically possible.
- Persistent worker threads rather than creating threads inside each operation.
- Thread parallelism across images or independent tiles.
- Weight packing once per training segment.
- No tensor-layout conversion between forward and backward.
- Dedicated AVX2, AVX-512 and NEON kernels behind the same scalar reference API.

Blocked memory formats and graph-level fusion are important sources of CPU performance in optimized DNN systems.  KSHIRA should reproduce the relevant principles in its own small static runtime rather than attempting to reproduce a general-purpose framework.

## 7. Progressive CPU training schedule

Use a training curriculum designed around the fixed time budget:

### Stage 1: rapid ranking warm-up

- 96×96 or 128×128 input.
- Train quality-class head, box head and mixer.
- Centre positives and head-only negatives.
- Encoder receives very limited updates.

### Stage 2: representation learning

- Move to 160×160.
- Enable primary positive tiles and the hardest negative tiles.
- Add the second training scale.
- Begin bounded pairwise ranking loss.

### Stage 3: precision transition

- Freeze stable early weights periodically.
- Move activations and most gradients to INT8.
- Retain wider accumulators for weight-gradient reductions.
- Finish with a short F32 calibration stage.

Low-precision training research shows that weights, activations and substantial portions of gradient computation can be reduced to 8-bit when scale handling and selected high-precision accumulations are preserved. 

Use a training-budget-aware linear or one-cycle-style decay rather than the current aggressive inverse-time schedule. Budgeted-training research found that schedules should be explicitly matched to the allowed number of iterations. 

# Implementation priority

### Phase 1 — establish accuracy

1. Finish quality-class training and persistence.
2. Add score-distribution instrumentation.
3. Add neighbour class-quality supervision.
4. Implement spatially diverse two-level hard-negative mining.
5. Add bounded ranking loss.
6. Expand unique images and include background-only hard negatives.

### Phase 2 — establish fast convergence

7. Add the training-only dense auxiliary assignment.
8. Implement fixed gradient budgets.
9. Add adaptive positive-versus-negative tile allocation.
10. Add progressive resolution and progressive encoder unlocking.
11. Measure CPU **time-to-F1** and **time-to-AP50**, not only epoch time.

### Phase 3 — improve architecture

12. Test information-preserving two-stage downsampling.
13. Add one shared P4 scale.
14. Ablate the contrast feature completely.
15. Add a normalized learnable contrast gate only when contrast proves useful.

### Phase 4 — optimize the C engine

16. SIMD-blocked kernels.
17. Fused operations.
18. Persistent thread pool.
19. INT8 training transition.
20. Architecture-specific kernel generation.

# Required benchmark definition

Before claiming “YOLO-like accuracy” or “lowest CPU training time,” compare KSHIRA and a nano-YOLO baseline under identical conditions:

- Same train/validation split.
- Same input resolution.
- Same augmentation allowance.
- Both initialized from random weights.
- Same CPU, thread count and time limit.
- AP50, mAP50–95, precision, recall and F1.
- Wall-clock time to reach fixed accuracy thresholds.
- Peak memory and model size.

A meaningful intermediate target would be:

- Pass the existing internal gate of F1 ≥ 0.07, FP/TP < 15 and TP at threshold 0.20 ≥ 8. 
- Then reach within approximately 10% relative AP50 of the selected nano-YOLO baseline.
- Train to that accuracy at least 2–3× faster on the same CPU.
- Preserve the sub-256 KiB compact training profile.

The strongest route is therefore **not simply “a smaller YOLO in C.”** It is a detector built around **quality-aligned supervision plus adaptive, tile-local gradient budgeting**, with the entire architecture and runtime designed around fastest possible CPU convergence.