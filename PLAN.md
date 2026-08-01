# C-First Edge Detector Family Plan

## 1. Goal and success contract

Build a dataset-neutral object and pattern detection framework in C17 that can
replace the detection workflow of YOLO:

- define a detector architecture;
- train it from zero learned weights;
- validate it;
- run inference;
- save and load models;
- export INT8 and INT4 deployment profiles;
- later lower the same model representation to an FPGA backend.

The first product is one fully learned edge detector, not a fast fixed-feature
model presented as a substitute for it. The same architecture has two training
engines:

- `LOCAL_FAST`: single-pass block-local learning designed for the sub-second
  research goal;
- `GLOBAL_BP`: conventional end-to-end backpropagation used as the correctness
  and accuracy reference.

The primary reference workload is dataset-neutral:

- 5,000 labeled images;
- 160 x 160 input;
- 1 to 4 input channels;
- up to 80 classes;
- configurable boxes per image;
- one streaming pass from newly initialized learned weights;
- axis-aligned object detection;
- a whole-image pattern can be represented as a full-frame box, so a separate
  classifier is not required in v1.

### KSHIRA integration direction

KSHIRA is an architecture profile inside the detector family, not a second
user-facing framework. The validated CDET graph remains the compatibility
profile while KSHIRA operators stay in focused source modules behind the same
`det_model`, `det_dataset`, `det_train`, and `det_predict` lifecycle. The
integration contract is:

- `src/det_*.inc` contains the existing detector in ordered core, quantization,
  model, forward, training, evaluation, and serialization units; `src/det.c`
  is only the implementation-unit aggregator.
- `include/kshira/` and `src/kshira_*.c` provide the new M0--M2 contracts:
  caller-owned arena/high-water accounting, symmetric INT4/INT8 packing and
  toggle, QAS gradient scaling, sparse channel masks/memory estimates, and
  PRE/TRAIN/ODT phase validation.
- The current multi-scale detector remains the measured baseline. KSHIRA's
  SMRE single-map RAD encoder and bounded top-K head are the next model branch,
  so their small-object and quantized-training hypotheses are measured against
  the baseline instead of silently changing the published graph.
- The first RAD edge harness now runs a 160 x 160, 8-feature, 16-candidate
  single-map model inside a 256 KiB caller-owned arena. Its current Release
  checkpoint uses 258,100 bytes high-water (1,892 parameter bytes plus 256,000
  activation bytes) and measured about 0.47 ms per image on the pinned
  development CPU; this is a software/memory checkpoint, not a 1 W claim.
- Real INT4/INT8 training, QAS inside the detector update path, sparse ODT, and
  the hard 256 KiB arena gate remain implementation milestones for the full
  detector. The RAD branch now has real INT8/INT4 forward arithmetic and a
  QAS-scaled sparse head update; the validated multi-scale detector's
  quantized modes remain inference-only until its own update path is wired.
- KSHIRA phase drivers now enforce the ordered PRE -> TRAIN -> ODT transition,
  and sparse plans reject conservative peak-memory estimates above the supplied
  arena cap before a layer schedule is admitted.
- `kshira_session` now composes those contracts with RAD: it owns no heap state,
  gates each train step by the active phase, carries a channel mask, and exposes
  one prediction path. A 16 KiB session fixture covers FP32 PRE, INT8 TRAIN,
  and INT4 ODT transitions under the same arena.
- `KSHIRA_UPDATE_FULL` now propagates the single-target supervised signal through
  RAD's projection, depthwise dilated branches, and stem with straight-through
  quantized updates. Encoder gradients are channel-maskable and preflighted for
  finite aggregate updates before commit; the session fixture exercises this
  path with nonzero input data.
- `kshira_domain` adds a balanced ten-domain curriculum stream with deterministic
  boxes and caller-owned image storage. The 160 x 160, 5,000-sample Release
  harness (500 samples/domain) trains the FP32 local receptive-field path in
  315--359 ms across the latest three Release WSL runs with all encoder channels
  enabled, and 258,365-byte high-water under the 256 KiB arena. Host load affects
  the timing; this is a synthetic throughput gate, not an accuracy, board-power,
  or real-dataset claim.
- `kshira_eval` adds allocation-free IoU and class-hit accumulation for
  calibration sets. The M9 harness runs the same 5,000-sample stream in FP32,
  INT8, and INT4, calibrates quantized models on one sample from each domain,
  then evaluates held-out samples at a 0.25 threshold. The latest Release run
  measured FP32/INT8/INT4 proxy IoU of 0.311/0.475/0.435 and quantized loss
  ratios of 0.426/0.509; the plain ASAN harness completes as well. Quantized
  full-encoder training now measures 0.92--1.08 s for 5,000 samples after
  per-step scale reuse, with 5.1--7.0 ms calibration and 1.06--1.07 s
  held-out evaluation. M11 adds a transactional encoder delta cache so the
  full-encoder path preflights all projection, branch, and stem updates before
  committing them. Three consecutive Release runs measured INT8 0.663--0.675 s
  and INT4 0.654--0.671 s for 5,000 samples, with `edge_train_gate=PASS` on
  every run; held-out evaluation is about 0.76--0.86 s aggregate (7.6--8.6 ms
  per image). ASAN timing remains diagnostic and slower. These are synthetic
  recovery/timing gates, not COCO accuracy, FPGA timing, or 1 W measurements.
  The harness emits per-mode `train_gate`, aggregate `edge_train_gate`, and
  per-image evaluation latency so host jitter cannot be mistaken for a
  universal guarantee. Non-noise domain backgrounds use an exact zero-fill
  fast path. M12 moves the delta cache into the caller-owned arena and sizes it
  from the configured feature/input channels. The current 160 x 160 / 8-feature
  profile uses 260,025 bytes high-water inside the 256 KiB arena, including this
  scratch, with no per-step stack allocation. Larger feature maps must still be
  admitted by the sparse/arena planner before use.
- M13 adds an inference-only pooled P3/P4/P5 view over the RAD fused map. P4
  and P5 are generated cell-by-cell from the stride-4 map and merged into the
  existing bounded top-K output without new arena buffers. Release held-out
  evaluation is now about 0.89--1.05 s aggregate (8.9--10.5 ms per image), with
  the same 260,025-byte high-water. The training step still supervises only
  P3, so this is an architectural/inference checkpoint, not a qualified
  multi-scale accuracy result.
- RAD construction now rolls back direct caller arenas on partial allocation
  or initialization failure; a focused test preserves a nonzero pre-existing
  arena offset and high-water.
- M15 adds an opt-in `multiscale_heads=1` profile. Its P4/P5 heads are trained
  only during ODT through `kshira_rad_train_multiscale_step` or the session
  wrapper; the RAD API supports FREEZE, BIAS, and channel-sparse updates, while
  the session contract selects channel-sparse ODT. FULL remains reserved for
  the validated encoder/P3 path. The bounded ODT forward
  computes every pooled source cell from one caller-owned region, and inference
  falls back to the base head until each scale has been updated. The Release
  harness processes 5,000 base and 5,000 ODT samples and reports separate base,
  ODT, combined, and inference gates. Recent host runs are about 0.78--1.03 s
  base, 0.97--1.21 s ODT, and 9.8--11.9 ms per eval image; high-water is
  261,729/262,144 bytes. The combined 10,000-step gate is still open, and the
  timing statuses are diagnostics rather than process-failure criteria so ASAN
  and host jitter remain observable. The measurements are synthetic host
  diagnostics rather than COCO, FPGA, or 1 W qualification. The default
  `multiscale_heads=0` path remains unchanged.
- M16a reduces quantized ODT inner-loop work by packing the fixed branch and
  projection weights once per sample and reusing their scales across all pooled
  cells. It does not change the arena layout or the validated P3/encoder path.
  Latest Release runs keep the 260,089-byte baseline high-water and
  261,729-byte optional-head high-water; repeated host runs put INT8 ODT near
  0.94--0.98 s and INT4 near 1.01--1.04 s. INT4 remains just outside the
  per-phase stretch gate and the combined 10,000-step gate remains open.
- M16b1 reuses one 32-channel branch quantization cache while materializing all
  pooled branch outputs before projection. This removes 576 bytes from the
  multiscale step's automatic cache footprint without changing arena usage or
  quantized results. Latest repeated Release runs put INT8 ODT near
  0.94--0.96 s and INT4 near 0.90--0.94 s; both per-phase gates passed in those
  runs. Arena headroom and the combined gate remain open for M16b2.
- M16b2 overlays the transactional encoder and optional multi-scale head delta
  descriptors on one caller-owned float workspace because those update paths
  are mutually exclusive. The optional profile high-water fell from 261,729 to
  261,181 bytes (963 bytes free in the 256 KiB harness) while the default
  profile remains covered by the same tests. The multiscale harness now reports
  P4/P5 sample counts, per-level proxy IoU/class accuracy, and per-level ODT
  loss; a current Release run measured INT8 P4/P5 IoU 0.451/0.562 and INT4
  0.427/0.497 on the synthetic held-out stream. These are scale-aware proxy
  diagnostics, not COCO accuracy, and the combined timing gate remains open.
- M17 begins the public-runtime unification. `det_model_spec.architecture`
  selects `DET_ARCH_CDET` or `DET_ARCH_KSHIRA`; the KSHIRA profile owns a
  per-model bounded arena and is driven through the normal detector build,
  reset, train, precision, prediction, evaluation, and dataset contracts.
  F32 local training, INT8 full-encoder training, and INT4 multi-scale ODT now
  share that API, and multiple KSHIRA models can coexist under one context
  without aliasing state. CDET W4A8 is kept distinct from KSHIRA W4A4/INT4
  semantics. KSHIRA currently rejects nonzero momentum, zero-box negative
  learning, and GLOBAL_BP instead of silently approximating those contracts.
- M18 completes the first shared-runtime checkpoint path. KSHIRA models use a
  pointer-free version-9 payload with CRC32 and exact-length validation; it
  preserves RAD parameters, P4/P5 heads, calibration, sparse-channel mask,
  precision, phase state, and update counters. Save/load produces identical
  detections, deterministic continued training produces identical checkpoints,
  and CRC-invalid or CRC-valid truncated files fail closed. The version-8 CDET
  loader remains backward compatible. `det_bench --architecture kshira` now
  drives the same synthetic or raw-manifest workflow in F32, INT8, and INT4.
  On the current WSL CPU, two Release checks over 5,000 in-memory 160 x 160
  samples measured 240.244--337.097 ms F32, 639.539--802.958 ms INT8, and
  610.714--777.428 ms INT4 through model serialization; average loaded-model
  inference measured 0.671--0.840, 6.869--8.726, and 7.994--9.916 ms
  respectively. These pass the synthetic edge timing gate over 5,000 generated
  moving-box samples, but they do not decode 5,000 image files and are not a
  raw-dataset, accuracy, COCO, FPGA, or power claim.

Every KSHIRA optimization must carry memory high-water, bit-mode,
proxy-detection, and latency measurements.

Timing targets:

- stretch target: at most 1 second;
- secondary target: at most 10 seconds;
- inference target: less than 33 ms per image on the current CPU;
- all failures are reported numerically and are not relabeled as success.

The timing gate is measured with a Release build, one pinned host CPU, and a
single worker unless the report explicitly says otherwise. Debug timings are
diagnostic only. The first executable baseline (a learned 1 x 1 stem plus
multi-scale heads, before the planned 3 x 3 backbone and split-stream blocks
are complete) measured 2,019 ms in Debug and 464--629 ms in Release across
the first repeated runs for 5,000 synthetic 160 x 160 images on the
development CPU. The same runs measured 0.142--0.325 ms single-image
inference and 6.8--21.0 ms model save/load. These are performance baselines,
not claims that the final detector has already passed the accuracy or
architecture gates.

After wiring the first spatial backbone, the compact edge profile initially
measured about 1.19--1.36 s `LOCAL_FAST` synthetic end-to-end training. A
specialized 3 x 3 stride-2 kernel for the two dense backbone stages now measures
0.874--0.958 s `synthetic_e2e_ms` across repeated 5,000-image Release runs,
with 0.08--0.16 ms repeated inference and 9--15 ms save/load. The optimization
keeps the generic convolution API for callers and only specializes the fixed
backbone shape. This is the first sub-second synthetic timing milestone; the
official training gate remains open until raw-input timing, pinned-target
repeatability, accuracy qualification, and the larger detector profile are
also measured.
The profile uses one learned 3 x 3 stem channel, one learned 3 x 3 expansion
stage, and three depthwise 3 x 3 pyramid stages. It is an executable edge
variant of the architecture, not yet the larger 16/24/40/64/96-channel research
configuration in the table below.
The current 5,000-image `LOCAL_FAST` smoke benchmark produces top-K detections
at the conventional 0.25 threshold after the background-control ablation; the
synthetic overfit test passes, but localization quality, confidence
calibration, and the larger functional/accuracy gate remain open. The
benchmark exposes `--threshold` for repeatable calibration measurements.
With the auxiliary one-to-many bank enabled, repeated post-fix runs on the
development CPU have measured approximately 0.923--0.988 s F32, 0.933--1.052 s
INT8, and 0.906--1.083 s W4A8 synthetic end-to-end. The auxiliary bank samples positive
3 x 3 neighborhoods plus up to four deterministic negatives per scale every
sixteenth image and does not change inference. The occasional W4A8 overrun
shows why the one-second result is still a stretch target, not a qualified
guarantee.
The top-down plus bottom-up build with the one-box local-update fast path and
damped bottom-up ablation measured 0.802 s F32, 0.780 s INT8, and 0.862 s W4A8
for 5,000 synthetic 160 x 160 images on a pinned development CPU; repeated
runs remain host-load sensitive. The same compact graph at 33 x 33 remains
about 100--118 ms.
These are timing checkpoints, not accuracy claims. All three 160 x 160 local
variants were under one second in this pinned run; repeated runs remain
host-load sensitive, so this is a qualification checkpoint rather than a
hardware-independent guarantee. W4A8 still needs an accuracy gate before it is
treated as a deployable packed format.
The fast path preserves the original row-major target/update order for the
common one-box stream and retains the original multi-box fallback. The C
convolution API now has dedicated 1-channel stride-2 and 1 x 1 kernels; generic
shape behavior remains covered by the parity tests.
The bottom-up extension is now wired as two depthwise 3 x 3 stride-2 stages;
its zero initialization preserves the validated local path while `GLOBAL_BP`
can learn the new fusion weights. LOCAL_FAST now qualifies a damped bottom-up
ablation after a cumulative 1,024-sample warm-up across the streamed training
run, updating it every sixteenth sample; this avoids destabilizing the early
compact detector while still exercising the full architecture on the
5,000-image profile.

Two training times must be published:

1. `train_core_ms`: prepared low-resolution tensors through final learned
   weights and model serialization;
2. `train_e2e_ms`: raw files, decoding, resizing, labels, training, and model
   serialization. An in-memory run reports its broader setup boundary as
   `synthetic_e2e_ms`; a manifest run reports the raw boundary as
   `train_e2e_ms`.

Both target one second. Separating them is diagnostic, not an exclusion:
`train_e2e_ms` is the user-visible raw-input result, while `train_core_ms`
identifies whether the bottleneck is learning or input I/O. A
`synthetic_e2e_ms` result is explicitly not a raw-input claim.
The raw boundary now has a small dependency-free adapter: `det_manifest_open`
streams P2/P3/P5/P6 PNM files from a manifest, performs nearest-neighbor resize,
converts 1/3-channel input to the model layout, and scales box coordinates. The
manifest format is one image path followed by optional whitespace-separated
`x1,y1,x2,y2,class` records; relative image paths resolve beside the manifest.
The adapter follows the model’s 4096-pixel dimension ceiling and rejects source
rasters above a 64 MiB decode budget.
`det_train` propagates a negative dataset callback as `DET_ERR_IO`, so a decode
failure cannot be reported as a short successful epoch. The benchmark accepts
`--manifest` and labels the full timer `train_e2e_ms`; because decoding is
streamed inside training, its companion is explicitly `train_plus_decode_ms`,
not `train_core_ms`. It remains a raw-adapter baseline, not a JPEG/COCO
implementation. Manifest runs now also emit streamed precision/recall, mean
IoU, AP50, mAP50:95, and small/medium/large ground-truth counts after the
serialized model is reloaded, keeping timing and detector quality visible
together. `--eval-manifest` can select a separate streamed validation manifest;
without it, a supplied training manifest is replayed for the smoke report.
The public `det_evaluate` path performs greedy class-aware point matching and
reports precision, recall, mean IoU, TP/FP/FN counts, class-macro 101-point
AP50/AP75, mAP50:95, and class-macro small/medium/large AP50 buckets in one
threshold-zero stream. The metric implementation is dataset-neutral;
COCO-specific adapters remain out of the core API.
The benchmark now stops both training timers immediately after checkpoint
serialization; inference warmups and repeated inference, plus save/load I/O,
are reported separately. It reserves a unique checkpoint path per process and
malformed or unknown benchmark options fail closed.

Verification status for this checkpoint: fresh Release, Debug, and ASan builds
pass CTest; 5,000-image Release LOCAL_FAST runs cover F32, INT8, and W4A8;
GLOBAL_BP remains the reference run; and boundary smoke tests cover minimum and
odd/even image sizes, including a serialized proof that bottom-up weights change
under global training. This specialization is verified; the official raw/full-
architecture training gate remains open.

The first raw-manifest smoke on a 33 x 33 P2 image completed successfully in
9.529--47.100 ms end-to-end across repeated runs (including decode, resize,
training, and checkpoint I/O); this is an adapter correctness measurement, not
a 5,000-image performance claim.

COCO is not a separate product: it is an optional 80-class scalability and
accuracy qualification for the same dataset-neutral replacement framework after
the 5,000-image gate works. Dataset adapters must never leak COCO-specific
assumptions into the model, loss, graph, or public API.

## 2. Research-driven detector

### 2.1 Architecture

Use a small convolutional detector whose dimensions are fixed for the first
experiment:

| Stage | Resolution | Channels | Blocks |
|---|---:|---:|---:|
| Stem | 80 x 80 | 16 | one 3 x 3 stride-2 convolution |
| S1 | 40 x 40 | 24 | one split-stream block |
| P3 | 20 x 20 | 40 | two split-stream blocks |
| P4 | 10 x 10 | 64 | two split-stream blocks |
| P5 | 5 x 5 | 96 | one split-stream block |

Each split-stream block:

1. creates two channel views without copying;
2. leaves one half as the short path;
3. applies two direct-tiled 3 x 3 convolutions to the other half;
4. merges channel ranges without materializing a concatenation;
5. applies a 1 x 1 projection for channel mixing;
6. uses clipped ReLU during quantization-aware training.

The default architecture excludes C2PSA, transformers, deformable convolution,
and physical PAN concatenations. These operators complicate low-clock CPU and
FPGA execution and are not needed to test the central learning hypothesis.

The neck projects P3, P4, and P5 to 48 channels and uses:

- nearest-neighbor upsampling;
- projected integer-friendly addition;
- one top-down pass;
- one bottom-up pass;
- no learned division or normalized fusion in the inference graph.

The shared detection tower produces, at every scale:

- four direct non-negative distances `(left, top, right, bottom)`;
- configurable class logits;
- per-scale integer gain and bias.

There is no DFL. The deployed one-to-one output requires thresholding, box
decoding, and fixed-size top-K selection but no NMS. The current model carries
a serialized training-only one-to-many auxiliary head per scale. It supervises
a 3 x 3 neighborhood periodically during training; inference reads only the
one-to-one bank, so the deployment path remains NMS-free.

These choices retain YOLO26's useful deployment ideas—DFL-free regression and
dual-head NMS-free detection—without copying its complete architecture:
[YOLO26 paper](https://arxiv.org/html/2606.03748v1) and
[YOLOv10 dual assignment](https://arxiv.org/abs/2405.14458).

The current C implementation has the same five-resolution topology but uses a
compact 1/4-channel edge variant while the kernels and training path are being
validated. The stage weights and both head banks are learned from scratch,
P3/P4/P5 are real spatial feature maps, and `GLOBAL_BP` uses convolutional
backward operators. The auxiliary bank is training-only and is never read by
`det_predict`.
The larger channel table remains the next scale-up experiment, not a claim
about the current binary.

The current deployment API exposes `det_model_set_precision` for FP32, INT8,
and W4A8. INT8 and W4A8 inference quantize activations per tensor, use
per-output-channel weight scales, accumulate integer products, and dequantize
only at each stage output. Training remains FP32 until quantization-aware
training is added. Quantized accuracy loss still requires the planned
validation gate.

On the same 160 x 160 Release benchmark, repeated quantized inference has
measured about 0.8--1.9 ms across INT8 and W4A8 runs. The model file now carries the
quantized buffers and scales, so a quantized load is self-contained; these are
latency measurements only. The test suite now includes a deterministic two-box,
two-class 64 x 64 streamed fixture: it requires two FP32 true positives with
finite IoU/AP, then requires INT8 and W4A8 to retain every true positive and
false negative, stay within a small false-positive/recall/precision allowance,
and remain within the 2-point and 5-point mAP-loss limits respectively. This
is a qualification smoke gate, not a COCO-scale accuracy claim. A separate
three-sample stream now exercises small, medium, and large boxes across three
classes through FP32, INT8, and W4A8 evaluation, including reset/replay and
finite per-class/per-size AP checks; larger real-data validation remains
outstanding.

### 2.2 Fast full-model learning

`LOCAL_FAST` must update the convolutional backbone, neck, and both assignment
heads once the neck is present. It must not use a frozen or pretrained detector.
The compact graph now has three learned 1 x 1 lateral projections, a top-down
P5-to-P3 nearest-neighbor fusion path, and two depthwise bottom-up stages
(P3-to-P4 and P4-to-P5). `GLOBAL_BP` propagates through both directions and
updates all neck stages; `LOCAL_FAST` applies the deterministic sparse local
rule to the three lateral stages and the damped warm-start bottom-up ablation.
This is intentionally smaller than the research table's 48-channel PAN target.

The current implementation's local stage update is explicitly a surrogate
local-learning rule: each stage receives a deterministic sparse local box
target, while the heads use the detector loss. It is not being presented as
full detector-loss backpropagation. The downstream-loss comparison against
`GLOBAL_BP` is a required acceptance experiment before selecting this rule. The
local stage rule currently uses a straight-through ReLU surrogate to recover
dead channels; this is intentional and must be included in the ablation.
The local head path also down-weights sparse background BCE updates to 0.001 of
the positive-cell signal; this is a measured imbalance control, not part of
`GLOBAL_BP`.

Training is a single forward stream with local updates:

1. Every block owns a small training-only projection that maps its output to
   local class-presence and coarse localization targets.
2. The block computes a local loss as soon as its output exists.
3. Backward computation is restricted to that block.
4. The block updates its weights immediately and releases its saved activation.
5. Deeper blocks consume the updated forward stream; there is no global
   activation tape or reverse traversal of the complete network.
6. A deterministic rotating channel mask limits each sample's gradient work,
   while guaranteeing that every trainable channel is updated during the
   5,000-image pass.
7. P3, P4, and P5 receive scale-appropriate local targets derived from the same
   ground-truth boxes.
8. The final one-to-one head receives exact class and direct-box supervision.

Three local-learning variants must be evaluated before choosing one:

- block-local SGD;
- distance/centroid local learning;
- block-local SGD with error-triggered sparse updates.

Selection order is:

1. passes numerical and stability tests;
2. beats the simple detector baseline;
3. has the lowest `train_e2e_ms`;
4. has the lowest peak memory.

Forward-only and local-learning research supports lower activation memory and
parallel local updates, but does not establish one-second detector training.
This framework treats the timing as a falsifiable experiment:
[Distance-Forward](https://arxiv.org/abs/2408.14925),
[Scalable Forward-Forward](https://arxiv.org/abs/2501.03176), and
[structured sparse backpropagation](https://openaccess.thecvf.com/content/CVPR2024W/EVW/html/Paissan_Structured_Sparse_Back-propagation_for_Lightweight_On-Device_Continual_Learning_on_Microcontroller_CVPRW_2024_paper.html).

### 2.3 Reference learning

`GLOBAL_BP` trains the identical current graph with ordinary end-to-end
backpropagation. It provides the gradient and convergence reference needed
before adding the planned neck:

- gradient and convergence truth for `LOCAL_FAST`;
- one-to-many auxiliary plus one-to-one deployment assignment;
- Progressive Loss moving from `(0.8, 0.2)` to `(0.1, 0.9)`;
- tiny-object assignment protection;
- BCE classification loss;
- IoU plus normalized L1 box loss;
- SGD with momentum as the first CDET optimizer; KSHIRA currently exposes
  direct SGD only and rejects nonzero momentum.

Training preserves weights by default so a loaded model can resume. Set the
public `reset_weights` flag for a deterministic fresh run; the benchmark and
tests set it explicitly.

MuSGD is implemented only after SGD is correct. It is retained only if it
improves validation accuracy per wall-clock second, because fewer epochs do not
automatically mean faster CPU training.

## 3. Framework architecture

### 3.1 Public C API

The public header exposes opaque runtime objects and explicit caller-owned
buffers:

- `det_context_create` / `det_context_destroy`;
- `det_model_build`;
- `det_train`;
- `det_evaluate`;
- `det_predict`;
- `det_compile`;
- `det_save` / `det_load`.

Core public types:

- `det_tensor`;
- `det_model_spec`;
- `det_dataset`;
- `det_sample`;
- `det_train_config`;
- `det_train_report`;
- `det_detection`;
- `det_precision`;
- `det_backend`.

`det_dataset` is callback-based. A dataset adapter supplies images, boxes, and
class IDs; the graph never knows whether they came from a directory, camera,
binary pack, COCO JSON, or another annotation format.

### 3.2 Static graph and memory

Use five planned arenas:

- persistent weights, biases, scales, and optimizer state;
- activations;
- operator workspace;
- gradients for the currently trainable block or global graph;
- saved state required by the selected training engine.

No allocation is allowed inside inference or training loops. Tensor liveness is
computed before execution, and non-overlapping buffers reuse arena offsets.

The small intermediate representation supports only:

- tensor views, slices, and channel splits;
- 3 x 3, 1 x 1, depthwise, and grouped convolution;
- addition and virtual concatenation;
- nearest upsampling and max pooling;
- activation and requantization;
- direct box decode, score filtering, and top-K;
- local and global losses;
- block-local and global backward operations;
- optimizer updates.

Compilation performs shape propagation, batch-normalization folding,
convolution/bias/activation fusion, precision propagation, backward-graph
pruning, liveness allocation, kernel selection, and INT4 packing.

### 3.3 CPU implementation

Implementation order:

1. readable scalar FP32 reference kernels;
2. direct-tiled scalar convolution;
3. separate 1 x 1 and depthwise kernels;
4. AVX2 INT8 and FP32 kernels;
5. packed W4A8 kernels;
6. multithreaded image decode and training;
7. double-buffered data loading;
8. cache and memory-traffic profiling.

Explicit im2col is not the default because its expanded workspace is unsuitable
for the eventual low-memory and FPGA targets. It is admitted only where direct
benchmarking proves it faster within the arena budget.

The current bring-up toolchain is WSL GCC. Report scalar single-thread, AVX2
single-thread, and AVX2 all-core results on the available i5-13500H. The future
Pentium-class result is a new hardware measurement, not an extrapolated claim.

Runtime dependencies are intentionally narrow:

- no Python, PyTorch, ONNX, OpenCV, or BLAS;
- OS threads and CPU intrinsics are allowed;
- `libjpeg-turbo` is allowed for the raw-JPEG benchmark because rewriting a
  slower JPEG decoder does not contribute to the detector research;
- the tensor engine, resize path, labels, augmentation, training, inference,
  quantization, and model format remain C framework code.

### 3.4 Quantization and model format

Validated precision profiles:

- CDET: `F32` reference training, `Q8` INT8 weights/activations, and `W4A8`
  packed signed INT4 middle-layer weights with INT8 activations;
- KSHIRA: `F32`, symmetric `INT8`, and symmetric `INT4` weight/activation
  execution with quantization-aware local updates. KSHIRA accuracy remains an
  experimental gate even though all three runtime modes are executable.

Use per-output-channel symmetric weight scales, per-tensor activation scales,
INT32 accumulation, and integer multiplier-plus-shift requantization. Precision
profiles are compiled separately rather than switched at runtime.

The current version-8 `CDET` model contains graph metadata, tensor shapes,
FP32 optimizer state, quantized buffers/scales, and a CRC32 over the payload.
It contains both the deployed one-to-one and training-only one-to-many head
banks and does not serialize pointers. On load, FP32 weights are authoritative and
quantized caches are regenerated, preventing contradictory CRC-valid cache
state. The on-disk integers/floats are still native ABI fields; a portable
endian-neutral format and atomic replacement are explicit follow-up hardening
work, not silently assumed properties.
The loader validates the fixed header and exact payload length before its CRC
buffer allocation. A separate device-memory budget check remains required
before accepting very large edge-hosted dimensions.

Integrated KSHIRA checkpoints use the same `CDET` magic with version 9 and an
architecture tag. Their deterministic pointer-free payload includes the phase
driver, channel mask, calibration state, base and optional scale heads, and all
persistent RAD weights. CRC32 and exact-length checks cover the full payload.
Like version 8, version 9 is currently native-endian/native-float and direct
file replacement is not atomic; portable encoding and atomic save remain
explicit hardening work.

## 4. Implementation phases

### Phase 0: specification and lower bounds

- Freeze all equations, layouts, rounding, saturation, assignments, losses,
  timer boundaries, and model sections in an executable specification.
- Build raw decode/resize and memory-bandwidth-only benchmarks.
- Establish how much of the one-second budget is already consumed before
  learning begins.

### Phase 1: scalar framework

- Implement graph construction, scalar forward kernels, direct box decoding,
  top-K, arenas, and model serialization.
- Implement scalar global backward kernels and finite-difference tests.
- Complete a deterministic synthetic overfit test before optimization.

### Phase 2: baseline and local learning

- Implement a simple HOG-pyramid detector with linear class and direct-box
  heads using the same 5,000-image workload.
- Implement the three `LOCAL_FAST` variants.
- Record unsuccessful variants in a failure journal.
- Select the fastest variant that beats the baseline.

### Phase 3: optimized C training

- Add AVX2, threading, tiled kernels, streaming decode, immediate block updates,
  hard-negative limits, and data-layout optimization.
- Optimize measured cycles and memory traffic rather than FLOP estimates.
- Re-run complete 5,000-image timing after every material optimization.

### Phase 4: global accuracy path

- Dual assignment is implemented in CDET v8 as a serialized auxiliary
  one-to-many training bank plus one-to-one deployment bank.
- Add Progressive Loss, tiny-object assignment protection, and optional
  contextual convolution at P4/P5.
- Compare `LOCAL_FAST` and `GLOBAL_BP` on the same architecture and data.
- Keep contextual convolution only if its accuracy gain exceeds its measured
  CPU and memory penalty:
  [Contextual Convolution](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/html/Duta_Contextual_Convolutional_Neural_Networks_ICCVW_2021_paper.html).

### Phase 5: quantized family

- Establish FP32 accuracy.
- Add INT8 PTQ, INT8 QAT, and W4A8 QAT in that order.
- Publish only variants on the measured accuracy, training-time,
  inference-time, and memory Pareto frontier.

### Phase 6: larger-scale and FPGA validation

- Run optional COCO adapters to test 80-class and 100,000-image scalability.
- Keep this separate from the 5,000-image sub-second acceptance test.
- Once an FPGA part is selected, add HLS-compatible lowering, line buffers,
  streaming requantization, channel folding, on-chip weight planning, and
  streaming top-K.
- Make no 1 W claim until power is measured at the board input.

## 5. Verification and acceptance

### Correctness

- FP32 operators agree with double-precision reference results.
- Every backward operator passes finite-difference checks.
- Scalar and AVX2 quantized paths are bit-exact.
- Empty images, crowded images, overlapping boxes, tiny boxes, and invalid
  labels are covered.
- Saving and reloading produces identical detections.
- CRC-valid truncated payloads are rejected, and inactive-cache mutations are
  canonicalized from FP32 weights on load.
- Arena instrumentation proves zero hot-loop allocations.
- Corrupt and truncated model files fail closed.

### Fast-training gate

For the official 5,000-image, 160 x 160 workload:

- all learned backbone, neck, and head stages start without pretrained weights;
- all stages receive updates during the single pass;
- `train_core_ms` and `train_e2e_ms` are both reported;
- at most 1 second is the stretch target;
- at most 10 seconds is the secondary target;
- validation mAP50:95 must exceed the HOG detector by at least 1 absolute point;
- median results across three deterministic seeds are reported;
- peak memory, images per second, cycles, thread count, and model size accompany
  every timing result.

The edge profile is not a separate relaxed product. It is the first complete
product profile: one Release-built, fully learned model must report both the
training timer and single-image inference latency on the same CPU. A faster
prepared-tensor path cannot be substituted for the end-to-end result; it is
reported only as a diagnostic alongside raw decoding, resizing, label parsing,
training, and serialization.

If raw decoding alone exceeds a time target, the report must show that lower
bound. The project continues toward the closest measured result, but it may not
claim that the missed target was achieved.

### Detector and quantization gates

- `LOCAL_FAST` and `GLOBAL_BP` use the same inference graph.
- Report precision, recall, mAP50, mAP50:95, and small/medium/large-object AP.
- INT8 may lose at most 2 mAP points relative to FP32.
- W4A8 may lose at most 5 mAP points relative to FP32.
- Validation must include decoded-box IoU and class agreement, not only tensor
  similarity.
- The 160 profile must infer in less than 33 ms all-core on the current CPU and
  less than one second single-thread.

## 6. Boundaries and claim policy

- V1 is object and localized-pattern detection only. Segmentation, pose,
  oriented boxes, tracking, and open-vocabulary detection are later heads.
- The framework is a YOLO replacement by capability, not an Ultralytics API or
  checkpoint clone.
- The implementation is clean-room from published algorithms and mathematics;
  no Ultralytics source is copied.
- The detector is called novel only after controlled ablations demonstrate a
  contribution beyond the individual cited techniques.
- Sub-second fully learned detector training is a research goal. It becomes a
  result only after the complete published benchmark satisfies the timer and
  accuracy gates.
