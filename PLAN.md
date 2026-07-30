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

Two training times must be published:

1. `train_core_ms`: prepared low-resolution tensors through final learned
   weights and model serialization;
2. `train_e2e_ms`: raw files, decoding, resizing, labels, training, and model
   serialization.

Both target one second. Separating them is diagnostic, not an exclusion:
`train_e2e_ms` is the complete user-visible result, while `train_core_ms`
identifies whether the bottleneck is learning or input I/O.

COCO is not part of the core API or the first timing contract. COCO 2017 becomes
an optional 80-class scalability and accuracy test after the 5,000-image gate
works. Dataset adapters must never leak COCO-specific assumptions into the
model, loss, graph, or public API.

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
decoding, and fixed-size top-K selection but no NMS. A one-to-many head remains
available during reference training and evaluation.

These choices retain YOLO26's useful deployment ideas—DFL-free regression and
dual-head NMS-free detection—without copying its complete architecture:
[YOLO26 paper](https://arxiv.org/html/2606.03748v1) and
[YOLOv10 dual assignment](https://arxiv.org/abs/2405.14458).

### 2.2 Fast full-model learning

`LOCAL_FAST` must update the convolutional backbone, neck, and head. It must not
use a frozen or pretrained detector.

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

`GLOBAL_BP` trains the identical graph with ordinary end-to-end backpropagation.
It provides:

- gradient and convergence truth for `LOCAL_FAST`;
- one-to-many plus one-to-one consistent assignment;
- Progressive Loss moving from `(0.8, 0.2)` to `(0.1, 0.9)`;
- tiny-object assignment protection;
- BCE classification loss;
- IoU plus normalized L1 box loss;
- SGD with momentum as the first optimizer.

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

- `F32` for reference training;
- `Q8` for INT8 weights and activations;
- `W4A8` for packed signed INT4 middle-layer weights and INT8 activations;
- W4A4 remains experimental until W4A8 passes accuracy gates.

Use per-output-channel symmetric weight scales, per-tensor activation scales,
INT32 accumulation, and integer multiplier-plus-shift requantization. Precision
profiles are compiled separately rather than switched at runtime.

The versioned `CDET` model contains graph metadata, tensor shapes, training
mode, quantization parameters, packed weights, arena requirements, checksums,
and an architecture hash. It never serializes native pointers or
compiler-dependent structs.

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

- Add dual assignment, Progressive Loss, tiny-object assignment protection,
  and optional contextual convolution at P4/P5.
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
