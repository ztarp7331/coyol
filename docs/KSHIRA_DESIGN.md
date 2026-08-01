# KSHIRA implementation contract

KSHIRA (Kernel-Sparse Hierarchical Inference & Runtime Adaptation) is the
research branch for the C detector. It combines the measured multi-scale
detector baseline with a lower-memory RAD/SMRE branch rather than silently
changing the baseline's accuracy and timing claims.

## Contracts already implemented

- `kshira_arena`: caller-owned bump allocation, reset, and high-water tracking.
- `kshira_quant`: symmetric INT4 (`[-7, 7]`) and INT8 (`[-127, 127]`) quantizers,
  packed nibbles, explicit INT4-to-INT8 toggle storage, and QAS gradient
  scaling.
- `kshira_sparse`: channel masks and conservative peak-memory estimates for
  freeze, bias-only, channel-sparse, and full updates, plus arena-cap plan
  admission.
- `kshira_phase`: validation and ordered driver transitions for PRE, TRAIN, and
  ODT bit/update/QAS contracts.
- `kshira_session`: caller-owned composition of the arena, RAD model, phase
  driver, and channel mask; the session test runs FP32 PRE, INT8 TRAIN, and
  INT4 ODT without reallocating.
- `kshira_rad_build` now treats model construction as a transaction: direct
  callers get the original arena offset/high-water and a null model on any
  allocation or initialization failure, matching the session rollback
  contract.
- Full-update training now uses a straight-through single-target gradient path
  through projection, depthwise dilated branches, and the stem. Encoder updates
  honor the sparse channel mask and validate aggregate finite deltas before
  committing them, including INT8/INT4 modes.
- `kshira_domain` provides ten deterministic, balanced curriculum domains
  (textures, edges, rings, gradients, sparse points, and noise) with one target
  box per sample. The local training path computes only the nine-by-nine map
  tile needed by the largest dilation. On the development WSL host, the
  160 x 160 / 5,000-sample Release harness measured 315--359 ms across the
  latest three WSL runs with all encoder channels enabled and 258,365 bytes
  high-water before the arena-backed delta scratch; host load and ASAN affect
  timing.
  The image buffer is caller-owned outside that arena (102,400 bytes for this
  harness), and the timer is a host diagnostic rather than an energy meter.
- `kshira_rad_calibrate` runs a full-map representative pass after quantized
  training and persists input, stem, and per-branch activation scales in the
  caller-owned model. Local target training and full-map inference then share
  those deployment scales; changing bit mode or resetting the model clears the
  calibration state. Head feature quantization remains per-cell so the bounded
  top-K head does not lose small activation signals.
- `kshira_eval` supplies allocation-free IoU/class-hit metrics. The M9 harness
  runs FP32, INT8, and INT4 over the same 5,000-sample curriculum, calibrates
  quantized models on one sample from each domain, and evaluates held-out
  samples using the highest-score detection at a 0.25 threshold. The latest
  release run reports FP32 IoU 0.311, INT8 IoU 0.475, and INT4 IoU 0.435, with
  quantized loss ratios 0.426 and 0.509; the plain ASAN harness also completes.
  These are synthetic proxy results, not COCO accuracy or board-power claims.
- M13 adds an inference-only three-scale view: P4 and P5 are formed by
  pooling the existing stride-4 fused map one cell at a time, then merged with
  P3 through the bounded top-K path without allocating additional feature maps.
  This keeps the 256 KiB high-water unchanged and raises recent held-out
  evaluation to about 0.89--1.05 s aggregate (8.9--10.5 ms per image). The
  current training step still supervises only the P3 head, so P4/P5 outputs are
  experimental and are not a qualified multi-scale accuracy result.
- Reusing the per-step input and stem-tile scales removes duplicate full-image
  scans from the scalar full-encoder path. M11 adds a transactional encoder
  delta cache: projection, dilated-branch, and stem deltas are preflighted
  against one frozen parameter snapshot, then committed only after the whole
  update is finite. Three consecutive Release runs on the development WSL
  host measured INT8 0.663--0.675 s and INT4 0.654--0.671 s for 5,000
  quantized samples; every run reports `edge_train_gate=PASS`. Recent held-out
  evaluation is about 0.76--0.86 s aggregate (7.6--8.6 ms per image).
  Instrumented ASAN timing is diagnostic and remains above the strict gate;
  the Release result is host-specific evidence, not a board or FPGA claim.
- M12 moves the encoder delta cache into the caller-owned arena and sizes it
  from the configured feature/input channels. The 160 x 160 / 8-feature
  benchmark now reports 260,025 bytes high-water inside the 256 KiB arena,
  including this scratch, with no per-step stack allocation. This removes the
  fixed stack burden but leaves only about 2.1 KiB of headroom for this profile;
  larger feature maps must be admitted by the sparse/arena planner before use.
- `kshira_domain_bench` now reports per-mode `train_gate`, aggregate INT8/INT4
  `edge_train_gate`, and per-image evaluation latency as explicit diagnostics.
- The balanced domain generator uses an exact zero-fill fast path for its
  non-noise backgrounds; this does not change the samples or close the
  network-bound INT8 timing gate.

All KSHIRA modules use caller-owned buffers and return explicit failure statuses.
The `kshira_tests` executable covers alignment, overflow/failure paths, pack /
unpack round trips, QAS identities, sparse masks, and phase compatibility.

The RAD M3--M4 branch is also present behind `kshira_rad.h`: its 160 x 160
8-feature configuration uses 260,025 bytes high-water inside a 256 KiB arena
including the reusable training-delta scratch and emits at most 16 top-K
candidates without NMS. M5 now dispatches real
integer MACs for FP32/INT8/INT4 modes and applies QAS to a sparse head update;
the Release harness measured approximately 1.35 ms F32, 9.52 ms INT8, and
9.10 ms INT4 per image on the pinned development CPU. These measurements do
not imply board power or detector accuracy yet, and the full multi-scale
detector still needs its own quantized update integration.

## Architecture combination

The existing `det_*` implementation remains the compatibility baseline: it has
real P3/P4/P5 features, learned neck stages, dual assignment banks, model I/O,
evaluation, and FP32 training with INT8/W4A8 inference. Its source is now split
into ordered implementation units instead of one monolithic translation file.

The KSHIRA RAD branch will use:

1. a compact depthwise-separable encoder;
2. parallel dilated depthwise branches (`d=1,2,4`) fused into one map;
3. a bounded top-K set head with quality scores and greedy one-to-one matching;
4. real quantized forward/backward arithmetic with QAS;
5. sparse channel/bias updates under a measured arena cap.

The branch is accepted only when it reports small-object proxy IoU, quantized
loss recovery, high-water bytes, toggle cost, and latency against the baseline.
No 1 W or 256 KiB claim is made until measured on selected hardware.

## Next milestones

- M3: static explicit forward/backward interfaces for the KSHIRA operators. (forward path landed)
- M4: RAD single-map detector and top-K head behind a separate API. (landed)
- M5: detector-integrated real INT8/INT4 forward plus QAS-scaled sparse head
  training. (landed; M7 adds the sparse full-encoder path)
- M6: PRE/TRAIN/ODT driver with checkpoint contracts and hard arena failure.
- M7: ten-domain generators, local receptive-field training, and the 5,000-image
  edge harness (landed; accuracy and energy gates remain open).
- M8: quantized multi-domain recovery and proxy detection qualification
  (harness landed; recovery gate open).
- M9: persistent full-map activation calibration, quantized class-balance
  updates, and INT4/INT8 proxy recovery (landed on the synthetic gate).
- M11: transactional full-encoder delta cache and a strict Release INT8/INT4
  sub-second training gate (landed on the synthetic 5,000-sample harness).
- M12: arena-backed, spec-sized delta scratch (landed); multi-scale RAD
  feature integration and end-to-end dataset-scale qualification remain open.
- M13: inference-only pooled P3/P4/P5 candidate integration (landed); scale-
  aware assignment and head training remain open.
- M14: direct RAD arena transaction/rollback hardening (landed); scale-aware
  training with separately qualified head/encoder gradients remains open.
- M15+: FPGA lowering and measured hardware energy qualification.
