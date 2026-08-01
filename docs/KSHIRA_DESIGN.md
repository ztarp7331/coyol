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
- Full-update training now uses a straight-through single-target gradient path
  through projection, depthwise dilated branches, and the stem. Encoder updates
  honor the sparse channel mask and validate aggregate finite deltas before
  committing them, including INT8/INT4 modes.
- `kshira_domain` provides ten deterministic, balanced curriculum domains
  (textures, edges, rings, gradients, sparse points, and noise) with one target
  box per sample. The local training path computes only the nine-by-nine map
  tile needed by the largest dilation. On the development WSL host, the
  160 x 160 / 5,000-sample Release harness measured 315--359 ms across the
  latest three WSL runs with all encoder channels enabled and 258,317 bytes
  high-water inside a 256 KiB arena; host load and ASAN affect timing.
  The image buffer is caller-owned outside that arena (102,400 bytes for this
  harness), and the timer is a host diagnostic rather than an energy meter.
- The local INT4/INT8 target path calibrates stem scales on its live receptive
  tile and branch scales at the supervised cell, while full-map inference uses
  full-map calibration. This is an explicit local-fast approximation; M8 must
  add calibrated persistent scales and a quantized proxy-recovery gate before
  claiming deployment-equivalent accuracy.
- `kshira_eval` supplies allocation-free IoU/class-hit metrics. The M8 harness
  runs FP32, INT8, and INT4 over the same 5,000-sample curriculum and a held-out
  calibration stream, using the highest-score detection at a 0.25 threshold.
  The current quantized profile uses sparse channel updates plus a bounded QAS
  gradient; it completes, but its latest top-1 proxy result is FP32 IoU about
  0.31 versus 0 for INT8/INT4, leaving recovery unqualified.

All KSHIRA modules use caller-owned buffers and return explicit failure statuses.
The `kshira_tests` executable covers alignment, overflow/failure paths, pack /
unpack round trips, QAS identities, sparse masks, and phase compatibility.

The RAD M3--M4 branch is also present behind `kshira_rad.h`: its 160 x 160
8-feature configuration uses 258,100 bytes high-water inside a 256 KiB arena
and emits at most 16 top-K candidates without NMS. M5 now dispatches real
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
- M9+: hardware-specific energy data and deployment-equivalent quantized scales.
