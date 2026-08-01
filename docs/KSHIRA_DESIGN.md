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
  freeze, bias-only, channel-sparse, and full updates.
- `kshira_phase`: validation of PRE, TRAIN, and ODT bit/update/QAS contracts.

All four modules use caller-owned buffers and return explicit failure statuses.
The `kshira_tests` executable covers alignment, overflow/failure paths, pack /
unpack round trips, QAS identities, sparse masks, and phase compatibility.

The RAD M3--M4 branch is also present behind `kshira_rad.h`: its 160 x 160
8-feature configuration uses 258,100 bytes high-water inside a 256 KiB arena
and emits at most 16 top-K candidates without NMS. The Release harness measured
about 0.47 ms per image on the pinned development CPU. These measurements do
not imply quantized training, board power, or detector accuracy yet.

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

- M3: static explicit forward/backward interfaces for the KSHIRA operators.
- M4: RAD single-map detector and top-K head behind a separate API.
- M5: detector-integrated real INT8/INT4 training, QAS, and sparse schedules.
- M6: PRE/TRAIN/ODT driver with checkpoint contracts and hard arena failure.
- M7+: ten-domain generators, edge harness, and hardware-specific energy data.
