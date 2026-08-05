# KSHIRA Updated
PLEASE FOLLOW THIS , YOU HAVE DONE SOME NEW IMPLEMENTATION TRY TO FIT THIS ALL IN WITHIN THAT AS WELL U SHOULD UPDATE THE CODE WHICH NEEDS TO BE CHANGED WHEREVER. WE NEED TO ACHIEVE VERY FAST TRAINING SPEED WITH AS CLOSE TO YOLO ACCURACY POSSIBLE
KSHIRA Fast-Convergence Implementation Plan

Accuracy-First, Time-Budgeted Object Detection in ISO C

Status: Proposed implementation specificationDate: 2026-08-05Repository basis: Current C-OLOY / KSHIRA tree described by docs/HANDOFF.md, docs/KSHIRA_DESIGN.md, docs/RESEARCH_CRITIQUE.md, PLAN_UPDATED.md, and the public det_model lifecycle.Product name: KSHIRAPurpose: Implement an accuracy-first path that approaches nano-YOLO accuracy under a measured approximately 30-second CPU time-to-accuracy budget, while retaining a compact, quantized ISO-C deployment graph.

0. Document Authority and Scope

This document defines the next implementation path on top of the current repository. It does not erase the existing RAD implementation, closed F32 result, quality-class experiment, quantized paths, or 256 KiB compact profile.

The current repository remains the source of truth for what is already implemented. This document is the source of truth for the proposed fast-convergence work after the current snapshot.

This plan deliberately separates:

the graph used to learn quickly;

the graph used for final deployment;

the compact 256 KiB research profile;

the accuracy-oriented 30-second research profile.

No code should be removed merely because a new path is proposed. Every major mechanism must be independently toggleable, benchmarked and reversible.

1. Current Repository Position

1.1 Existing public lifecycle

KSHIRA is an architecture profile of det_model and must remain inside the existing lifecycle:

det_context_create
  -> det_model_build
  -> det_train
  -> det_predict / det_evaluate
  -> det_save / det_load
  -> det_model_destroy / det_context_destroy

The detector must remain dataset-neutral. Dataset adapters provide images and boxes; kernels must not contain dataset-specific parsing or class assumptions.

1.2 Current shipped KSHIRA mechanisms

The current tree already contains or has locked implementations for:

3x3 stride-4 stem producing a 40x40 map at 160 input;

depthwise dilated branches with dilation 1, 2 and 4;

sequential branch execution and fusion;

pointwise channel mixer;

learned-feature contrast channel;

quality-class head path with 4 + K outputs;

historical objectness-times-class head path;

Varifocal-style quality supervision;

bounded pairwise ranking;

two-level hard-negative mining;

identity mixer initialization;

neighbour class-quality supervision;

13x13 dependency-tile backward propagation;

transactional updates;

F32, INT8 and INT4 runtime paths;

checkpoint import/export;

fixed top-K and duplicate suppression;

optional P4/P5 ODT heads;

arena high-water reporting;

current compact f8 high-water around 159 KiB.

1.3 Locked measured accuracy bars

The implementation must preserve two distinct baselines:

Closed historical F32 accuracy bar

F1: 0.0430

TP: 26

FP: 730

precision: 0.0344

recall: 0.0573

matched-box mean IoU: approximately 0.618

AP50: approximately 0.0012

training time: approximately 35 seconds for eight epochs on the current host

arena: approximately 159 KiB

Surgical quality-class F32 path

F1: 0.0355

TP: 20

FP: 654

precision: approximately 0.0297

recall: approximately 0.0441

training time: approximately 62 seconds

ranking band remains glued across thresholds

The historical F32 path remains the primary accuracy bar until a new path beats it in repeated runs.

1.4 Current scientific diagnosis

The existing evidence indicates:

localization is not the dominant failure;

true and false candidates occupy a similar score range;

sparse SGD supervision has not learned strong ranking;

adding channels to the same weak training formulation did not improve accuracy;

adding P4/P5 supervision before improving representation and assignment regressed accuracy;

more ordinary epochs did not consistently help;

the compact memory problem has been substantially improved;

the next bottleneck is time-to-useful-representation and time-to-score-separation.

2. Revised Product and Research Objective

2.1 Primary objective

The primary objective is now:

Train or adapt KSHIRA to approach the accuracy of a nano-YOLO baseline under an approximately 30-second measured CPU wall-clock budget, then deploy a folded and quantized ISO-C graph with substantially lower runtime and infrastructure cost.

2.2 Variables that are no longer fixed constraints

The following are tunable research variables:

parameter count;

feature-channel width;

number of unique images;

RGB versus grayscale input;

training resolution;

training arena size;

number of feature scales;

training-only auxiliary branches;

pretrained initialization;

cached teacher supervision;

number of CPU threads.

The following remain important engineering properties:

ISO C runtime;

no dynamic allocation in hot loops;

deterministic execution modes;

explicit memory planning;

F32/INT8/INT4 deployment;

CPU-first implementation;

future FPGA lowering;

honest reporting of preparation time, training time and inference time.

2.3 Three operating modes

KSHIRA shall support three explicit operating modes.

KSHIRA Scratch

all learned weights start from random or mathematical initialization;

no teacher targets;

no external pretrained backbone;

evaluated against nano-YOLO trained from scratch under the same data, resolution, hardware and time budget.

KSHIRA Adapt

starts from reusable KSHIRA foundation weights;

may use cached teacher targets;

trains dataset-specific adapters, fusion and heads within the 30-second budget;

evaluated against pretrained nano-YOLO fine-tuned under the same wall-clock budget.

KSHIRA Full Research

unrestricted time for architecture and compression studies;

used to establish the attainable accuracy ceiling;

not described as a 30-second result.

The selected mode must be persisted in checkpoints and printed in every benchmark result.

3. Central Innovation Direction

The new KSHIRA path is based on five coordinated mechanisms.

3.1 Train-deploy graph separation

The learning graph may contain:

reparameterizable branches;

dense auxiliary heads;

additional feature scales;

normalization;

teacher-distillation projections;

larger temporary candidate pools.

The deployment graph contains only:

fused backbone blocks;

selected P3/P4/P5 feature path;

shared detection head;

required contrast path if validated;

fixed top-K and duplicate control;

quantized weights and scales.

Training-only components must be removable or foldable without changing the final detection interface.

3.2 Dense-to-residual learning curriculum

Early training uses dense box-derived supervision to make the representation spatially meaningful. Later training uses only high-residual locations for expensive encoder updates.

The curriculum is:

box-derived dense fields
  -> dense quality learning
  -> analytic head refresh
  -> residual map
  -> budgeted tile-local backward
  -> final sparse deployment-head refinement

3.3 Streaming analytic readout

The final class-quality head is low dimensional relative to the backbone. KSHIRA will periodically solve a regularized linear readout from streaming sufficient statistics instead of relying entirely on short-horizon SGD.

The analytic readout is used to:

initialize or refresh class-quality rows;

test whether the representation is linearly useful;

calculate residuals for gradient-budget allocation;

reduce the time spent fitting a small linear head.

3.4 Residual-guided gradient budgeting

After the best available readout is fitted, the remaining errors identify where the representation itself is inadequate.

Full encoder backward work is allocated to cells with high combined residual:

missed positive residual;

false-positive residual;

box residual;

assignment instability;

underrepresented class or object size.

This replaces score-only hard-negative selection with representation-aware update selection.

3.5 Time-budget controller

A runtime controller observes:

elapsed wall time;

remaining wall time;

images processed;

milliseconds per forward sample;

milliseconds per full tile;

milliseconds per head-only update;

recent validation proxy movement;

current resolution;

current trainable stage set.

It adjusts:

resolution;

number of images;

number of dense targets;

number of encoder tiles;

number of head-only negatives;

trainable stages;

teacher-loss weight;

auxiliary-loss weight;

finalization reserve.

The 30-second schedule is therefore a measured controller, not a fixed epoch count.

4. Proposed Architecture

4.1 Architecture profiles

KSHIRA remains one architecture with multiple admitted profiles.

Profile

Intended use

Estimated parameter range

Training arena target

Deployment target

Compact

extreme deployment

50k-150k

1-4 MiB

256 KiB-1 MiB

Balanced

primary 30-second research

200k-600k

4-24 MiB

1-4 MiB INT8

Accuracy

ceiling and distillation teacher-student study

700k-1.5M

16-64 MiB

2-8 MiB INT8

Exact parameter and memory values must come from the model builder. The ranges above are planning targets, not claims.

4.2 RGB-capable input

The new primary accuracy profile shall accept RGB.

Supported input transforms:

direct RGB;

luminance plus two compressed chroma channels;

grayscale compatibility mode.

The input transform must be declared in the model spec and persisted in the checkpoint.

4.3 Information-preserving two-stage stem

The current single stride-4 stem remains available for the compact legacy path.

The accuracy path uses:

input
  -> 3x3 stride 2 spatial stem
  -> normalization / calibrated scale
  -> reparameterizable stride 2 block
  -> P2 or P3 base feature

An alternative space-to-depth stem should also be implemented behind an ablation flag:

2x2 or 4x4 space-to-depth
  -> 1x1 learned projection
  -> spatial refinement block

The objective is to avoid discarding fine information before the network learns useful filters.

4.4 Reparameterizable spatial block

During training, each block may contain:

3x3 depthwise or grouped branch;

1x1 channel branch;

identity branch when dimensions permit;

optional dilated branch;

learned branch scales;

batch-independent normalization or fixed calibrated scale.

At deployment, compatible branches are fused into one convolution and bias.

Every block must provide:

training forward;

training backward;

fold operation;

folded forward;

fold-parity test;

quantized folded export.

4.5 Backbone stages

The first balanced prototype should use three or four stages.

Example starting point at 224 input:

Stage

Approximate resolution

Starting channels

Block count

S1

112x112

16-24

1

S2 / P2

56x56

24-32

1-2

S3 / P3

28x28

40-56

2

S4 / P4

14x14

72-96

2

S5 / P5

7x7

112-160

1-2

Channel widths must be generated from a profile table rather than hard-coded throughout kernels.

4.6 Lightweight feature fusion

Use a small bidirectional feature path:

lateral 1x1 projections to a shared feature width;

P5 upsample and add into P4;

fused P4 upsample and add into P3;

one optional bottom-up refinement from P3 to P4;

one optional bottom-up refinement from P4 to P5.

Avoid large concatenations in the default path. Addition and virtual concatenation remain preferred because they reduce memory traffic.

The previous independent P4/P5 ODT heads remain available but are not the canonical accuracy path.

4.7 Shared detection head

The primary deployment head shall be shared across scales.

Outputs per location:

four box distances;

K class-quality logits;

optional one scalar proposal quality if candidate generation is separated from class ranking.

Scale-specific adaptation is limited to:

learned bias;

learned gain;

distance normalization;

optional small scale embedding.

The default final score is the maximum class-quality score. A separate proposal signal may limit candidate evaluation but must not recreate the old weak-score multiplication pathology.

4.8 Contrast path

Contrast remains an ablation, not a mandatory identity of KSHIRA.

The new path uses:

normalized learned features;

local feature contrast;

learnable non-negative gate;

gate initialized near zero or a small value;

explicit pruning when the learned gate remains negligible.

The contrast feature must be tested as:

disabled;

fixed scale;

calibrated scale;

learnable gate;

assignment input only;

head input only;

both.

No final claim may assume contrast is useful without this evidence.

5. Training-Only Supervision

5.1 Dense auxiliary fields

Ground-truth boxes generate low-cost dense fields for each admitted scale.

Centre field

A polynomial centre prior over cells inside or near the object.

Inside-object field

A binary or soft field indicating box interior.

Distance field

Normalized left, top, right and bottom distances.

Size field

Log width and log height or scale-bin target.

Boundary field

Distance to nearest box boundary, normalized by object size.

These fields provide early dense supervision even when final one-to-one assignment is unstable.

5.2 Training-only auxiliary heads

Each selected scale may have a small auxiliary projection predicting:

centre field;

inside field;

size field;

optional boundary field.

Auxiliary heads:

are active only during training;

are not serialized into compact deployment unless requested for ODT;

may share weights across scales;

must use bounded output dimensions;

must have separately reported compute and memory.

5.3 One-to-many and one-to-one paths

The learning graph uses two assignment roles.

Dense auxiliary path

multiple candidates per object;

broad spatial supervision;

high weight early;

weight decays over time.

Deployment path

bounded one-to-one or small top-k assignment;

directly trains final class-quality and box outputs;

weight grows over time.

The inference graph contains only the deployment path.

5.4 Matchability weighting

Dense candidates vary in quality. Their training weight should depend on:

centre prior;

current IoU;

object-size compatibility with scale;

teacher confidence when available;

assignment stability across refreshes.

Low-quality dense matches should contribute feature supervision without dominating final class-quality calibration.

6. Streaming Analytic Head Refresh

6.1 Purpose

The analytic readout determines whether the current features contain linearly separable object/class information and rapidly fits the low-dimensional class-quality head.

It is not a replacement for backbone learning. It is a fast readout and diagnostic.

6.2 Feature vector

For candidate i:

x_i = [shared-head feature vector, optional contrast, bias 1]

The feature dimension should normally remain below 128 so the solve remains small.

6.3 Target transform

For class k:

positive target is IoU-aware quality;

non-target classes are zero;

background is zero.

For the analytic solve, targets may be converted to clipped logits:

t = clamp(log((q + eps) / (1 - q + eps)), t_min, t_max)

Background receives a configured negative target logit.

The target transform must be an ablation. Direct ridge fitting to quality values is the simpler reference.

6.4 Streaming sufficient statistics

Accumulate:

A = lambda * I + sum_i w_i x_i x_i^T
B = sum_i w_i x_i y_i^T

Solve:

W = solve(A, B)

Recommended solver:

Cholesky factorization for positive-definite A;

diagonal regularization;

finite checks;

fallback to previous head when solve fails;

deterministic accumulation order for reproducibility mode.

6.5 Sample composition

The statistics must contain a balanced mixture of:

primary positives;

neighbour positives;

small-object positives;

rare-class positives;

near-object negatives;

diverse far-background negatives;

current false positives;

teacher-selected hard locations when available.

Per-group weights must be explicit and reported.

6.6 Refresh schedule

Candidate schedules:

after bootstrap;

after each resolution transition;

after a configured number of images;

when validation proxy stagnates;

before final calibration.

The default 30-second controller should reserve time for at least two readout refreshes.

6.7 Post-solve fine-tuning

After analytic refresh:

copy solved weights into the class-quality head;

retain the previous head if validation proxy worsens;

run bounded VFL/ranking fine-tuning;

do not reset optimizer state blindly;

optionally interpolate old and solved rows.

6.8 Diagnostic value

If the analytic head substantially improves ranking with a frozen encoder, the primary limitation was readout optimization.

If it does not improve ranking, the representation remains the primary limitation and more head-loss changes should stop.

7. Residual-Guided Gradient Budgeting

7.1 Residual definition

For candidate i, compute a composite priority:

R_i = a * class_quality_residual
    + b * box_residual
    + c * false_positive_cost
    + d * assignment_instability
    + e * rarity_weight
    + f * small_object_weight

Possible terms:

absolute target minus prediction;

VFL residual;

1 minus IoU;

current score for background;

disagreement between teacher and student;

class-frequency inverse square root;

repeated failure across refreshes.

7.2 Update tiers

Tier 0: no update

Low-residual candidates receive no backward work.

Tier 1: head-only

Moderate residuals update only:

class-quality head;

box head;

scale bias/gain.

Tier 2: late-stage tile

High residuals update:

shared head;

neck/fusion;

final one or two backbone stages.

Tier 3: full dependency tile

Highest residuals update:

stem or early backbone when required;

all relevant feature stages;

fusion;

head.

Tier 3 must be rare under the 30-second controller.

7.3 Tile construction

The existing 13x13 RAD dependency tile remains for the legacy path.

The multistage path requires graph-derived dependency rectangles.

The compiler/planner must calculate:

source support per output location;

scale transitions;

upsample/add dependencies;

halo required by every active branch;

overlap between selected residual tiles.

Tiles are merged when doing so reduces recomputation without exceeding workspace or time budget.

7.4 Gradient budget

The controller sets a budget in measured milliseconds, not only number of tiles.

For example:

remaining_update_ms = remaining_ms - finalization_reserve_ms
max_tier3 = floor(remaining_update_ms * tier3_share / measured_tier3_ms)
max_tier2 = floor(...)
max_head = floor(...)

A fixed-budget reference must be implemented before adaptive budgeting.

7.5 Stagnation response

When the validation proxy does not improve:

first increase data diversity;

then increase head statistics;

then increase late-stage tiles;

only then unfreeze earlier stages;

do not automatically widen the model during a run.

8. Target-Domain Bootstrap

8.1 Box-crop stream

Before full-image detection, construct object-centric samples from existing boxes:

tight object crop;

object plus context;

translated crop;

scaled crop;

horizontal flip when valid;

same-size background crop;

object-boundary crop.

The crop stage trains:

class discrimination;

foreground/background;

centre offset;

object size;

feature consistency across crop variants.

8.2 Use in scratch mode

In scratch mode, the crop stream gives the random backbone a high positive-signal density before dense detection begins.

8.3 Use in adaptation mode

In adaptation mode, crop samples rapidly align the reusable backbone to the target classes and visual domain.

8.4 Cache policy

Crops may be:

generated ahead of time and cached;

generated deterministically from decoded images;

generated from compact resize caches.

Preparation time must be reported separately from the 30-second adaptation time.

9. Cached Teacher Supervision

9.1 Optional, mode-specific mechanism

Teacher supervision is allowed only in KSHIRA Adapt or explicitly teacher-assisted research runs.

Scratch results must not use teacher targets.

9.2 Teacher cache contents

The initial compact cache should store per image:

image identifier and dataset hash;

teacher model identifier and checksum;

top teacher boxes;

class IDs;

class-quality scores;

optional box uncertainty;

selected hard-background locations;

per-object preferred feature scale;

optional low-dimensional projected feature vectors at object locations.

Avoid storing full teacher feature maps in the first implementation.

9.3 Teacher losses

Box loss

Student box output matches teacher boxes, subject to ground-truth trust policy.

Quality loss

Student class-quality logits match teacher quality at consistent candidate locations.

Assignment loss

Teacher-selected cells provide stable early assignments.

Feature loss

Optional projected student features match compact teacher projections only at object-aware locations.

9.4 Trust policy

Teacher outputs must not override ground truth blindly.

Priority order:

ground-truth labels and boxes;

teacher correction within bounded tolerance;

teacher-only targets when confidence and consistency pass configured gates;

ignored otherwise.

9.5 Cache preparation boundary

Report:

teacher-cache generation time;

cache size;

KSHIRA 30-second adaptation time;

combined first-use time;

repeated-use adaptation time.

A cached result must never be described as 30-second end-to-end first-use training without this distinction.

10. Progressive Resolution and Stage Unlocking

10.1 Resolution curriculum

The first implementation shall support:

96 -> 160 -> 224

and:

128 -> 192 -> 256

The exact schedule is selected by profile and measured throughput.

10.2 Shared weights

Backbone and head weights are shared across resolutions. Scale calibration parameters may be resolution-specific only when justified by measurement.

10.3 Unlock schedule

Initial stage

Train:

auxiliary heads;

class-quality head;

box head;

fusion adapters;

final backbone stage.

Middle stage

Unfreeze:

neck;

middle backbone stage;

selected reparameterization branches.

Final stage

Use residual budget to decide whether early stages need updates.

In adaptation mode, the stem should normally remain frozen unless colour/domain residuals remain high.

10.4 Resolution-transition refresh

After changing resolution:

refresh scale statistics;

optionally refresh analytic readout;

reset only resolution-dependent caches;

retain learned weights;

preserve deterministic seed state.

11. Time-Budget Controller

11.1 Controller configuration

Add an optional time-budget structure to the KSHIRA training specification containing:

total wall-clock budget;

preparation included/excluded flag;

finalization reserve;

allowed resolutions;

minimum samples per stage;

maximum encoder-tile share;

maximum teacher-loss share;

allowed thread count;

deterministic versus throughput mode.

11.2 Throughput calibration

At the beginning of training, measure a small bounded sample of:

forward-only time;

head-update time;

late-stage tile time;

full tile time;

analytic accumulation time;

analytic solve time;

validation-proxy time;

branch-fold time;

serialization time.

Use these measurements to calculate quotas.

11.3 Default adaptation schedule

An initial schedule, subject to controller adjustment:

Budget window

Main work

0-2 s

load model/cache, calibrate throughput

2-6 s

object-crop adaptation and first analytic readout

6-14 s

low/mid-resolution dense auxiliary learning

14-23 s

multiscale deployment-head and fusion training

23-27 s

residual-guided late-stage/full tiles

27-29 s

final analytic refresh and ranking refinement

29-30 s

fold, calibrate, serialize

11.4 Default scratch schedule

Budget window

Main work

0-5 s

target-domain box-crop bootstrap

5-12 s

low-resolution dense geometry learning

12-21 s

multiscale dense-to-deploy transition

21-27 s

residual-guided representation updates

27-29 s

analytic head refresh

29-30 s

fold, calibrate, serialize

11.5 Early stopping and reallocation

When a phase reaches its metric gate early, remaining time is transferred to the next phase.

When a phase fails to improve after its configured patience:

stop repeating the same updates;

refresh head or data selection;

move to a new resolution;

increase residual diversity;

preserve finalization reserve.

12. Repository Change Map

12.1 Public API: include/det.h

Add optional, backward-compatible fields at the end of relevant structures.

Required concepts:

KSHIRA profile: compact, balanced, accuracy;

training mode: scratch, adapt, full research;

time budget in milliseconds;

resolution schedule identifier;

input colour mode;

teacher-cache enable flag;

auxiliary-training enable flag;

analytic-readout enable flag;

residual-budget enable flag;

training-arena bytes separate from deployment-arena bytes;

requested thread count;

deterministic mode.

Older callers that zero-initialize structures must retain current behaviour.

12.2 Adapter/orchestrator: src/det_kshira.inc

Responsibilities to add:

choose architecture profile;

initialize training controller;

run throughput calibration;

select scratch/adapt schedule;

transition resolutions;

transition trainable stage masks;

call analytic refresh;

build residual candidate queues;

apply tile budgets;

trigger fold and quantization;

print phase timing and metric movement;

preserve legacy recipe selection.

Do not place low-level convolution math in the adapter.

12.3 Core detector: src/kshira_rad.c

Retain legacy RAD functions.

Add or route to:

multi-stage RGB stem;

reparameterizable blocks;

shared pyramid;

shared multiscale head;

dense auxiliary forward/loss;

residual extraction;

trainable-stage masks;

folded deployment forward;

contrast gate.

The file should not become a single monolith. New components should be separated as described below.

12.4 Proposed new modules

src/kshira_reparam.c

branch forward/backward;

branch-scale handling;

exact folding;

folded-kernel export;

fold-parity checks.

src/kshira_pyramid.c

P3/P4/P5 production;

top-down addition;

optional bottom-up refinement;

shared projection scheduling;

scale-support queries.

src/kshira_aux.c

dense target generation;

centre/interior/size/boundary losses;

auxiliary head forward/backward;

loss-weight schedule.

src/kshira_solver.c

sufficient-statistic accumulation;

regularization;

Cholesky solve;

head interpolation;

solve diagnostics;

rollback on regression.

src/kshira_budget.c

elapsed-time tracking;

throughput calibration;

quota calculation;

residual priority queue;

update-tier selection;

finalization reserve.

src/kshira_bootstrap.c

deterministic crop generation;

object/background crop targets;

bootstrap schedule;

crop-level auxiliary losses.

src/kshira_distill.c

teacher-cache parsing;

ground-truth/teacher trust policy;

teacher assignment targets;

prediction and optional feature distillation.

src/kshira_cache.c

predecoded tensor cache;

multiresolution cache indexing;

dataset fingerprint;

target cache validation.

12.5 Internal definitions

Extend src/kshira_rad_internal.h or split it into internal headers containing:

stage descriptors;

reparameterization block descriptors;

pyramid descriptors;

auxiliary-head state;

solver state;

budget-controller state;

training/deployment graph identifiers;

fold state;

profile dimensions;

graph fingerprint.

12.6 State persistence: src/kshira_rad_state.c

Bump state format after the current v2 path.

The new format must record:

format version;

graph/profile fingerprint;

input colour transform;

training mode;

resolution schedule;

unfolded training weights when saving resumable state;

folded deployment weights when exporting deploy state;

analytic-head metadata;

normalization/calibration state;

contrast gate;

scale gains and biases;

quantization scales;

optional foundation-model identifier;

optional teacher-cache identifier;

CRC and exact payload length.

Training checkpoints and deployment blobs should be distinguishable.

12.7 Bench and tools

tools/bench.c

Add reporting for:

mode and profile;

preparation time;

adaptation/training time;

fold time;

serialization time;

time-to-first metric gate;

time-to-best F1/AP50;

resolution transitions;

update-tier counts;

solver refresh count;

teacher-cache usage;

training and deployment memory;

folded and unfolded parameter counts.

New cache tool

Create a tool for:

decoding source images once;

generating multiresolution RGB tensors;

writing dataset fingerprints;

generating dense box fields;

optionally importing teacher predictions.

A Python preparation tool is acceptable, but the product runtime must not require Python.

Teacher import tool

Support common teacher output formats without copying external detector source code.

The imported cache becomes a KSHIRA-owned versioned binary format.

13. Memory Planning

13.1 Separate training and deployment arenas

The public model configuration must distinguish:

training arena capacity;

deployment arena capacity.

The compact path may continue to use approximately 256 KiB. Balanced and accuracy training profiles may use larger caller-owned arenas.

13.2 Liveness and reuse

The graph planner should reuse buffers across:

sequential reparameterization branches;

scale production;

auxiliary heads;

analytic accumulation;

residual queues;

tile backward;

quantization calibration.

Training-only buffers must not remain live during folded inference.

13.3 Memory report

Every result must report:

parameter bytes;

optimizer bytes;

activation bytes;

auxiliary bytes;

solver bytes;

residual-queue bytes;

teacher-cache resident bytes;

fold workspace;

training high-water;

deployment high-water;

checkpoint bytes;

deployment blob bytes.

13.4 No hidden allocation

The no-hot-loop-allocation contract remains.

All candidate queues, sufficient statistics, tile lists and cache pages must be preallocated or arena-planned.

14. CPU Execution Plan

14.1 Scalar reference first

Every new operator requires a readable scalar FP32 reference.

14.2 Kernel order

Implementation sequence:

direct scalar FP32;

specialized 1x1 and depthwise kernels;

fused add/activation;

blocked RGB stem;

AVX2 FP32;

AVX2 INT8;

thread-pool image or tile parallelism;

double-buffered cache loading;

packed INT4 deployment.

14.3 Parallelism

The 30-second benchmark must report:

single-thread result;

configured all-core result;

thread count;

CPU affinity policy.

The deterministic reference may use a fixed accumulation order. Throughput mode may permit deterministic per-thread partial sums followed by ordered reduction.

14.4 Fusion opportunities

Prioritize:

stem convolution + activation;

depthwise + pointwise fusion where numerically valid;

lateral projection + addition;

head projection + top-K insertion;

quantization + activation;

teacher-target lookup + loss.

15. Quantization and Compression

15.1 Accuracy first

Do not quantize an architecture that has not established useful F32 accuracy.

15.2 Folding order

Recommended order:

finish training;

fold reparameterization branches;

fold frozen normalization/scales;

prune disabled contrast or auxiliary paths;

calibrate activations;

run INT8 QAT or short adaptation;

optionally run W4A8/INT4 adaptation;

export packed deployment graph.

15.3 Distillation to compact profile

When the balanced profile reaches useful accuracy, train a compact profile using cached balanced-model targets.

This creates a KSHIRA-to-KSHIRA compression path without requiring the external teacher at deployment.

15.4 Required parity tests

unfolded versus folded FP32 detections;

folded FP32 versus INT8 decoded detections;

INT8 versus INT4 class agreement;

box IoU agreement;

score-order agreement;

deployment memory and latency.

16. Benchmark Protocol

16.1 Fair nano-YOLO comparison

Create two nano-YOLO baselines on the exact same:

train split;

validation split;

RGB preprocessing;

class mapping;

maximum detections;

evaluator;

IoU matching;

CPU hardware;

thread count;

wall-clock budget.

Baseline A: scratch

Nano-YOLO starts from random weights.

Baseline B: adaptation

Nano-YOLO starts from its normal pretrained checkpoint.

KSHIRA Scratch compares with Baseline A. KSHIRA Adapt compares with Baseline B.

16.2 Time boundaries

Report:

cache preparation;

teacher preparation;

model load;

training/adaptation;

folding;

quantization;

serialization;

end-to-end first use;

repeated adaptation using existing caches.

16.3 Accuracy metrics

Required:

precision;

recall;

F1-max;

AP50;

AP75;

mAP50:95;

per-class AP;

small/medium/large AP;

TP/FP/FN;

matched-box IoU;

score histograms;

calibration error;

predictions per image.

16.4 Time-to-accuracy curves

Record metrics at fixed times:

2 s, 5 s, 10 s, 15 s, 20 s, 25 s, 30 s

Primary comparisons:

time to AP50 threshold;

time to F1 threshold;

AP50 at 30 seconds;

mAP50:95 at 30 seconds;

area under the time-to-accuracy curve.

17. Implementation Phases

Phase 0: Baseline preservation and instrumentation

Work

lock current closed F32 recipe;

lock current surgical QC recipe;

add score histograms;

add GT-count versus max-detections audit;

add time-to-metric logging;

add graph/profile fingerprinting;

add feature flags for every proposed mechanism.

Gate

existing repeated results remain reproducible;

no default behaviour change;

tests green.

Phase 1: Analytic readout experiment

Work

implement kshira_solver.c;

freeze existing encoder;

accumulate positive and diverse negative statistics;

solve class-quality head;

compare ranking with SGD head;

add rollback.

Decision gate

Continue analytic refresh if it improves any of:

F1 by at least 0.005;

TP at threshold 0.20;

FP/TP ratio;

AP50;

positive/negative score separation.

If it does not improve, prioritize representation before further head work.

Phase 2: Target-domain bootstrap

Work

deterministic object/background crops;

crop-level class and foreground losses;

fixed wall-clock bootstrap window;

compare full-image-only versus bootstrap.

Gate

At equal total time, bootstrap must improve time-to-F1 or AP50.

Phase 3: Information-preserving RGB backbone

Work

two-stage stem;

RGB cache;

compact reparameterizable block;

profile-based channels;

normalization/scaling;

legacy stem remains available.

Gate

A small balanced prototype must clearly exceed the current closed accuracy bar before multiscale expansion.

Phase 4: Dense auxiliary supervision

Work

target-field generation;

centre/interior/size heads;

dense-to-deployment loss schedule;

auxiliary heads removed from deploy graph.

Gate

At equal 30-second budget, dense supervision must improve time-to-AP50 and not merely training loss.

Phase 5: Shared multiscale pyramid

Work

P3/P4/P5 shared projections;

top-down addition;

shared head;

scale bias/gain;

scale-aware dense and deploy assignment.

Gate

small-object AP improves;

prediction flood remains controlled;

fold/deploy memory is reported;

no regression from untrained scale heads.

Phase 6: Residual-guided gradient budget

Work

residual map;

update tiers;

fixed budget;

adaptive budget;

graph-derived dependency tiles;

measured milliseconds per tier.

Gate

At the same accuracy, measured training time decreases; or at the same 30 seconds, AP/F1 increases.

Phase 7: Time-budget controller

Work

throughput calibration;

resolution transitions;

unlock schedule;

finalization reserve;

metric-based reallocation.

Gate

Repeated 30-second runs complete folding and serialization without overrunning the budget tolerance.

Phase 8: Teacher-assisted adaptation

Work

teacher cache format;

import tool;

prediction/assignment distillation;

optional projected feature targets;

trust policy.

Gate

KSHIRA Adapt materially closes the gap to pretrained nano-YOLO at the same 30-second adaptation budget.

Phase 9: Folding and deployment compression

Work

fold branches;

prune auxiliary paths;

INT8 calibration/QAT;

compact-profile distillation;

packed deploy export.

Gate

fold parity passes;

quantized AP loss remains within declared limit;

deployment latency and memory meet selected profile target.

18. Tests

18.1 Unit tests

Add tests for:

solver accumulation;

Cholesky solve against known systems;

singular-system fallback;

residual priority ordering;

fixed-budget determinism;

time-controller quota calculation;

reparameterization fold equality;

RGB stem parity;

space-to-depth ordering;

dense target fields;

scale assignment;

teacher-cache validation;

profile builder rejection;

state-format mismatch;

quantized folded blocks.

18.2 Integration tests

current legacy recipe unchanged;

scratch schedule completes;

adaptation schedule completes;

resolution transition preserves model state;

solver refresh can roll back;

auxiliary heads absent from deployment blob;

folded checkpoint reload matches prediction;

deterministic run reproduces metrics and state checksum.

18.3 Numerical tests

dense versus tile gradient parity on small graph;

unfolded versus folded output tolerance;

scalar versus SIMD parity;

F32 versus INT8 decoded output comparison;

teacher-target coordinate alignment.

19. Experiment Matrix

19.1 Representation isolation

ID

Stem

RGB

Norm

Reparam

Solver

R0

current stride 4

no

no

no

no

R1

current stride 4

no

no

no

yes

R2

two-stage

yes

no

no

yes

R3

two-stage

yes

yes

no

yes

R4

two-stage

yes

yes

yes

yes

Fit the same head/evaluator to isolate representation gains.

19.2 Supervision isolation

ID

Crop bootstrap

Dense fields

Multiscale

Residual budget

S0

no

no

no

score HNM

S1

yes

no

no

score HNM

S2

yes

yes

no

score HNM

S3

yes

yes

yes

score HNM

S4

yes

yes

yes

residual

19.3 Training-mode comparison

random KSHIRA Scratch;

foundation KSHIRA Adapt;

foundation plus cached teacher;

nano-YOLO random;

nano-YOLO pretrained;

balanced KSHIRA Full Research ceiling.

19.4 Capacity sweep

Measure at approximately:

100k parameters;

300k parameters;

600k parameters;

1.0M parameters.

Report AP at 10, 20 and 30 seconds. Do not select by parameter count alone.

19.5 Resolution sweep

fixed 160;

fixed 224;

96 to 160 to 224;

128 to 192 to 256.

20. Acceptance Gates

20.1 Stage gates

No phase becomes canonical merely because it compiles.

A phase must improve at least one primary axis without unacceptable regression on the others:

AP50;

mAP50:95;

F1;

time-to-accuracy;

training memory;

deployment memory;

inference latency.

20.2 Accuracy-development gates

Initial progression:

beat closed F1 0.043 and TP 26 in repeated F32 runs;

obtain a non-trivial AP50 improvement over 0.0012;

reduce FP/TP materially;

establish useful score separation;

reach F1 0.10;

reach F1 0.20 with useful AP;

begin nano-YOLO relative comparison only after the evaluator is shared.

20.3 Nano-YOLO race gates

After fair baselines exist:

Scratch goal

At 30 seconds, KSHIRA Scratch should exceed nano-YOLO Scratch on at least one of:

AP50;

mAP50:95;

time to fixed AP;

memory to fixed AP.

Adapt goal

Target:

within approximately 10-15% relative AP50 of pretrained nano-YOLO at the same 30-second adaptation budget;

or reach the same AP two to three times faster;

with materially lower training memory or simpler deployment runtime.

These percentages are research targets, not current claims.

20.4 Reproducibility

Every promoted result requires:

at least three deterministic seeds;

median and range;

complete command/config;

data fingerprint;

CPU and thread count;

compiler and flags;

training/deployment memory;

time boundary definition.

21. Rollback and Branch Policy

21.1 Preserve legacy implementation

Keep the current RAD path selectable through a profile or compatibility flag.

21.2 Feature flags

Every major feature requires an independent toggle:

RGB stem;

two-stage stem;

reparameterization;

normalization;

dense auxiliary fields;

analytic readout;

residual budget;

progressive resolution;

shared pyramid;

contrast gate;

teacher cache;

adaptation mode.

21.3 Promotion rule

A new default is promoted only after:

unit and integration tests pass;

repeated metric gate passes;

baseline command remains available;

checkpoint compatibility policy is documented;

memory and time are reported.

21.4 Stop rule

If three consecutive architecture ablations fail to improve AP50 or F1 meaningfully, stop architecture churn and investigate:

data quality;

evaluator correctness;

target alignment;

model capacity;

teacher/domain mismatch;

CPU bottleneck.

22. Paper and Novelty Position

The proposed paper contribution should not claim that each component is individually new.

The potentially distinctive contribution is the integrated system:

KSHIRA is a time-budgeted ISO-C object detector that uses a richer foldable training graph, dense box-derived supervision, streaming analytic readout refresh and residual-guided local backpropagation to maximize detection accuracy under a fixed CPU training budget before exporting a compact quantized deployment graph.

The strongest proposed novelty is the combination of:

train-deploy graph separation inside a custom C detector runtime;

streaming analytic quality-head refresh;

analytic-residual-driven gradient allocation;

dense-to-residual detection curriculum;

time-budget-controlled resolution and stage unlocking;

optional cached assignment distillation with no teacher execution during adaptation;

structural folding into a low-bit deployment graph.

The contribution becomes valid only if controlled ablations show that these mechanisms improve time-to-accuracy over simpler alternatives.

23. Immediate Next Actions

The first implementation cycle should not attempt the complete architecture.

Execute in this order:

Action 1: instrumentation and fair baseline

lock current recipes;

add score histograms;

add time-to-metric checkpoints;

establish nano-YOLO scratch and adaptation commands on the same evaluator.

Action 2: analytic readout

implement solver on frozen current KSHIRA features;

determine whether current representation can support stronger ranking.

This is the highest-information experiment.

Action 3: RGB target-domain bootstrap

add predecoded RGB cache;

add deterministic object/background crop stream;

compare equal-time learning.

Action 4: two-stage information-preserving stem

implement the smallest balanced profile;

retain current RAD neck/head initially;

compare representation before building the full pyramid.

Action 5: dense auxiliary fields

add centre/interior/size supervision;

measure time-to-AP50.

Only after these five actions should the repository proceed to:

shared multiscale pyramid;

adaptive residual budgets;

teacher-assisted adaptation;

full train-deploy folding.

24. Final Implementation Principle

KSHIRA should no longer be optimized by asking:

How can the smallest existing graph be trained a little better?

The new engineering question is:

Given a fixed wall-clock budget, which supervision, parameters, resolutions and gradient paths produce the largest increase in validated detection accuracy, and how can the resulting learned system be folded into a compact ISO-C deployment graph?

The implementation must therefore optimize:

validated accuracy gained per millisecond

rather than only:

milliseconds per epoch

or:

minimum parameter count

This is the central direction for the next KSHIRA repository version.

25. Research References Informing the Direction

The implementation should cite and compare against the following research families without copying source code:

deeply supervised detector training from scratch;

stable normalization and information-preserving stems for scratch detection;

dense matching for faster detector convergence;

structural/online convolutional reparameterization;

object-aware and assignment-aware detector distillation;

sparse or local backward computation;

quality-aware classification and ranking.

The KSHIRA contribution is the measured integration of these principles into a time-budgeted, arena-planned ISO-C training and deployment system.