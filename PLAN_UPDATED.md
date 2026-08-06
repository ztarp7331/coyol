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

26. Current implementation status (2026-08-06)

The instrumentation, analytic readout, cache/teacher, stem, and dense-control
actions have been implemented or measured in the current tree.
The highest-yield native adaptation refinement is now direct training of the
three final per-scale class-head biases in the owned C checkpoint blobs. It
keeps the imported feature graph frozen and uses fractional accumulator
rounding for quantized bias updates. With the learning_rate=0.02 recipe, three
real 30-second cars runs processed 64, 67 and 68 images in 28.03, 27.99 and
27.77 seconds. The common 250-image evaluator at threshold 0.05 reached
mAP50:95 0.1907, 0.1903 and 0.1909, with AP50 0.3074, 0.3075 and 0.3075.
The seed-7 checkpoint preserved mAP50:95=0.1907 and AP50=0.3074 under INT8.

This refinement is promoted for native Adapt mode. It does not close the
Scratch gate and must not be described as full graph backpropagation. The
remaining research gate is trainable feature/head weights or a stronger
representation under the same real-data wall-clock budget; additional stem
or cache ablations remain subordinate to that gate.

A center-based dense-cell head update was also measured and rejected: it
processed 57 images in the 30-second run, produced 4,220 false positives on
the 250-image validation split, and reduced mAP50:95 to 0.1567. It remains
opt-in research work and is not part of the promoted path.

A frozen-backbone final-head-weight experiment was also measured and rejected:
it reached AP50=0.3110 but mAP50:95=0.1899 with more false positives than the
promoted bias-only path. Its feature capture and weight-update code was
reverted, leaving full trainable head weights as the next research gate.

27. Class-and-scale localization adapter (2026-08-06)

The promoted native Adapt path now learns four bounded box corrections for
each class and output scale: center-x, center-y, log-width and log-height.
The correction is applied in both native C prediction paths and serialized in
checkpoint version 2; the C loader accepts both version 1 and version 2.
Three real 30-second cars runs processed 75, 77 and 89 images in 27.95, 27.90
and 27.72 seconds. The common 250-image evaluator at threshold 0.05 reached
mAP50:95 0.1927, 0.1927 and 0.1923, with AP50 0.3081, 0.3081 and 0.3093.
The seed-7 checkpoint preserved 0.1927 mAP50:95 and 0.3081 AP50 under INT8.
This is promoted as the current native Adapt profile. It is still a compact
adaptation over a frozen feature graph, so the Scratch and full trainable
feature/head-weight gates remain open.

The 0.30-IoU matching-threshold ablation was rejected: it processed 83 images
in 27.68 seconds and improved AP50 to 0.3122, but increased predictions to 856
and reduced common mAP50:95 to 0.1854. The 0.50-IoU matcher remains promoted.

28. Three-scale native graph runtime and arena correction (2026-08-06)

The complete three-scale native C graph is now exercised by the public
adaptation/checkpoint/evaluation path. Its FP32 forward had been blocked by
the 2 MiB single-scale arena; a 4 MiB multi-scale arena now covers the measured
3,458,900-byte high-water peak while the compact single-scale profile remains
at 2 MiB. Standalone S8 and FP32 forward both pass, and a real 30-second
adaptation processed 108 samples in 28.05 seconds. The current FP32 result on
the 250-image evaluator is AP50=0.1247 and mAP50:95=0.0703, with no predictions
at the 0.05 operating threshold. The alternate quantized calibration reached
mAP50:95=0.0075 with 9,246 predictions. This closes the runtime/arena gate but
rejects the current multi-scale export as an accuracy profile; export score
calibration and trainable graph-weight adaptation remain open gates.

29. Dense final-head weight update rejection (2026-08-06)

A native C experiment retained the final class-head feature tiles and applied
positive plus hard-negative gradients directly to mutable INT8 final-head
weights using fractional accumulators. It completed a real 30-second run in
27.93 seconds and processed 76 images, but the full 250-image evaluator
reached only mAP50:95=0.1601, below the promoted 0.1927 profile. The path was
rejected and removed, including its extra forward-pass feature retention. The
promoted implementation therefore remains the scalar/class/scale/class-box
adaptation over the frozen native graph; a stronger measured representation or
scratch path is still required before claiming nano-like accuracy.

30. Quality-aware score target rejection (2026-08-06)

Matched detections were temporarily assigned a continuous 0.5--1.0 target
based on IoU rather than the existing binary positive target. Three real
30-second runs completed in 27.77--27.91 seconds and reached full-evaluator
mAP50:95 0.1958, 0.1909 and 0.1921 for seeds 7, 2 and 3; the median 0.1921
was below the promoted 0.1927 median. The binary target was restored, and
score-target tuning is no longer the active research gate.

31. OpenMP kernel rejection (2026-08-06)

An optional OpenMP convolution path appeared faster, but repeated native
single-image inference was nondeterministic under the parallel build. The
change was removed after failing the parity gate; the scalar C kernel remains
the verified runtime until a race-free parallel implementation is measured.

The retained seed-7 class-and-scale checkpoint was rechecked after the
reversion on all 250 validation images and reproduced AP50=0.3081 and
mAP50:95=0.1927. This remains the current verified native Adapt baseline.

32. Geometry quality readout rejection (2026-08-06)

A zero-initialized class-conditioned C readout over candidate area, position,
width and height was measured as a ranking improvement. Its real 30-second
run processed 64 images, but full validation fell to AP50=0.3027 and
mAP50:95=0.1885. The readout and temporary checkpoint-version extension were
removed; ranking adaptation is no longer the active gate.

33. Base FP32 native graph path (2026-08-06)

The base graph now has a genuine C FP32 forward and post-processing route
through its loaded full-precision sidecars; the compact integer route remains
available. It compiled and passed all existing tests, then completed a real
30-second seed-7 adaptation over 72 images in 27.94 seconds. Full validation
reached AP50=0.3081 and mAP50:95=0.1921, slightly below the retained scalar
integer baseline of 0.1927. The FP32 route is retained as a valid precision
option, but it is not promoted as an accuracy improvement; the stronger
representation/trainable-graph gate remains open. An eval-only run of the
retained seed-7 checkpoint through FP32 reproduced AP50=0.3081 and
mAP50:95=0.1927, so the small probe gap is a training-trajectory difference,
not a forward or post-processing parity failure.

34. FP32 final class-head weight rejection (2026-08-06)

A bounded C experiment used graph class-feature tiles with center-cell
positives and hard negatives to update the final FP32 class convolutions. It
completed 66 real samples in 27.75 seconds, but full validation fell to
AP50=0.3081 and mAP50:95=0.1916 versus the retained 0.1927 baseline. The
feature retention and weight-update path was removed; the validated profile
still updates only the existing native adaptation parameters.

35. FP32 DFL regression-bias rejection (2026-08-06)

A direct C FP32 DFL regression-bias experiment used the same native forward
pass and real ground-truth center-cell geometry. It completed 64 samples in
27.89 seconds, but full validation reached AP50=0.3080 and mAP50:95=0.1917,
below the retained 0.1927 baseline. The regression-bias path was removed and
the native adaptation profile remains unchanged.

36. 128x128 resolution rejection (2026-08-06)

A 128x128 INT8 native adaptation processed 95 real samples in 27.66 seconds,
but full validation fell to AP50=0.2715 and mAP50:95=0.1532 versus the
160x160 baseline. The lower-resolution profile was rejected; 160x160 remains
the retained operating point.

37. Raw-grid class-bias rejection (2026-08-06)

The retained 160x160 profile was given a C raw-grid supervision experiment:
ground-truth center cells were assigned to output scales and existing class
biases were updated from hard positives and negatives without another forward
pass. It processed 62 real samples in 27.69 seconds, but full validation
reached AP50=0.3080 and mAP50:95=0.1917, below 0.1927. The instrumentation
was removed and the deterministic validated adapter remains.

38. Full planned KSHIRA switch combination (2026-08-06)

The exact planned trainable combination was measured on real data: 32
features, space-preserving stem, shared multi-scale/context fusion, dense
auxiliary budget 8, quality alignment, adaptive/residual budgeting, 4-second
crop bootstrap, and analytic readout. With a 16 MiB arena it completed 179
samples in 30.06 seconds, but full validation produced 7,532 predictions,
7,512 false positives, and mAP50:95=0.0000. The switches are implemented and
serializable, but this current scratch representation is rejected as an
accuracy profile; the native Adapt baseline remains strongest.

39. Native NMS threshold sweep (2026-08-06)

Class-aware NMS thresholds 0.35, 0.45 and 0.55 were measured on the retained
checkpoint. Full validation mAP50:95 was 0.1923, 0.1927 and 0.1918; the
existing 0.45 setting remains the validated default.

40. Native target-domain crop phase (2026-08-06)

The native C Adapt path now exposes the existing deterministic tight/context
crop stream as an opt-in training phase. Crops update only the owned score,
class/scale, and localization adapters around the frozen deployment graph;
they do not alter the default recipe or introduce an external runtime. A
real 30-second INT8 probe with a 4-second crop budget processed 12 crop
samples and 65 full images. On the 126-image test split it reached
mAP50:95=0.2254 versus 0.2242 for the same-checkpoint no-crop control. The
seed-2 repeat reached 0.2246 versus 0.2251 for its no-crop control. The
signal is not repeatable, so crop adaptation remains an opt-in research path
and is rejected for promotion; the default native profile is unchanged.

41. Correct native padding and int8 throughput (2026-08-06)

The native int8 reference convolution had a stale horizontal-padding bounds
bug. The generic path now guards both spatial dimensions, and exact dense and
depthwise 3x3 fast paths were added in C. Fast and corrected-reference graph
detections agree exactly on a real image; the focused C regression test covers
the padded case. The fast path measured 3.12 images/s versus 2.16 images/s for
the corrected reference. A fresh corrected-padding 30-second run processed 92
real images in 27.89 seconds and reached mAP50:95=0.2070 on the 126-image
test split. Old checkpoints trained against the stale kernel are not used as
comparisons.

42. Resolution and learning-rate checks after the correction (2026-08-06)

A 192x192 run processed 63 images and reached mAP50:95=0.1948, below the
corrected 160x160 result. A learning rate of 0.05 reached 0.2052 versus
0.2070 at 0.02. Both are rejected; 160x160 and learning_rate=0.02 remain the
current measured operating point.

43. Activation-scale selection and precision parity (2026-08-06)

A fixed real-data screen selected the 0.0125 native activation-scale profile:
it reached mAP50:95=0.1379 on 20 images versus 0.0872 for 0.025. The 0.00625
profile was rejected after full adaptation reached 0.2117. Two 30-second
0.0125 adaptations processed 91 and 94 real images and reached 250-image
validation mAP50:95=0.2259 and 0.2256, with identical AP50=0.3617 and
208 TP / 606 FP / 246 FN. FP32 evaluation of the seed-7 checkpoint reproduced
the INT8 metrics exactly, at about 293 ms/image. The 0.0125 profile is the
strongest retained native Adapt candidate; Scratch and full trainable-graph
gates remain open.

44. C-only FP32 sidecar reconstruction and repeatability (2026-08-06)

The native C loader now reconstructs absent FP32 sidecars from the packaged
integer weights, biases, multipliers, shifts, and activation scale. Checkpoint
reload now expands the native arena before the first FP32 forward when those
sidecars are present. Release CTest and the native lifecycle round-trip pass.

With the base graph and a 30,000 ms real-data budget, the C FP32 adaptation
processed 85 images in 28.00 s at seed 7 and reached AP50=0.3813 /
mAP50:95=0.2583; seed 2 processed 91 images in 27.93 s and reached
AP50=0.3813 / mAP50:95=0.2557 on the locked 250-image validation manifest.
This is the strongest current C-side profile, but it is still Adapt over a
loaded graph, not full scratch training.

The opt-in feature-map class readout was measured and rejected for promotion:
its conservative blend reached AP50=0.3303 / mAP50:95=0.2103 on a 5-second
screen, below the untouched FP32 baseline. It remains an explicit research
ablation; the default path is unchanged.

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
