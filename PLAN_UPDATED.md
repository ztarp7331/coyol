# KSHIRA Updated
PLEASE FOLLOW THIS , YOU HAVE DONE SOME NEW IMPLEMENTATION TRY TO FIT THIS ALL IN WITHIN THAT AS WELL U SHOULD UPDATE THE CODE WHICH NEEDS TO BE CHANGED WHEREVER. WE NEED TO ACHIEVE VERY FAST TRAINING SPEED WITH AS CLOSE TO YOLO ACCURACY POSSIBLE
## Contrast-Modulated Object Detection Under Extreme Memory Constraints

### A Dataset-Neutral Object Detection Framework for Arena-Bounded Training and Deployment in ISO C

---

## Executive Summary

KSHIRA is a compact object-detection architecture and training framework designed for environments where conventional deep-learning systems are impractical.

The framework targets:

* implementation in ISO C;
* training from randomly initialized weights;
* dataset-neutral image and bounding-box input;
* caller-owned, statically bounded memory;
* FP32, INT8 and INT4 execution;
* low-clock CPU operation;
* future FPGA lowering;
* deterministic training and inference;
* no dynamic allocation inside training or inference loops;
* direct bounding-box and class prediction;
* fixed-size top-K deployment;
* sub-second training as a falsifiable research target.

KSHIRA does not attempt to reproduce a conventional YOLO network at a smaller scale. Its purpose is to explore a different detector design that is co-developed with its training algorithm, memory system, quantization method and deployment hardware constraints.

The updated KSHIRA architecture is built around five core mechanisms:

1. **Receptive-Field Aggregation**
2. **Pointwise Channel Mixing**
3. **Contrast-Modulated Objectness**
4. **Task-Aligned Assignment**
5. **Block-Local Sparse Training**

The central hypothesis is that a very small detector can improve its spatial decision-making when it combines:

* learned semantic features;
* explicit local feature contrast;
* localization-aware target assignment;
* carefully balanced objectness supervision;
* spatially bounded backward computation.

KSHIRA retains a learned objectness pathway. Contrast is not treated as a complete replacement for semantic objectness. Instead, contrast becomes an explicit feature that the learned detection head may use, suppress or ignore depending on the data.

The framework is therefore neither a fixed-feature detector nor a handcrafted saliency system. All backbone, channel-mixing and detection-head parameters remain trainable from random initialization.

---

# Part I — Design Objective

## 1. Primary Goal

KSHIRA shall provide a complete object-detection lifecycle in C:

1. define a detector architecture;
2. initialize it without pretrained weights;
3. stream arbitrary images, classes and bounding boxes;
4. train the model;
5. validate the model;
6. perform inference;
7. save and reload training state;
8. export deployment-oriented INT8 and INT4 representations;
9. report timing, memory and accuracy;
10. support future lowering to FPGA-compatible operators.

The framework must remain dataset-neutral. The detector must not depend internally on:

* a particular dataset directory structure;
* COCO JSON;
* YOLO text files;
* a fixed number of classes;
* one particular image format;
* hard-coded vehicle classes;
* synthetic training patterns.

Dataset adapters are responsible for supplying normalized image tensors, boxes and class identifiers.

## 2. Reference Workload

The primary research workload is:

* 5,000 unique labelled images;
* 160 × 160 input resolution;
* one to four input channels;
* up to 80 object classes;
* configurable numbers of boxes per image;
* one streaming pass from random initialization;
* axis-aligned bounding boxes;
* Release compilation;
* deterministic seeds;
* bounded memory;
* no hidden pretrained features.

Multi-epoch experiments may be used for architecture discovery and convergence analysis, but they do not replace the official one-pass qualification.

## 3. Memory Contract

KSHIRA operates inside a caller-provided arena.

The arena contains the memory required for:

* trainable parameters;
* persistent quantization information;
* activation buffers;
* contrast buffers;
* local backward workspace;
* temporary parameter deltas;
* phase state;
* calibration information;
* fixed-size detection candidates.

The target compact profile shall fit within:

[
256\ \text{KiB}=262,144\ \text{bytes}.
]

Every build must report:

* arena capacity;
* arena high-water mark;
* parameter bytes;
* activation bytes;
* workspace bytes;
* calibration bytes;
* checkpoint bytes;
* unused headroom.

An architecture is not admitted when its conservative memory plan exceeds the supplied arena.

## 4. Execution Contract

KSHIRA shall not allocate memory inside:

* the per-image training loop;
* convolution kernels;
* box assignment;
* contrast computation;
* prediction;
* top-K selection;
* quantization;
* evaluation.

Temporary memory must be:

* statically planned;
* drawn from the caller-owned arena;
* stack-local only when its size is small and fixed;
* reusable between mutually exclusive operations.

## 5. Claim Boundaries

KSHIRA is a YOLO replacement by workflow capability, not by source compatibility or checkpoint compatibility.

It aims to provide the same broad lifecycle:

* architecture definition;
* from-scratch training;
* bounding-box detection;
* class prediction;
* validation;
* quantized deployment;
* model persistence.

It does not initially claim:

* COCO-scale YOLO accuracy;
* segmentation;
* pose estimation;
* tracking;
* open-vocabulary detection;
* oriented boxes;
* automatic mixed-precision GPU training;
* one-watt operation before board-level measurement;
* sub-second training before the complete timing and accuracy gates pass.

---

# Part II — Architectural Motivation

## 6. Current Learning Behaviour

The compact KSHIRA detector has demonstrated that its box-regression pathway can sometimes produce meaningful localization. In the strongest measured five-class real-image experiment, matched predictions reached mean IoU of approximately 0.62.

However, the same run produced:

* 18 true positives;
* 890 predictions;
* precision of approximately 0.020;
* recall of approximately 0.040;
* poorly separated confidence scores.

The measured real-data training-plus-decode time was approximately 14.57 seconds for the five-epoch experiment, while the bounded arena and inference-latency checks passed.

These results indicate that the primary failure is not exclusively box geometry. The detector has difficulty answering:

> Which spatial locations contain meaningful objects?

Two architectural causes are central.

## 7. Depthwise Representation Bottleneck

The original RAD encoder uses depthwise spatial convolutions.

Depthwise convolutions are efficient because they process each channel independently. However, depthwise operations alone cannot learn combinations between channels.

For example, the detector cannot directly learn that:

* one channel responds to a vertical boundary;
* another responds to a horizontal transition;
* a third responds to texture;
* their combination is characteristic of a vehicle.

Without learned channel interaction, the fused feature vector remains weak even when individual channels respond to useful image patterns.

## 8. Objectness Sparsity

Dense detection produces a severe positive-to-negative imbalance.

For a 40 × 40 feature map:

[
40\times40=1,600\text{ cells}.
]

With only a few objects in an image, fewer than one percent of cells are usually positive. A naïve binary loss may therefore be dominated by background cells.

The detector may reduce its average loss by producing low objectness almost everywhere. This creates:

* flat confidence maps;
* weak threshold separation;
* low recall;
* unstable hard-negative training;
* objectness collapse during longer training.

KSHIRA Updated addresses this through representation improvement, balanced supervision and bounded hard-negative selection.

---

# Part III — Complete Architecture

## 9. Architectural Overview

The primary KSHIRA path is:

```text
Input image
    ↓
Spatial stem
    ↓
Parallel dilated depthwise branches
    ↓
Branch accumulation and average fusion
    ↓
Pointwise channel mixer
    ↓
Mixed semantic feature map
    ↓
Local contrast extraction
    ↓
Semantic channels + contrast channel
    ↓
Detection head
    ↓
Box distances + objectness + classes
    ↓
Fixed-size top-K predictions
```

For the reference profile:

* input size: 160 × 160;
* base feature resolution: 40 × 40;
* semantic channels: 8;
* contrast channels: 1;
* head input channels: 9.

## 10. Input Representation

The dataset adapter provides an input tensor with:

* width;
* height;
* number of channels;
* numeric precision;
* row stride;
* caller-owned memory.

The compact reference input is transformed to the configured 160 × 160 model layout.

Permitted input channel counts are:

* one channel;
* three channels;
* four channels.

No image decoder is embedded into the mathematical detector graph. Decoding and resizing belong to the input adapter but are included in end-to-end timing.

---

## 11. Spatial Stem

The stem converts the input image into the base feature map.

Reference behaviour:

* convolution: 3 × 3;
* output stride: 4;
* output resolution: 40 × 40;
* output channels: 8;
* activation: clipped ReLU or another quantization-safe bounded activation.

The stem is learned from random weights.

Its purpose is to:

* reduce spatial resolution;
* extract low-level edges and gradients;
* map different input channel counts into the fixed internal feature width;
* provide the input to all dilated branches.

---

## 12. Receptive-Field Aggregation

The stem output is processed by three parallel depthwise branches:

[
d\in{1,2,4}.
]

Each branch uses:

* depthwise 3 × 3 convolution;
* identical channel count;
* fixed feature resolution;
* independent trainable weights.

The branches provide three receptive-field scales:

* local detail;
* medium-range spatial context;
* broader context.

The branch outputs are fused by accumulation:

\frac{F_{d=1}+F_{d=2}+F_{d=4}}{3}.
]

The implementation need not materialize all three complete branch maps simultaneously.

A memory-efficient execution schedule is:

1. clear the fused map;
2. compute one branch;
3. accumulate it into the fused map;
4. reuse the branch workspace;
5. repeat for the remaining branches;
6. apply the final scale factor.

This keeps the architecture logically parallel while avoiding three persistent full-resolution maps.

---

## 13. Pointwise Channel Mixer

The fused RAD map is passed through an 8-to-8 pointwise convolution:

[
M_x=W_{\text{mix}}F_{\text{fused},x}+b_{\text{mix}}.
]

Where:

[
W_{\text{mix}}\in\mathbb{R}^{8\times8},
\qquad
b_{\text{mix}}\in\mathbb{R}^{8}.
]

Parameter cost:

[
8\times8+8=72\text{ parameters}.
]

FP32 parameter memory:

[
72\times4=288\text{ bytes}.
]

Compute per spatial cell:

[
8\times8=64\text{ multiply-accumulate operations}.
]

Across the complete 40 × 40 map:

[
1,600\times64=102,400\text{ MACs}.
]

The pointwise mixer provides the channel interaction missing from a purely depthwise encoder.

It allows the detector to learn:

* combinations of edge directions;
* texture and boundary conjunctions;
* channel suppression;
* channel amplification;
* class-relevant feature mixtures;
* features that are useful for localization but not classification;
* features that are useful for objectness but not localization.

### 13.1 Memory-Safe Mixer Execution

The mixer may operate cell by cell.

For each cell:

1. load the eight fused input values;
2. compute eight mixed output values into a temporary vector;
3. write the eight results back to the feature map.

The temporary vector requires eight values and prevents in-place overwrite from corrupting later output channels.

No second 40 × 40 × 8 map is required.

---

# Part IV — Contrast-Modulated Objectness

## 14. Purpose of Contrast Modulation

The mixed semantic feature map contains learned information, but the detector may still have difficulty learning spatial objectness from extremely sparse positives.

KSHIRA therefore computes an explicit local contrast feature.

The contrast feature does not directly decide whether a cell contains an object.

Instead, it answers:

> How different is this cell’s learned feature vector from its immediate spatial context?

The detection head receives this value as an additional input and learns whether it is useful for the current dataset.

## 15. Local Feature Mean

Let:

[
M_x\in\mathbb{R}^{C}
]

be the mixed feature vector at cell (x), with (C=8).

Let (\mathcal N_r(x)) be the local spatial neighbourhood around (x), with radius:

[
r=2.
]

The nominal neighbourhood is therefore 5 × 5.

The local mean is:

\frac{1}{N_x}
\sum_{j\in\mathcal N_r(x)}M_j,
]

where:

[
N_x=|\mathcal N_r(x)|.
]

At boundaries, (N_x) is reduced to the number of valid cells. No implicit zero-padding is used in the default contrast definition.

## 16. Raw Spatial Contrast

The raw contrast is:

\sum_{c=1}^{C}
\left(M_{x,c}-\bar M_{x,c}\right)^2.
]

Properties:

* (\kappa_x\ge0);
* uniform regions produce low values;
* feature discontinuities produce larger values;
* the signal depends on learned features rather than raw pixels;
* the signal remains class-agnostic until interpreted by the head.

## 17. Robust Contrast Transform

Raw contrast may have a long-tailed distribution because of:

* sharp illumination boundaries;
* image noise;
* highly textured background;
* saturated feature channels;
* quantization artifacts.

KSHIRA applies:

[
C_x=\log(1+\kappa_x).
]

This provides:

* non-negative output;
* strong resolution near zero;
* compression of large outliers;
* stable percentile calibration;
* reduced domination by extreme background texture.

## 18. Deployment-Compatible Contrast Transform

The deployment path must not depend on a general-purpose floating-point logarithm.

KSHIRA therefore defines two representations of the same monotonic transform.

### 18.1 FP32 Reference Transform

During FP32 reference training and operator validation:

[
C_x^{F32}=\log(1+\kappa_x).
]

### 18.2 Integer Deployment Transform

The deployment profile uses a monotonic piecewise-linear approximation:

a_s\kappa_x+b_s,
]

where segment (s) is selected from a small fixed set of contrast intervals.

The default design uses a bounded number of segments, such as:

* 8 segments for the smallest profile;
* 16 segments for the reference profile.

Each segment stores:

* lower boundary;
* integer slope;
* integer intercept;
* output shift.

The approximation must satisfy:

1. monotonicity;
2. non-negative output;
3. bounded maximum output;
4. deterministic rounding;
5. no dynamic allocation;
6. bit-exact scalar and accelerated behaviour.

Quantization-aware training simulates the same piecewise transform so that the detection head sees deployment-representative contrast values.

A lookup-table implementation may also be admitted when its memory and latency are explicitly reported.

---

## 19. Head Input

The detection head receives:

[
H_x=
[M_{x,1},M_{x,2},\ldots,M_{x,8},C_x].
]

Thus:

[
H_x\in\mathbb{R}^{9}.
]

The contrast value is a feature, not a score.

The head is free to learn:

* positive contrast weight;
* negative contrast weight;
* near-zero contrast weight;
* class-dependent use of contrast through the semantic channels;
* interaction between contrast and learned channel combinations.

---

# Part V — Detection Head

## 20. Head Outputs

For (K) classes, the head predicts:

1. four non-negative box distances;
2. one objectness logit;
3. (K) class logits.

For the five-class reference experiment:

[
4+1+5=10\text{ outputs}.
]

The head parameter count is:

[
9\times10+10=100.
]

The previous eight-input head required:

[
8\times10+10=90.
]

The contrast feature therefore adds only:

[
10\text{ parameters}=40\text{ FP32 bytes}.
]

## 21. Bounding-Box Representation

At each cell (x), the head predicts:

[
(l_x,t_x,r_x,b_x).
]

These are distances from the cell location to the corresponding box boundaries.

The decoded box is:

[
B_x=
(x-l_x,\ y-t_x,\ x+r_x,\ y+b_x).
]

Distances must be constrained to non-negative values through:

* bounded activation;
* positive parameterization;
* clamping;
* or another quantization-safe monotonic mapping.

No distribution focal loss is required in the compact profile.

## 22. Learned Objectness

The head produces an objectness logit (z_x^{obj}).

The probability is:

[
O_x=\sigma(z_x^{obj}).
]

Objectness represents the detector’s learned estimate that the location contains a valid object prediction.

Contrast is not multiplied directly into objectness during deployment. It influences objectness through the learned head weights.

## 23. Class Prediction

For mutually exclusive classes, the default class probability is:

\frac{\exp(z_{x,k})}
{\sum_j\exp(z_{x,j})}.
]

The quantized implementation may use:

* bounded exponential lookup;
* integer softmax approximation;
* logit subtraction from the maximum;
* fixed-point reciprocal approximation.

The same approximation must be validated against the FP32 reference.

## 24. Deployment Score

For class (k):

[
S_{x,k}=O_xP_{x,k}.
]

The predicted class is:

[
\hat k_x=\arg\max_kP_{x,k}.
]

The location score is:

[
S_x=O_x\max_kP_{x,k}.
]

This scoring function avoids the previous squared-objectness suppression:

[
O_x^2P_{x,k}.
]

The deployed score uses one final multiplication after objectness and class confidence are available.

---

# Part VI — Multi-Scale Detection

## 25. Single Stored Map, Multiple Logical Scales

To preserve the arena, KSHIRA stores one mixed semantic map at 40 × 40.

Additional logical scales are produced from that map without allocating complete additional feature pyramids.

The reference logical levels are:

* level 0: 40 × 40;
* level 1: 20 × 20;
* level 2: 10 × 10.

Pooled views are generated on demand.

## 26. Pooled Feature Views

For level 1:

\operatorname{pool}_{2\times2}
M^{(0)}.
]

For level 2:

\operatorname{pool}_{2\times2}
M^{(1)}.
]

The pooling operation may be:

* average pooling;
* maximum pooling;
* an integer-friendly weighted average.

The selected operation must be fixed in the model specification.

## 27. Contrast at Each Scale

Contrast should be computed after scale pooling:

\bar M_x^{(s)}
\right|_2^2
\right).
]

This ensures that contrast is measured relative to the semantic resolution of each detection scale.

The scale views and their contrast values may be generated cell by cell during inference.

No complete 20 × 20 × 8 or 10 × 10 × 8 persistent map is required.

## 28. Shared Head

The compact profile uses one shared nine-input head across scales.

Scale-specific adaptation is provided through small affine parameters:

[
z^{(s)}=g_s\odot z+b_s.
]

This avoids three complete independent heads while allowing:

* scale-specific confidence calibration;
* scale-specific distance magnitudes;
* different response distributions.

## 29. Scale Assignment

Ground-truth objects are assigned to scales according to their dimensions in input pixels.

A configurable rule maps:

* small objects to the high-resolution level;
* medium objects to the middle level;
* large objects to the low-resolution level.

Near scale boundaries, an object may temporarily supervise two adjacent levels during training.

Tiny objects must always retain at least one positive candidate on the highest-resolution level.

---

# Part VII — Task-Aligned Assignment

## 30. Assignment Objective

Assignment determines which feature cells supervise a ground-truth object.

KSHIRA avoids:

* one fixed centre cell;
* unrestricted all-inside-box assignment;
* contrast-only assignment;
* prediction-IoU-only assignment.

The assignment combines:

1. spatial centrality;
2. learned feature contrast;
3. predicted localization quality.

## 31. Candidate Pool

For each ground-truth box (G), define candidate cells whose centres satisfy:

* the cell centre lies inside (G);
* the cell belongs to an eligible detection scale;
* the cell lies within a bounded centre radius.

For a box with centre ((c_x,c_y)), candidate distance is:

[
d_x^2=(x-c_x)^2+(y-c_y)^2.
]

The radius may be derived from box dimensions and bounded by configured minimum and maximum values.

## 32. Centre Prior

To avoid exponential computation, KSHIRA uses a bounded polynomial prior:

\max
\left(
0,
1-\frac{d_x^2}{R_G^2+\epsilon}
\right).
]

Properties:

* maximum at the ground-truth centre;
* monotonic decrease with distance;
* zero outside the configured radius;
* no transcendental functions;
* deterministic scalar implementation.

## 33. Contrast Normalization

Contrast is normalized inside the candidate pool:

\frac{C_x}
{\max_{j\in\mathcal P_G}C_j+\epsilon}.
]

Candidate-local normalization avoids allowing one unrelated high-contrast background region to suppress all candidates.

## 34. Localization Quality

For each candidate:

[
Q_x=\operatorname{IoU}(B_x,G).
]

The IoU used for assignment is treated as a detached selection signal. Assignment does not backpropagate through the discrete top-k decision.

## 35. Staged Assignment Score

Assignment changes progressively during the training stream.

The general score is:

P_{\text{center}}(x,G)
\left[
a_t+(1-a_t)\widetilde C_x
\right]
\left[
b_t+(1-b_t)Q_x
\right].
]

### 35.1 Bootstrap Stage

During the first portion of training:

[
a_t=1,\qquad b_t=1.
]

Therefore:

[
A_t=P_{\text{center}}.
]

The detector begins with deterministic centre-prior assignment while its features and boxes are still random.

### 35.2 Contrast Introduction Stage

After initial feature formation:

[
0<a_t<1,\qquad b_t=1.
]

Contrast begins influencing assignment, while unreliable predicted IoU is ignored.

### 35.3 Localization-Aligned Stage

Once the box head produces non-degenerate predictions:

[
0<a_t<1,\qquad0<b_t<1.
]

Assignment is influenced by:

* centre proximity;
* feature distinctiveness;
* predicted localization quality.

### 35.4 Mature Stage

Near the later part of the stream:

[
a_t\rightarrow0,\qquad b_t\rightarrow0.
]

The full task-aligned product is used:

[
A_t
\approx
P_{\text{center}}\widetilde C_xQ_x.
]

## 36. One-Pass Scheduling

The schedule is based on sample progress, not epoch count.

For 5,000 samples, a possible default is:

| Stream range | Assignment behaviour               |
| ------------ | ---------------------------------- |
| 0–10%        | Centre prior only                  |
| 10–35%       | Centre prior + increasing contrast |
| 35–65%       | Introduce predicted IoU            |
| 65–100%      | Full task-aligned assignment       |

This schedule remains configurable and must be ablated.

## 37. Positive Count

For each ground-truth box:

\operatorname{clamp}
\left(
K_{\min}
+
f(\text{box area}),
K_{\min},
K_{\max}
\right).
]

Reference bounds may be:

* minimum: one positive;
* maximum: nine positives.

Small objects receive fewer, carefully protected positives. Larger objects may receive more spatial supervision.

## 38. Deterministic Tie-Breaking

When two candidates have equal assignment scores, priority is determined by:

1. smaller squared distance to GT centre;
2. higher predicted IoU;
3. higher contrast;
4. lower row-major cell index.

## 39. Multi-Ground-Truth Conflict

When one cell is selected by multiple objects:

1. retain the object with highest assignment score;
2. if tied, retain the object with smaller centre distance;
3. if still tied, retain the smaller ground-truth area;
4. if still tied, retain the lower annotation index.

Every ground-truth object must receive at least one positive. A centre-cell fallback is used when conflict resolution removes all candidates.

## 40. Assignment Weights

Positive losses may be weighted using normalized assignment quality:

w_{\min}
+
(1-w_{\min})
\frac{A_t(x,G)}
{\max_{j\in P_G}A_t(j,G)+\epsilon}.
]

The positive floor (w_{\min}) prevents low-ranked assigned cells from receiving negligible gradients.

---

# Part VIII — Training Objectives

## 41. Total Detection Loss

The total loss is:

\lambda_{\text{box}}\mathcal L_{\text{box}}
+
\lambda_{\text{obj}}\mathcal L_{\text{obj}}
+
\lambda_{\text{cls}}\mathcal L_{\text{cls}}.
]

The loss weights may change over the training stream, but no epoch-dependent behaviour appears in the deployed graph.

---

## 42. Box Loss

The box loss combines distance regression and decoded IoU:

\frac{1}{|P|}
\sum_{x\in P}
w_x
\left[
\lambda_d
\operatorname{SmoothL1}
(\hat d_x,d_x)
+
\lambda_i
(1-\operatorname{IoU}(B_x,G_x))
\right].
]

Where:

* (\hat d_x) is the predicted distance vector;
* (d_x) is the target distance vector;
* (B_x) is the decoded predicted box;
* (G_x) is the assigned ground-truth box.

Distance targets should be normalized by the scale stride or configured distance range.

## 43. Classification Loss

For one class per object:

-\frac{1}{|P|}
\sum_{x\in P}
w_x,
\omega_{k_x}
\log P_{x,k_x}.
]

The class weight is based on streamed class frequency.

A stable reference weighting is:

\operatorname{clamp}
\left(
\sqrt{
\frac{\bar f+\epsilon}
{f_k+\epsilon}
},
\omega_{\min},
\omega_{\max}
\right).
]

The square root prevents extreme rare-class amplification.

Class-frequency estimates are:

* supplied by the adapter when known;
* or maintained as deterministic streaming counts.

---

## 44. Objectness Targets

For an assigned positive cell:

\operatorname{stopgrad}
\left[
\operatorname{clamp}
(\operatorname{IoU}(B_x,G_x),q_{\min},1)
\right].
]

This creates localization-aware objectness.

A well-localized prediction receives a larger target than a poor box at the same location.

During early bootstrap, when predicted IoU is unreliable, the target may be blended with one:

\eta_t
+
(1-\eta_t)
\operatorname{IoU}(B_x,G_x).
]

The coefficient (\eta_t) decreases during training.

For negative cells:

[
y_x^{obj}=0.
]

---

## 45. Balanced Focal Objectness

The objectness loss is:

\mathcal L_{\text{pos}}
+
\lambda_n\mathcal L_{\text{neg}}.
]

Positive term:

*

\frac{1}{|P|}
\sum_{x\in P}
w_x
|y_x^{obj}-O_x|^\gamma
\left[
y_x^{obj}\log O_x+
(1-y_x^{obj})\log(1-O_x)
\right].
]

Negative term:

*

\frac{1}{|N_s|}
\sum_{x\in N_s}
O_x^\gamma\log(1-O_x).
]

Where:

* (P) is the positive set;
* (N_s) is the bounded sampled-negative set;
* (\gamma) is the focal exponent;
* (\lambda_n) controls total negative strength.

Positive and negative terms are normalized independently. This prevents the number of background cells from automatically determining the gradient magnitude.

## 46. Hard-Negative Mining

Negative candidates must:

* lie outside all ground-truth boxes;
* not overlap an ignored region;
* not be selected positives;
* have valid feature support.

A bounded number of negatives are selected according to learned objectness:

[
N_s=\operatorname{topM}_{x\in N}O_x.
]

A reference ratio is:

\operatorname{clamp}
(r_n|P|,N_{\min},N_{\max}).
]

Possible starting values:

* (r_n=3) or (5);
* fixed minimum for empty images;
* bounded maximum to protect training time.

Hard-negative mining begins gradually. Early training uses a mixture of deterministic spatial negatives and predicted hard negatives so that random initial outputs do not dominate selection.

---

# Part IX — Exact Contrast Gradient

## 47. Contrast Dependency

Each mixed feature value participates in:

* its own contrast calculation;
* the local mean of neighbouring contrast calculations.

Therefore, contrast backpropagation must accumulate all contributions.

Define:

\frac{\partial\mathcal L}{\partial C_y}.
]

Because:

[
C_y=\log(1+\kappa_y),
]

then:

\frac{1}{1+\kappa_y}.
]

For:

\sum_c(M_{y,c}-\bar M_{y,c})^2,
]

the complete feature derivative is:

\sum_{y:,x\in\mathcal N(y)}
g_y^C
\frac{2(M_{y,c}-\bar M_{y,c})}
{1+\kappa_y}
\left[
\mathbf 1(x=y)-\frac{1}{N_y}
\right]
}
]

where:

* (\mathbf 1(x=y)=1) when (x=y);
* otherwise it is zero;
* (N_y) is the valid neighbourhood size for cell (y).

This formulation handles:

* interior cells;
* image boundaries;
* direct contributions;
* indirect mean contributions;
* multiple overlapping contrast windows.

## 48. Gradient Through Deployment Approximation

During quantization-aware training, the forward contrast transform uses the deployment approximation.

The backward path may use:

* the exact segment slope;
* a straight-through derivative;
* or the derivative of the FP32 `log1p` reference.

The chosen method must be explicitly recorded in the model’s training configuration and tested through ablation.

The default should use the actual piecewise-linear segment slope so the training gradient matches deployment behaviour.

---

# Part X — Block-Local Sparse Training

## 49. Objective

KSHIRA avoids full-image backward propagation.

The forward pass may produce the complete base feature map because inference and hard-negative ranking require spatial predictions.

The backward pass is restricted to regions that can affect:

* assigned positive cells;
* selected hard-negative cells;
* optional local calibration cells.

This reduces backward convolution work and activation retention.

## 50. Dependency Radius

For one supervised base-map cell:

* contrast radius: two cells;
* maximum RAD dilation radius: four cells.

Therefore, the required base dependency radius is:

[
r_{\text{dep}}=2+4=6.
]

The minimum dependency region is:

13\times13.
]

Thus, the updated KSHIRA local training unit is not a 9 × 9 tile.

It is a **13 × 13 dependency tile** for one central supervised output cell.

## 51. Tile Union

Multiple assigned cells may have overlapping dependency tiles.

KSHIRA merges overlapping tiles deterministically.

The process is:

1. generate one dependency rectangle per supervised cell;
2. clip each rectangle to map boundaries;
3. sort rectangles by row-major origin;
4. merge rectangles whose overlap exceeds a configured threshold;
5. reject merges that exceed the maximum tile workspace;
6. process the resulting tile list in deterministic order.

This prevents repeated computation when positives are spatially close.

## 52. Large Tile Handling

If merged tiles exceed the available workspace:

* split them into bounded strips;
* preserve overlap halos;
* accumulate parameter deltas transactionally;
* commit only after all strips complete with finite values.

## 53. Tile Backward Sequence

For each tile:

1. reconstruct or load the required stem features;
2. compute required dilated branch values;
3. accumulate the fused RAD output;
4. apply pointwise mixing;
5. compute contrast dependencies;
6. compute local head gradients;
7. backpropagate through the contrast channel;
8. backpropagate through the mixer;
9. backpropagate through fused branches;
10. backpropagate through the stem;
11. accumulate parameter deltas;
12. verify finite deltas;
13. commit transactionally.

The implementation need not retain a global reverse-mode activation tape.

## 54. Head and Encoder Update Frequency

The head receives updates for every supervised positive and sampled negative.

The encoder receives updates through the dependency tiles.

A deterministic coverage scheduler ensures that:

* all stem channels receive updates;
* all branch channels receive updates;
* all pointwise mixer outputs receive updates;
* no channel remains permanently inactive because of sparse selection.

## 55. Sparse Channel Scheduling

Spatial tile sparsity may be combined with a rotating channel mask.

The channel mask must be:

* deterministic;
* independent of instantaneous gradient magnitude;
* guaranteed to cover every trainable channel during the reference stream;
* explicitly included in the training report.

The official full-update experiment must also be retained as a correctness and convergence reference.

## 56. Transactional Updates

Parameter updates are committed only when:

* all tile calculations complete;
* every gradient is finite;
* every proposed parameter delta is finite;
* quantized scales are valid;
* the arena remains within bounds.

On failure:

* no partial parameter update is committed;
* the caller receives an explicit failure status;
* the model remains in its previous valid state.

---

# Part XI — Training Phases

## 57. PRE Phase

The PRE phase establishes stable initial representations.

Typical behaviour:

* FP32 arithmetic;
*
* centre-prior assignment;
* no predicted-IoU assignment;
* limited contrast influence;
* broad channel coverage;
* conservative hard-negative sampling;
* full or high-density local updates.

## 58. TRAIN Phase

The TRAIN phase performs the main learning process.

Behaviour includes:

* contrast-modulated head input;
* staged task-aligned assignment;
* localization-aware objectness targets;
* bounded hard-negative mining;
* class balancing;
* block-local backward;
* optional quantization-aware arithmetic;
* learning-rate decay;
* measured tile statistics.

## 59. ODT Phase

The On-Device Training phase performs limited adaptation after deployment.

ODT may permit:

* bias-only updates;
* mixer updates;
* head updates;
* selected channel updates;
* bounded scale-head adaptation;
* calibration refresh.

ODT may reject:

* unrestricted full-image global backpropagation;
* architecture changes;
* arena resizing;
* unsupported precision transitions.

---

# Part XII — Quantization

## 60. Precision Modes

KSHIRA supports:

* FP32 reference;
* INT8;
* INT4.

Quantized modes must use real integer arithmetic in the forward path rather than merely storing quantized values and converting everything back to FP32.

## 61. Weight Quantization

Weights use symmetric quantization.

For INT8:

\operatorname{clamp}
\left(
\operatorname{round}(w/s_w),
-127,
127
\right).
]

For signed INT4:

\operatorname{clamp}
\left(
\operatorname{round}(w/s_w),
-7,
7
\right).
]

Per-output-channel scales are preferred for:

* stem convolution;
* depthwise branches;
* pointwise mixer;
* detection head.

## 62. Activation Quantization

Semantic activations use signed quantization because learned activations may include values on both sides of zero, depending on the selected activation function and folding strategy.

Activation scales are established through representative calibration and updated only through controlled phase transitions.

## 63. Contrast Quantization

Contrast is non-negative.

When stored in the existing signed formats, the compact profile uses:

* INT8 non-negative code range: (0\ldots127);
* INT4 non-negative code range: (0\ldots7).

These are non-negative values represented inside signed storage, not full unsigned 8-bit or unsigned 4-bit formats.

## 64. Streaming Percentile Calibration

KSHIRA must not store every contrast sample.

Percentiles are estimated using a bounded streaming method.

The default design uses a fixed-bin histogram.

Calibration sequence:

1. first pass determines a stable upper range;
2. second pass fills a bounded histogram;
3. cumulative counts identify the desired percentile;
4. the resulting clipping value is persisted;
5. histogram workspace is released or reused.

The histogram must reuse phase-exclusive workspace whenever possible.

## 65. Bit-Specific Contrast Ranges

Reference defaults:

* INT8 clipping percentile: (P_{99});
* INT4 clipping percentile: (P_{95}).

These are initial experimental values, not universal constants.

The quantized code is:

\operatorname{clamp}
\left(
\operatorname{round}
\left(
C\frac{Q_{\max}}{P}
\right),
0,
Q_{\max}
\right).
]

Where:

* (P=P_{99}), (Q_{\max}=127) for INT8;
* (P=P_{95}), (Q_{\max}=7) for signed-storage INT4.

The calibration study must compare:

* maximum-based scaling;
* percentile scaling;
* mean-plus-standard-deviation clipping;
* learned clipping;
* exact `log1p`;
* piecewise-linear `log1p`;
* clipped raw contrast.

## 66. Quantization-Aware Training

Quantization-aware training uses:

* quantized forward simulation;
* bit-specific activation clipping;
* persisted calibration scales;
* QAS or other bounded gradient scaling;
* straight-through treatment of rounding where required;
* transactional parameter updates.

The contrast path must simulate:

* the deployment contrast transform;
* the deployment clipping range;
* the deployment integer code range;
* the deployment dequantization scale.

## 67. Packed INT4 Deployment

The deployment model must eventually be independent of FP32 master weights.

A qualified packed INT4 model contains:

* packed weight nibbles;
* quantization scales;
* contrast transform parameters;
* head calibration information;
* architecture dimensions;
* class count;
* checksum;
* exact payload length;
* version identifier.

Training checkpoints may retain FP32 master state. Deployment blobs must be separately identifiable.

---

# Part XIII — Memory Plan

## 68. Parameter Count

For the five-class reference profile:

### Existing model

[
2,648\text{ parameters}.
]

### Pointwise mixer

[
72\text{ parameters}.
]

### Additional ninth-channel head weights

[
10\text{ parameters}.
]

### Updated total

2,730\text{ parameters}.
]

FP32 parameter bytes:

10,920\text{ bytes}.
]

## 69. Incremental Memory Estimate

Approximate additional requirements:

| Addition                              | Estimated bytes         |
| ------------------------------------- | ----------------------- |
| Pointwise mixer parameters            | 288                     |
| Expanded head parameters              | 40                      |
| Expanded parameter-delta workspace    | 328                     |
| Contrast percentile/scales            | 8–16                    |
| Piecewise contrast-transform metadata | Configuration-dependent |
| Alignment and descriptors             | Configuration-dependent |

The known minimum increase is approximately:

[
664\text{ to }672\text{ bytes},
]

before transform metadata and alignment.

## 70. Memory Reuse Requirements

To remain under 256 KiB:

* the mixer must operate cell by cell;
* no second semantic feature map may be retained;
* the contrast map must reuse an existing scalar-map buffer;
* calibration histograms must reuse phase-exclusive scratch;
* parameter delta workspace must be sized for 2,730 parameters;
* multi-scale maps must be virtual or cell-generated;
* tile buffers must remain bounded;
* model construction must reject insufficient arenas.

## 71. Hard Memory Gate

The final memory claim is not based on arithmetic estimates alone.

The builder must report:

[
\text{high-water}\le262,144.
]

The qualification record must include:

* FP32 profile;
* INT8 profile;
* INT4 profile;
* single-scale profile;
* multi-scale profile;
* training profile;
* inference-only deployment profile.

---

# Part XIV — Inference Pipeline

## 72. Inference Sequence

For one image:

1. validate input dimensions and channels;
2. normalize or quantize input;
3. compute stem map;
4. compute and fuse RAD branches;
5. apply pointwise channel mixing;
6. generate logical scale views;
7. compute local contrast;
8. apply deployment contrast transform;
9. concatenate semantic and contrast features logically;
10. execute shared detection head;
11. decode box distances;
12. compute objectness and class confidence;
13. apply per-class score;
14. update fixed-size top-K candidates;
15. apply configured duplicate-control policy;
16. return caller-owned detections.

## 73. Candidate Storage

Candidate count is fixed at model build time.

Each candidate stores:

* box coordinates;
* score;
* class;
* scale;
* source cell index.

No unbounded candidate list is created.

## 74. Duplicate Control

The desired deployment mode is learned one-to-one selection with bounded top-K output.

During transition and validation, an optional class-aware overlap suppressor may remain available.

Reports must clearly distinguish:

* pure top-K mode;
* learned one-to-one mode;
* top-K plus duplicate suppression.

The framework must not describe a suppressor-assisted result as fully NMS-free.

---

# Part XV — Correctness and Validation

## 75. Operator Correctness

The following operators require independent reference tests:

* stem convolution;
* dilated depthwise convolution;
* branch fusion;
* pointwise mixer;
* local mean;
* raw contrast;
* contrast transform;
* contrast backward;
* box decode;
* IoU;
* objectness loss;
* class loss;
* assignment;
* top-K;
* pooling;
* quantization;
* packed INT4 operations.

## 76. Gradient Validation

Finite-difference tests must cover:

* mixer weights;
* mixer biases;
* head weights connected to contrast;
* direct contrast gradient;
* indirect neighbourhood gradient;
* boundary cells;
* overlapping contrast windows;
* box regression;
* objectness;
* classification.

## 77. Dense-versus-Tile Parity

A small deterministic fixture must compare:

* dense backward;
* block-local backward.

For an identical supervised-cell set, parameter deltas must agree within the configured numerical tolerance.

Separate tests are required for:

* one positive;
* overlapping positive tiles;
* positive plus hard negative;
* boundary tile;
* multiple boxes;
* quantized path.

## 78. Quantized Validation

Tests must verify:

* scalar and accelerated paths are bit-exact;
* INT8 packing and unpacking;
* INT4 nibble packing;
* contrast clipping;
* percentile calibration;
* piecewise transform monotonicity;
* saturation handling;
* score stability;
* decoded-box parity;
* class agreement.

## 79. Persistence

Saving and loading must preserve:

* architecture;
* parameters;
* mixer;
* head;
* precision;
* calibration;
* phase state;
* sparse schedule;
* update counters;
* class count;
* multiscale configuration.

Reloaded inference must produce identical detections.

Corrupt, truncated or length-inconsistent model files must fail closed.

---

# Part XVI — Experimental Programme

## 80. Principle

Mechanisms must be introduced through controlled ablation.

No full KSHIRA Updated claim is made until the effect of each major change can be isolated.

---

## 81. Experiment A — Existing Control

Configuration:

* original RAD encoder;
* depthwise branches;
* no pointwise mixer;
* learned objectness;
* current assignment;
* current scoring.

Report:

* TP, FP and FN;
* precision;
* recall;
* F1;
* AP50;
* mAP50:95;
* mean IoU on matched boxes;
* score histogram;
* prediction count;
* training time;
* inference time;
* memory high-water.

---

## 82. Experiment B — Pointwise Mixer

Change:

* add the 8-to-8 pointwise mixer;
* keep all other training and scoring behaviour identical.

Measure:

* matched-box IoU;
* class-aware AP;
* class-agnostic localization recall;
* channel covariance;
* effective feature covariance rank;
* objectness separation;
* stem gradient magnitude;
* training-time increase;
* inference-time increase;
* arena increase.

Success means measurable improvement in useful detector metrics, not merely increased feature variance.

---

## 83. Experiment C — Contrast Diagnostic

Change:

* compute contrast;
* do not feed it into the head;
* do not change losses or assignment.

Analyse contrast for:

* ground-truth centre cells;
* ground-truth interior cells;
* ground-truth boundary cells;
* near-object background;
* hard background;
* random background.

Report:

* contrast histograms;
* median and percentiles;
* AUROC for object-region versus background-region discrimination;
* precision-recall AUC;
* object-size breakdown;
* per-image normalization sensitivity;
* before-training and after-training distributions.

This determines whether contrast contains useful information before it is allowed to affect training.

---

## 84. Experiment D — Hybrid Contrast Head

Change:

* append contrast as the ninth feature;
* retain learned objectness;
* use balanced objectness loss;
* keep centre-based assignment initially.

Report all detector metrics and compare with Experiment B.

Required evidence:

* improved score separation;
* no uncontrolled FP increase;
* stable objectness across thresholds;
* no loss of box quality;
* acceptable latency and memory cost.

---

## 85. Experiment E — Staged Task-Aligned Assignment

Change:

* introduce candidate-local contrast;
* introduce centre prior;
* gradually introduce predicted IoU;
* apply deterministic conflict resolution.

Compare:

* centre-only assignment;
* centre-plus-contrast;
* centre-plus-IoU;
* complete centre-plus-contrast-plus-IoU.

Report:

* positives per GT;
* unassigned GT count;
* per-size AP;
* early-training stability;
* assignment churn;
* class imbalance;
* convergence.

---

## 86. Experiment F — Block-Local Sparse Training

Change:

* preserve Experiment E architecture and losses;
* replace dense encoder backward with dependency-tile backward.

Measure:

* total tile count;
* average tile area;
* merged-tile ratio;
* recomputed cells;
* backward MACs;
* parameter-update count;
* train-core time;
* end-to-end time;
* memory;
* AP;
* precision and recall.

The success criterion is:

* materially lower measured train-core time;
* no unacceptable degradation in detector accuracy;
* no hidden full-map backward fallback;
* deterministic tile scheduling.

A nominal FLOP reduction is not sufficient. Wall-clock reduction must be measured.

---

## 87. Experiment G — Quantization

Run:

* FP32;
* INT8;
* INT4.

Compare:

* exact versus piecewise contrast transform;
* max versus percentile clipping;
* quantized objectness calibration;
* box IoU;
* class agreement;
* mAP loss;
* score saturation;
* training stability;
* inference latency;
* model size.

Qualification follows the detector-level metrics rather than TP count alone.

---

# Part XVII — Official Qualification

## 88. Real-Image Workload

The final model must run on:

* 5,000 unique real images;
* raw manifest input;
* one streaming pass;
* randomly initialized learned weights;
* separate validation data.

Repeated exposures of a smaller dataset do not count as 5,000 unique images.

## 89. Timing

Report:

### Train core

Prepared model tensors through:

* training;
* parameter updates;
* final serialization.

### Train end to end

Raw files through:

* image decode;
* resizing;
* label parsing;
* training;
* serialization.

Both values remain visible.

The targets are:

* stretch target: at most one second;
* secondary target: at most ten seconds.

A prepared-tensor result does not replace the end-to-end result.

## 90. Accuracy

Report:

* precision;
* recall;
* F1;
* AP50;
* AP75;
* mAP50:95;
* per-class AP;
* small-object AP;
* medium-object AP;
* large-object AP;
* class-agnostic localization recall;
* matched-box IoU;
* objectness calibration;
* prediction count.

## 91. Quantized Accuracy

Quantized validation must include decoded detections.

Targets include:

* INT8 within two mAP points of FP32;
* W4A8 within five mAP points of FP32;
* explicit INT4 KSHIRA threshold defined before qualification;
* class agreement;
* decoded-box IoU;
* no objectness collapse.

The official plan requires detector-level quantization validation rather than tensor similarity alone.

## 92. Reproducibility

The final result must report:

* three deterministic random seeds;
* median performance;
* result range;
* pinned CPU configuration;
* compiler and flags;
* thread count;
* cycles where available;
* peak memory;
* model size;
* dataset identity;
* unique-image count;
* exposure count;
* precision mode.

---

# Part XVIII — Research Contribution

## 93. Central Contribution

KSHIRA investigates whether useful object detection can be trained from random initialization under a hard memory and execution budget by co-designing:

* a compact receptive-field encoder;
* learned pointwise channel mixing;
* explicit local semantic contrast;
* task-aligned positive assignment;
* localization-aware objectness;
* balanced hard-negative supervision;
* dependency-tile backward propagation;
* real low-bit arithmetic;
* deterministic ISO-C execution.

## 94. Distinctive System Hypothesis

The strongest research hypothesis is not that any individual operation is entirely new.

The hypothesis is that their constrained composition can produce a useful detector under conditions where conventional detector training is unsuitable:

> A contrast-modulated, task-aligned detector can be trained from random weights using bounded local backward tiles and real low-bit arithmetic while remaining inside a fixed caller-owned ISO-C arena.

## 95. Required Evidence

The framework may be described as a validated research contribution only after controlled experiments demonstrate that:

1. pointwise mixing improves learned representation;
2. contrast contains useful object-region information;
3. the hybrid contrast channel improves detector calibration or accuracy;
4. staged assignment improves training stability or positive quality;
5. block-local training reduces actual wall-clock time;
6. quantized training retains detector accuracy;
7. the complete model fits the arena;
8. results repeat across seeds;
9. gains are not caused solely by changed threshold calibration.

---

# Part XIX — Implementation Order

## 96. Stage 1 — Representation

Implement and validate:

* pointwise mixer;
* in-place cellwise execution;
* mixer forward and backward;
* updated serialization;
* updated memory planner.

## 97. Stage 2 — Contrast Diagnostics

Implement and validate:

* neighbourhood mean;
* raw contrast;
* FP32 `log1p`;
* contrast statistics;
* contrast maps;
* object-region versus background diagnostics.

Do not yet change the head.

## 98. Stage 3 — Hybrid Head

Implement:

* ninth input channel;
* expanded head;
* exact contrast backward;
* balanced focal objectness;
* bounded negatives.

## 99. Stage 4 — Assignment

Implement:

* candidate generation;
* centre prior;
* contrast normalization;
* staged IoU introduction;
* deterministic conflict resolution;
* positive fallback.

## 100. Stage 5 — Local Backward

Implement:

* 13 × 13 dependency tiles;
* tile merging;
* tile splitting;
* transactional delta accumulation;
* dense-versus-local parity tests;
* measured compute reporting.

## 101. Stage 6 — Quantization

Implement:

* deployment-compatible contrast approximation;
* streaming percentile calibration;
* INT8 contrast path;
* INT4 contrast path;
* quantization-aware contrast training;
* packed deployment representation.

## 102. Stage 7 — Multi-Scale Qualification

Implement:

* virtual pooled views;
* shared head;
* scale calibration;
* scale-aware assignment;
* pooled-source local gradients;
* per-size validation.

## 103. Stage 8 — Product Qualification

Run:

* 5,000 unique images;
* one-pass training;
* raw end-to-end timing;
* HOG baseline;
* three seeds;
* F32/INT8/INT4 comparison;
* complete memory report;
* low-clock CPU profile;
* FPGA preparation.

---

# Part XX — Final Framework Definition

KSHIRA Updated is defined as:

> A dataset-neutral object-detection architecture and runtime in ISO C that trains learned spatial, channel-mixing and detection parameters from random initialization inside a statically bounded caller-owned arena.

Its encoder combines:

* a learned spatial stem;
* parallel dilated depthwise receptive fields;
* memory-efficient branch fusion;
* learned pointwise channel interaction.

Its objectness system combines:

* learned semantic features;
* an explicit centre-surround contrast feature;
* localization-aware objectness targets;
* balanced focal supervision;
* bounded hard-negative mining.

Its assignment system combines:

* deterministic centre priors;
* candidate-local contrast;
* predicted IoU;
* staged one-pass scheduling;
* conflict-safe top-k selection.

Its training system combines:

* full forward spatial evaluation;
* dependency-bounded local backward propagation;
* deterministic tile merging;
* transactional updates;
* sparse channel coverage;
* no hot-loop allocation.

Its quantization system combines:

* real INT8 and INT4 arithmetic;
* per-channel weight scales;
* bounded activation calibration;
* non-negative contrast codes;
* deployment-matched contrast approximation;
* packed deployment artifacts.

Its qualification system requires:

* real-image detection accuracy;
* raw and core timing;
* memory high-water;
* three-seed reproducibility;
* HOG-relative validation;
* detector-level quantization parity;
* explicit failure reporting.

KSHIRA does not define success by parameter count or synthetic speed alone.

Success requires one complete detector to demonstrate, on the same qualified configuration:

* training from random weights;
* useful localization;
* meaningful objectness separation;
* class prediction;
* bounded memory;
* quantized operation;
* deterministic persistence;
* measured training latency;
* measured inference latency.

Until these gates pass, sub-second training, stable INT4 learning, FPGA performance and low-power operation remain research objectives.

The architecture is designed so that every such objective can be tested numerically, independently and reproducibly.
