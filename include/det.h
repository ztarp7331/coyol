#ifndef DET_H
#define DET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DET_MAX_SCALES 3
#define DET_MAX_CLASSES 80

typedef enum {
    DET_OK = 0,
    DET_ERR_ARGUMENT = 1,
    DET_ERR_MEMORY = 2,
    DET_ERR_SHAPE = 3,
    DET_ERR_IO = 4,
    DET_ERR_FORMAT = 5,
    DET_ERR_UNSUPPORTED = 6
} det_status;

typedef enum {
    DET_PRECISION_F32 = 0,
    DET_PRECISION_INT8 = 1,
    DET_PRECISION_W4A8 = 2,
    DET_PRECISION_INT4 = 3
} det_precision;

typedef enum {
    DET_TRAIN_LOCAL_FAST = 0,
    DET_TRAIN_GLOBAL_BP = 1,
    DET_TRAIN_ODT = 2
} det_train_mode;

typedef enum {
    DET_RESEARCH_SCRATCH = 0,
    DET_RESEARCH_ADAPT = 1,
    DET_RESEARCH_FULL = 2
} det_research_mode;

typedef enum {
    DET_ARCH_CDET = 0,
    DET_ARCH_KSHIRA = 1
} det_architecture;

typedef enum {
    DET_STEM_LEGACY = 0,
    DET_STEM_SPACE_TO_DEPTH = 1,
    DET_STEM_TWO_STAGE = 2
} det_stem_mode;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    /* Optional manifest confidence; zero means an ordinary ground-truth box. */
    float target_weight;
} det_box;

typedef struct {
    const float *data;
    int channels;
    int height;
    int width;
} det_image;

typedef struct {
    det_image image;
    const det_box *boxes;
    int box_count;
} det_sample;

typedef int (*det_dataset_next_fn)(void *user, det_sample *sample);
typedef void (*det_dataset_reset_fn)(void *user);

typedef struct {
    void *user;
    det_dataset_next_fn next;
    det_dataset_reset_fn reset;
    size_t sample_count;
} det_dataset;

typedef struct {
    int width;
    int height;
    int channels;
    int num_classes;
    int max_detections;
    int seed;
    det_architecture architecture;
    int feature_channels; /* KSHIRA: 0 selects the measured 8-channel profile. */
    det_stem_mode stem_mode; /* KSHIRA: legacy, space-to-depth, or two-stage stem. */
    int one_to_one_head; /* KSHIRA: separate deployment head; zero preserves legacy. */
    int shared_multiscale_head; /* KSHIRA: shared deployment head across P3/P4/P5. */
    det_research_mode research_mode;
    int context_fusion; /* KSHIRA: opt-in learned pooled-context fusion ablation. */
    int raw_input_features; /* KSHIRA: append fixed local image statistics. */
    int p3_only_deployment; /* KSHIRA: retain auxiliary scales for training, emit P3 only. */
    int smooth_box_decode; /* KSHIRA: smooth positive box-side parameterization. */
    /* Optional exported native graph directory for the KSHIRA accuracy profile.
       NULL keeps the ordinary trainable KSHIRA predictor. */
    const char *native_graph_dir;
} det_model_spec;

typedef struct {
    det_train_mode mode;
    det_precision precision;
    int epochs;
    float learning_rate;
    float momentum;
    float score_threshold;
    int max_samples;
    int seed;
    int reset_weights;
    /* Optional Phase 0/1 controls. Zero preserves the legacy fixed-epoch path. */
    double time_budget_ms;
    int analytic_readout;
    /* Optional Phase 2 target-domain bootstrap. Zero preserves the legacy path. */
    double bootstrap_budget_ms;
    int bootstrap_enabled;
    /* Adaptation profile: update the head while keeping encoder channels fixed. */
    int freeze_encoder;
    /* Adaptation profile: retain weights but restart stream-dependent schedules. */
    int reset_schedule;
    /* Training-only one-to-many positive budget; zero preserves the default. */
    int dense_aux_budget;
    /* Optional total stream cap across epochs; zero preserves per-epoch behavior. */
    int max_total_samples;
    /* Train pooled P4/P5 heads as auxiliary scales; zero preserves P3-only. */
    int multiscale_aux;
    /* Use measured dense/mixed/residual stage transitions; zero preserves the
       established epoch schedule. */
    int adaptive_budget;
    /* Use a full-map residual candidate pass for hard-negative mining. */
    int residual_budget;
    /* Use score/IoU-ranked positive cells instead of a fixed neighborhood. */
    int quality_aligned_assignment;
    /* Fit the native graph's final class projection from cached feature maps. */
    int feature_readout;
    /* Adapt final class weights at the cells that produced training candidates. */
    int candidate_head_adapt;
} det_train_config;

typedef struct {
    size_t samples_seen;
    size_t updates;
    double elapsed_ms;
    float mean_loss;
    int used_global_backward;
    int stopped_by_budget;
    int analytic_readout_applied;
    size_t readout_examples;
    size_t readout_positives;
    size_t readout_negatives;
    double readout_pre_objective;
    double readout_post_objective;
    size_t bootstrap_samples;
    double bootstrap_elapsed_ms;
    int feature_readout_applied;
    size_t feature_readout_examples;
    size_t candidate_head_updates;
} det_train_report;

/* Model-owned payload bytes. Allocator metadata and automatic stack temporaries
   are intentionally excluded; arena fields expose the bounded KSHIRA region. */
typedef struct {
    size_t parameter_bytes;
    size_t optimizer_bytes;
    size_t quant_cache_bytes; /* Persistent caches, not on-the-fly quantization. */
    size_t activation_workspace_bytes;
    /* Zero for heap-backed profiles; nonzero for bounded-arena profiles. */
    size_t arena_high_water_bytes;
    size_t arena_capacity_bytes;
} det_memory_report;

typedef struct {
    size_t samples_seen;
    size_t ground_truths;
    size_t predictions;
    size_t true_positives;
    size_t false_positives;
    size_t false_negatives;
    float precision;
    float recall;
    float mean_iou;
    float ap50;
    float ap75;
    float map50_95;
    float class_ap50[DET_MAX_CLASSES];
    float size_ap50[3];
    /* Number of streamed ground-truth boxes in small, medium, and large
       COCO-style area buckets. */
    size_t size_ground_truths[3];
    /* Score bins are [0,.1), ... [.9,1]; row 0 is loose IoU-positive and row 1
       is non-matching. They are filled before the requested score threshold. */
    size_t score_histogram[2][10];
} det_eval_report;

typedef struct {
    det_box box;
    float score;
} det_detection;

typedef struct {
    float *data;
    int channels;
    int height;
    int width;
} det_tensor_f32;

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t offset;
} det_arena;

typedef struct {
    float scale;
    int32_t zero_point;
    int32_t multiplier;
    int shift;
} det_quant_params;

typedef struct det_context det_context;
typedef struct det_model det_model;
typedef struct det_manifest_dataset det_manifest_dataset;

det_status det_context_create(size_t arena_bytes, det_context **out);
void det_context_destroy(det_context *ctx);

det_status det_arena_init(det_arena *arena, void *memory, size_t capacity);
void *det_arena_alloc(det_arena *arena, size_t bytes, size_t alignment);

det_status det_tensor_alloc(det_arena *arena, int channels, int height, int width,
                            det_tensor_f32 *out);
void det_tensor_fill(det_tensor_f32 *tensor, float value);

det_status det_conv2d_f32(const det_tensor_f32 *input, const float *weights,
                         const float *bias, int out_channels, int kernel,
                         int stride, int padding, det_tensor_f32 *output);
det_status det_conv2d_backward_f32(const det_tensor_f32 *input, const float *weights,
                                   const det_tensor_f32 *grad_output, int out_channels,
                                   int kernel, int stride, int padding,
                                   det_tensor_f32 *grad_input, float *grad_weights,
                                   float *grad_bias);
det_status det_add_f32(const det_tensor_f32 *a, const det_tensor_f32 *b,
                       det_tensor_f32 *output);
void det_relu_inplace(det_tensor_f32 *tensor);
det_status det_upsample_nearest(const det_tensor_f32 *input, int scale,
                                det_tensor_f32 *output);

det_status det_model_build(det_context *ctx, const det_model_spec *spec,
                           det_model **out);
det_status det_model_set_precision(det_model *model, det_precision precision);
/* Attach or replace the optional exported native graph accuracy profile. */
det_status det_model_set_native_graph(det_model *model, const char *directory);
det_status det_model_native_graph_calibration(const det_model *model,
                                             float *gain, float *bias);
det_precision det_model_precision(const det_model *model);
det_architecture det_model_architecture(const det_model *model);
int det_model_one_to_one_head(const det_model *model);
int det_model_shared_multiscale(const det_model *model);
int det_model_context_fusion(const det_model *model);
int det_model_p3_only_deployment(const det_model *model);
det_research_mode det_model_research_mode(const det_model *model);
det_status det_model_memory(const det_model *model, det_memory_report *report);
void det_model_destroy(det_model *model);
det_status det_model_reset(det_model *model, int seed);

det_status det_train(det_model *model, const det_dataset *dataset,
                     const det_train_config *config, det_train_report *report);
/* Greedy class-aware matching over a streamed validation dataset. AP fields use
   class-macro AP at IoU .50, .75, and .50:.05:.95; size buckets use COCO pixel
   area cutoffs (small, medium, large). */
det_status det_evaluate(const det_model *model, const det_dataset *dataset,
                        float score_threshold, det_eval_report *report);
det_status det_predict(const det_model *model, const det_image *image,
                       float score_threshold, det_detection *detections,
                       int capacity, int *count);

det_status det_save(const det_model *model, const char *path);
det_status det_load(det_context *ctx, const char *path, det_model **out);

/* Dataset-neutral raw-file adapter. Manifest lines are:
   image_path [x1,y1,x2,y2,class ...]
   Paths cannot contain whitespace; relative paths resolve beside the manifest.
   P2/P3/P5/P6 PNM images are decoded, resized to the requested shape, and exposed as
   a normal det_dataset callback stream. */
det_status det_manifest_open(const char *manifest_path, int width, int height,
                             int channels, int max_boxes,
                             det_manifest_dataset **out);
det_status det_manifest_dataset_view(det_manifest_dataset *dataset,
                                     det_dataset *out);
/* Optional predecoded cache. Preparation is outside det_train's budget and is
   reported separately by the benchmark; the streamed API remains unchanged. */
det_status det_manifest_enable_cache(det_manifest_dataset *dataset);
/* Optionally permute manifest samples deterministically at every reset. This
   uses a lightweight line index and does not require the pixel cache. */
det_status det_manifest_set_shuffle(det_manifest_dataset *dataset, int enabled,
                                    int seed);
det_status det_manifest_status(const det_manifest_dataset *dataset);
void det_manifest_close(det_manifest_dataset *dataset);

float det_sigmoid(float x);
float det_iou(const det_box *a, const det_box *b);
int8_t det_quantize_s8(float value, float scale);
int8_t det_requantize_i32(int64_t accumulator, int32_t multiplier, int shift,
                          int32_t zero_point);
det_status det_quantize_per_channel_s8(const float *weights, int rows, int cols,
                                        int8_t *quantized, float *scales);
det_status det_quantize_pack_w4(const float *weights, int rows, int cols,
                                uint8_t *packed, float *scales);

#ifdef __cplusplus
}
#endif

#endif
