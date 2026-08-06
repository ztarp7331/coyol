#ifndef KSHIRA_RAD_H
#define KSHIRA_RAD_H

#include <stddef.h>

#include "kshira/core.h"
#include "kshira/sparse.h"
#include "kshira/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *data;
    int channels;
    int height;
    int width;
} kshira_image_f32;

typedef struct {
    int width;
    int height;
    int channels;
    int classes;
    int feature_channels;
    int top_k;
    int seed;
    int multiscale_heads; /* 0: pooled inference shares P3 head; 1: allocate P4/P5 ODT heads */
    int stem_mode; /* 0: legacy; 1: space-to-depth; 2: learned stride-2+2 stem */
    int one_to_one_head; /* 0: shared train/deploy head; 1: center-trained deploy head */
    int shared_multiscale_head; /* 0: independent scale heads; 1: shared P3/P4/P5 head */
    int context_fusion; /* 0: pooled scale baseline; 1: learned wider-context blend */
    int raw_input_features; /* 0: semantic+contrast head; 1: add fixed local image cues */
    int p3_only_deployment; /* 0: emit configured scales; 1: emit P3 while retaining auxiliaries */
    int smooth_box_decode; /* 0: clipped linear sides; 1: smooth positive sides */
} kshira_rad_spec;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
} kshira_rad_box;

typedef struct {
    kshira_rad_box box;
    float score;
    float quality;
} kshira_rad_detection;

/* FULL updates the RAD encoder as well as the head; CHANNELS keeps the sparse
 * channel mask on the head-only path. All updates are caller-arena resident. */
typedef struct {
    kshira_bit_mode bits;
    kshira_update_mode update_mode;
    const kshira_sparse_mask *channel_mask;
    float learning_rate;
    /* Number of one-to-many positive cells; zero selects the default. */
    int dense_aux_budget;
    /* 1: select positive cells by score^0.5 * IoU^6, 0: geometric neighborhood. */
    int quality_aligned_assignment;
} kshira_rad_train_config;

typedef struct {
    double *gram;
    double *rhs;
    double *target_square_sum;
    size_t feature_count;
    size_t class_count;
    size_t examples;
    size_t positives;
    size_t negatives;
} kshira_rad_readout_stats;

typedef struct {
    size_t examples;
    size_t positives;
    size_t negatives;
    double pre_objective;
    double post_objective;
    int applied;
} kshira_rad_readout_report;

typedef struct kshira_rad_model kshira_rad_model;

kshira_status kshira_rad_build(kshira_arena *arena, const kshira_rad_spec *spec,
                                kshira_rad_model **out);
kshira_status kshira_rad_reset(kshira_rad_model *model, int seed);
kshira_status kshira_rad_reset_training_state(kshira_rad_model *model);
kshira_status kshira_rad_set_bits(kshira_rad_model *model, kshira_bit_mode bits);
kshira_bit_mode kshira_rad_bits(const kshira_rad_model *model);
kshira_status kshira_rad_calibrate(kshira_rad_model *model,
                                    const kshira_image_f32 *image);
int kshira_rad_calibration_ready(const kshira_rad_model *model);
int kshira_rad_multiscale_ready(const kshira_rad_model *model);
kshira_status kshira_rad_predict(kshira_rad_model *model,
                                 const kshira_image_f32 *image, float threshold,
                                 kshira_rad_detection *detections, int capacity, int *count);
kshira_status kshira_rad_train_step(kshira_rad_model *model, const kshira_image_f32 *image,
                                     const kshira_rad_box *target,
                                     const kshira_rad_train_config *config, float *loss);
/* Peek max class-quality sigmoid at one map cell (for hard-negative mining). */
kshira_status kshira_rad_objectness_at(kshira_rad_model *model, const kshira_image_f32 *image,
                                        int cell_y, int cell_x, float *probability);
/* Build a sorted top-score residual candidate list from the current P3 map.
 * Callers filter cells covered by ground-truth boxes before updating them. */
kshira_status kshira_rad_hard_negative_candidates(
    kshira_rad_model *model, const kshira_image_f32 *image,
    size_t *cells, float *scores, int capacity, int *count);
/* Surgical FP background: score-gate easy BG; update only argmax quality head
 * row (mid-band LR boost). FULL = 13×13 encoder tile from winning-class grad. */
kshira_status kshira_rad_train_background_step(
    kshira_rad_model *model, const kshira_image_f32 *image, int cell_y, int cell_x,
    const kshira_rad_train_config *config, float *loss);
/* Head-only background update that reuses the complete map produced by the
 * immediately preceding hard-negative candidate scan. */
kshira_status kshira_rad_train_cached_background_step(
    kshira_rad_model *model, const kshira_image_f32 *image, int cell_y, int cell_x,
    const kshira_rad_train_config *config, float *loss);
/* Head-only ranking hinge: s_pos(class) ≥ s_neg_max + margin; surgical neg
 * updates only the winning class head row on the FP cell. */
kshira_status kshira_rad_train_rank_pair(
    kshira_rad_model *model, const kshira_image_f32 *image,
    int pos_y, int pos_x, int class_id, int neg_y, int neg_x, float margin,
    const kshira_rad_train_config *config, float *loss);
kshira_status kshira_rad_train_multiscale_rank_pair(
    kshira_rad_model *model, const kshira_image_f32 *image, int level,
    int pos_y, int pos_x, int class_id, int neg_y, int neg_x, float margin,
    const kshira_rad_train_config *config, float *loss);
kshira_status kshira_rad_train_multiscale_background_step(
    kshira_rad_model *model, const kshira_image_f32 *image, int level,
    int cell_y, int cell_x, const kshira_rad_train_config *config, float *loss);
kshira_status kshira_rad_train_multiscale_step(kshira_rad_model *model,
                                                const kshira_image_f32 *image,
                                                const kshira_rad_box *target, int level,
                                                const kshira_rad_train_config *config,
                                                float *loss);
kshira_status kshira_rad_readout_accumulate(
    kshira_rad_model *model, const kshira_image_f32 *image,
    const kshira_rad_box *targets, int target_count,
    kshira_rad_readout_stats *stats, int max_background_cells);
kshira_status kshira_rad_readout_solve(kshira_rad_model *model,
                                        const kshira_rad_readout_stats *stats,
                                        double regularization,
                                        kshira_rad_readout_report *report);
int kshira_rad_map_height(const kshira_rad_model *model);
int kshira_rad_map_width(const kshira_rad_model *model);
size_t kshira_rad_parameter_bytes(const kshira_rad_model *model);
size_t kshira_rad_activation_bytes(const kshira_rad_model *model);
size_t kshira_rad_state_bytes(const kshira_rad_model *model);
kshira_status kshira_rad_export_state(const kshira_rad_model *model, void *buffer,
                                       size_t capacity, size_t *written);
kshira_status kshira_rad_import_state(kshira_rad_model *model, const void *buffer,
                                       size_t bytes);

#ifdef __cplusplus
}
#endif

#endif
