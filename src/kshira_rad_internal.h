#ifndef KSHIRA_RAD_INTERNAL_H
#define KSHIRA_RAD_INTERNAL_H

#include "kshira/rad.h"

enum {
    RAD_BRANCHES = 3,
    RAD_SCALES = 3,
    RAD_KERNEL = 3,
    RAD_STRIDE = 4,
    RAD_MAX_CLASSES = 80,
    /* PLAN_UPDATED: contrast radius r=2 (5x5), max dilation 4 → dep radius 6. */
    RAD_CONTRAST_RADIUS = 2,
    RAD_DEP_RADIUS = 6,
    RAD_CONTRAST_CHANNELS = 1,
    RAD_RAW_CHANNELS = 4,
    RAD_MAX_FEATURES = 32,
    RAD_MAX_HEAD_IN = RAD_MAX_FEATURES + RAD_CONTRAST_CHANNELS + RAD_RAW_CHANNELS
};

typedef struct {
    float *project_weights;
    float *project_bias;
    float *branch_weights[RAD_BRANCHES];
    float *branch_bias[RAD_BRANCHES];
    float *stem_weights;
    float *stem_bias;
    float *stem_pre_weights;
    float *stem_pre_bias;
} rad_encoder_delta_buffer;

typedef struct {
    float *weights;
    float *bias;
} rad_head_delta_buffer;

struct kshira_rad_model {
    kshira_rad_spec spec;
    kshira_arena *arena;
    kshira_bit_mode bits;
    int map_height;
    int map_width;
    int outputs;
    /* Head input = semantic feature_channels + one contrast channel. */
    int head_in;
    int stem_input_channels;
    size_t parameter_bytes;
    size_t activation_bytes;
    float *stem_weights;
    float *stem_bias;
    /* Two-stage profile: learned stride-2 pre-stem feeding the existing
       stride-2 semantic stem. */
    float *stem_pre_weights;
    float *stem_pre_bias;
    float *stem_mid;
    int stem_mid_height;
    int stem_mid_width;
    float *branch_weights[RAD_BRANCHES];
    float *branch_bias[RAD_BRANCHES];
    /* Pointwise channel mixer (PLAN_UPDATED §13); historically "project". */
    float *project_weights;
    float *project_bias;
    float *head_weights;
    float *head_bias;
    float *one_to_one_head_weights;
    float *one_to_one_head_bias;
    float *scale_head_weights[RAD_SCALES];
    float *scale_head_bias[RAD_SCALES];
    /* Lightweight scale adaptation retained by the shared deployment head. */
    float scale_head_gain[RAD_SCALES];
    float scale_head_offset[RAD_SCALES];
    /* Trainable on-demand P4/P5 transforms and P5 -> P4 top-down add. */
    float pyramid_feature_gain[RAD_SCALES][RAD_MAX_FEATURES];
    float pyramid_feature_bias[RAD_SCALES][RAD_MAX_FEATURES];
    float pyramid_topdown_gain[RAD_SCALES];
    /* Context profile: learned cross-channel lateral projections and a
       P5-to-P4 top-down projection. Identity initialization preserves the
       previous pooled-affine behavior before training. */
    float *pyramid_projection_weights;
    float *pyramid_projection_bias;
    float *pyramid_topdown_weights;
    float *pyramid_refine_weights;
    float *pyramid_refine_bias;
    float *pyramid_cache[RAD_SCALES];
    int pyramid_cache_valid;
    rad_head_delta_buffer *scale_head_deltas;
    /* Both descriptors overlay one mutually exclusive update workspace. */
    rad_encoder_delta_buffer encoder_delta_storage;
    rad_head_delta_buffer scale_head_delta_storage;
    int scale_heads_ready;
    int scale_head_trained_mask;
    float *stem;
    /* Fixed local image statistics appended to the train/deploy head. */
    float *raw_features;
    /* branches[0..2] may alias one workspace (PLAN_UPDATED sequential schedule). */
    float *branches[RAD_BRANCHES];
    int branch_maps_shared; /* 1: single workspace reused across dilations */
    float *fused;
    /* Single-channel contrast map C_x = log1p(κ_x); H*W floats. */
    float *contrast;
    /* Transient validity marker for head-only hard-negative updates that reuse
       the complete feature map produced by candidate mining. */
    int full_map_ready;
    rad_encoder_delta_buffer *encoder_deltas;
    float calibration_input_scale;
    float calibration_stem_scale;
    float calibration_branch_scale[RAD_BRANCHES];
    size_t calibration_samples;
    float transient_image_scale;
    float transient_stem_scale;
    int transient_scales_valid;
    /* Online class frequencies for inverse-frequency CE (training only). */
    float class_count[RAD_MAX_CLASSES];
    float class_total;
    /* Stream progress for staged task-aligned assignment (sample count). */
    size_t stream_samples;
    size_t stream_horizon;
};

void kshira_rad_pyramid_features(const kshira_rad_model *model, int level, int y,
                                 int x, float *features);
void kshira_rad_pyramid_prepare(kshira_rad_model *model);
void kshira_rad_pyramid_gradients(
    const kshira_rad_model *model, int level, int y, int x,
    const float *feature_gradient, const kshira_sparse_mask *channel_mask,
    float *topdown_gradient, float *gain_gradient, float *bias_gradient);
void kshira_rad_pyramid_update(
    kshira_rad_model *model, int level, int y, int x,
    const float *feature_gradient, const kshira_sparse_mask *channel_mask,
    float learning_rate);

#endif
