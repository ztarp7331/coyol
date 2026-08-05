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
    RAD_MAX_FEATURES = 32,
    RAD_MAX_HEAD_IN = RAD_MAX_FEATURES + RAD_CONTRAST_CHANNELS
};

typedef struct {
    float *project_weights;
    float *project_bias;
    float *branch_weights[RAD_BRANCHES];
    float *branch_bias[RAD_BRANCHES];
    float *stem_weights;
    float *stem_bias;
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
    size_t parameter_bytes;
    size_t activation_bytes;
    float *stem_weights;
    float *stem_bias;
    float *branch_weights[RAD_BRANCHES];
    float *branch_bias[RAD_BRANCHES];
    /* Pointwise channel mixer (PLAN_UPDATED §13); historically "project". */
    float *project_weights;
    float *project_bias;
    float *head_weights;
    float *head_bias;
    float *scale_head_weights[RAD_SCALES];
    float *scale_head_bias[RAD_SCALES];
    rad_head_delta_buffer *scale_head_deltas;
    /* Both descriptors overlay one mutually exclusive update workspace. */
    rad_encoder_delta_buffer encoder_delta_storage;
    rad_head_delta_buffer scale_head_delta_storage;
    int scale_heads_ready;
    int scale_head_trained_mask;
    float *stem;
    /* branches[0..2] may alias one workspace (PLAN_UPDATED sequential schedule). */
    float *branches[RAD_BRANCHES];
    int branch_maps_shared; /* 1: single workspace reused across dilations */
    float *fused;
    /* Single-channel contrast map C_x = log1p(κ_x); H*W floats. */
    float *contrast;
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

#endif
