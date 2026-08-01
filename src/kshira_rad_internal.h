#ifndef KSHIRA_RAD_INTERNAL_H
#define KSHIRA_RAD_INTERNAL_H

#include "kshira/rad.h"

enum {
    RAD_BRANCHES = 3,
    RAD_SCALES = 3,
    RAD_KERNEL = 3,
    RAD_STRIDE = 4,
    RAD_MAX_CLASSES = 80
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
    size_t parameter_bytes;
    size_t activation_bytes;
    float *stem_weights;
    float *stem_bias;
    float *branch_weights[RAD_BRANCHES];
    float *branch_bias[RAD_BRANCHES];
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
    float *branches[RAD_BRANCHES];
    float *fused;
    rad_encoder_delta_buffer *encoder_deltas;
    float calibration_input_scale;
    float calibration_stem_scale;
    float calibration_branch_scale[RAD_BRANCHES];
    size_t calibration_samples;
    float transient_image_scale;
    float transient_stem_scale;
    int transient_scales_valid;
};

#endif
