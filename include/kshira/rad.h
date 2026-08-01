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
} kshira_rad_train_config;

typedef struct kshira_rad_model kshira_rad_model;

kshira_status kshira_rad_build(kshira_arena *arena, const kshira_rad_spec *spec,
                                kshira_rad_model **out);
kshira_status kshira_rad_reset(kshira_rad_model *model, int seed);
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
kshira_status kshira_rad_train_multiscale_step(kshira_rad_model *model,
                                                const kshira_image_f32 *image,
                                                const kshira_rad_box *target, int level,
                                                const kshira_rad_train_config *config,
                                                float *loss);
int kshira_rad_map_height(const kshira_rad_model *model);
int kshira_rad_map_width(const kshira_rad_model *model);
size_t kshira_rad_parameter_bytes(const kshira_rad_model *model);
size_t kshira_rad_activation_bytes(const kshira_rad_model *model);

#ifdef __cplusplus
}
#endif

#endif
