/*
 * Preprocess and full detection post-processing for the int8 native graph.
 * No heap allocation. Thermal-aware letterbox uses centered padding, configurable
 * fill, scale-to-frame resizing, then maps to int8 activations.
 */
#ifndef NATIVE_GRAPH_POST_H
#define NATIVE_GRAPH_POST_H

#include "native_graph_ops.h"
#include "native_graph_model.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NG_MAX_CANDIDATES
#define NG_MAX_CANDIDATES 8400
#endif

#ifndef NG_MAX_KEEP
#define NG_MAX_KEEP 100
#endif

typedef struct {
    int32_t orig_w;
    int32_t orig_h;
    int32_t imgsz;
    float scale;   /* new / old (uniform) */
    int32_t pad_x; /* left pad in pixels (floor of half pad) */
    int32_t pad_y; /* top pad */
} Y8LetterboxInfo;

typedef struct {
    float conf_threshold;      /* 0..1 */
    float iou_threshold;       /* 0..1 */
    float logit_scale;         /* float value represented by one int8 output */
    int32_t max_candidates;
    int32_t max_keep;
    const NGDflConfig *dfl;
} Y8DetectPostConfig;

typedef struct {
    NGDetection dets[NG_MAX_KEEP];
    int32_t count;
} Y8DetectResult;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int32_t class_id;
} NGF32Detection;

typedef struct {
    float conf_threshold;
    float iou_threshold;
    int32_t max_candidates;
    int32_t max_keep;
} Y8F32DetectPostConfig;

typedef struct {
    NGF32Detection dets[NG_MAX_KEEP];
    int32_t count;
} Y8F32DetectResult;

/*
 * Letterbox uint8 grayscale thermal (or single-channel) into NCHW int8.
 * If out_channels==3, replicates the single channel (RGB-compatible transfer).
 * Mapping: q = (pixel - 128) roughly via (pixel * 2 - 255) style:
 *   float f = pixel / 255;  q = round(f / act_scale) clamped, with act_scale=1/127
 *   => q â‰ˆ round(pixel * 127 / 255)
 * pad_value defaults to 114 unless overridden.
 */
int ng_letterbox_u8_to_s8(const uint8_t *src, int32_t src_w, int32_t src_h,
                          int8_t *dst_nchw, int32_t imgsz, int32_t out_channels,
                          uint8_t pad_value, Y8LetterboxInfo *info);

/* Same preprocessing with the activation scale used by the exported graph. */
int ng_letterbox_u8_to_s8_scaled(const uint8_t *src, int32_t src_w, int32_t src_h,
                                 int8_t *dst_nchw, int32_t imgsz, int32_t out_channels,
                                 uint8_t pad_value, float act_scale, Y8LetterboxInfo *info);

/* Exact normalized FP32 counterpart used by the reference graph. */
int ng_letterbox_u8_to_f32(const uint8_t *src, int32_t src_w, int32_t src_h,
                           float *dst_nchw, int32_t imgsz, int32_t out_channels,
                           uint8_t pad_value, Y8LetterboxInfo *info);

/* 14-bit radiometric (0..16383) â†’ same letterbox path after scale to 8-bit. */
int ng_letterbox_u16_to_s8(const uint16_t *src, int32_t src_w, int32_t src_h, uint16_t max_value,
                           int8_t *dst_nchw, int32_t imgsz, int32_t out_channels,
                           uint8_t pad_value, Y8LetterboxInfo *info);

/*
 * Decode raw Detect tensors (3 scales, channels = 4*reg_max + nc) into NMS'd boxes
 * in letterboxed image coordinates (Q12 pixels). Then use ng_map_boxes_to_original
 * to remove letterbox.
 */
int ng_detect_post_s8(const NGModelOutput *raw, int32_t class_count,
                      const Y8DetectPostConfig *cfg, Y8DetectResult *out);

int ng_detect_post_f32(const NGF32ModelOutput *raw, int32_t class_count,
                       const Y8F32DetectPostConfig *cfg, Y8F32DetectResult *out);

/* Map Q12 letterbox boxes â†’ Q12 original-image boxes using letterbox meta. */
int ng_map_boxes_to_original(NGDetection *dets, int32_t count, const Y8LetterboxInfo *lb);

/*
 * Docking: require approximately `expected_count` detections of class `class_id`
 * with pairwise spacing ratio check. pose_out[0..5] optional:
 *   cx, cy (Q12 image), mean_radius_q12, angle_q15 (atan2 approx stub), score_q15, reserved.
 * Returns 1 if pattern ok, 0 if not, -1 on error.
 */
int ng_check_docking_pattern(const NGDetection *dets, int32_t count, int16_t class_id,
                             int32_t expected_count, int32_t spacing_tol_q12,
                             int32_t *pose_out6);

/* Convert Q15 score threshold from float conf in [0,1]. */
uint16_t ng_conf_to_q15(float conf);
uint16_t ng_iou_to_q15(float iou);

#ifdef __cplusplus
}
#endif

#endif
