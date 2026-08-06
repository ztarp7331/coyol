/*
 * Fixed-point native graph deployment primitives.
 *
 * This interface deliberately contains no allocator, filesystem access, Python
 * dependency, floating-point arithmetic, or GPU dependency.  An FPGA wrapper
 * can replace individual functions while keeping the model call graph intact.
 */
#ifndef NATIVE_GRAPH_OPS_H
#define NATIVE_GRAPH_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NG_QUANT_INT8 = 8,
    NG_QUANT_INT4 = 4,
} NGQuantMode;

typedef enum {
    NG_ACT_IDENTITY = 0,
    NG_ACT_SILU = 1,
} NGActivation;

/* NCHW signed-int8 activations.  The caller owns data and all scratch memory. */
typedef struct {
    int8_t *data;
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
} NGTensor;

/*
 * Fused Conv+BN parameters, ready for inference.  multiplier and shift are
 * per-output-channel calibration values: out = clamp((acc * multiplier) >> shift).
 * For INT4, weights contains signed two's-complement nibbles, low nibble first.
 */
typedef struct {
    const void *weights;
    const float *weights_f32; /* Optional exact FP32 weights for reference graphs. */
    const int32_t *bias;
    const float *bias_f32; /* Optional exact FP32 bias for reference graphs. */
    const int32_t *multiplier;
    const uint8_t *shift;
    const int8_t *silu_lut; /* 256 entries indexed by (uint8_t)pre_activation */
    int32_t in_channels;
    int32_t out_channels;
    int32_t kernel;
    int32_t stride;
    int32_t padding;
    int32_t groups;
    NGQuantMode weight_quant;
    NGActivation activation;
    /* Optional activation-domain metadata for calibrated native graphs. */
    float input_scale;
    float output_scale;
} NGConv2D;

typedef struct {
    int32_t x1_q12;
    int32_t y1_q12;
    int32_t x2_q12;
    int32_t y2_q12;
    int16_t score_q15;
    int16_t class_id;
} NGDetection;

/* A 256-entry exp(logit) table in Q15, indexed by (uint8_t)int8_logit. */
typedef struct {
    const uint16_t *exp_lut_q15;
} NGDflConfig;

int ng_conv2d_s8(const NGTensor *input, NGTensor *output, const NGConv2D *layer);
int ng_maxpool2d_s8(const NGTensor *input, NGTensor *output, int32_t kernel, int32_t stride, int32_t padding);
int ng_upsample_nearest2_s8(const NGTensor *input, NGTensor *output);
int ng_concat_channels_s8(const NGTensor *inputs, int32_t input_count, NGTensor *output);
/* Element-wise residual add. Inputs and output must use the same quantization scale. */
int ng_add_s8(const NGTensor *a, const NGTensor *b, NGTensor *output);

/* Decode one distributional box candidate. logits is channel-major [4][16]. */
int ng_decode_dfl_s8(const int8_t *logits, int32_t grid_x, int32_t grid_y, int32_t stride,
                     const NGDflConfig *config, NGDetection *out);
uint16_t ng_iou_q15(const NGDetection *a, const NGDetection *b);
int32_t ng_nms(NGDetection *detections, int32_t count, uint16_t iou_threshold_q15,
               NGDetection *kept, int32_t kept_capacity);

#ifdef __cplusplus
}
#endif

#endif
