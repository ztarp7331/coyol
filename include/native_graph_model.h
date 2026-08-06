/*
 * Native C implementation of a compact multi-scale detection graph.
 * This API is inference-only and uses no heap allocation.
 */
#ifndef NATIVE_GRAPH_MODEL_H
#define NATIVE_GRAPH_MODEL_H

#include "native_graph_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NG_SCALE_N = 0,
    NG_SCALE_S,
    NG_SCALE_M,
    NG_SCALE_L,
    NG_SCALE_X,
} NGModelScale;

typedef struct {
    NGModelScale scale;
    int32_t input_channels;
    int32_t class_count;
} NGModelConfig;

/*
 * The caller owns this fixed activation arena. `used` grows during a forward
 * pass; reset it only after its NGModelOutput is no longer needed.
 */
typedef struct {
    int8_t *data;
    int32_t capacity;
    int32_t used;
} NGArena;

/*
 * Fused Conv+BN descriptors in forward-execution order. The required count
 * is returned by ng_native_graph_conv_count(). The parameter exporter must emit the
 * descriptors in this order, independently of PyTorch state_dict ordering.
 */
typedef struct {
    const NGConv2D *convs;
    int32_t conv_count;
} NGModelWeights;

/* Raw [4 * reg_max + class_count, H, W] tensors for strides 8, 16, and 32. */
typedef struct {
    NGTensor prediction[3];
    /* Feature maps immediately before each scale's final class projection. */
    NGTensor class_feature[3];
    int32_t stride[3];
} NGModelOutput;

typedef struct {
    float *data;
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
} NGF32Tensor;

typedef struct {
    NGF32Tensor prediction[3];
    NGF32Tensor class_feature[3];
    int32_t stride[3];
} NGF32ModelOutput;

void ng_arena_init(NGArena *arena, int8_t *data, int32_t capacity);
void ng_arena_reset(NGArena *arena);
int32_t ng_native_graph_conv_count(NGModelScale scale);

/*
 * Runs one N=1 frame through the complete native detector
 * backbone, PAN/FPN neck, and Detect head. Input H and W must be divisible by
 * 32. The output tensors borrow storage from arena until the arena is reset.
 */
int ng_native_graph_forward_s8(const NGModelConfig *config, const NGModelWeights *weights, const NGTensor *input,
                          NGArena *arena, NGModelOutput *output);

int ng_native_graph_forward_f32(const NGModelConfig *config, const NGModelWeights *weights,
                                const float *input, int32_t input_channels,
                                int32_t height, int32_t width, NGArena *arena,
                                NGF32ModelOutput *output);

#ifdef __cplusplus
}
#endif

#endif
