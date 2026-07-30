#include "det.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct det_context {
    uint8_t *arena_memory;
    size_t arena_bytes;
};

typedef struct {
    int height;
    int width;
    int channels;
    int outputs;
    float *weights;
    float *bias;
    float *velocity_w;
    float *velocity_b;
} det_head;

struct det_model {
    det_context *ctx;
    det_model_spec spec;
    int feature_channels;
    float *feature_weights;
    float *feature_bias;
    float *feature_velocity_w;
    float *feature_velocity_b;
    float *stem_weights;
    float *stem_bias;
    float *stem_velocity_w;
    float *stem_velocity_b;
    float input_mean[4];
    float *integral;
    float *integral_sq;
    float *input_integral;
    det_head heads[DET_MAX_SCALES];
    size_t train_updates;
};

static const int k_strides[DET_MAX_SCALES] = {8, 16, 32};

static double now_ms(void) {
    struct timespec ts;
    (void)timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int checked_size3(int a, int b, int c, size_t *out) {
    if (a <= 0 || b <= 0 || c <= 0) return 0;
    size_t sa = (size_t)a;
    size_t sb = (size_t)b;
    size_t sc = (size_t)c;
    if (sa > SIZE_MAX / sb) return 0;
    size_t ab = sa * sb;
    if (ab > SIZE_MAX / sc) return 0;
    *out = ab * sc;
    return 1;
}

static size_t align_up_size(size_t value, size_t alignment) {
    size_t mask = alignment - 1U;
    return (value + mask) & ~mask;
}

det_status det_context_create(size_t arena_bytes, det_context **out) {
    if (out == NULL || arena_bytes == 0U) return DET_ERR_ARGUMENT;
    det_context *ctx = (det_context *)calloc(1U, sizeof(*ctx));
    if (ctx == NULL) return DET_ERR_MEMORY;
    ctx->arena_memory = (uint8_t *)malloc(arena_bytes);
    if (ctx->arena_memory == NULL) {
        free(ctx);
        return DET_ERR_MEMORY;
    }
    ctx->arena_bytes = arena_bytes;
    *out = ctx;
    return DET_OK;
}

void det_context_destroy(det_context *ctx) {
    if (ctx == NULL) return;
    free(ctx->arena_memory);
    free(ctx);
}

det_status det_arena_init(det_arena *arena, void *memory, size_t capacity) {
    if (arena == NULL || memory == NULL || capacity == 0U) return DET_ERR_ARGUMENT;
    arena->data = (uint8_t *)memory;
    arena->capacity = capacity;
    arena->offset = 0U;
    return DET_OK;
}

void *det_arena_alloc(det_arena *arena, size_t bytes, size_t alignment) {
    if (arena == NULL || arena->data == NULL || bytes == 0U || alignment == 0U ||
        (alignment & (alignment - 1U)) != 0U) return NULL;
    size_t start = align_up_size(arena->offset, alignment);
    if (start > arena->capacity || bytes > arena->capacity - start) return NULL;
    void *result = arena->data + start;
    arena->offset = start + bytes;
    return result;
}

det_status det_tensor_alloc(det_arena *arena, int channels, int height, int width,
                            det_tensor_f32 *out) {
    if (arena == NULL || out == NULL) return DET_ERR_ARGUMENT;
    size_t elements = 0U;
    if (!checked_size3(channels, height, width, &elements) ||
        elements > SIZE_MAX / sizeof(float)) return DET_ERR_SHAPE;
    float *data = (float *)det_arena_alloc(arena, elements * sizeof(float),
                                           _Alignof(float));
    if (data == NULL) return DET_ERR_MEMORY;
    out->data = data;
    out->channels = channels;
    out->height = height;
    out->width = width;
    return DET_OK;
}

void det_tensor_fill(det_tensor_f32 *tensor, float value) {
    if (tensor == NULL || tensor->data == NULL) return;
    size_t elements = 0U;
    if (!checked_size3(tensor->channels, tensor->height, tensor->width, &elements)) return;
    for (size_t i = 0U; i < elements; ++i) tensor->data[i] = value;
}

static size_t tensor_index(const det_tensor_f32 *tensor, int c, int y, int x) {
    return ((size_t)c * (size_t)tensor->height + (size_t)y) *
           (size_t)tensor->width + (size_t)x;
}

det_status det_conv2d_f32(const det_tensor_f32 *input, const float *weights,
                         const float *bias, int out_channels, int kernel,
                         int stride, int padding, det_tensor_f32 *output) {
    if (input == NULL || weights == NULL || bias == NULL || output == NULL ||
        input->data == NULL || output->data == NULL || out_channels <= 0 ||
        kernel <= 0 || stride <= 0 || padding < 0) return DET_ERR_ARGUMENT;
    int numerator_h = input->height + 2 * padding - kernel;
    int numerator_w = input->width + 2 * padding - kernel;
    if (numerator_h < 0 || numerator_w < 0) return DET_ERR_SHAPE;
    int expected_h = numerator_h / stride + 1;
    int expected_w = numerator_w / stride + 1;
    if (output->channels != out_channels || output->height != expected_h ||
        output->width != expected_w) return DET_ERR_SHAPE;
    size_t weight_count = (size_t)out_channels * (size_t)input->channels *
                          (size_t)kernel * (size_t)kernel;
    (void)weight_count;
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oy = 0; oy < expected_h; ++oy) {
            for (int ox = 0; ox < expected_w; ++ox) {
                float sum = bias[oc];
                for (int ic = 0; ic < input->channels; ++ic) {
                    for (int ky = 0; ky < kernel; ++ky) {
                        int iy = oy * stride + ky - padding;
                        if (iy < 0 || iy >= input->height) continue;
                        for (int kx = 0; kx < kernel; ++kx) {
                            int ix = ox * stride + kx - padding;
                            if (ix < 0 || ix >= input->width) continue;
                            size_t wi = ((((size_t)oc * (size_t)input->channels +
                                           (size_t)ic) * (size_t)kernel +
                                          (size_t)ky) * (size_t)kernel +
                                         (size_t)kx);
                            sum += weights[wi] * input->data[tensor_index(input, ic, iy, ix)];
                        }
                    }
                }
                output->data[tensor_index(output, oc, oy, ox)] = sum;
            }
        }
    }
    return DET_OK;
}

det_status det_conv2d_backward_f32(const det_tensor_f32 *input, const float *weights,
                                   const det_tensor_f32 *grad_output, int out_channels,
                                   int kernel, int stride, int padding,
                                   det_tensor_f32 *grad_input, float *grad_weights,
                                   float *grad_bias) {
    if (input == NULL || weights == NULL || grad_output == NULL || grad_input == NULL ||
        grad_weights == NULL || grad_bias == NULL || input->data == NULL ||
        grad_output->data == NULL || grad_input->data == NULL || out_channels <= 0 ||
        kernel <= 0 || stride <= 0 || padding < 0) return DET_ERR_ARGUMENT;
    int numerator_h = input->height + 2 * padding - kernel;
    int numerator_w = input->width + 2 * padding - kernel;
    if (numerator_h < 0 || numerator_w < 0) return DET_ERR_SHAPE;
    int output_h = numerator_h / stride + 1;
    int output_w = numerator_w / stride + 1;
    if (grad_output->channels != out_channels || grad_output->height != output_h ||
        grad_output->width != output_w || grad_input->channels != input->channels ||
        grad_input->height != input->height || grad_input->width != input->width) {
        return DET_ERR_SHAPE;
    }
    size_t weight_count = (size_t)out_channels * (size_t)input->channels *
                          (size_t)kernel * (size_t)kernel;
    memset(grad_input->data, 0, (size_t)grad_input->channels *
           (size_t)grad_input->height * (size_t)grad_input->width * sizeof(float));
    memset(grad_weights, 0, weight_count * sizeof(float));
    memset(grad_bias, 0, (size_t)out_channels * sizeof(float));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oy = 0; oy < output_h; ++oy) {
            for (int ox = 0; ox < output_w; ++ox) {
                float go = grad_output->data[tensor_index(grad_output, oc, oy, ox)];
                grad_bias[oc] += go;
                for (int ic = 0; ic < input->channels; ++ic) {
                    for (int ky = 0; ky < kernel; ++ky) {
                        int iy = oy * stride + ky - padding;
                        if (iy < 0 || iy >= input->height) continue;
                        for (int kx = 0; kx < kernel; ++kx) {
                            int ix = ox * stride + kx - padding;
                            if (ix < 0 || ix >= input->width) continue;
                            size_t wi = ((((size_t)oc * (size_t)input->channels +
                                           (size_t)ic) * (size_t)kernel +
                                          (size_t)ky) * (size_t)kernel +
                                         (size_t)kx);
                            grad_weights[wi] += go * input->data[tensor_index(input, ic, iy, ix)];
                            grad_input->data[tensor_index(grad_input, ic, iy, ix)] +=
                                go * weights[wi];
                        }
                    }
                }
            }
        }
    }
    return DET_OK;
}

det_status det_add_f32(const det_tensor_f32 *a, const det_tensor_f32 *b,
                       det_tensor_f32 *output) {
    if (a == NULL || b == NULL || output == NULL || a->data == NULL || b->data == NULL ||
        output->data == NULL || a->channels != b->channels || a->height != b->height ||
        a->width != b->width || output->channels != a->channels ||
        output->height != a->height || output->width != a->width) return DET_ERR_SHAPE;
    size_t elements = (size_t)a->channels * (size_t)a->height * (size_t)a->width;
    for (size_t i = 0U; i < elements; ++i) output->data[i] = a->data[i] + b->data[i];
    return DET_OK;
}

void det_relu_inplace(det_tensor_f32 *tensor) {
    if (tensor == NULL || tensor->data == NULL) return;
    size_t elements = 0U;
    if (!checked_size3(tensor->channels, tensor->height, tensor->width, &elements)) return;
    for (size_t i = 0U; i < elements; ++i) {
        if (tensor->data[i] < 0.0f) tensor->data[i] = 0.0f;
    }
}

det_status det_upsample_nearest(const det_tensor_f32 *input, int scale,
                                det_tensor_f32 *output) {
    if (input == NULL || output == NULL || input->data == NULL || output->data == NULL ||
        scale <= 0 || output->channels != input->channels ||
        output->height != input->height * scale || output->width != input->width * scale) {
        return DET_ERR_ARGUMENT;
    }
    for (int c = 0; c < input->channels; ++c) {
        for (int y = 0; y < output->height; ++y) {
            for (int x = 0; x < output->width; ++x) {
                output->data[tensor_index(output, c, y, x)] =
                    input->data[tensor_index(input, c, y / scale, x / scale)];
            }
        }
    }
    return DET_OK;
}

float det_sigmoid(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

float det_iou(const det_box *a, const det_box *b) {
    if (a == NULL || b == NULL) return 0.0f;
    float left = fmaxf(a->x1, b->x1);
    float top = fmaxf(a->y1, b->y1);
    float right = fminf(a->x2, b->x2);
    float bottom = fminf(a->y2, b->y2);
    float iw = fmaxf(0.0f, right - left);
    float ih = fmaxf(0.0f, bottom - top);
    float inter = iw * ih;
    float aa = fmaxf(0.0f, a->x2 - a->x1) * fmaxf(0.0f, a->y2 - a->y1);
    float ab = fmaxf(0.0f, b->x2 - b->x1) * fmaxf(0.0f, b->y2 - b->y1);
    float denom = aa + ab - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

int8_t det_quantize_s8(float value, float scale) {
    if (!(scale > 0.0f) || !isfinite(value)) return 0;
    float scaled = value / scale;
    if (scaled > 127.0f) scaled = 127.0f;
    if (scaled < -127.0f) scaled = -127.0f;
    long rounded = lroundf(scaled);
    if (rounded > 127L) rounded = 127L;
    if (rounded < -127L) rounded = -127L;
    return (int8_t)rounded;
}

int8_t det_requantize_i32(int64_t accumulator, int32_t multiplier, int shift,
                          int32_t zero_point) {
    if (shift < 0 || shift > 62) return 0;
    int64_t scaled = accumulator * (int64_t)multiplier;
    int64_t rounded = scaled;
    if (shift > 0) {
        int64_t offset = (int64_t)1 << (shift - 1);
        rounded = scaled >= 0 ? (scaled + offset) >> shift : -((-scaled + offset) >> shift);
    }
    rounded += (int64_t)zero_point;
    if (rounded > 127) rounded = 127;
    if (rounded < -128) rounded = -128;
    return (int8_t)rounded;
}

det_status det_quantize_per_channel_s8(const float *weights, int rows, int cols,
                                        int8_t *quantized, float *scales) {
    if (weights == NULL || quantized == NULL || scales == NULL || rows <= 0 || cols <= 0) {
        return DET_ERR_ARGUMENT;
    }
    for (int row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = fabsf(weights[(size_t)row * (size_t)cols + (size_t)col]);
            if (value > max_abs) max_abs = value;
        }
        scales[row] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        for (int col = 0; col < cols; ++col) {
            quantized[(size_t)row * (size_t)cols + (size_t)col] =
                det_quantize_s8(weights[(size_t)row * (size_t)cols + (size_t)col], scales[row]);
        }
    }
    return DET_OK;
}

det_status det_quantize_pack_w4(const float *weights, int rows, int cols,
                                uint8_t *packed, float *scales) {
    if (weights == NULL || packed == NULL || scales == NULL || rows <= 0 || cols <= 0) {
        return DET_ERR_ARGUMENT;
    }
    int packed_cols = (cols + 1) / 2;
    for (int row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = fabsf(weights[(size_t)row * (size_t)cols + (size_t)col]);
            if (value > max_abs) max_abs = value;
        }
        scales[row] = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
        for (int byte_index = 0; byte_index < packed_cols; ++byte_index) {
            int col0 = byte_index * 2;
            int col1 = col0 + 1;
            int8_t q0 = col0 < cols ? det_quantize_s8(
                weights[(size_t)row * (size_t)cols + (size_t)col0], scales[row]) : 0;
            int8_t q1 = col1 < cols ? det_quantize_s8(
                weights[(size_t)row * (size_t)cols + (size_t)col1], scales[row]) : 0;
            if (q0 > 7) q0 = 7;
            if (q0 < -8) q0 = -8;
            if (q1 > 7) q1 = 7;
            if (q1 < -8) q1 = -8;
            packed[(size_t)row * (size_t)packed_cols + (size_t)byte_index] =
                (uint8_t)(((uint8_t)q0 & 0x0fU) | (((uint8_t)q1 & 0x0fU) << 4U));
        }
    }
    return DET_OK;
}

static void free_head(det_head *head) {
    if (head == NULL) return;
    free(head->weights);
    free(head->bias);
    free(head->velocity_w);
    free(head->velocity_b);
    memset(head, 0, sizeof(*head));
}

static void free_feature_adapter(det_model *model) {
    if (model == NULL) return;
    free(model->feature_weights);
    free(model->feature_bias);
    free(model->feature_velocity_w);
    free(model->feature_velocity_b);
    free(model->stem_weights);
    free(model->stem_bias);
    free(model->stem_velocity_w);
    free(model->stem_velocity_b);
    free(model->integral);
    free(model->integral_sq);
    free(model->input_integral);
    model->feature_weights = NULL;
    model->feature_bias = NULL;
    model->feature_velocity_w = NULL;
    model->feature_velocity_b = NULL;
    model->stem_weights = NULL;
    model->stem_bias = NULL;
    model->stem_velocity_w = NULL;
    model->stem_velocity_b = NULL;
    model->integral = NULL;
    model->integral_sq = NULL;
    model->input_integral = NULL;
}

static int alloc_integrals(det_model *model) {
    size_t stride = (size_t)model->spec.width + 1U;
    size_t count = stride * ((size_t)model->spec.height + 1U);
    model->integral = (float *)calloc(count, sizeof(float));
    model->integral_sq = (float *)calloc(count, sizeof(float));
    model->input_integral = (float *)calloc((size_t)model->spec.channels * count, sizeof(float));
    return model->integral != NULL && model->integral_sq != NULL && model->input_integral != NULL;
}

static void build_integrals(det_model *model, const det_image *image) {
    size_t stride = (size_t)image->width + 1U;
    size_t count = stride * ((size_t)image->height + 1U);
    memset(model->integral, 0, count * sizeof(float));
    memset(model->integral_sq, 0, count * sizeof(float));
    for (int y = 1; y <= image->height; ++y) {
        float row_sum = 0.0f;
        float row_sq = 0.0f;
        for (int x = 1; x <= image->width; ++x) {
            float value = 0.0f;
            for (int c = 0; c < image->channels; ++c) {
                value += image->data[((size_t)c * (size_t)image->height + (size_t)(y - 1)) *
                                     (size_t)image->width + (size_t)(x - 1)];
            }
            value /= (float)image->channels;
            row_sum += value;
            row_sq += value * value;
            size_t current = (size_t)y * stride + (size_t)x;
            size_t above = (size_t)(y - 1) * stride + (size_t)x;
            model->integral[current] = model->integral[above] + row_sum;
            model->integral_sq[current] = model->integral_sq[above] + row_sq;
        }
    }
    size_t plane = stride * ((size_t)image->height + 1U);
    for (int c = 0; c < image->channels; ++c) {
        float *integral = model->input_integral + (size_t)c * plane;
        memset(integral, 0, plane * sizeof(float));
        for (int y = 1; y <= image->height; ++y) {
            float row_sum = 0.0f;
            for (int x = 1; x <= image->width; ++x) {
                row_sum += image->data[((size_t)c * (size_t)image->height + (size_t)(y - 1)) *
                                       (size_t)image->width + (size_t)(x - 1)];
                size_t current = (size_t)y * stride + (size_t)x;
                size_t above = (size_t)(y - 1) * stride + (size_t)x;
                integral[current] = integral[above] + row_sum;
            }
        }
    }
}

static det_status alloc_head(det_head *head, int height, int width, int channels,
                             int outputs) {
    if (head == NULL || height <= 0 || width <= 0 || channels <= 0 || outputs <= 0) {
        return DET_ERR_ARGUMENT;
    }
    size_t count = (size_t)channels * (size_t)outputs;
    head->weights = (float *)calloc(count, sizeof(float));
    head->bias = (float *)calloc((size_t)outputs, sizeof(float));
    head->velocity_w = (float *)calloc(count, sizeof(float));
    head->velocity_b = (float *)calloc((size_t)outputs, sizeof(float));
    if (head->weights == NULL || head->bias == NULL || head->velocity_w == NULL ||
        head->velocity_b == NULL) {
        free_head(head);
        return DET_ERR_MEMORY;
    }
    head->height = height;
    head->width = width;
    head->channels = channels;
    head->outputs = outputs;
    return DET_OK;
}

det_status det_model_build(det_context *ctx, const det_model_spec *spec,
                           det_model **out) {
    if (ctx == NULL || spec == NULL || out == NULL || spec->width <= 0 ||
        spec->height <= 0 || spec->channels <= 0 || spec->num_classes <= 0 ||
        spec->num_classes > DET_MAX_CLASSES || spec->max_detections <= 0) {
        return DET_ERR_ARGUMENT;
    }
    det_model *model = (det_model *)calloc(1U, sizeof(*model));
    if (model == NULL) return DET_ERR_MEMORY;
    model->ctx = ctx;
    model->spec = *spec;
    model->feature_channels = 4;
    size_t feature_weight_count = (size_t)model->feature_channels * 4U;
    model->feature_weights = (float *)calloc(feature_weight_count, sizeof(float));
    model->feature_bias = (float *)calloc((size_t)model->feature_channels, sizeof(float));
    model->feature_velocity_w = (float *)calloc(feature_weight_count, sizeof(float));
    model->feature_velocity_b = (float *)calloc((size_t)model->feature_channels, sizeof(float));
    size_t stem_weight_count = (size_t)model->feature_channels * (size_t)spec->channels;
    model->stem_weights = (float *)calloc(stem_weight_count, sizeof(float));
    model->stem_bias = (float *)calloc((size_t)model->feature_channels, sizeof(float));
    model->stem_velocity_w = (float *)calloc(stem_weight_count, sizeof(float));
    model->stem_velocity_b = (float *)calloc((size_t)model->feature_channels, sizeof(float));
    if (model->feature_weights == NULL || model->feature_bias == NULL ||
        model->feature_velocity_w == NULL || model->feature_velocity_b == NULL ||
        model->stem_weights == NULL || model->stem_bias == NULL ||
        model->stem_velocity_w == NULL || model->stem_velocity_b == NULL ||
        !alloc_integrals(model)) {
        free_feature_adapter(model);
        free(model);
        return DET_ERR_MEMORY;
    }
    for (int c = 0; c < model->feature_channels; ++c) {
        model->feature_weights[(size_t)c * 4U + (size_t)c] = 1.0f;
        if (c < spec->channels) model->stem_weights[(size_t)c * (size_t)spec->channels + (size_t)c] = 1.0f;
    }
    for (int i = 0; i < DET_MAX_SCALES; ++i) {
        int h = (spec->height + k_strides[i] - 1) / k_strides[i];
        int w = (spec->width + k_strides[i] - 1) / k_strides[i];
        int outputs = 4 + spec->num_classes;
        det_status status = alloc_head(&model->heads[i], h, w, model->feature_channels,
                                       outputs);
        if (status != DET_OK) {
            det_model_destroy(model);
            return status;
        }
    }
    *out = model;
    return DET_OK;
}

void det_model_destroy(det_model *model) {
    if (model == NULL) return;
    for (int i = 0; i < DET_MAX_SCALES; ++i) free_head(&model->heads[i]);
    free_feature_adapter(model);
    free(model);
}

det_status det_model_reset(det_model *model, int seed) {
    if (model == NULL) return DET_ERR_ARGUMENT;
    uint32_t state = (uint32_t)(seed == 0 ? 1 : seed);
    for (int o = 0; o < model->feature_channels; ++o) {
        for (int i = 0; i < 4; ++i) {
            size_t index = (size_t)o * 4U + (size_t)i;
            model->feature_weights[index] = o == i ? 1.0f : 0.0f;
            model->feature_velocity_w[index] = 0.0f;
        }
        model->feature_bias[o] = 0.0f;
        model->feature_velocity_b[o] = 0.0f;
        model->stem_bias[o] = 0.0f;
        model->stem_velocity_b[o] = 0.0f;
        for (int i = 0; i < model->spec.channels; ++i) {
            size_t stem_index = (size_t)o * (size_t)model->spec.channels + (size_t)i;
            model->stem_weights[stem_index] = o == i ? 1.0f : 0.0f;
            model->stem_velocity_w[stem_index] = 0.0f;
        }
    }
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head = &model->heads[s];
        size_t wc = (size_t)head->channels * (size_t)head->outputs;
        for (size_t i = 0U; i < wc; ++i) {
            state = state * 1664525U + 1013904223U;
            float r = (float)(state & 0xffffU) / 65535.0f;
            head->weights[i] = (r - 0.5f) * 0.02f;
            head->velocity_w[i] = 0.0f;
        }
        for (int i = 0; i < head->outputs; ++i) {
            head->bias[i] = 0.0f;
            head->velocity_b[i] = 0.0f;
        }
    }
    model->train_updates = 0U;
    return DET_OK;
}

static void build_stem(det_model *model, const det_image *image) {
    int pixels = image->width * image->height;
    float sums[4] = {0.0f};
    for (int i = 0; i < image->channels; ++i)
        for (int pixel = 0; pixel < pixels; ++pixel)
            sums[i] += image->data[(size_t)i * (size_t)pixels + (size_t)pixel];
    for (int i = 0; i < image->channels; ++i) model->input_mean[i] = sums[i] / (float)pixels;
}

static void extract_feature(const det_model *model, const det_image *image, int scale,
                            int cell_y, int cell_x, float features[4], float base[4]) {
    int stride = k_strides[scale];
    int x0 = cell_x * stride;
    int y0 = cell_y * stride;
    int x1 = x0 + stride;
    int y1 = y0 + stride;
    if (x1 > image->width) x1 = image->width;
    if (y1 > image->height) y1 = image->height;
    int count = (x1 - x0) * (y1 - y0);
    float pooled[4] = {0.0f};
    size_t integral_stride = (size_t)image->width + 1U;
    size_t a = (size_t)y0 * integral_stride + (size_t)x0;
    size_t b = (size_t)y0 * integral_stride + (size_t)x1;
    size_t below = (size_t)y1 * integral_stride + (size_t)x0;
    size_t d = (size_t)y1 * integral_stride + (size_t)x1;
    for (int i = 0; i < model->feature_channels; ++i) {
        float transformed = model->stem_bias[i];
        for (int j = 0; j < image->channels; ++j) {
            const float *channel_integral = model->input_integral + (size_t)j *
                                            integral_stride * ((size_t)image->height + 1U);
            float channel_sum = channel_integral[d] - channel_integral[b] -
                                channel_integral[below] + channel_integral[a];
            float channel_mean = count > 0 ? channel_sum / (float)count : 0.0f;
            transformed += model->stem_weights[(size_t)i * (size_t)image->channels + (size_t)j] * channel_mean;
        }
        pooled[i] = transformed;
    }
    for (int i = 0; i < model->feature_channels; ++i) {
        if (base != NULL) base[i] = pooled[i];
        float value = model->feature_bias[i];
        for (int j = 0; j < 4; ++j) {
            value += model->feature_weights[(size_t)i * 4U + (size_t)j] * pooled[j];
        }
        features[i] = value;
    }
}

static void head_forward(const det_head *head, const float features[4], float *output) {
    for (int o = 0; o < head->outputs; ++o) {
        float value = head->bias[o];
        for (int c = 0; c < head->channels; ++c) {
            value += head->weights[(size_t)o * (size_t)head->channels + (size_t)c] * features[c];
        }
        output[o] = value;
    }
}

static int choose_scale(const det_box *box, int width, int height) {
    float bw = (box->x2 - box->x1) / (float)width;
    float bh = (box->y2 - box->y1) / (float)height;
    float size = fmaxf(bw, bh);
    return size < 0.2f ? 0 : (size < 0.5f ? 1 : 2);
}

static void update_head(det_head *head, const float features[4], const float *target,
                        float lr, float momentum) {
    float prediction[4 + DET_MAX_CLASSES];
    head_forward(head, features, prediction);
    for (int o = 0; o < head->outputs; ++o) {
        float error = prediction[o] - target[o];
        size_t base = (size_t)o * (size_t)head->channels;
        for (int c = 0; c < head->channels; ++c) {
            float grad = error * features[c];
            head->velocity_w[base + (size_t)c] = momentum * head->velocity_w[base + (size_t)c] + grad;
            head->weights[base + (size_t)c] -= lr * head->velocity_w[base + (size_t)c];
        }
        head->velocity_b[o] = momentum * head->velocity_b[o] + error;
        head->bias[o] -= lr * head->velocity_b[o];
    }
}

static void feature_gradient_from_error(const det_head *head, const float *prediction,
                                        const float *target, float gradient[4]) {
    for (int c = 0; c < 4; ++c) gradient[c] = 0.0f;
    for (int o = 0; o < head->outputs; ++o) {
        float error = prediction[o] - target[o];
        for (int c = 0; c < 4; ++c) {
            gradient[c] += error * head->weights[(size_t)o * (size_t)head->channels + (size_t)c];
        }
    }
}

static void update_feature_adapter(det_model *model, const float base[4],
                                    const float gradient[4], float lr, float momentum) {
    for (int o = 0; o < model->feature_channels; ++o) {
        for (int i = 0; i < 4; ++i) {
            size_t index = (size_t)o * 4U + (size_t)i;
            float grad = gradient[o] * base[i];
            model->feature_velocity_w[index] = momentum * model->feature_velocity_w[index] + grad;
            model->feature_weights[index] -= lr * model->feature_velocity_w[index];
        }
        model->feature_velocity_b[o] = momentum * model->feature_velocity_b[o] + gradient[o];
        model->feature_bias[o] -= lr * model->feature_velocity_b[o];
    }
}

static void update_stem(det_model *model, const float gradient[4], float lr, float momentum) {
    float pooled_gradient[4] = {0.0f};
    for (int j = 0; j < model->feature_channels; ++j) {
        for (int o = 0; o < model->feature_channels; ++o) {
            pooled_gradient[j] += gradient[o] *
                                  model->feature_weights[(size_t)o * 4U + (size_t)j];
        }
    }
    for (int o = 0; o < model->feature_channels; ++o) {
        for (int i = 0; i < model->spec.channels; ++i) {
            size_t index = (size_t)o * (size_t)model->spec.channels + (size_t)i;
            float grad = pooled_gradient[o] * model->input_mean[i];
            model->stem_velocity_w[index] = momentum * model->stem_velocity_w[index] + grad;
            model->stem_weights[index] -= lr * model->stem_velocity_w[index];
        }
        model->stem_velocity_b[o] = momentum * model->stem_velocity_b[o] + pooled_gradient[o];
        model->stem_bias[o] -= lr * model->stem_velocity_b[o];
    }
}

static void accumulate_stem_gradient(const det_model *model, const float gradient[4],
                                     float gradient_w[4][4], float gradient_b[4]) {
    float pooled_gradient[4] = {0.0f};
    for (int j = 0; j < model->feature_channels; ++j) {
        for (int o = 0; o < model->feature_channels; ++o) {
            pooled_gradient[j] += gradient[o] *
                                  model->feature_weights[(size_t)o * 4U + (size_t)j];
        }
    }
    for (int o = 0; o < model->feature_channels; ++o) {
        gradient_b[o] += pooled_gradient[o];
        for (int i = 0; i < model->spec.channels; ++i) {
            gradient_w[o][i] += pooled_gradient[o] * model->input_mean[i];
        }
    }
}

static void apply_head_gradient(det_head *head, float gradient_w[][4],
                                const float *gradient_b, float lr, float momentum,
                                float scale) {
    for (int o = 0; o < head->outputs; ++o) {
        size_t base = (size_t)o * (size_t)head->channels;
        for (int c = 0; c < head->channels; ++c) {
            float grad = gradient_w[o][c] * scale;
            head->velocity_w[base + (size_t)c] = momentum * head->velocity_w[base + (size_t)c] + grad;
            head->weights[base + (size_t)c] -= lr * head->velocity_w[base + (size_t)c];
        }
        float grad_b = gradient_b[o] * scale;
        head->velocity_b[o] = momentum * head->velocity_b[o] + grad_b;
        head->bias[o] -= lr * head->velocity_b[o];
    }
}

static float train_sample(det_model *model, const det_sample *sample,
                           const det_train_config *config, size_t sample_index,
                           size_t *updates) {
    build_integrals(model, &sample->image);
    build_stem(model, &sample->image);
    float loss = 0.0f;
    float target[4 + DET_MAX_CLASSES];
    float prediction[4 + DET_MAX_CLASSES];
    float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4] = {{{0.0f}}};
    float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES] = {{0.0f}};
    float feature_gradient_w[4][4] = {{0.0f}};
    float feature_gradient_b[4] = {0.0f};
    float stem_gradient_w[4][4] = {{0.0f}};
    float stem_gradient_b[4] = {0.0f};
    int use_global = config->mode == DET_TRAIN_GLOBAL_BP;
    size_t terms = 0U;
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head = &model->heads[s];
        terms += (size_t)head->height * (size_t)head->width * (size_t)head->outputs;
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                float features[4];
                float base_features[4];
                extract_feature(model, &sample->image, s, y, x, features, base_features);
                for (int o = 0; o < head->outputs; ++o) target[o] = 0.0f;
                int positive = -1;
                const det_box *assigned = NULL;
                for (int b = 0; b < sample->box_count; ++b) {
                    if (sample->boxes[b].class_id < 0 || sample->boxes[b].class_id >= model->spec.num_classes) continue;
                    if (choose_scale(&sample->boxes[b], sample->image.width, sample->image.height) != s) continue;
                    int tx = (int)(((sample->boxes[b].x1 + sample->boxes[b].x2) * 0.5f) /
                                   (float)k_strides[s]);
                    int ty = (int)(((sample->boxes[b].y1 + sample->boxes[b].y2) * 0.5f) /
                                   (float)k_strides[s]);
                    if (tx == x && ty == y) {
                        positive = sample->boxes[b].class_id;
                        assigned = &sample->boxes[b];
                        break;
                    }
                }
                if (positive >= 0 && assigned != NULL) {
                    target[positive + 4] = 1.0f;
                    float gx = ((assigned->x1 + assigned->x2) * 0.5f) / (float)sample->image.width;
                    float gy = ((assigned->y1 + assigned->y2) * 0.5f) / (float)sample->image.height;
                    float cx = ((float)x + 0.5f) / (float)head->width;
                    float cy = ((float)y + 0.5f) / (float)head->height;
                    target[0] = fmaxf(0.0f, (cx - assigned->x1 / (float)sample->image.width) * (float)head->width);
                    target[1] = fmaxf(0.0f, (cy - assigned->y1 / (float)sample->image.height) * (float)head->height);
                    target[2] = fmaxf(0.0f, (assigned->x2 / (float)sample->image.width - cx) * (float)head->width);
                    target[3] = fmaxf(0.0f, (assigned->y2 / (float)sample->image.height - cy) * (float)head->height);
                    (void)gx;
                    (void)gy;
                }
                head_forward(head, features, prediction);
                for (int o = 0; o < head->outputs; ++o) {
                    float error = prediction[o] - target[o];
                    loss += error * error;
                    if (use_global) {
                        gradient_b[s][o] += error;
                        for (int c = 0; c < 4; ++c) {
                            gradient_w[s][o][c] += error * features[c];
                        }
                    }
                }
                int should_update = positive >= 0 || ((x + y + (int)sample_index) % 8 == 0);
                if (use_global) {
                    float feature_gradient[4];
                    feature_gradient_from_error(head, prediction, target, feature_gradient);
                    for (int o = 0; o < model->feature_channels; ++o) {
                        feature_gradient_b[o] += feature_gradient[o];
                        for (int i = 0; i < 4; ++i) {
                            feature_gradient_w[o][i] += feature_gradient[o] * base_features[i];
                        }
                    }
                    accumulate_stem_gradient(model, feature_gradient, stem_gradient_w, stem_gradient_b);
                    ++(*updates);
                } else if (should_update) {
                    float feature_gradient[4];
                    feature_gradient_from_error(head, prediction, target, feature_gradient);
                    update_head(head, features, target, config->learning_rate, config->momentum);
                    update_feature_adapter(model, base_features, feature_gradient,
                                           config->learning_rate, config->momentum);
                    update_stem(model, feature_gradient, config->learning_rate, config->momentum);
                    ++(*updates);
                }
            }
        }
    }
    if (use_global && terms > 0U) {
        float scale = 1.0f / (float)terms;
        for (int s = 0; s < DET_MAX_SCALES; ++s) {
            apply_head_gradient(&model->heads[s], gradient_w[s], gradient_b[s],
                                config->learning_rate, config->momentum, scale);
        }
        for (int o = 0; o < model->feature_channels; ++o) {
            for (int i = 0; i < 4; ++i) {
                size_t index = (size_t)o * 4U + (size_t)i;
                float grad = feature_gradient_w[o][i] * scale;
                model->feature_velocity_w[index] = config->momentum * model->feature_velocity_w[index] + grad;
                model->feature_weights[index] -= config->learning_rate * model->feature_velocity_w[index];
            }
            model->feature_velocity_b[o] = config->momentum * model->feature_velocity_b[o] +
                                           feature_gradient_b[o] * scale;
            model->feature_bias[o] -= config->learning_rate * model->feature_velocity_b[o];
            for (int i = 0; i < model->spec.channels; ++i) {
                size_t index = (size_t)o * (size_t)model->spec.channels + (size_t)i;
                float grad = stem_gradient_w[o][i] * scale;
                model->stem_velocity_w[index] = config->momentum * model->stem_velocity_w[index] + grad;
                model->stem_weights[index] -= config->learning_rate * model->stem_velocity_w[index];
            }
            model->stem_velocity_b[o] = config->momentum * model->stem_velocity_b[o] +
                                        stem_gradient_b[o] * scale;
            model->stem_bias[o] -= config->learning_rate * model->stem_velocity_b[o];
        }
    }
    return terms > 0U ? loss / (float)terms : 0.0f;
}

det_status det_train(det_model *model, const det_dataset *dataset,
                     const det_train_config *config, det_train_report *report) {
    if (model == NULL || dataset == NULL || config == NULL || report == NULL ||
        dataset->next == NULL || config->epochs <= 0 || config->learning_rate <= 0.0f ||
        config->momentum < 0.0f || config->momentum >= 1.0f) return DET_ERR_ARGUMENT;
    if (config->precision != DET_PRECISION_F32) {
        return DET_ERR_UNSUPPORTED;
    }
    if (det_model_reset(model, config->seed) != DET_OK) return DET_ERR_MEMORY;
    memset(report, 0, sizeof(*report));
    double start = now_ms();
    float loss_sum = 0.0f;
    size_t updates = 0U;
    for (int epoch = 0; epoch < config->epochs; ++epoch) {
        if (dataset->reset != NULL) dataset->reset(dataset->user);
        det_sample sample;
        size_t seen = 0U;
        while ((config->max_samples <= 0 || seen < (size_t)config->max_samples) &&
               dataset->next(dataset->user, &sample) > 0) {
            loss_sum += train_sample(model, &sample, config, seen, &updates);
            ++seen;
            ++report->samples_seen;
        }
        if (seen == 0U) return DET_ERR_ARGUMENT;
    }
    model->train_updates += updates;
    report->updates = updates;
    report->elapsed_ms = now_ms() - start;
    report->mean_loss = report->samples_seen > 0U ? loss_sum / (float)report->samples_seen : 0.0f;
    report->used_global_backward = config->mode == DET_TRAIN_GLOBAL_BP ? 1 : 0;
    return DET_OK;
}

static void insert_detection(det_detection *detections, int *count, int capacity,
                             const det_detection *candidate) {
    if (capacity <= 0 || count == NULL || candidate == NULL) return;
    if (*count < capacity) {
        detections[*count] = *candidate;
        ++(*count);
        return;
    }
    int lowest = 0;
    for (int i = 1; i < *count; ++i) {
        if (detections[i].score < detections[lowest].score) lowest = i;
    }
    if (candidate->score > detections[lowest].score) detections[lowest] = *candidate;
}

static void sort_detections_descending(det_detection *detections, int count) {
    for (int i = 1; i < count; ++i) {
        det_detection value = detections[i];
        int j = i;
        while (j > 0 && detections[j - 1].score < value.score) {
            detections[j] = detections[j - 1];
            --j;
        }
        detections[j] = value;
    }
}

det_status det_predict(const det_model *model, const det_image *image,
                       float score_threshold, det_detection *detections,
                       int capacity, int *count) {
    if (model == NULL || image == NULL || image->data == NULL || detections == NULL ||
        count == NULL || capacity < 0 || image->channels != model->spec.channels ||
        image->width != model->spec.width || image->height != model->spec.height) {
        return DET_ERR_ARGUMENT;
    }
    *count = 0;
    build_integrals((det_model *)model, image);
    build_stem((det_model *)model, image);
    float output[4 + DET_MAX_CLASSES];
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        const det_head *head = &model->heads[s];
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                float features[4];
                extract_feature(model, image, s, y, x, features, NULL);
                head_forward(head, features, output);
                int class_id = 0;
                float best = output[4];
                for (int c = 1; c < model->spec.num_classes; ++c) {
                    if (output[c + 4] > best) {
                        best = output[c + 4];
                        class_id = c;
                    }
                }
                float score = det_sigmoid(best);
                if (score < score_threshold) continue;
                float cx = ((float)x + 0.5f) * (float)k_strides[s];
                float cy = ((float)y + 0.5f) * (float)k_strides[s];
                float left = fmaxf(0.0f, output[0]) * (float)k_strides[s];
                float top = fmaxf(0.0f, output[1]) * (float)k_strides[s];
                float right = fmaxf(0.0f, output[2]) * (float)k_strides[s];
                float bottom = fmaxf(0.0f, output[3]) * (float)k_strides[s];
                det_detection candidate;
                candidate.score = score;
                candidate.box.x1 = fmaxf(0.0f, cx - left);
                candidate.box.y1 = fmaxf(0.0f, cy - top);
                candidate.box.x2 = fminf((float)image->width, cx + right);
                candidate.box.y2 = fminf((float)image->height, cy + bottom);
                candidate.box.class_id = class_id;
                insert_detection(detections, count, capacity, &candidate);
            }
        }
    }
    sort_detections_descending(detections, *count);
    return DET_OK;
}

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t classes;
    uint32_t max_detections;
    uint32_t feature_channels;
} det_file_header;

det_status det_save(const det_model *model, const char *path) {
    if (model == NULL || path == NULL) return DET_ERR_ARGUMENT;
    FILE *file = fopen(path, "wb");
    if (file == NULL) return DET_ERR_IO;
    det_file_header header = {{'C', 'D', 'E', 'T'}, 1U, (uint32_t)model->spec.width,
                              (uint32_t)model->spec.height, (uint32_t)model->spec.channels,
                              (uint32_t)model->spec.num_classes, (uint32_t)model->spec.max_detections,
                              (uint32_t)model->feature_channels};
    int ok = fwrite(&header, sizeof(header), 1U, file) == 1U;
    if (ok) {
        size_t stem_weight_count = (size_t)model->feature_channels * (size_t)model->spec.channels;
        ok = fwrite(model->feature_weights, sizeof(float),
                    (size_t)model->feature_channels * 4U, file) ==
                 (size_t)model->feature_channels * 4U &&
             fwrite(model->feature_bias, sizeof(float),
                    (size_t)model->feature_channels, file) ==
                 (size_t)model->feature_channels &&
             fwrite(model->stem_weights, sizeof(float), stem_weight_count, file) == stem_weight_count &&
             fwrite(model->stem_bias, sizeof(float),
                    (size_t)model->feature_channels, file) ==
                 (size_t)model->feature_channels;
    }
    for (int s = 0; ok && s < DET_MAX_SCALES; ++s) {
        const det_head *head = &model->heads[s];
        size_t wc = (size_t)head->channels * (size_t)head->outputs;
        ok = fwrite(&head->height, sizeof(head->height), 1U, file) == 1U &&
             fwrite(&head->width, sizeof(head->width), 1U, file) == 1U &&
             fwrite(head->weights, sizeof(float), wc, file) == wc &&
             fwrite(head->bias, sizeof(float), (size_t)head->outputs, file) == (size_t)head->outputs;
    }
    int close_result = fclose(file);
    return ok && close_result == 0 ? DET_OK : DET_ERR_IO;
}

det_status det_load(det_context *ctx, const char *path, det_model **out) {
    if (ctx == NULL || path == NULL || out == NULL) return DET_ERR_ARGUMENT;
    FILE *file = fopen(path, "rb");
    if (file == NULL) return DET_ERR_IO;
    det_file_header header;
    int ok = fread(&header, sizeof(header), 1U, file) == 1U;
    if (!ok || memcmp(header.magic, "CDET", 4U) != 0 || header.version != 1U) {
        fclose(file);
        return DET_ERR_FORMAT;
    }
    det_model_spec spec = {(int)header.width, (int)header.height, (int)header.channels,
                           (int)header.classes, (int)header.max_detections, 1};
    det_model *model = NULL;
    det_status status = det_model_build(ctx, &spec, &model);
    if (status != DET_OK) {
        fclose(file);
        return status;
    }
    if ((int)header.feature_channels != model->feature_channels) {
        det_model_destroy(model);
        fclose(file);
        return DET_ERR_FORMAT;
    }
    size_t feature_weight_count = (size_t)model->feature_channels * 4U;
    size_t stem_weight_count = (size_t)model->feature_channels * (size_t)model->spec.channels;
    ok = fread(model->feature_weights, sizeof(float), feature_weight_count, file) ==
             feature_weight_count &&
         fread(model->feature_bias, sizeof(float), (size_t)model->feature_channels, file) ==
             (size_t)model->feature_channels &&
         fread(model->stem_weights, sizeof(float), stem_weight_count, file) == stem_weight_count &&
         fread(model->stem_bias, sizeof(float), (size_t)model->feature_channels, file) ==
             (size_t)model->feature_channels;
    for (int s = 0; ok && s < DET_MAX_SCALES; ++s) {
        det_head *head = &model->heads[s];
        int h = 0;
        int w = 0;
        size_t wc = (size_t)head->channels * (size_t)head->outputs;
        ok = fread(&h, sizeof(h), 1U, file) == 1U && fread(&w, sizeof(w), 1U, file) == 1U &&
             h == head->height && w == head->width &&
             fread(head->weights, sizeof(float), wc, file) == wc &&
             fread(head->bias, sizeof(float), (size_t)head->outputs, file) == (size_t)head->outputs;
    }
    fclose(file);
    if (!ok) {
        det_model_destroy(model);
        return DET_ERR_FORMAT;
    }
    *out = model;
    return DET_OK;
}
