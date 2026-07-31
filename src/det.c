#include "det.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif

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
    int8_t *quant_weights;
    uint8_t *packed_weights;
    float *quant_scales;
} det_head;

enum { DET_GRAPH_CHANNELS = 4, DET_GRAPH_STEM_CHANNELS = 1, DET_GRAPH_STAGES = 5 };

typedef struct {
    int input_channels;
    int output_channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int kernel;
    int stride;
    int padding;
    int depthwise;
    float *weights;
    float *bias;
    float *velocity_w;
    float *velocity_b;
    float *gradient_w;
    float *gradient_b;
    int8_t *quant_weights;
    uint8_t *packed_weights;
    float *quant_scales;
    float *activation;
    float *gradient_output;
    float *gradient_input;
} det_stage;

struct det_model {
    det_context *ctx;
    det_model_spec spec;
    det_stage stages[DET_GRAPH_STAGES];
    det_head heads[DET_MAX_SCALES];
    det_head aux_heads[DET_MAX_SCALES];
    det_precision precision;
    size_t train_updates;
};

static const int k_strides[DET_MAX_SCALES] = {8, 16, 32};

static double now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (QueryPerformanceCounter(&counter) && QueryPerformanceFrequency(&frequency) &&
        frequency.QuadPart > 0) {
        return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
    }
#elif defined(CLOCK_MONOTONIC)
    struct timespec monotonic_ts;
    if (clock_gettime(CLOCK_MONOTONIC, &monotonic_ts) == 0) {
        return (double)monotonic_ts.tv_sec * 1000.0 +
               (double)monotonic_ts.tv_nsec / 1000000.0;
    }
#endif
    struct timespec fallback_ts;
    (void)timespec_get(&fallback_ts, TIME_UTC);
    return (double)fallback_ts.tv_sec * 1000.0 +
           (double)fallback_ts.tv_nsec / 1000000.0;
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
    if (value > SIZE_MAX - mask) return SIZE_MAX;
    return (value + mask) & ~mask;
}

static int checked_add_size(size_t left, size_t right, size_t *out) {
    if (out == NULL || left > SIZE_MAX - right) return 0;
    *out = left + right;
    return 1;
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
    if (start == SIZE_MAX || start > arena->capacity || bytes > arena->capacity - start) return NULL;
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

static int convolution_output_dim(int input, int kernel, int stride, int padding,
                                  int *output) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || padding < 0 || output == NULL) return 0;
    int64_t numerator = (int64_t)input + 2LL * (int64_t)padding - (int64_t)kernel;
    if (numerator < 0 || numerator > INT_MAX) return 0;
    int64_t result = numerator / (int64_t)stride + 1LL;
    if (result <= 0 || result > INT_MAX) return 0;
    *output = (int)result;
    return 1;
}

det_status det_conv2d_f32(const det_tensor_f32 *input, const float *weights,
                         const float *bias, int out_channels, int kernel,
                         int stride, int padding, det_tensor_f32 *output) {
    if (input == NULL || weights == NULL || bias == NULL || output == NULL ||
        input->data == NULL || output->data == NULL || out_channels <= 0 ||
        kernel <= 0 || stride <= 0 || padding < 0) return DET_ERR_ARGUMENT;
    int expected_h = 0;
    int expected_w = 0;
    if (!convolution_output_dim(input->height, kernel, stride, padding, &expected_h) ||
        !convolution_output_dim(input->width, kernel, stride, padding, &expected_w)) {
        return DET_ERR_SHAPE;
    }
    if (output->channels != out_channels || output->height != expected_h ||
        output->width != expected_w) return DET_ERR_SHAPE;
    if (kernel == 3 && stride == 2 && padding == 1) {
        size_t input_plane = (size_t)input->height * (size_t)input->width;
        size_t output_plane = (size_t)expected_h * (size_t)expected_w;
        for (int oc = 0; oc < out_channels; ++oc) {
            const float *output_weights = weights +
                (size_t)oc * (size_t)input->channels * 9U;
            float *output_channel = output->data + (size_t)oc * output_plane;
            for (int oy = 0; oy < expected_h; ++oy) {
                int iy_base = oy * 2 - 1;
                int ky_start = iy_base < 0 ? -iy_base : 0;
                int ky_end = input->height - iy_base;
                if (ky_end > 3) ky_end = 3;
                if (ky_start > ky_end) ky_start = ky_end;
                for (int ox = 0; ox < expected_w; ++ox) {
                    int ix_base = ox * 2 - 1;
                    int kx_start = ix_base < 0 ? -ix_base : 0;
                    int kx_end = input->width - ix_base;
                    if (kx_end > 3) kx_end = 3;
                    if (kx_start > kx_end) kx_start = kx_end;
                    float sum = bias[oc];
                    for (int ic = 0; ic < input->channels; ++ic) {
                        const float *input_channel = input->data +
                            (size_t)ic * input_plane;
                        const float *channel_weights = output_weights + (size_t)ic * 9U;
                        for (int ky = ky_start; ky < ky_end; ++ky) {
                            const float *input_row = input_channel +
                                (size_t)(iy_base + ky) * (size_t)input->width +
                                (size_t)(ix_base + kx_start);
                            const float *weight_row = channel_weights + (size_t)ky * 3U +
                                (size_t)kx_start;
                            for (int kx = kx_start; kx < kx_end; ++kx) {
                                sum += weight_row[kx - kx_start] *
                                       input_row[kx - kx_start];
                            }
                        }
                    }
                    output_channel[(size_t)oy * (size_t)expected_w + (size_t)ox] = sum;
                }
            }
        }
        return DET_OK;
    }
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
    int output_h = 0;
    int output_w = 0;
    if (!convolution_output_dim(input->height, kernel, stride, padding, &output_h) ||
        !convolution_output_dim(input->width, kernel, stride, padding, &output_w)) {
        return DET_ERR_SHAPE;
    }
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
    if (!(scale > 0.0f) || !isfinite(scale) || !isfinite(value)) return 0;
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
#if defined(__SIZEOF_INT128__)
    __extension__ typedef __int128 det_int128_t;
    det_int128_t scaled = (det_int128_t)accumulator * (det_int128_t)multiplier;
    det_int128_t rounded = scaled;
    if (shift > 0) {
        det_int128_t divisor = (det_int128_t)1 << shift;
        det_int128_t offset = divisor >> 1;
        rounded = scaled >= 0 ? (scaled + offset) / divisor
                              : -((-scaled + offset) / divisor);
    }
    rounded += (det_int128_t)zero_point;
    if (rounded > 127) rounded = 127;
    if (rounded < -128) rounded = -128;
    return (int8_t)rounded;
#else
    long double scaled = (long double)accumulator * (long double)multiplier;
    if (shift > 0) scaled /= ldexpl(1.0L, shift);
    long double rounded = roundl(scaled) + (long double)zero_point;
    if (rounded > 127.0L) rounded = 127.0L;
    if (rounded < -128.0L) rounded = -128.0L;
    return (int8_t)rounded;
#endif
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
            if (!isfinite(value)) return DET_ERR_ARGUMENT;
            if (value > max_abs) max_abs = value;
        }
        scales[row] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        if (!isfinite(scales[row]) || !(scales[row] > 0.0f)) return DET_ERR_ARGUMENT;
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
            if (!isfinite(value)) return DET_ERR_ARGUMENT;
            if (value > max_abs) max_abs = value;
        }
        scales[row] = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
        if (!isfinite(scales[row]) || !(scales[row] > 0.0f)) return DET_ERR_ARGUMENT;
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
    free(head->quant_weights);
    free(head->packed_weights);
    free(head->quant_scales);
    memset(head, 0, sizeof(*head));
}

static void free_stage(det_stage *stage) {
    if (stage == NULL) return;
    free(stage->weights);
    free(stage->bias);
    free(stage->velocity_w);
    free(stage->velocity_b);
    free(stage->gradient_w);
    free(stage->gradient_b);
    free(stage->quant_weights);
    free(stage->packed_weights);
    free(stage->quant_scales);
    free(stage->activation);
    free(stage->gradient_output);
    free(stage->gradient_input);
    memset(stage, 0, sizeof(*stage));
}

static size_t stage_weight_count(const det_stage *stage) {
    size_t channels = stage->depthwise ? (size_t)stage->output_channels :
                      (size_t)stage->output_channels * (size_t)stage->input_channels;
    return channels * (size_t)stage->kernel * (size_t)stage->kernel;
}

static size_t stage_quant_cols(const det_stage *stage) {
    return (size_t)(stage->depthwise ? 1 : stage->input_channels) *
           (size_t)stage->kernel * (size_t)stage->kernel;
}

static size_t stage_packed_weight_count(const det_stage *stage) {
    return (size_t)stage->output_channels * ((stage_quant_cols(stage) + 1U) / 2U);
}

static size_t stage_weight_index(const det_stage *stage, int oc, int ic, int ky, int kx) {
    int channel = stage->depthwise ? oc : oc * stage->input_channels + ic;
    return (((size_t)channel * (size_t)stage->kernel + (size_t)ky) *
            (size_t)stage->kernel + (size_t)kx);
}

static int finite_values(const float *values, size_t count) {
    if (values == NULL) return 0;
    for (size_t i = 0U; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static int positive_finite_values(const float *values, size_t count) {
    if (values == NULL) return 0;
    for (size_t i = 0U; i < count; ++i) {
        if (!(values[i] > 0.0f) || !isfinite(values[i])) return 0;
    }
    return 1;
}

static uint32_t crc32_bytes(const uint8_t *data, size_t length) {
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0U; i < length; ++i) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1U);
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

static det_status alloc_stage(det_stage *stage, int input_channels, int input_height,
                              int input_width, int output_channels, int kernel,
                              int stride, int padding, int depthwise) {
    if (stage == NULL || input_channels <= 0 || input_height <= 0 || input_width <= 0 ||
        output_channels <= 0 || kernel <= 0 || stride <= 0 || padding < 0) {
        return DET_ERR_ARGUMENT;
    }
    int output_height = 0;
    int output_width = 0;
    if (!convolution_output_dim(input_height, kernel, stride, padding, &output_height) ||
        !convolution_output_dim(input_width, kernel, stride, padding, &output_width)) {
        return DET_ERR_SHAPE;
    }
    if (depthwise && input_channels != output_channels) return DET_ERR_SHAPE;
    size_t weight_count = (size_t)(depthwise ? output_channels :
                                   output_channels * input_channels) *
                          (size_t)kernel * (size_t)kernel;
    size_t output_count = (size_t)output_channels * (size_t)output_height *
                          (size_t)output_width;
    size_t input_count = (size_t)input_channels * (size_t)input_height *
                         (size_t)input_width;
    size_t quant_cols = (size_t)(depthwise ? 1 : input_channels) *
                        (size_t)kernel * (size_t)kernel;
    size_t packed_count = (size_t)output_channels * ((quant_cols + 1U) / 2U);
    stage->weights = (float *)calloc(weight_count, sizeof(float));
    stage->bias = (float *)calloc((size_t)output_channels, sizeof(float));
    stage->velocity_w = (float *)calloc(weight_count, sizeof(float));
    stage->velocity_b = (float *)calloc((size_t)output_channels, sizeof(float));
    stage->gradient_w = (float *)calloc(weight_count, sizeof(float));
    stage->gradient_b = (float *)calloc((size_t)output_channels, sizeof(float));
    stage->quant_weights = (int8_t *)calloc(weight_count, sizeof(int8_t));
    stage->packed_weights = (uint8_t *)calloc(packed_count, sizeof(uint8_t));
    stage->quant_scales = (float *)calloc((size_t)output_channels, sizeof(float));
    stage->activation = (float *)calloc(output_count, sizeof(float));
    stage->gradient_output = (float *)calloc(output_count, sizeof(float));
    stage->gradient_input = (float *)calloc(input_count, sizeof(float));
    if (stage->weights == NULL || stage->bias == NULL || stage->velocity_w == NULL ||
        stage->velocity_b == NULL || stage->gradient_w == NULL || stage->gradient_b == NULL ||
        stage->quant_weights == NULL || stage->packed_weights == NULL ||
        stage->quant_scales == NULL ||
        stage->activation == NULL || stage->gradient_output == NULL ||
        stage->gradient_input == NULL) {
        free_stage(stage);
        return DET_ERR_MEMORY;
    }
    stage->input_channels = input_channels;
    stage->output_channels = output_channels;
    stage->input_height = input_height;
    stage->input_width = input_width;
    stage->output_height = output_height;
    stage->output_width = output_width;
    stage->kernel = kernel;
    stage->stride = stride;
    stage->padding = padding;
    stage->depthwise = depthwise;
    return DET_OK;
}

static det_status alloc_head(det_head *head, int height, int width, int channels,
                             int outputs) {
    if (head == NULL || height <= 0 || width <= 0 || channels <= 0 || outputs <= 0) {
        return DET_ERR_ARGUMENT;
    }
    size_t count = (size_t)channels * (size_t)outputs;
    size_t packed_count = (size_t)outputs * (((size_t)channels + 1U) / 2U);
    head->weights = (float *)calloc(count, sizeof(float));
    head->bias = (float *)calloc((size_t)outputs, sizeof(float));
    head->velocity_w = (float *)calloc(count, sizeof(float));
    head->velocity_b = (float *)calloc((size_t)outputs, sizeof(float));
    head->quant_weights = (int8_t *)calloc(count, sizeof(int8_t));
    head->packed_weights = (uint8_t *)calloc(packed_count, sizeof(uint8_t));
    head->quant_scales = (float *)calloc((size_t)outputs, sizeof(float));
    if (head->weights == NULL || head->bias == NULL || head->velocity_w == NULL ||
        head->velocity_b == NULL || head->quant_weights == NULL ||
        head->packed_weights == NULL || head->quant_scales == NULL) {
        free_head(head);
        return DET_ERR_MEMORY;
    }
    head->height = height;
    head->width = width;
    head->channels = channels;
    head->outputs = outputs;
    return DET_OK;
}

static int valid_precision(det_precision precision) {
    return precision == DET_PRECISION_F32 || precision == DET_PRECISION_INT8 ||
           precision == DET_PRECISION_W4A8;
}

static det_status quantize_stage(det_stage *stage, det_precision precision) {
    if (precision == DET_PRECISION_F32) return DET_OK;
    size_t cols = (size_t)(stage->depthwise ? 1 : stage->input_channels) *
                  (size_t)stage->kernel * (size_t)stage->kernel;
    if (cols > (size_t)INT_MAX) return DET_ERR_SHAPE;
    det_status status;
    if (precision == DET_PRECISION_INT8) {
        status = det_quantize_per_channel_s8(stage->weights, stage->output_channels,
                                             (int)cols, stage->quant_weights,
                                             stage->quant_scales);
    } else {
        status = det_quantize_pack_w4(stage->weights, stage->output_channels,
                                      (int)cols, stage->packed_weights,
                                      stage->quant_scales);
    }
    return status;
}

static det_status quantize_head(det_head *head, det_precision precision) {
    if (precision == DET_PRECISION_F32) return DET_OK;
    if (precision == DET_PRECISION_INT8) {
        return det_quantize_per_channel_s8(head->weights, head->outputs, head->channels,
                                           head->quant_weights, head->quant_scales);
    }
    return det_quantize_pack_w4(head->weights, head->outputs, head->channels,
                                head->packed_weights, head->quant_scales);
}

det_status det_model_build(det_context *ctx, const det_model_spec *spec,
                           det_model **out) {
    if (ctx == NULL || spec == NULL || out == NULL || spec->width <= 0 ||
        spec->height <= 0 || spec->width > 4096 || spec->height > 4096 ||
        spec->channels <= 0 || spec->channels > 4 || spec->num_classes <= 0 ||
        spec->num_classes > DET_MAX_CLASSES || spec->max_detections <= 0) {
        return DET_ERR_ARGUMENT;
    }
    det_model *model = (det_model *)calloc(1U, sizeof(*model));
    if (model == NULL) return DET_ERR_MEMORY;
    model->ctx = ctx;
    model->spec = *spec;
    int stage_height = spec->height;
    int stage_width = spec->width;
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
        int stage_input_channels = i == 0 ? spec->channels :
                                   model->stages[i - 1].output_channels;
        int stage_output_channels = i == 0 ? DET_GRAPH_STEM_CHANNELS : DET_GRAPH_CHANNELS;
        det_status status = alloc_stage(&model->stages[i], stage_input_channels,
                                        stage_height, stage_width, stage_output_channels,
                                        3, 2, 1, i > 1 ? 1 : 0);
        if (status != DET_OK) {
            det_model_destroy(model);
            return status;
        }
        stage_height = model->stages[i].output_height;
        stage_width = model->stages[i].output_width;
    }
    for (int i = 0; i < DET_MAX_SCALES; ++i) {
        int h = (spec->height + k_strides[i] - 1) / k_strides[i];
        int w = (spec->width + k_strides[i] - 1) / k_strides[i];
        int outputs = 4 + spec->num_classes;
        det_status status = alloc_head(&model->heads[i], h, w, DET_GRAPH_CHANNELS,
                                       outputs);
        if (status != DET_OK) {
            det_model_destroy(model);
            return status;
        }
        status = alloc_head(&model->aux_heads[i], h, w, DET_GRAPH_CHANNELS, outputs);
        if (status != DET_OK) {
            det_model_destroy(model);
            return status;
        }
    }
    *out = model;
    return DET_OK;
}

det_status det_model_set_precision(det_model *model, det_precision precision) {
    if (model == NULL || !valid_precision(precision)) return DET_ERR_ARGUMENT;
    if (precision != DET_PRECISION_F32) {
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            const det_stage *stage = &model->stages[i];
            if (!finite_values(stage->weights, stage_weight_count(stage)) ||
                !finite_values(stage->bias, (size_t)stage->output_channels)) {
                return DET_ERR_FORMAT;
            }
        }
        for (int i = 0; i < DET_MAX_SCALES; ++i) {
            const det_head *head = &model->heads[i];
            size_t count = (size_t)head->channels * (size_t)head->outputs;
            if (!finite_values(head->weights, count) ||
                !finite_values(head->bias, (size_t)head->outputs)) return DET_ERR_FORMAT;
            head = &model->aux_heads[i];
            count = (size_t)head->channels * (size_t)head->outputs;
            if (!finite_values(head->weights, count) ||
                !finite_values(head->bias, (size_t)head->outputs)) return DET_ERR_FORMAT;
        }
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            det_status status = quantize_stage(&model->stages[i], precision);
            if (status != DET_OK) return status;
        }
        for (int i = 0; i < DET_MAX_SCALES; ++i) {
            det_status status = quantize_head(&model->heads[i], precision);
            if (status != DET_OK) return status;
            status = quantize_head(&model->aux_heads[i], precision);
            if (status != DET_OK) return status;
        }
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            det_stage *stage = &model->stages[i];
            size_t wc = stage_weight_count(stage);
            if (precision == DET_PRECISION_INT8) {
                memset(stage->packed_weights, 0,
                       stage_packed_weight_count(stage) * sizeof(*stage->packed_weights));
            } else {
                memset(stage->quant_weights, 0, wc * sizeof(*stage->quant_weights));
            }
        }
        for (int i = 0; i < DET_MAX_SCALES; ++i) {
            det_head *head = &model->heads[i];
            size_t wc = (size_t)head->channels * (size_t)head->outputs;
            size_t packed = (size_t)head->outputs * (((size_t)head->channels + 1U) / 2U);
            if (precision == DET_PRECISION_INT8) {
                memset(head->packed_weights, 0, packed * sizeof(*head->packed_weights));
            } else {
                memset(head->quant_weights, 0, wc * sizeof(*head->quant_weights));
            }
            head = &model->aux_heads[i];
            wc = (size_t)head->channels * (size_t)head->outputs;
            packed = (size_t)head->outputs * (((size_t)head->channels + 1U) / 2U);
            if (precision == DET_PRECISION_INT8) {
                memset(head->packed_weights, 0, packed * sizeof(*head->packed_weights));
            } else {
                memset(head->quant_weights, 0, wc * sizeof(*head->quant_weights));
            }
        }
    } else {
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            det_stage *stage = &model->stages[i];
            memset(stage->quant_weights, 0,
                   stage_weight_count(stage) * sizeof(*stage->quant_weights));
            memset(stage->packed_weights, 0,
                   stage_packed_weight_count(stage) * sizeof(*stage->packed_weights));
            memset(stage->quant_scales, 0,
                   (size_t)stage->output_channels * sizeof(*stage->quant_scales));
        }
        for (int i = 0; i < DET_MAX_SCALES; ++i) {
            det_head *head = &model->heads[i];
            size_t wc = (size_t)head->channels * (size_t)head->outputs;
            size_t packed = (size_t)head->outputs * (((size_t)head->channels + 1U) / 2U);
            memset(head->quant_weights, 0, wc * sizeof(*head->quant_weights));
            memset(head->packed_weights, 0, packed * sizeof(*head->packed_weights));
            memset(head->quant_scales, 0,
                   (size_t)head->outputs * sizeof(*head->quant_scales));
            head = &model->aux_heads[i];
            wc = (size_t)head->channels * (size_t)head->outputs;
            packed = (size_t)head->outputs * (((size_t)head->channels + 1U) / 2U);
            memset(head->quant_weights, 0, wc * sizeof(*head->quant_weights));
            memset(head->packed_weights, 0, packed * sizeof(*head->packed_weights));
            memset(head->quant_scales, 0,
                   (size_t)head->outputs * sizeof(*head->quant_scales));
        }
    }
    model->precision = precision;
    return DET_OK;
}

det_precision det_model_precision(const det_model *model) {
    return model == NULL ? DET_PRECISION_F32 : model->precision;
}

void det_model_destroy(det_model *model) {
    if (model == NULL) return;
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) free_stage(&model->stages[i]);
    for (int i = 0; i < DET_MAX_SCALES; ++i) free_head(&model->heads[i]);
    for (int i = 0; i < DET_MAX_SCALES; ++i) free_head(&model->aux_heads[i]);
    free(model);
}

det_status det_model_reset(det_model *model, int seed) {
    if (model == NULL) return DET_ERR_ARGUMENT;
    model->precision = DET_PRECISION_F32;
    uint32_t state = (uint32_t)(seed == 0 ? 1 : seed);
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head_banks[2] = {&model->heads[s], &model->aux_heads[s]};
        for (int bank = 0; bank < 2; ++bank) {
            det_head *head = head_banks[bank];
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
    }
    for (int s = 0; s < DET_GRAPH_STAGES; ++s) {
        det_stage *stage = &model->stages[s];
        size_t wc = stage_weight_count(stage);
        for (size_t i = 0U; i < wc; ++i) {
            state = state * 1664525U + 1013904223U;
            float r = (float)(state & 0xffffU) / 65535.0f;
            stage->weights[i] = (r - 0.5f) * 0.04f;
            stage->velocity_w[i] = 0.0f;
            stage->gradient_w[i] = 0.0f;
        }
        for (int i = 0; i < stage->output_channels; ++i) {
            stage->bias[i] = 0.0f;
            stage->velocity_b[i] = 0.0f;
            stage->gradient_b[i] = 0.0f;
        }
        size_t out_count = (size_t)stage->output_channels * (size_t)stage->output_height *
                           (size_t)stage->output_width;
        memset(stage->activation, 0, out_count * sizeof(float));
        memset(stage->gradient_output, 0, out_count * sizeof(float));
    }
    model->train_updates = 0U;
    return det_model_set_precision(model, DET_PRECISION_F32);
}

static void stage_forward(det_stage *stage, const det_tensor_f32 *input) {
    det_tensor_f32 output = {stage->activation, stage->output_channels,
                             stage->output_height, stage->output_width};
    if (!stage->depthwise) {
        (void)det_conv2d_f32(input, stage->weights, stage->bias,
                             stage->output_channels, stage->kernel, stage->stride,
                             stage->padding, &output);
        det_relu_inplace(&output);
        return;
    }
    for (int c = 0; c < stage->output_channels; ++c) {
        for (int oy = 0; oy < stage->output_height; ++oy) {
            for (int ox = 0; ox < stage->output_width; ++ox) {
                float sum = stage->bias[c];
                for (int ky = 0; ky < stage->kernel; ++ky) {
                    int iy = oy * stage->stride + ky - stage->padding;
                    if (iy < 0 || iy >= stage->input_height) continue;
                    for (int kx = 0; kx < stage->kernel; ++kx) {
                        int ix = ox * stage->stride + kx - stage->padding;
                        if (ix < 0 || ix >= stage->input_width) continue;
                        size_t weight_index = stage_weight_index(stage, c, c, ky, kx);
                        sum += stage->weights[weight_index] *
                               input->data[tensor_index(input, c, iy, ix)];
                    }
                }
                if (sum < 0.0f) sum = 0.0f;
                output.data[tensor_index(&output, c, oy, ox)] = sum;
            }
        }
    }
}

static int quantized_stage_weight(const det_stage *stage, det_precision precision,
                                  size_t index) {
    if (precision == DET_PRECISION_INT8) return (int)stage->quant_weights[index];
    size_t cols = (size_t)(stage->depthwise ? 1 : stage->input_channels) *
                  (size_t)stage->kernel * (size_t)stage->kernel;
    size_t row = index / cols;
    size_t col = index % cols;
    size_t packed_index = row * ((cols + 1U) / 2U) + col / 2U;
    unsigned int packed = (unsigned int)stage->packed_weights[packed_index];
    int value = (int)((packed >> ((col & 1U) * 4U)) & 0x0fU);
    return value >= 8 ? value - 16 : value;
}

static float tensor_quant_scale(const det_tensor_f32 *tensor) {
    size_t count = (size_t)tensor->channels * (size_t)tensor->height *
                   (size_t)tensor->width;
    float maximum = 0.0f;
    for (size_t i = 0U; i < count; ++i) {
        float value = fabsf(tensor->data[i]);
        if (value > maximum) maximum = value;
    }
    return maximum > 0.0f ? maximum / 127.0f : 1.0f;
}

static void stage_forward_quant(const det_stage *stage, const det_tensor_f32 *input,
                                det_precision precision, float *output_data) {
    det_tensor_f32 output = {output_data, stage->output_channels,
                             stage->output_height, stage->output_width};
    float input_scale = tensor_quant_scale(input);
    for (int oc = 0; oc < stage->output_channels; ++oc) {
        for (int oy = 0; oy < stage->output_height; ++oy) {
            for (int ox = 0; ox < stage->output_width; ++ox) {
                int32_t accumulator = 0;
                int channel_start = stage->depthwise ? oc : 0;
                int channel_end = stage->depthwise ? oc + 1 : stage->input_channels;
                for (int ic = channel_start; ic < channel_end; ++ic) {
                    for (int ky = 0; ky < stage->kernel; ++ky) {
                        int iy = oy * stage->stride + ky - stage->padding;
                        if (iy < 0 || iy >= stage->input_height) continue;
                        for (int kx = 0; kx < stage->kernel; ++kx) {
                            int ix = ox * stage->stride + kx - stage->padding;
                            if (ix < 0 || ix >= stage->input_width) continue;
                            size_t weight_index = stage_weight_index(stage, oc, ic, ky, kx);
                            int input_q = (int)det_quantize_s8(
                                input->data[tensor_index(input, ic, iy, ix)], input_scale);
                            accumulator += input_q * quantized_stage_weight(stage, precision,
                                                                             weight_index);
                        }
                    }
                }
                float value = (float)accumulator * input_scale * stage->quant_scales[oc] +
                              stage->bias[oc];
                if (value < 0.0f) value = 0.0f;
                output.data[tensor_index(&output, oc, oy, ox)] = value;
            }
        }
    }
}

static int quantized_head_weight(const det_head *head, det_precision precision,
                                 size_t index) {
    if (precision == DET_PRECISION_INT8) return (int)head->quant_weights[index];
    size_t col = index % (size_t)head->channels;
    size_t row = index / (size_t)head->channels;
    size_t packed_index = row * (((size_t)head->channels + 1U) / 2U) + col / 2U;
    unsigned int packed = (unsigned int)head->packed_weights[packed_index];
    int value = (int)((packed >> ((col & 1U) * 4U)) & 0x0fU);
    return value >= 8 ? value - 16 : value;
}

static void head_forward_quant(const det_head *head, const float features[4],
                               det_precision precision, float *output) {
    float maximum = 0.0f;
    for (int c = 0; c < head->channels; ++c) {
        float value = fabsf(features[c]);
        if (value > maximum) maximum = value;
    }
    float input_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    for (int o = 0; o < head->outputs; ++o) {
        int32_t accumulator = 0;
        for (int c = 0; c < head->channels; ++c) {
            int input_q = (int)det_quantize_s8(features[c], input_scale);
            size_t weight_index = (size_t)o * (size_t)head->channels + (size_t)c;
            accumulator += input_q * quantized_head_weight(head, precision, weight_index);
        }
        output[o] = (float)accumulator * input_scale * head->quant_scales[o] + head->bias[o];
    }
}

static void stage_backward(det_stage *stage, const det_tensor_f32 *input,
                           const det_tensor_f32 *grad_output,
                           det_tensor_f32 *grad_input) {
    if (!stage->depthwise) {
        (void)det_conv2d_backward_f32(input, stage->weights, grad_output,
                                      stage->output_channels, stage->kernel,
                                      stage->stride, stage->padding, grad_input,
                                      stage->gradient_w, stage->gradient_b);
        return;
    }
    size_t input_count = (size_t)stage->input_channels * (size_t)stage->input_height *
                         (size_t)stage->input_width;
    size_t weight_count = stage_weight_count(stage);
    memset(grad_input->data, 0, input_count * sizeof(float));
    memset(stage->gradient_w, 0, weight_count * sizeof(float));
    memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
    for (int c = 0; c < stage->output_channels; ++c) {
        for (int oy = 0; oy < stage->output_height; ++oy) {
            for (int ox = 0; ox < stage->output_width; ++ox) {
                float go = grad_output->data[tensor_index(grad_output, c, oy, ox)];
                stage->gradient_b[c] += go;
                for (int ky = 0; ky < stage->kernel; ++ky) {
                    int iy = oy * stage->stride + ky - stage->padding;
                    if (iy < 0 || iy >= stage->input_height) continue;
                    for (int kx = 0; kx < stage->kernel; ++kx) {
                        int ix = ox * stage->stride + kx - stage->padding;
                        if (ix < 0 || ix >= stage->input_width) continue;
                        size_t weight_index = stage_weight_index(stage, c, c, ky, kx);
                        size_t input_index = tensor_index(input, c, iy, ix);
                        stage->gradient_w[weight_index] += go * input->data[input_index];
                        grad_input->data[input_index] += go * stage->weights[weight_index];
                    }
                }
            }
        }
    }
}

static void backbone_forward(det_model *model, const det_image *image) {
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
        det_stage *stage = &model->stages[i];
        det_tensor_f32 input;
        if (i == 0) {
            input.data = (float *)image->data;
            input.channels = image->channels;
            input.height = image->height;
            input.width = image->width;
        } else {
            det_stage *previous = &model->stages[i - 1];
            input.data = previous->activation;
            input.channels = previous->output_channels;
            input.height = previous->output_height;
            input.width = previous->output_width;
        }
        if (model->precision == DET_PRECISION_F32) {
            stage_forward(stage, &input);
        } else {
            stage_forward_quant(stage, &input, model->precision, stage->activation);
        }
    }
}

static void extract_feature(const det_model *model, const det_image *image, int scale,
                            int cell_y, int cell_x, float features[4], float base[4]) {
    (void)image;
    const det_stage *stage = &model->stages[scale + 2];
    det_tensor_f32 tensor = {stage->activation, stage->output_channels,
                             stage->output_height, stage->output_width};
    for (int c = 0; c < DET_GRAPH_CHANNELS; ++c) {
        float value = tensor.data[tensor_index(&tensor, c, cell_y, cell_x)];
        features[c] = value;
        if (base != NULL) base[c] = value;
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

static float smooth_l1_loss(float value) {
    float magnitude = fabsf(value);
    return magnitude < 1.0f ? 0.5f * magnitude * magnitude : magnitude - 0.5f;
}

static float smooth_l1_gradient(float value) {
    float magnitude = fabsf(value);
    if (magnitude < 1.0f) return value;
    return value < 0.0f ? -1.0f : 1.0f;
}

static float centered_distance_iou(const float prediction[4], const float target[4]) {
    float pl = fmaxf(0.0f, prediction[0]);
    float pt = fmaxf(0.0f, prediction[1]);
    float pr = fmaxf(0.0f, prediction[2]);
    float pb = fmaxf(0.0f, prediction[3]);
    float tl = fmaxf(0.0f, target[0]);
    float tt = fmaxf(0.0f, target[1]);
    float tr = fmaxf(0.0f, target[2]);
    float tb = fmaxf(0.0f, target[3]);
    float intersection = (fminf(pl, tl) + fminf(pr, tr)) *
                         (fminf(pt, tt) + fminf(pb, tb));
    float prediction_area = (pl + pr) * (pt + pb);
    float target_area = (tl + tr) * (tt + tb);
    float union_area = prediction_area + target_area - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static void centered_iou_gradient(const float prediction[4], const float target[4],
                                  float gradient[4]) {
    const float epsilon = 1.0e-3f;
    for (int i = 0; i < 4; ++i) {
        float plus[4] = {prediction[0], prediction[1], prediction[2], prediction[3]};
        float minus[4] = {prediction[0], prediction[1], prediction[2], prediction[3]};
        plus[i] += epsilon;
        minus[i] -= epsilon;
        gradient[i] = (centered_distance_iou(plus, target) -
                       centered_distance_iou(minus, target)) / (2.0f * epsilon);
    }
}

static float prediction_loss_gradient(const float *prediction, const float *target,
                                      int outputs, int positive, int regression_enabled,
                                      float gradient[4 + DET_MAX_CLASSES]) {
    float loss = 0.0f;
    for (int o = 0; o < outputs; ++o) gradient[o] = 0.0f;
    if (positive >= 0 && regression_enabled) {
        float iou = centered_distance_iou(prediction, target);
        float iou_gradient[4];
        centered_iou_gradient(prediction, target, iou_gradient);
        loss += 0.5f * (1.0f - iou);
        for (int o = 0; o < 4; ++o) {
            float difference = prediction[o] - target[o];
            loss += 0.5f * smooth_l1_loss(difference);
            gradient[o] = 0.5f * smooth_l1_gradient(difference) - 0.5f * iou_gradient[o];
        }
    }
    for (int o = 4; o < outputs; ++o) {
        float logit = prediction[o];
        float target_value = target[o];
        float softplus = logit > 0.0f ? logit + log1pf(expf(-logit)) : log1pf(expf(logit));
        loss += softplus - target_value * logit;
        gradient[o] = det_sigmoid(logit) - target_value;
    }
    return loss;
}

static void update_head(det_head *head, const float features[4], const float *target,
                        int positive, int regression_enabled, float lr, float momentum) {
    float prediction[4 + DET_MAX_CLASSES] = {0.0f};
    float gradient[4 + DET_MAX_CLASSES];
    head_forward(head, features, prediction);
    (void)prediction_loss_gradient(prediction, target, head->outputs, positive,
                                   regression_enabled, gradient);
    if (positive < 0) {
        /* LOCAL_FAST sees many more background cells than positives. Keep a
           sparse negative signal without allowing BCE bias updates to erase
           the one-to-one class evidence. GLOBAL_BP keeps the full loss. */
        for (int o = 4; o < head->outputs; ++o) gradient[o] *= 0.001f;
    }
    for (int o = 0; o < head->outputs; ++o) {
        size_t base = (size_t)o * (size_t)head->channels;
        for (int c = 0; c < head->channels; ++c) {
            float grad = gradient[o] * features[c];
            head->velocity_w[base + (size_t)c] = momentum * head->velocity_w[base + (size_t)c] + grad;
            head->weights[base + (size_t)c] -= lr * head->velocity_w[base + (size_t)c];
        }
        head->velocity_b[o] = momentum * head->velocity_b[o] + gradient[o];
        head->bias[o] -= lr * head->velocity_b[o];
    }
}

static void feature_gradient_from_error(const det_head *head, const float *gradient,
                                        float feature_gradient[4]) {
    for (int c = 0; c < 4; ++c) feature_gradient[c] = 0.0f;
    for (int o = 0; o < head->outputs; ++o) {
        for (int c = 0; c < 4; ++c) {
            feature_gradient[c] += gradient[o] *
                                   head->weights[(size_t)o * (size_t)head->channels + (size_t)c];
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

static void clear_backbone_gradients(det_model *model) {
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
        det_stage *stage = &model->stages[i];
        size_t output_count = (size_t)stage->output_channels *
                              (size_t)stage->output_height * (size_t)stage->output_width;
        size_t weight_count = stage_weight_count(stage);
        memset(stage->gradient_output, 0, output_count * sizeof(float));
        memset(stage->gradient_w, 0, weight_count * sizeof(float));
        memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
    }
}

static void add_feature_gradient_to_backbone(det_model *model, int scale, int y, int x,
                                             const float gradient[4]) {
    det_stage *stage = &model->stages[scale + 2];
    for (int c = 0; c < DET_GRAPH_CHANNELS; ++c) {
        float value = gradient[c];
        size_t index = ((size_t)c * (size_t)stage->output_height + (size_t)y) *
                       (size_t)stage->output_width + (size_t)x;
        if (stage->activation[index] <= 0.0f) value = 0.0f;
        stage->gradient_output[index] += value;
    }
}

static void accumulate_previous_stage_gradient(det_stage *stage, det_stage *previous) {
    size_t count = (size_t)stage->input_channels * (size_t)stage->input_height *
                   (size_t)stage->input_width;
    for (size_t i = 0U; i < count; ++i) previous->gradient_output[i] += stage->gradient_input[i];
}

static void backbone_backward(det_model *model, const det_image *image) {
    for (int i = DET_GRAPH_STAGES - 1; i >= 0; --i) {
        det_stage *stage = &model->stages[i];
        size_t output_count = (size_t)stage->output_channels *
                              (size_t)stage->output_height * (size_t)stage->output_width;
        for (size_t j = 0U; j < output_count; ++j) {
            if (stage->activation[j] <= 0.0f) stage->gradient_output[j] = 0.0f;
        }
        det_tensor_f32 input;
        if (i == 0) {
            input.data = (float *)image->data;
            input.channels = image->channels;
            input.height = image->height;
            input.width = image->width;
        } else {
            det_stage *previous = &model->stages[i - 1];
            input.data = previous->activation;
            input.channels = previous->output_channels;
            input.height = previous->output_height;
            input.width = previous->output_width;
        }
        det_tensor_f32 grad_output = {stage->gradient_output, stage->output_channels,
                                      stage->output_height, stage->output_width};
        det_tensor_f32 grad_input = {stage->gradient_input, stage->input_channels,
                                     stage->input_height, stage->input_width};
        stage_backward(stage, &input, &grad_output, &grad_input);
        if (i > 0) accumulate_previous_stage_gradient(stage, &model->stages[i - 1]);
    }
}

static void apply_stage_gradient(det_stage *stage, float lr, float momentum, float scale) {
    size_t weight_count = stage_weight_count(stage);
    for (size_t i = 0U; i < weight_count; ++i) {
        float gradient = stage->gradient_w[i] * scale;
        stage->velocity_w[i] = momentum * stage->velocity_w[i] + gradient;
        stage->weights[i] -= lr * stage->velocity_w[i];
    }
    for (int i = 0; i < stage->output_channels; ++i) {
        float gradient = stage->gradient_b[i] * scale;
        stage->velocity_b[i] = momentum * stage->velocity_b[i] + gradient;
        stage->bias[i] -= lr * stage->velocity_b[i];
    }
}

static int stage_box_target(const det_sample *sample, int stage_index, int x, int y,
                            float target[4]) {
    for (int i = 0; i < 4; ++i) target[i] = 0.0f;
    int stride = 1 << (stage_index + 1);
    for (int b = 0; b < sample->box_count; ++b) {
        const det_box *box = &sample->boxes[b];
        if (box->class_id < 0) continue;
        int tx = (int)(((box->x1 + box->x2) * 0.5f) / (float)stride);
        int ty = (int)(((box->y1 + box->y2) * 0.5f) / (float)stride);
        if (tx != x || ty != y) continue;
        target[0] = 1.0f;
        target[1] = ((box->x1 + box->x2) * 0.5f) / (float)sample->image.width;
        target[2] = ((box->y1 + box->y2) * 0.5f) / (float)sample->image.height;
        target[3] = fmaxf((box->x2 - box->x1) / (float)sample->image.width,
                          (box->y2 - box->y1) / (float)sample->image.height);
        return 1;
    }
    return 0;
}

static void local_update_stage(det_model *model, const det_sample *sample, int stage_index,
                               size_t sample_index, float lr, float momentum,
                               size_t *updates) {
    det_stage *stage = &model->stages[stage_index];
    const float *input_data = stage_index == 0 ? sample->image.data :
                              model->stages[stage_index - 1].activation;
    for (int y = 0; y < stage->output_height; ++y) {
        for (int x = 0; x < stage->output_width; ++x) {
            float target[4];
            int positive = stage_box_target(sample, stage_index, x, y, target);
            if (!positive && ((x + y + (int)sample_index + stage_index) % 64 != 0)) continue;
            for (int oc = 0; oc < stage->output_channels; ++oc) {
                size_t output_index = ((size_t)oc * (size_t)stage->output_height + (size_t)y) *
                                      (size_t)stage->output_width + (size_t)x;
                /* LOCAL_FAST uses a straight-through ReLU surrogate so a dead
                   stage can receive a positive local target and recover. */
                float error = stage->activation[output_index] - target[oc];
                size_t bias_index = (size_t)oc;
                stage->velocity_b[bias_index] = momentum * stage->velocity_b[bias_index] + error;
                stage->bias[bias_index] -= lr * stage->velocity_b[bias_index];
                for (int ic = 0; ic < stage->input_channels; ++ic) {
                    if (stage->depthwise && ic != oc) continue;
                    for (int ky = 0; ky < stage->kernel; ++ky) {
                        int iy = y * stage->stride + ky - stage->padding;
                        if (iy < 0 || iy >= stage->input_height) continue;
                        for (int kx = 0; kx < stage->kernel; ++kx) {
                            int ix = x * stage->stride + kx - stage->padding;
                            if (ix < 0 || ix >= stage->input_width) continue;
                            size_t input_index = ((size_t)ic * (size_t)stage->input_height +
                                                  (size_t)iy) * (size_t)stage->input_width +
                                                 (size_t)ix;
                            size_t weight_index = stage_weight_index(stage, oc, ic, ky, kx);
                            float gradient = error * input_data[input_index];
                            stage->velocity_w[weight_index] = momentum * stage->velocity_w[weight_index] + gradient;
                            stage->weights[weight_index] -= lr * stage->velocity_w[weight_index];
                        }
                    }
                }
            }
            ++(*updates);
        }
    }
}

static int validate_sample(const det_model *model, const det_sample *sample) {
    if (model == NULL || sample == NULL || sample->image.data == NULL ||
        sample->image.channels != model->spec.channels ||
        sample->image.height != model->spec.height || sample->image.width != model->spec.width ||
        sample->box_count < 0 || (sample->box_count > 0 && sample->boxes == NULL)) {
        return 0;
    }
    for (int i = 0; i < sample->box_count; ++i) {
        const det_box *box = &sample->boxes[i];
        if (box->class_id < 0 || box->class_id >= model->spec.num_classes ||
            !isfinite(box->x1) || !isfinite(box->y1) || !isfinite(box->x2) ||
            !isfinite(box->y2) || box->x1 < 0.0f || box->y1 < 0.0f ||
            box->x2 > (float)model->spec.width || box->y2 > (float)model->spec.height ||
            box->x2 <= box->x1 || box->y2 <= box->y1) {
            return 0;
        }
    }
    size_t pixels = (size_t)sample->image.channels * (size_t)sample->image.height *
                    (size_t)sample->image.width;
    return finite_values(sample->image.data, pixels);
}

static const det_box *assigned_training_box(const det_sample *sample, int scale,
                                             int cell_x, int cell_y, int radius,
                                             int image_width, int image_height) {
    const det_box *best = NULL;
    int best_distance = INT_MAX;
    for (int b = 0; b < sample->box_count; ++b) {
        const det_box *box = &sample->boxes[b];
        if (choose_scale(box, image_width, image_height) != scale) continue;
        int target_x = (int)(((box->x1 + box->x2) * 0.5f) /
                             (float)k_strides[scale]);
        int target_y = (int)(((box->y1 + box->y2) * 0.5f) /
                             (float)k_strides[scale]);
        int distance = abs(cell_x - target_x) + abs(cell_y - target_y);
        if (abs(cell_x - target_x) > radius || abs(cell_y - target_y) > radius ||
            distance >= best_distance) continue;
        best = box;
        best_distance = distance;
    }
    return best;
}

static int fill_head_target(const det_box *box, int scale, int cell_x, int cell_y,
                            int classes, float *target) {
    for (int o = 0; o < 4 + DET_MAX_CLASSES; ++o) target[o] = 0.0f;
    if (box == NULL || box->class_id < 0 || box->class_id >= classes) return -1;
    target[box->class_id + 4] = 1.0f;
    float center_x = ((float)cell_x + 0.5f) * (float)k_strides[scale];
    float center_y = ((float)cell_y + 0.5f) * (float)k_strides[scale];
    target[0] = fmaxf(0.0f, (center_x - box->x1) / (float)k_strides[scale]);
    target[1] = fmaxf(0.0f, (center_y - box->y1) / (float)k_strides[scale]);
    target[2] = fmaxf(0.0f, (box->x2 - center_x) / (float)k_strides[scale]);
    target[3] = fmaxf(0.0f, (box->y2 - center_y) / (float)k_strides[scale]);
    return box->class_id;
}

static float train_head_cell(det_model *model, const det_sample *sample,
                             det_head *head, int scale, int cell_x, int cell_y,
                             const det_box *assigned, size_t sample_index,
                             int regression_enabled, int force_update, int use_global,
                             float learning_rate, float momentum,
                             float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4],
                             float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES],
                             size_t *terms, size_t *updates) {
    float features[4];
    float target[4 + DET_MAX_CLASSES];
    float prediction[4 + DET_MAX_CLASSES] = {0.0f};
    extract_feature(model, &sample->image, scale, cell_y, cell_x, features, NULL);
    int positive = fill_head_target(assigned, scale, cell_x, cell_y,
                                    model->spec.num_classes, target);
    head_forward(head, features, prediction);
    float prediction_gradient[4 + DET_MAX_CLASSES];
    float loss = prediction_loss_gradient(prediction, target, head->outputs,
                                          positive, regression_enabled, prediction_gradient);
    *terms += (size_t)head->outputs;
    if (use_global) {
        for (int o = 0; o < head->outputs; ++o) {
            gradient_b[scale][o] += prediction_gradient[o];
            for (int c = 0; c < 4; ++c) {
                gradient_w[scale][o][c] += prediction_gradient[o] * features[c];
            }
        }
        float feature_gradient[4];
        feature_gradient_from_error(head, prediction_gradient, feature_gradient);
        add_feature_gradient_to_backbone(model, scale, cell_y, cell_x, feature_gradient);
        ++(*updates);
    } else if (positive >= 0 || force_update ||
               ((cell_x + cell_y + (int)sample_index + scale) % 16 == 0)) {
        update_head(head, features, target, positive, regression_enabled,
                    learning_rate, momentum);
        ++(*updates);
    }
    return loss;
}

static float train_head_bank(det_model *model, const det_sample *sample,
                             det_head *heads, int assignment_radius,
                             size_t sample_index, int use_global,
                             float learning_rate, float momentum,
                             float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4],
                             float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES],
                             size_t *terms, size_t *updates) {
    float loss = 0.0f;
    float target[4 + DET_MAX_CLASSES];
    float prediction[4 + DET_MAX_CLASSES] = {0.0f};
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head = &heads[s];
        *terms += (size_t)head->height * (size_t)head->width * (size_t)head->outputs;
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                float features[4];
                extract_feature(model, &sample->image, s, y, x, features, NULL);
                for (int o = 0; o < head->outputs; ++o) target[o] = 0.0f;
                int positive = -1;
                const det_box *assigned = NULL;
                for (int b = 0; b < sample->box_count; ++b) {
                    if (sample->boxes[b].class_id < 0 ||
                        sample->boxes[b].class_id >= model->spec.num_classes ||
                        choose_scale(&sample->boxes[b], sample->image.width,
                                     sample->image.height) != s) continue;
                    int tx = (int)(((sample->boxes[b].x1 + sample->boxes[b].x2) * 0.5f) /
                                   (float)k_strides[s]);
                    int ty = (int)(((sample->boxes[b].y1 + sample->boxes[b].y2) * 0.5f) /
                                   (float)k_strides[s]);
                    if (abs(x - tx) <= assignment_radius &&
                        abs(y - ty) <= assignment_radius) {
                        positive = sample->boxes[b].class_id;
                        assigned = &sample->boxes[b];
                        break;
                    }
                }
                if (positive >= 0 && assigned != NULL) {
                    float center_x = ((float)x + 0.5f) * (float)k_strides[s];
                    float center_y = ((float)y + 0.5f) * (float)k_strides[s];
                    target[positive + 4] = 1.0f;
                    target[0] = fmaxf(0.0f, (center_x - assigned->x1) /
                                      (float)k_strides[s]);
                    target[1] = fmaxf(0.0f, (center_y - assigned->y1) /
                                      (float)k_strides[s]);
                    target[2] = fmaxf(0.0f, (assigned->x2 - center_x) /
                                      (float)k_strides[s]);
                    target[3] = fmaxf(0.0f, (assigned->y2 - center_y) /
                                      (float)k_strides[s]);
                }
                head_forward(head, features, prediction);
                float prediction_gradient[4 + DET_MAX_CLASSES];
                loss += prediction_loss_gradient(prediction, target, head->outputs,
                                                  positive, 1, prediction_gradient);
                if (use_global) {
                    for (int o = 0; o < head->outputs; ++o) {
                        gradient_b[s][o] += prediction_gradient[o];
                        for (int c = 0; c < 4; ++c) {
                            gradient_w[s][o][c] += prediction_gradient[o] * features[c];
                        }
                    }
                    float feature_gradient[4];
                    feature_gradient_from_error(head, prediction_gradient, feature_gradient);
                    add_feature_gradient_to_backbone(model, s, y, x, feature_gradient);
                    ++(*updates);
                } else if (positive >= 0 ||
                           ((x + y + (int)sample_index) % 16 == 0)) {
                    update_head(head, features, target, positive, 1,
                                learning_rate, momentum);
                    ++(*updates);
                }
            }
        }
    }
    return loss;
}

static float train_auxiliary_head_bank(det_model *model, const det_sample *sample,
                                       size_t sample_index, int use_global,
                                       float learning_rate, float momentum,
                                       float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4],
                                       float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES],
                                       size_t *terms, size_t *updates) {
    float loss = 0.0f;
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head = &model->aux_heads[s];
        for (int b = 0; b < sample->box_count; ++b) {
            const det_box *box = &sample->boxes[b];
            if (choose_scale(box, sample->image.width, sample->image.height) != s) continue;
            int target_x = (int)(((box->x1 + box->x2) * 0.5f) /
                                 (float)k_strides[s]);
            int target_y = (int)(((box->y1 + box->y2) * 0.5f) /
                                 (float)k_strides[s]);
            for (int y = target_y - 1; y <= target_y + 1; ++y) {
                if (y < 0 || y >= head->height) continue;
                for (int x = target_x - 1; x <= target_x + 1; ++x) {
                    if (x < 0 || x >= head->width) continue;
                    const det_box *assigned = assigned_training_box(
                        sample, s, x, y, 1,
                        sample->image.width, sample->image.height);
                    if (assigned != box) continue;
                    float center_x = ((float)x + 0.5f) * (float)k_strides[s];
                    float center_y = ((float)y + 0.5f) * (float)k_strides[s];
                    int center_inside = center_x >= assigned->x1 && center_x <= assigned->x2 &&
                                        center_y >= assigned->y1 && center_y <= assigned->y2;
                    if (!center_inside && (x != target_x || y != target_y)) continue;
                    loss += train_head_cell(model, sample, head, s, x, y, assigned,
                                            sample_index, center_inside, 0, use_global,
                                            learning_rate, momentum,
                                            gradient_w, gradient_b, terms, updates);
                }
            }
        }
        int cell_count = head->height * head->width;
        int negatives = 0;
        size_t start = sample_index * 17U + (size_t)s * 313U;
        for (int step = 0; step < cell_count && negatives < 4; ++step) {
            int flat = (int)((start + (size_t)step) % (size_t)cell_count);
            int x = flat % head->width;
            int y = flat / head->width;
            if (assigned_training_box(sample, s, x, y, 1,
                                      sample->image.width, sample->image.height) != NULL) {
                continue;
            }
            loss += train_head_cell(model, sample, head, s, x, y, NULL,
                                    sample_index, 0, 1, use_global,
                                    learning_rate, momentum,
                                    gradient_w, gradient_b, terms, updates);
            ++negatives;
        }
    }
    return loss;
}

static float train_sample(det_model *model, const det_sample *sample,
                           const det_train_config *config, size_t sample_index,
                           size_t *updates) {
    backbone_forward(model, &sample->image);
    float loss = 0.0f;
    float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4] = {{{0.0f}}};
    float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES] = {{0.0f}};
    float aux_gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4] = {{{0.0f}}};
    float aux_gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES] = {{0.0f}};
    int use_global = config->mode == DET_TRAIN_GLOBAL_BP;
    int aux_sampled = (sample_index & 15U) == 0U;
    if (use_global) clear_backbone_gradients(model);
    size_t terms = 0U;
    loss += train_head_bank(model, sample, model->heads, 0, sample_index, use_global,
                            config->learning_rate, config->momentum,
                            gradient_w, gradient_b, &terms, updates);
    /* The auxiliary one-to-many bank is a training-only regularizer. Updating
       it every sixteenth sample bounds its CPU cost while preserving a full
       one-to-many neighborhood whenever the bank is sampled. */
    if (aux_sampled) {
        loss += train_auxiliary_head_bank(model, sample, sample_index, use_global,
                                          config->learning_rate, config->momentum,
                                          aux_gradient_w, aux_gradient_b, &terms, updates);
    }
    if (use_global && terms > 0U) {
        float scale = 1.0f / (float)terms;
        for (int s = 0; s < DET_MAX_SCALES; ++s) {
            apply_head_gradient(&model->heads[s], gradient_w[s], gradient_b[s],
                                config->learning_rate, config->momentum, scale);
            if (aux_sampled) {
                apply_head_gradient(&model->aux_heads[s], aux_gradient_w[s], aux_gradient_b[s],
                                    config->learning_rate, config->momentum, scale);
            }
        }
        backbone_backward(model, &sample->image);
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            apply_stage_gradient(&model->stages[i], config->learning_rate,
                                  config->momentum, scale);
        }
    } else {
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            local_update_stage(model, sample, i, sample_index, config->learning_rate,
                               config->momentum, updates);
        }
    }
    return terms > 0U ? loss / (float)terms : 0.0f;
}

det_status det_train(det_model *model, const det_dataset *dataset,
                     const det_train_config *config, det_train_report *report) {
    if (model == NULL || dataset == NULL || config == NULL || report == NULL ||
        dataset->next == NULL || config->epochs <= 0 || config->learning_rate <= 0.0f ||
        config->momentum < 0.0f || config->momentum >= 1.0f ||
        config->reset_weights < 0 || config->reset_weights > 1 ||
        (config->mode != DET_TRAIN_LOCAL_FAST && config->mode != DET_TRAIN_GLOBAL_BP) ||
        (config->precision != DET_PRECISION_F32 && config->precision != DET_PRECISION_INT8 &&
         config->precision != DET_PRECISION_W4A8) || !isfinite(config->learning_rate) ||
        !isfinite(config->momentum) || !isfinite(config->score_threshold)) {
        return DET_ERR_ARGUMENT;
    }
    if (config->precision != DET_PRECISION_F32) {
        return DET_ERR_UNSUPPORTED;
    }
    if (model->precision != DET_PRECISION_F32) return DET_ERR_UNSUPPORTED;
    if (config->reset_weights != 0 && det_model_reset(model, config->seed) != DET_OK) {
        return DET_ERR_MEMORY;
    }
    memset(report, 0, sizeof(*report));
    double start = now_ms();
    float loss_sum = 0.0f;
    size_t updates = 0U;
    for (int epoch = 0; epoch < config->epochs; ++epoch) {
        if (dataset->reset != NULL) dataset->reset(dataset->user);
        det_sample sample;
        size_t seen = 0U;
        while (config->max_samples <= 0 || seen < (size_t)config->max_samples) {
            int next_status = dataset->next(dataset->user, &sample);
            if (next_status < 0) return DET_ERR_IO;
            if (next_status == 0) break;
            if (!validate_sample(model, &sample)) return DET_ERR_ARGUMENT;
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

typedef struct {
    det_detection detection;
    size_t sample_index;
} det_eval_prediction;

typedef struct {
    det_box box;
    size_t sample_index;
} det_eval_ground_truth;

static int append_eval_prediction(det_eval_prediction **items, size_t *count,
                                  size_t *capacity, const det_eval_prediction *item) {
    if (*count == *capacity) {
        size_t next = *capacity == 0U ? 64U : *capacity * 2U;
        if (next < *capacity || next > SIZE_MAX / sizeof(**items)) return 0;
        det_eval_prediction *grown = (det_eval_prediction *)realloc(
            *items, next * sizeof(**items));
        if (grown == NULL) return 0;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = *item;
    return 1;
}

static int append_eval_ground_truth(det_eval_ground_truth **items, size_t *count,
                                    size_t *capacity, const det_eval_ground_truth *item) {
    if (*count == *capacity) {
        size_t next = *capacity == 0U ? 64U : *capacity * 2U;
        if (next < *capacity || next > SIZE_MAX / sizeof(**items)) return 0;
        det_eval_ground_truth *grown = (det_eval_ground_truth *)realloc(
            *items, next * sizeof(**items));
        if (grown == NULL) return 0;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = *item;
    return 1;
}

static int compare_eval_predictions(const void *left, const void *right) {
    const det_eval_prediction *a = (const det_eval_prediction *)left;
    const det_eval_prediction *b = (const det_eval_prediction *)right;
    if (a->detection.score < b->detection.score) return 1;
    if (a->detection.score > b->detection.score) return -1;
    if (a->sample_index < b->sample_index) return -1;
    if (a->sample_index > b->sample_index) return 1;
    return 0;
}

static det_status evaluate_ap50(const det_model *model, const det_dataset *dataset,
                                float *out_ap50) {
    if (out_ap50 == NULL) return DET_ERR_ARGUMENT;
    *out_ap50 = 0.0f;
    if (dataset->reset == NULL) return DET_OK;
    det_eval_prediction *predictions = NULL;
    det_eval_ground_truth *ground_truths = NULL;
    size_t prediction_count = 0U, prediction_capacity = 0U;
    size_t ground_truth_count = 0U, ground_truth_capacity = 0U;
    det_detection *detections = (det_detection *)malloc(
        (size_t)model->spec.max_detections * sizeof(*detections));
    if (detections == NULL) return DET_ERR_MEMORY;
    det_status status = DET_OK;
    dataset->reset(dataset->user);
    size_t sample_index = 0U;
    for (;;) {
        det_sample sample;
        int next_status = dataset->next(dataset->user, &sample);
        if (next_status < 0) {
            status = DET_ERR_IO;
            break;
        }
        if (next_status == 0) break;
        if (!validate_sample(model, &sample)) {
            status = DET_ERR_ARGUMENT;
            break;
        }
        for (int b = 0; b < sample.box_count; ++b) {
            det_eval_ground_truth item = {sample.boxes[b], sample_index};
            if (!append_eval_ground_truth(&ground_truths, &ground_truth_count,
                                          &ground_truth_capacity, &item)) {
                status = DET_ERR_MEMORY;
                break;
            }
        }
        if (status != DET_OK) break;
        int detection_count = 0;
        status = det_predict(model, &sample.image, 0.0f, detections,
                             model->spec.max_detections, &detection_count);
        if (status != DET_OK) break;
        for (int d = 0; d < detection_count; ++d) {
            det_eval_prediction item = {detections[d], sample_index};
            if (!append_eval_prediction(&predictions, &prediction_count,
                                        &prediction_capacity, &item)) {
                status = DET_ERR_MEMORY;
                break;
            }
        }
        if (status != DET_OK) break;
        ++sample_index;
    }
    free(detections);
    if (status != DET_OK) {
        free(predictions);
        free(ground_truths);
        return status;
    }
    if (prediction_count > 1U) {
        qsort(predictions, prediction_count, sizeof(*predictions), compare_eval_predictions);
    }
    unsigned char *matched = (unsigned char *)calloc(ground_truth_count, 1U);
    if (ground_truth_count > 0U && matched == NULL) {
        free(predictions);
        free(ground_truths);
        return DET_ERR_MEMORY;
    }
    size_t true_positives = 0U;
    double ap = 0.0;
    double previous_recall = 0.0;
    for (size_t i = 0U; i < prediction_count; ++i) {
        size_t best_index = SIZE_MAX;
        float best_iou = 0.0f;
        for (size_t g = 0U; g < ground_truth_count; ++g) {
            if (matched[g] != 0U ||
                ground_truths[g].sample_index != predictions[i].sample_index ||
                ground_truths[g].box.class_id != predictions[i].detection.box.class_id) {
                continue;
            }
            float iou = det_iou(&predictions[i].detection.box, &ground_truths[g].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_index = g;
            }
        }
        if (best_index != SIZE_MAX && best_iou >= 0.5f) {
            matched[best_index] = 1U;
            ++true_positives;
            double recall = ground_truth_count > 0U ?
                (double)true_positives / (double)ground_truth_count : 0.0;
            double precision = (double)true_positives / (double)(i + 1U);
            ap += precision * (recall - previous_recall);
            previous_recall = recall;
        }
    }
    *out_ap50 = ground_truth_count > 0U ? (float)ap : 0.0f;
    free(matched);
    free(predictions);
    free(ground_truths);
    return DET_OK;
}

det_status det_predict(const det_model *model, const det_image *image,
                       float score_threshold, det_detection *detections,
                       int capacity, int *count) {
    if (model == NULL || image == NULL || image->data == NULL || detections == NULL ||
        count == NULL || capacity < 0 || !isfinite(score_threshold) ||
        image->channels != model->spec.channels ||
        image->width != model->spec.width || image->height != model->spec.height) {
        return DET_ERR_ARGUMENT;
    }
    size_t image_elements = 0U;
    if (!checked_size3(image->channels, image->height, image->width, &image_elements) ||
        !finite_values(image->data, image_elements)) return DET_ERR_ARGUMENT;
    *count = 0;
    backbone_forward((det_model *)model, image);
    float output[4 + DET_MAX_CLASSES];
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        const det_head *head = &model->heads[s];
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                float features[4];
                extract_feature(model, image, s, y, x, features, NULL);
                if (model->precision == DET_PRECISION_F32) {
                    head_forward(head, features, output);
                } else {
                    head_forward_quant(head, features, model->precision, output);
                }
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

det_status det_evaluate(const det_model *model, const det_dataset *dataset,
                        float score_threshold, det_eval_report *report) {
    if (model == NULL || dataset == NULL || report == NULL || dataset->next == NULL ||
        !isfinite(score_threshold) || score_threshold < 0.0f || score_threshold > 1.0f) {
        return DET_ERR_ARGUMENT;
    }
    if ((size_t)model->spec.max_detections > SIZE_MAX / sizeof(det_detection)) {
        return DET_ERR_MEMORY;
    }
    det_detection *detections = (det_detection *)malloc(
        (size_t)model->spec.max_detections * sizeof(*detections));
    if (detections == NULL) return DET_ERR_MEMORY;
    memset(report, 0, sizeof(*report));
    if (dataset->reset != NULL) dataset->reset(dataset->user);
    double iou_sum = 0.0;
    det_status status = DET_OK;
    unsigned char *matched = NULL;
    size_t matched_capacity = 0U;
    for (;;) {
        det_sample sample;
        int next_status = dataset->next(dataset->user, &sample);
        if (next_status < 0) {
            status = DET_ERR_IO;
            break;
        }
        if (next_status == 0) break;
        if (!validate_sample(model, &sample)) {
            status = DET_ERR_ARGUMENT;
            break;
        }
        if ((size_t)sample.box_count > matched_capacity) {
            unsigned char *grown = (unsigned char *)realloc(
                matched, (size_t)sample.box_count);
            if (grown == NULL) {
                status = DET_ERR_MEMORY;
                break;
            }
            matched = grown;
            matched_capacity = (size_t)sample.box_count;
        }
        if (sample.box_count > 0) memset(matched, 0, (size_t)sample.box_count);
        int detection_count = 0;
        status = det_predict(model, &sample.image, score_threshold, detections,
                             model->spec.max_detections, &detection_count);
        if (status != DET_OK) {
            break;
        }
        report->samples_seen += 1U;
        report->ground_truths += (size_t)sample.box_count;
        report->predictions += (size_t)detection_count;
        size_t matched_count = 0U;
        for (int d = 0; d < detection_count; ++d) {
            int best_index = -1;
            float best_iou = 0.0f;
            for (int b = 0; b < sample.box_count; ++b) {
                if (matched[b] != 0U || sample.boxes[b].class_id != detections[d].box.class_id) {
                    continue;
                }
                float iou = det_iou(&detections[d].box, &sample.boxes[b]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_index = b;
                }
            }
            if (best_index >= 0 && best_iou >= 0.5f) {
                matched[best_index] = 1U;
                ++matched_count;
                ++report->true_positives;
                iou_sum += (double)best_iou;
            } else {
                ++report->false_positives;
            }
        }
        report->false_negatives += (size_t)sample.box_count - matched_count;
    }
    free(matched);
    free(detections);
    if (status != DET_OK) return status;
    if (report->samples_seen == 0U) return DET_ERR_ARGUMENT;
    size_t predicted_positive = report->true_positives + report->false_positives;
    report->precision = predicted_positive > 0U ?
        (float)report->true_positives / (float)predicted_positive : 0.0f;
    report->recall = report->ground_truths > 0U ?
        (float)report->true_positives / (float)report->ground_truths : 0.0f;
    report->mean_iou = report->true_positives > 0U ?
        (float)(iou_sum / (double)report->true_positives) : 0.0f;
    return evaluate_ap50(model, dataset, &report->ap50);
}

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t classes;
    uint32_t max_detections;
    uint32_t graph_channels;
    uint32_t stage_count;
    uint32_t precision;
    uint64_t train_updates;
} det_file_header;

static int expected_payload_size(const det_file_header *header, size_t *out) {
    if (header == NULL || out == NULL || header->width == 0U || header->height == 0U ||
        header->width > 4096U || header->height > 4096U || header->channels == 0U ||
        header->channels > 4U || header->classes == 0U || header->classes > DET_MAX_CLASSES ||
        header->max_detections == 0U || header->max_detections > (uint32_t)INT_MAX) {
        return 0;
    }
    int input_height = (int)header->height;
    int input_width = (int)header->width;
    int input_channels = (int)header->channels;
    size_t total = sizeof(det_file_header);
    for (int s = 0; s < DET_GRAPH_STAGES; ++s) {
        int output_channels = s == 0 ? DET_GRAPH_STEM_CHANNELS : DET_GRAPH_CHANNELS;
        int output_height = 0;
        int output_width = 0;
        int depthwise = s > 1;
        if (!convolution_output_dim(input_height, 3, 2, 1, &output_height) ||
            !convolution_output_dim(input_width, 3, 2, 1, &output_width)) return 0;
        size_t wc = (size_t)(depthwise ? output_channels : output_channels * input_channels) * 9U;
        size_t packed = (size_t)output_channels *
                        ((((size_t)(depthwise ? 1 : input_channels) * 9U) + 1U) / 2U);
        size_t bytes = 10U * sizeof(int);
        if (!checked_add_size(bytes, (2U * wc + 2U * (size_t)output_channels) * sizeof(float), &bytes) ||
            !checked_add_size(bytes, wc * sizeof(int8_t), &bytes) ||
            !checked_add_size(bytes, packed * sizeof(uint8_t), &bytes) ||
            !checked_add_size(bytes, (size_t)output_channels * sizeof(float), &bytes) ||
            !checked_add_size(total, bytes, &total)) return 0;
        input_height = output_height;
        input_width = output_width;
        input_channels = output_channels;
    }
    int outputs = 4 + (int)header->classes;
    for (int bank = 0; bank < 2; ++bank) {
        for (int s = 0; s < DET_MAX_SCALES; ++s) {
            int height = ((int)header->height + k_strides[s] - 1) / k_strides[s];
            int width = ((int)header->width + k_strides[s] - 1) / k_strides[s];
            size_t wc = (size_t)DET_GRAPH_CHANNELS * (size_t)outputs;
            size_t packed = (size_t)outputs * ((DET_GRAPH_CHANNELS + 1U) / 2U);
            size_t bytes = 2U * sizeof(int);
            if (!checked_add_size(bytes, (2U * wc + 2U * (size_t)outputs) * sizeof(float), &bytes) ||
                !checked_add_size(bytes, wc * sizeof(int8_t), &bytes) ||
                !checked_add_size(bytes, packed * sizeof(uint8_t), &bytes) ||
                !checked_add_size(bytes, (size_t)outputs * sizeof(float), &bytes) ||
                !checked_add_size(total, bytes, &total)) return 0;
            (void)height;
            (void)width;
        }
    }
    *out = total;
    return 1;
}

det_status det_save(const det_model *model, const char *path) {
    if (model == NULL || path == NULL) return DET_ERR_ARGUMENT;
    if (!valid_precision(model->precision)) return DET_ERR_FORMAT;
    for (int s = 0; s < DET_GRAPH_STAGES; ++s) {
        const det_stage *stage = &model->stages[s];
        if (!finite_values(stage->weights, stage_weight_count(stage)) ||
            !finite_values(stage->bias, (size_t)stage->output_channels) ||
            !finite_values(stage->velocity_w, stage_weight_count(stage)) ||
            !finite_values(stage->velocity_b, (size_t)stage->output_channels) ||
            (model->precision != DET_PRECISION_F32 &&
             !positive_finite_values(stage->quant_scales, (size_t)stage->output_channels))) {
            return DET_ERR_FORMAT;
        }
    }
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        const det_head *head_banks[2] = {&model->heads[s], &model->aux_heads[s]};
        for (int bank = 0; bank < 2; ++bank) {
            const det_head *head = head_banks[bank];
            if (!finite_values(head->weights, (size_t)head->channels * (size_t)head->outputs) ||
                !finite_values(head->bias, (size_t)head->outputs) ||
                !finite_values(head->velocity_w, (size_t)head->channels * (size_t)head->outputs) ||
                !finite_values(head->velocity_b, (size_t)head->outputs) ||
                (model->precision != DET_PRECISION_F32 &&
                 !positive_finite_values(head->quant_scales, (size_t)head->outputs))) {
                return DET_ERR_FORMAT;
            }
        }
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) return DET_ERR_IO;
    det_file_header header = {{'C', 'D', 'E', 'T'}, 6U, (uint32_t)model->spec.width,
                              (uint32_t)model->spec.height, (uint32_t)model->spec.channels,
                              (uint32_t)model->spec.num_classes, (uint32_t)model->spec.max_detections,
                              DET_GRAPH_CHANNELS, DET_GRAPH_STAGES, (uint32_t)model->precision,
                              (uint64_t)model->train_updates};
    int ok = fwrite(&header, sizeof(header), 1U, file) == 1U;
    for (int s = 0; ok && s < DET_GRAPH_STAGES; ++s) {
        const det_stage *stage = &model->stages[s];
        size_t wc = stage_weight_count(stage);
        ok = fwrite(&stage->input_channels, sizeof(stage->input_channels), 1U, file) == 1U &&
             fwrite(&stage->output_channels, sizeof(stage->output_channels), 1U, file) == 1U &&
             fwrite(&stage->input_height, sizeof(stage->input_height), 1U, file) == 1U &&
             fwrite(&stage->input_width, sizeof(stage->input_width), 1U, file) == 1U &&
             fwrite(&stage->output_height, sizeof(stage->output_height), 1U, file) == 1U &&
             fwrite(&stage->output_width, sizeof(stage->output_width), 1U, file) == 1U &&
             fwrite(&stage->kernel, sizeof(stage->kernel), 1U, file) == 1U &&
             fwrite(&stage->stride, sizeof(stage->stride), 1U, file) == 1U &&
             fwrite(&stage->padding, sizeof(stage->padding), 1U, file) == 1U &&
             fwrite(&stage->depthwise, sizeof(stage->depthwise), 1U, file) == 1U &&
             fwrite(stage->weights, sizeof(float), wc, file) == wc &&
             fwrite(stage->bias, sizeof(float), (size_t)stage->output_channels, file) ==
                 (size_t)stage->output_channels &&
             fwrite(stage->velocity_w, sizeof(float), wc, file) == wc &&
             fwrite(stage->velocity_b, sizeof(float), (size_t)stage->output_channels, file) ==
                 (size_t)stage->output_channels &&
             fwrite(stage->quant_weights, sizeof(int8_t), wc, file) == wc &&
             fwrite(stage->packed_weights, sizeof(uint8_t), stage_packed_weight_count(stage), file) ==
                 stage_packed_weight_count(stage) &&
             fwrite(stage->quant_scales, sizeof(float), (size_t)stage->output_channels, file) ==
                 (size_t)stage->output_channels;
    }
    for (int bank = 0; ok && bank < 2; ++bank) {
        for (int s = 0; ok && s < DET_MAX_SCALES; ++s) {
            const det_head *head = bank == 0 ? &model->heads[s] : &model->aux_heads[s];
            size_t wc = (size_t)head->channels * (size_t)head->outputs;
            ok = fwrite(&head->height, sizeof(head->height), 1U, file) == 1U &&
                 fwrite(&head->width, sizeof(head->width), 1U, file) == 1U &&
                 fwrite(head->weights, sizeof(float), wc, file) == wc &&
                 fwrite(head->bias, sizeof(float), (size_t)head->outputs, file) == (size_t)head->outputs &&
                 fwrite(head->velocity_w, sizeof(float), wc, file) == wc &&
                 fwrite(head->velocity_b, sizeof(float), (size_t)head->outputs, file) ==
                     (size_t)head->outputs &&
                 fwrite(head->quant_weights, sizeof(int8_t), wc, file) == wc &&
                 fwrite(head->packed_weights, sizeof(uint8_t), (size_t)head->outputs *
                        (((size_t)head->channels + 1U) / 2U), file) ==
                     (size_t)head->outputs * (((size_t)head->channels + 1U) / 2U) &&
                 fwrite(head->quant_scales, sizeof(float), (size_t)head->outputs, file) ==
                     (size_t)head->outputs;
        }
    }
    int close_result = fclose(file);
    if (!ok || close_result != 0) return DET_ERR_IO;
    FILE *check = fopen(path, "rb");
    if (check == NULL || fseek(check, 0L, SEEK_END) != 0) {
        if (check != NULL) fclose(check);
        return DET_ERR_IO;
    }
    long length = ftell(check);
    if (length < 0L || (unsigned long)length > 256UL * 1024UL * 1024UL) {
        fclose(check);
        return DET_ERR_IO;
    }
    size_t payload_length = (size_t)length;
    uint8_t *payload = (uint8_t *)malloc(payload_length == 0U ? 1U : payload_length);
    if (payload == NULL || fseek(check, 0L, SEEK_SET) != 0 ||
        fread(payload, 1U, payload_length, check) != payload_length) {
        free(payload);
        fclose(check);
        return DET_ERR_IO;
    }
    fclose(check);
    uint32_t checksum = crc32_bytes(payload, payload_length);
    free(payload);
    FILE *append = fopen(path, "ab");
    if (append == NULL || fwrite(&checksum, sizeof(checksum), 1U, append) != 1U) {
        if (append != NULL) fclose(append);
        return DET_ERR_IO;
    }
    return fclose(append) == 0 ? DET_OK : DET_ERR_IO;
}

det_status det_load(det_context *ctx, const char *path, det_model **out) {
    if (ctx == NULL || path == NULL || out == NULL) return DET_ERR_ARGUMENT;
    FILE *file = fopen(path, "rb");
    if (file == NULL) return DET_ERR_IO;
    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        return DET_ERR_FORMAT;
    }
    long file_length = ftell(file);
    if (file_length < (long)(sizeof(det_file_header) + sizeof(uint32_t)) ||
        (unsigned long)file_length > 256UL * 1024UL * 1024UL) {
        fclose(file);
        return DET_ERR_FORMAT;
    }
    size_t encoded_length = (size_t)file_length;
    size_t payload_length = encoded_length - sizeof(uint32_t);
    det_file_header header;
    if (fseek(file, 0L, SEEK_SET) != 0 ||
        fread(&header, sizeof(header), 1U, file) != 1U ||
        memcmp(header.magic, "CDET", 4U) != 0 || header.version != 6U ||
        header.graph_channels != DET_GRAPH_CHANNELS || header.stage_count != DET_GRAPH_STAGES ||
        !valid_precision((det_precision)header.precision)) {
        fclose(file);
        return DET_ERR_FORMAT;
    }
    size_t expected_payload = 0U;
    if (!expected_payload_size(&header, &expected_payload) || expected_payload != payload_length) {
        fclose(file);
        return DET_ERR_FORMAT;
    }
    uint8_t *encoded = (uint8_t *)malloc(payload_length);
    if (encoded == NULL || fseek(file, 0L, SEEK_SET) != 0 ||
        fread(encoded, 1U, payload_length, file) != payload_length) {
        free(encoded);
        fclose(file);
        return DET_ERR_FORMAT;
    }
    uint32_t stored_checksum = 0U;
    if (fread(&stored_checksum, sizeof(stored_checksum), 1U, file) != 1U ||
        crc32_bytes(encoded, payload_length) != stored_checksum) {
        free(encoded);
        fclose(file);
        return DET_ERR_FORMAT;
    }
    free(encoded);
    if (fseek(file, (long)sizeof(det_file_header), SEEK_SET) != 0) {
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
    int ok = 1;
    for (int s = 0; ok && s < DET_GRAPH_STAGES; ++s) {
        det_stage *stage = &model->stages[s];
        int input_channels = 0;
        int output_channels = 0;
        int input_height = 0;
        int input_width = 0;
        int output_height = 0;
        int output_width = 0;
        int kernel = 0;
        int stride = 0;
        int padding = 0;
        int depthwise = 0;
        size_t wc = stage_weight_count(stage);
        ok = fread(&input_channels, sizeof(input_channels), 1U, file) == 1U &&
             fread(&output_channels, sizeof(output_channels), 1U, file) == 1U &&
             fread(&input_height, sizeof(input_height), 1U, file) == 1U &&
             fread(&input_width, sizeof(input_width), 1U, file) == 1U &&
             fread(&output_height, sizeof(output_height), 1U, file) == 1U &&
             fread(&output_width, sizeof(output_width), 1U, file) == 1U &&
             fread(&kernel, sizeof(kernel), 1U, file) == 1U &&
             fread(&stride, sizeof(stride), 1U, file) == 1U &&
             fread(&padding, sizeof(padding), 1U, file) == 1U &&
             fread(&depthwise, sizeof(depthwise), 1U, file) == 1U &&
             input_channels == stage->input_channels && output_channels == stage->output_channels &&
             input_height == stage->input_height && input_width == stage->input_width &&
             output_height == stage->output_height && output_width == stage->output_width &&
             kernel == stage->kernel && stride == stage->stride && padding == stage->padding &&
             depthwise == stage->depthwise &&
             fread(stage->weights, sizeof(float), wc, file) == wc &&
             fread(stage->bias, sizeof(float), (size_t)stage->output_channels, file) ==
                 (size_t)stage->output_channels &&
             fread(stage->velocity_w, sizeof(float), wc, file) == wc &&
             fread(stage->velocity_b, sizeof(float), (size_t)stage->output_channels, file) ==
                 (size_t)stage->output_channels &&
             fread(stage->quant_weights, sizeof(int8_t), wc, file) == wc &&
             fread(stage->packed_weights, sizeof(uint8_t), stage_packed_weight_count(stage), file) ==
                 stage_packed_weight_count(stage) &&
             fread(stage->quant_scales, sizeof(float), (size_t)stage->output_channels, file) ==
                 (size_t)stage->output_channels &&
             finite_values(stage->weights, wc) &&
             finite_values(stage->bias, (size_t)stage->output_channels) &&
             finite_values(stage->velocity_w, wc) &&
             finite_values(stage->velocity_b, (size_t)stage->output_channels);
    }
    for (int bank = 0; ok && bank < 2; ++bank) {
        for (int s = 0; ok && s < DET_MAX_SCALES; ++s) {
            det_head *head = bank == 0 ? &model->heads[s] : &model->aux_heads[s];
            int h = 0;
            int w = 0;
            size_t wc = (size_t)head->channels * (size_t)head->outputs;
            ok = fread(&h, sizeof(h), 1U, file) == 1U && fread(&w, sizeof(w), 1U, file) == 1U &&
                 h == head->height && w == head->width &&
                 fread(head->weights, sizeof(float), wc, file) == wc &&
                 fread(head->bias, sizeof(float), (size_t)head->outputs, file) == (size_t)head->outputs &&
                 fread(head->velocity_w, sizeof(float), wc, file) == wc &&
                 fread(head->velocity_b, sizeof(float), (size_t)head->outputs, file) ==
                     (size_t)head->outputs &&
                 fread(head->quant_weights, sizeof(int8_t), wc, file) == wc &&
                 fread(head->packed_weights, sizeof(uint8_t), (size_t)head->outputs *
                       (((size_t)head->channels + 1U) / 2U), file) ==
                     (size_t)head->outputs * (((size_t)head->channels + 1U) / 2U) &&
                 fread(head->quant_scales, sizeof(float), (size_t)head->outputs, file) ==
                     (size_t)head->outputs &&
                 finite_values(head->weights, wc) &&
                 finite_values(head->bias, (size_t)head->outputs) &&
                 finite_values(head->velocity_w, wc) &&
                 finite_values(head->velocity_b, (size_t)head->outputs);
        }
    }
    fclose(file);
    if (!ok) {
        det_model_destroy(model);
        return DET_ERR_FORMAT;
    }
    if (header.train_updates > SIZE_MAX) {
        det_model_destroy(model);
        return DET_ERR_FORMAT;
    }
    model->train_updates = (size_t)header.train_updates;
    /* FP32 weights are canonical; regenerate caches so a CRC-valid file cannot
       carry a contradictory quantized representation or inactive garbage. */
    status = det_model_set_precision(model, (det_precision)header.precision);
    if (status != DET_OK) {
        det_model_destroy(model);
        return status;
    }
    *out = model;
    return DET_OK;
}
