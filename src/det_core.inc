/* Core public tensors, arena, convolution, and scalar math operators. */
#include "det.h"
#include "kshira/session.h"

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

enum { DET_GRAPH_CHANNELS = 4, DET_GRAPH_STEM_CHANNELS = 1, DET_GRAPH_STAGES = 5,
       DET_NECK_STAGES = DET_MAX_SCALES, DET_BOTTOMUP_STAGES = DET_MAX_SCALES - 1 };

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
    det_architecture architecture;
    uint8_t *kshira_memory;
    size_t kshira_memory_bytes;
    kshira_session kshira;
    det_stage stages[DET_GRAPH_STAGES];
    det_stage neck[DET_NECK_STAGES];
    det_stage bottomup[DET_BOTTOMUP_STAGES];
    float *neck_topdown[DET_NECK_STAGES];
    float *neck_fused[DET_NECK_STAGES];
    float *neck_gradient[DET_NECK_STAGES];
    det_head heads[DET_MAX_SCALES];
    det_head aux_heads[DET_MAX_SCALES];
    det_precision precision;
    size_t train_updates;
};

static const int k_strides[DET_MAX_SCALES] = {8, 16, 32};

static det_status det_kshira_build(det_model *model);
static void det_kshira_release(det_model *model);
static det_status det_kshira_reset(det_model *model, int seed);
static det_status det_kshira_set_precision(det_model *model,
                                            det_precision precision);
static det_status det_kshira_train(det_model *model, const det_dataset *dataset,
                                    const det_train_config *config,
                                    det_train_report *report);
static det_status det_kshira_predict(det_model *model, const det_image *image,
                                      float score_threshold,
                                      det_detection *detections, int capacity,
                                      int *count);

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
    ctx->arena_bytes = arena_bytes;
    *out = ctx;
    return DET_OK;
}

void det_context_destroy(det_context *ctx) {
    if (ctx == NULL) return;
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
        if (input->channels == 1) {
            for (int oc = 0; oc < out_channels; ++oc) {
                const float *output_weights = weights + (size_t)oc * 9U;
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
                        for (int ky = ky_start; ky < ky_end; ++ky) {
                            const float *input_row = input->data +
                                (size_t)(iy_base + ky) * (size_t)input->width +
                                (size_t)(ix_base + kx_start);
                            const float *weight_row = output_weights + (size_t)ky * 3U +
                                (size_t)kx_start;
                            for (int kx = kx_start; kx < kx_end; ++kx) {
                                sum += weight_row[kx - kx_start] *
                                       input_row[kx - kx_start];
                            }
                        }
                        output_channel[(size_t)oy * (size_t)expected_w + (size_t)ox] = sum;
                    }
                }
            }
            return DET_OK;
        }
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
    if (kernel == 1 && stride == 1 && padding == 0) {
        size_t input_plane = (size_t)input->height * (size_t)input->width;
        for (int oc = 0; oc < out_channels; ++oc) {
            const float *output_weights = weights +
                (size_t)oc * (size_t)input->channels;
            float *output_channel = output->data + (size_t)oc * input_plane;
            for (size_t index = 0U; index < input_plane; ++index) {
                float sum = bias[oc];
                for (int ic = 0; ic < input->channels; ++ic) {
                    sum += output_weights[ic] * input->data[
                        (size_t)ic * input_plane + index];
                }
                output_channel[index] = sum;
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
