/* Purpose: static single-map RAD encoder and bounded top-K detector head.
 * Ownership: the model and every buffer live in the caller's KSHIRA arena.
 * Failure: invalid specs, exhausted arena, malformed images, or output capacity
 * return explicit status; prediction performs no allocation and no NMS. */
#include "kshira/rad.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

enum { RAD_BRANCHES = 3, RAD_KERNEL = 3, RAD_STRIDE = 4, RAD_MAX_CLASSES = 80 };
static const float RAD_QAS_GRADIENT_LIMIT = 1.0f;

typedef struct {
    float *project_weights;
    float *project_bias;
    float *branch_weights[RAD_BRANCHES];
    float *branch_bias[RAD_BRANCHES];
    float *stem_weights;
    float *stem_bias;
} rad_encoder_delta_buffer;

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

static int valid_spec(const kshira_rad_spec *spec) {
    return spec != NULL && spec->width >= 8 && spec->height >= 8 &&
           spec->width <= INT_MAX - (RAD_STRIDE - 1) &&
           spec->height <= INT_MAX - (RAD_STRIDE - 1) &&
           spec->channels >= 1 && spec->channels <= 4 && spec->classes >= 1 &&
           spec->classes <= RAD_MAX_CLASSES && spec->feature_channels >= 1 &&
           spec->feature_channels <= 32 && spec->top_k >= 1 && spec->top_k <= 64;
}

static int checked_elements(int a, int b, int c, size_t *out) {
    size_t first;
    if (a <= 0 || b <= 0 || c <= 0 || out == NULL) return 0;
    if ((size_t)a > SIZE_MAX / (size_t)b) return 0;
    first = (size_t)a * (size_t)b;
    if (first > SIZE_MAX / (size_t)c) return 0;
    *out = first * (size_t)c;
    return 1;
}

static size_t rad_index(int channels, int height, int width, int c, int y, int x) {
    (void)channels;
    return ((size_t)c * (size_t)height + (size_t)y) * (size_t)width + (size_t)x;
}

static float *alloc_floats(kshira_arena *arena, size_t count) {
    if (count == 0U || count > SIZE_MAX / sizeof(float)) return NULL;
    return (float *)kshira_arena_alloc(arena, count * sizeof(float), _Alignof(float));
}

static rad_encoder_delta_buffer *alloc_encoder_deltas(kshira_arena *arena,
                                                       const kshira_rad_spec *spec) {
    rad_encoder_delta_buffer *buffer;
    if (arena == NULL || spec == NULL) return NULL;
    buffer = (rad_encoder_delta_buffer *)kshira_arena_alloc(
        arena, sizeof(*buffer), _Alignof(rad_encoder_delta_buffer));
    if (buffer == NULL) return NULL;
    buffer->project_weights = alloc_floats(
        arena, (size_t)spec->feature_channels * (size_t)spec->feature_channels);
    buffer->project_bias = alloc_floats(arena, (size_t)spec->feature_channels);
    buffer->stem_weights = alloc_floats(
        arena, (size_t)spec->feature_channels * (size_t)spec->channels * 9U);
    buffer->stem_bias = alloc_floats(arena, (size_t)spec->feature_channels);
    if (buffer->project_weights == NULL || buffer->project_bias == NULL ||
        buffer->stem_weights == NULL || buffer->stem_bias == NULL) return NULL;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        buffer->branch_weights[branch] = alloc_floats(
            arena, (size_t)spec->feature_channels * 9U);
        buffer->branch_bias[branch] = alloc_floats(
            arena, (size_t)spec->feature_channels);
        if (buffer->branch_weights[branch] == NULL ||
            buffer->branch_bias[branch] == NULL) return NULL;
    }
    return buffer;
}

static float sigmoid(float value) {
    if (value >= 0.0f) {
        float e = expf(-value);
        return 1.0f / (1.0f + e);
    }
    {
        float e = expf(value);
        return e / (1.0f + e);
    }
}

static float relu(float value) {
    if (!isfinite(value) || value <= 0.0f) return 0.0f;
    return value > 1000000.0f ? 1000000.0f : value;
}

static int clamp_qas_gradient(float *gradient) {
    if (gradient == NULL || !isfinite(*gradient)) return 0;
    if (*gradient > RAD_QAS_GRADIENT_LIMIT) *gradient = RAD_QAS_GRADIENT_LIMIT;
    if (*gradient < -RAD_QAS_GRADIENT_LIMIT) *gradient = -RAD_QAS_GRADIENT_LIMIT;
    return 1;
}

static int normalize_qas_gradient(float *gradient, int quantized) {
    if (gradient == NULL || !isfinite(*gradient)) return 0;
    if (quantized) return clamp_qas_gradient(gradient);
    return 1;
}

static float class_gradient_scale(const kshira_rad_model *model, int output,
                                  int class_id) {
    return model->bits != KSHIRA_BITS_FLOAT && output >= 5 && output - 5 == class_id ?
        (float)model->spec.classes : 1.0f;
}

static float quant_scale_values(const float *values, size_t count, kshira_bit_mode bits) {
    float maximum = 0.0f;
    if (values == NULL || count == 0U) return 1.0f;
    for (size_t i = 0U; i < count; ++i) {
        float value = values[i];
        if (!isfinite(value)) continue;
        value = fabsf(value);
        if (value > maximum) maximum = value;
    }
    return kshira_symmetric_scale(maximum, bits);
}

static float calibrated_or_dynamic(float calibrated, const float *values, size_t count,
                                   kshira_bit_mode bits, size_t samples) {
    if (samples > 0U && isfinite(calibrated) && calibrated > 0.0f) return calibrated;
    return quant_scale_values(values, count, bits);
}

static float activation_scale(const float *values, size_t count, kshira_bit_mode bits) {
    float maximum = 0.0f;
    if (values == NULL || count == 0U) return 1.0f;
    for (size_t i = 0U; i < count; ++i) {
        if (isfinite(values[i]) && fabsf(values[i]) > maximum) maximum = fabsf(values[i]);
    }
    return kshira_symmetric_scale(maximum, bits);
}

static void clear_calibration(kshira_rad_model *model) {
    model->calibration_input_scale = 1.0f;
    model->calibration_stem_scale = 1.0f;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        model->calibration_branch_scale[branch] = 1.0f;
    }
    model->calibration_samples = 0U;
}

static void clear_transient_scales(kshira_rad_model *model) {
    model->transient_image_scale = 1.0f;
    model->transient_stem_scale = 1.0f;
    model->transient_scales_valid = 0;
}

static float quantized_dot(const float *weights, const float *values, size_t count,
                           float weight_scale, float value_scale, kshira_bit_mode bits) {
    int64_t accumulator = 0;
    for (size_t i = 0U; i < count; ++i) {
        int8_t weight = kshira_quantize_symmetric(weights[i], weight_scale, bits);
        int8_t value = kshira_quantize_symmetric(values[i], value_scale, bits);
        accumulator += (int64_t)weight * (int64_t)value;
    }
    return (float)accumulator * weight_scale * value_scale;
}

static int8_t quantized_value(float value, float scale, kshira_bit_mode bits) {
    return kshira_quantize_symmetric(value, scale, bits == KSHIRA_BITS_INT4 ?
                                     KSHIRA_BITS_INT4 : KSHIRA_BITS_INT8);
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

static void fill_random(float *values, size_t count, uint32_t *state, float amplitude) {
    for (size_t i = 0U; i < count; ++i) {
        uint32_t raw = next_random(state) >> 8U;
        values[i] = ((float)raw / 16777215.0f * 2.0f - 1.0f) * amplitude;
    }
}

static void fill_zero(float *values, size_t count) {
    for (size_t i = 0U; i < count; ++i) values[i] = 0.0f;
}

static void conv_stem(const kshira_rad_model *model, const kshira_image_f32 *image) {
    int c = model->spec.feature_channels;
    for (int oc = 0; oc < c; ++oc) {
        for (int y = 0; y < model->map_height; ++y) {
            for (int x = 0; x < model->map_width; ++x) {
                float sum = model->stem_bias[oc];
                for (int ic = 0; ic < image->channels; ++ic) {
                    for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                        int iy = y * RAD_STRIDE + ky - 1;
                        if (iy < 0 || iy >= image->height) continue;
                        for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                            int ix = x * RAD_STRIDE + kx - 1;
                            if (ix < 0 || ix >= image->width) continue;
                            size_t wi = (((size_t)oc * (size_t)image->channels +
                                          (size_t)ic) * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                                        (size_t)kx;
                            size_t xi = ((size_t)ic * (size_t)image->height + (size_t)iy) *
                                         (size_t)image->width + (size_t)ix;
                            sum += model->stem_weights[wi] * image->data[xi];
                        }
                    }
                }
                model->stem[rad_index(c, model->map_height, model->map_width, oc, y, x)] =
                    relu(sum);
            }
        }
    }
}

static void conv_stem_quant(const kshira_rad_model *model, const kshira_image_f32 *image) {
    int c = model->spec.feature_channels;
    size_t weight_count = (size_t)c * (size_t)image->channels * 9U;
    size_t image_count = (size_t)image->channels * (size_t)image->height *
                         (size_t)image->width;
    float weight_scale = quant_scale_values(model->stem_weights, weight_count, model->bits);
    float input_scale = calibrated_or_dynamic(model->calibration_input_scale, image->data,
                                              image_count, model->bits,
                                              model->calibration_samples);
    for (int oc = 0; oc < c; ++oc) {
        for (int y = 0; y < model->map_height; ++y) {
            for (int x = 0; x < model->map_width; ++x) {
                int64_t accumulator = 0;
                for (int ic = 0; ic < image->channels; ++ic) {
                    for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                        int iy = y * RAD_STRIDE + ky - 1;
                        if (iy < 0 || iy >= image->height) continue;
                        for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                            int ix = x * RAD_STRIDE + kx - 1;
                            if (ix < 0 || ix >= image->width) continue;
                            size_t wi = (((size_t)oc * (size_t)image->channels +
                                          (size_t)ic) * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                                        (size_t)kx;
                            size_t xi = ((size_t)ic * (size_t)image->height + (size_t)iy) *
                                         (size_t)image->width + (size_t)ix;
                            int8_t qweight = kshira_quantize_symmetric(
                                model->stem_weights[wi], weight_scale, model->bits);
                            int8_t qinput = quantized_value(image->data[xi], input_scale,
                                                            model->bits);
                            accumulator += (int64_t)qweight * (int64_t)qinput;
                        }
                    }
                }
                model->stem[rad_index(c, model->map_height, model->map_width, oc, y, x)] =
                    relu((float)accumulator * weight_scale * input_scale + model->stem_bias[oc]);
            }
        }
    }
}

static void depthwise_branch(const kshira_rad_model *model, int branch) {
    int c = model->spec.feature_channels;
    int dilation = 1 << branch;
    const float *input = model->stem;
    float *output = model->branches[branch];
    for (int channel = 0; channel < c; ++channel) {
        for (int y = 0; y < model->map_height; ++y) {
            for (int x = 0; x < model->map_width; ++x) {
                float sum = model->branch_bias[branch][channel];
                for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                    int iy = y + (ky - 1) * dilation;
                    if (iy < 0 || iy >= model->map_height) continue;
                    for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                        int ix = x + (kx - 1) * dilation;
                        if (ix < 0 || ix >= model->map_width) continue;
                        size_t wi = ((size_t)channel * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                                    (size_t)kx;
                        size_t xi = rad_index(c, model->map_height, model->map_width,
                                              channel, iy, ix);
                        sum += model->branch_weights[branch][wi] * input[xi];
                    }
                }
                output[rad_index(c, model->map_height, model->map_width, channel, y, x)] =
                    relu(sum);
            }
        }
    }
}

static void depthwise_branch_quant(const kshira_rad_model *model, int branch) {
    int c = model->spec.feature_channels;
    int dilation = 1 << branch;
    size_t map_count = (size_t)c * (size_t)model->map_height * (size_t)model->map_width;
    float weight_scale = quant_scale_values(model->branch_weights[branch],
                                            (size_t)c * 9U, model->bits);
    float input_scale = calibrated_or_dynamic(model->calibration_stem_scale, model->stem,
                                              map_count, model->bits,
                                              model->calibration_samples);
    for (int channel = 0; channel < c; ++channel) {
        for (int y = 0; y < model->map_height; ++y) {
            for (int x = 0; x < model->map_width; ++x) {
                int64_t accumulator = 0;
                for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                    int iy = y + (ky - 1) * dilation;
                    if (iy < 0 || iy >= model->map_height) continue;
                    for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                        int ix = x + (kx - 1) * dilation;
                        if (ix < 0 || ix >= model->map_width) continue;
                        size_t wi = ((size_t)channel * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                                    (size_t)kx;
                        size_t xi = rad_index(c, model->map_height, model->map_width,
                                              channel, iy, ix);
                        int8_t qweight = kshira_quantize_symmetric(
                            model->branch_weights[branch][wi], weight_scale, model->bits);
                        int8_t qinput = quantized_value(model->stem[xi], input_scale, model->bits);
                        accumulator += (int64_t)qweight * (int64_t)qinput;
                    }
                }
                model->branches[branch][rad_index(c, model->map_height, model->map_width,
                                                  channel, y, x)] =
                    relu((float)accumulator * weight_scale * input_scale +
                         model->branch_bias[branch][channel]);
            }
        }
    }
}

static void project_fused(const kshira_rad_model *model) {
    int c = model->spec.feature_channels;
    for (int oc = 0; oc < c; ++oc) {
        for (int y = 0; y < model->map_height; ++y) {
            for (int x = 0; x < model->map_width; ++x) {
                float sum = model->project_bias[oc];
                for (int ic = 0; ic < c; ++ic) {
                    float branches = 0.0f;
                    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                        branches += model->branches[branch][rad_index(
                            c, model->map_height, model->map_width, ic, y, x)];
                    }
                    sum += model->project_weights[(size_t)oc * (size_t)c + (size_t)ic] *
                           (branches / (float)RAD_BRANCHES);
                }
                model->fused[rad_index(c, model->map_height, model->map_width, oc, y, x)] =
                    relu(sum);
            }
        }
    }
}

static void project_fused_quant(const kshira_rad_model *model) {
    int c = model->spec.feature_channels;
    size_t map_count = (size_t)c * (size_t)model->map_height * (size_t)model->map_width;
    float input_scale = 0.0f;
    float weight_scale = quant_scale_values(model->project_weights, (size_t)c * (size_t)c,
                                            model->bits);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        float scale = calibrated_or_dynamic(model->calibration_branch_scale[branch],
                                            model->branches[branch], map_count, model->bits,
                                            model->calibration_samples);
        if (scale > input_scale) input_scale = scale;
    }
    if (input_scale <= 0.0f) input_scale = 1.0f;
    for (int oc = 0; oc < c; ++oc) {
        for (int y = 0; y < model->map_height; ++y) {
            for (int x = 0; x < model->map_width; ++x) {
                int64_t accumulator = 0;
                for (int ic = 0; ic < c; ++ic) {
                    int64_t branch_sum = 0;
                    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                        size_t index = rad_index(c, model->map_height, model->map_width,
                                                 ic, y, x);
                        int8_t qbranch = quantized_value(model->branches[branch][index],
                                                         input_scale, model->bits);
                        branch_sum += (int64_t)qbranch;
                    }
                    int8_t qweight = kshira_quantize_symmetric(
                        model->project_weights[(size_t)oc * (size_t)c + (size_t)ic],
                        weight_scale, model->bits);
                    accumulator += (int64_t)qweight * branch_sum;
                }
                model->fused[rad_index(c, model->map_height, model->map_width, oc, y, x)] =
                    relu((float)accumulator * weight_scale * input_scale /
                         (float)RAD_BRANCHES + model->project_bias[oc]);
            }
        }
    }
}

static void target_bounds(const kshira_rad_model *model, int target_y, int target_x,
                          int *y0, int *y1, int *x0, int *x1) {
    *y0 = target_y - 4;
    *y1 = target_y + 5;
    *x0 = target_x - 4;
    *x1 = target_x + 5;
    if (*y0 < 0) *y0 = 0;
    if (*x0 < 0) *x0 = 0;
    if (*y1 > model->map_height) *y1 = model->map_height;
    if (*x1 > model->map_width) *x1 = model->map_width;
}

static float quant_scale_region(const float *values, int channels, int height, int width,
                                int y0, int y1, int x0, int x1, kshira_bit_mode bits) {
    float maximum = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                float value = values[rad_index(channels, height, width, channel, y, x)];
                if (isfinite(value) && fabsf(value) > maximum) maximum = fabsf(value);
            }
        }
    }
    return kshira_symmetric_scale(maximum, bits);
}

static void conv_stem_target(kshira_rad_model *model, const kshira_image_f32 *image,
                             int target_y, int target_x) {
    int c = model->spec.feature_channels;
    int y0, y1, x0, x1;
    model->transient_scales_valid = 0;
    target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
    for (int oc = 0; oc < c; ++oc) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                float sum = model->stem_bias[oc];
                for (int ic = 0; ic < image->channels; ++ic) {
                    for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                        int iy = y * RAD_STRIDE + ky - 1;
                        if (iy < 0 || iy >= image->height) continue;
                        for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                            int ix = x * RAD_STRIDE + kx - 1;
                            size_t wi;
                            size_t xi;
                            if (ix < 0 || ix >= image->width) continue;
                            wi = (((size_t)oc * (size_t)image->channels + (size_t)ic) *
                                  RAD_KERNEL + (size_t)ky) * RAD_KERNEL + (size_t)kx;
                            xi = ((size_t)ic * (size_t)image->height + (size_t)iy) *
                                 (size_t)image->width + (size_t)ix;
                            sum += model->stem_weights[wi] * image->data[xi];
                        }
                    }
                }
                model->stem[rad_index(c, model->map_height, model->map_width, oc, y, x)] =
                    relu(sum);
            }
        }
    }
}

static void conv_stem_target_quant(kshira_rad_model *model,
                                   const kshira_image_f32 *image,
                                   int target_y, int target_x) {
    int c = model->spec.feature_channels;
    int y0, y1, x0, x1;
    size_t weight_count = (size_t)c * (size_t)image->channels * 9U;
    size_t image_count = (size_t)image->channels * (size_t)image->height *
                         (size_t)image->width;
    float weight_scale = quant_scale_values(model->stem_weights, weight_count, model->bits);
    float input_scale = calibrated_or_dynamic(model->calibration_input_scale, image->data,
                                              image_count, model->bits,
                                              model->calibration_samples);
    model->transient_image_scale = input_scale;
    model->transient_scales_valid = 1;
    target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
    for (int oc = 0; oc < c; ++oc) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                int64_t accumulator = 0;
                for (int ic = 0; ic < image->channels; ++ic) {
                    for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                        int iy = y * RAD_STRIDE + ky - 1;
                        if (iy < 0 || iy >= image->height) continue;
                        for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                            int ix = x * RAD_STRIDE + kx - 1;
                            size_t wi;
                            size_t xi;
                            if (ix < 0 || ix >= image->width) continue;
                            wi = (((size_t)oc * (size_t)image->channels + (size_t)ic) *
                                  RAD_KERNEL + (size_t)ky) * RAD_KERNEL + (size_t)kx;
                            xi = ((size_t)ic * (size_t)image->height + (size_t)iy) *
                                 (size_t)image->width + (size_t)ix;
                            accumulator += (int64_t)kshira_quantize_symmetric(
                                model->stem_weights[wi], weight_scale, model->bits) *
                                (int64_t)quantized_value(image->data[xi], input_scale,
                                                          model->bits);
                        }
                    }
                }
                model->stem[rad_index(c, model->map_height, model->map_width, oc, y, x)] =
                    relu((float)accumulator * weight_scale * input_scale +
                         model->stem_bias[oc]);
            }
        }
    }
    if (model->calibration_samples == 0U) {
        model->transient_stem_scale = quant_scale_region(
            model->stem, c, model->map_height, model->map_width, y0, y1, x0, x1,
            model->bits);
    }
}

static void depthwise_branch_target(const kshira_rad_model *model, int branch,
                                    int target_y, int target_x) {
    int c = model->spec.feature_channels;
    int dilation = 1 << branch;
    for (int channel = 0; channel < c; ++channel) {
        float sum = model->branch_bias[branch][channel];
        for (int ky = 0; ky < RAD_KERNEL; ++ky) {
            int iy = target_y + (ky - 1) * dilation;
            if (iy < 0 || iy >= model->map_height) continue;
            for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                int ix = target_x + (kx - 1) * dilation;
                size_t wi;
                if (ix < 0 || ix >= model->map_width) continue;
                wi = ((size_t)channel * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                     (size_t)kx;
                sum += model->branch_weights[branch][wi] * model->stem[rad_index(
                    c, model->map_height, model->map_width, channel, iy, ix)];
            }
        }
        model->branches[branch][rad_index(c, model->map_height, model->map_width,
                                          channel, target_y, target_x)] = relu(sum);
    }
}

static void depthwise_branch_target_quant(const kshira_rad_model *model, int branch,
                                          int target_y, int target_x) {
    int c = model->spec.feature_channels;
    int y0, y1, x0, x1;
    int dilation = 1 << branch;
    float weight_scale = quant_scale_values(model->branch_weights[branch],
                                            (size_t)c * 9U, model->bits);
    float input_scale;
    if (model->calibration_samples > 0U) {
        input_scale = model->calibration_stem_scale;
    } else if (model->transient_scales_valid) {
        input_scale = model->transient_stem_scale;
    } else {
        target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
        input_scale = quant_scale_region(model->stem, c, model->map_height,
                                         model->map_width, y0, y1, x0, x1, model->bits);
    }
    for (int channel = 0; channel < c; ++channel) {
        int64_t accumulator = 0;
        for (int ky = 0; ky < RAD_KERNEL; ++ky) {
            int iy = target_y + (ky - 1) * dilation;
            if (iy < 0 || iy >= model->map_height) continue;
            for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                int ix = target_x + (kx - 1) * dilation;
                size_t wi;
                if (ix < 0 || ix >= model->map_width) continue;
                wi = ((size_t)channel * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                     (size_t)kx;
                accumulator += (int64_t)kshira_quantize_symmetric(
                    model->branch_weights[branch][wi], weight_scale, model->bits) *
                    (int64_t)quantized_value(model->stem[rad_index(
                        c, model->map_height, model->map_width, channel, iy, ix)],
                        input_scale, model->bits);
            }
        }
        model->branches[branch][rad_index(c, model->map_height, model->map_width,
                                          channel, target_y, target_x)] =
            relu((float)accumulator * weight_scale * input_scale +
                 model->branch_bias[branch][channel]);
    }
}

static void project_fused_target(const kshira_rad_model *model, int target_y, int target_x) {
    int c = model->spec.feature_channels;
    for (int oc = 0; oc < c; ++oc) {
        float sum = model->project_bias[oc];
        for (int ic = 0; ic < c; ++ic) {
            float branches = 0.0f;
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                branches += model->branches[branch][rad_index(
                    c, model->map_height, model->map_width, ic, target_y, target_x)];
            }
            sum += model->project_weights[(size_t)oc * (size_t)c + (size_t)ic] *
                   (branches / (float)RAD_BRANCHES);
        }
        model->fused[rad_index(c, model->map_height, model->map_width, oc, target_y, target_x)] =
            relu(sum);
    }
}

static void project_fused_target_quant(const kshira_rad_model *model,
                                       int target_y, int target_x) {
    int c = model->spec.feature_channels;
    float weight_scale = quant_scale_values(model->project_weights,
                                            (size_t)c * (size_t)c, model->bits);
    float input_scale = 0.0f;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        float scale;
        if (model->calibration_samples > 0U) {
            scale = model->calibration_branch_scale[branch];
        } else {
            float maximum = 0.0f;
            for (int channel = 0; channel < c; ++channel) {
                float value = model->branches[branch][rad_index(
                    c, model->map_height, model->map_width, channel, target_y, target_x)];
                if (isfinite(value) && fabsf(value) > maximum) maximum = fabsf(value);
            }
            scale = kshira_symmetric_scale(maximum, model->bits);
        }
        if (scale > input_scale) input_scale = scale;
    }
    if (input_scale <= 0.0f) input_scale = 1.0f;
    for (int oc = 0; oc < c; ++oc) {
        int64_t accumulator = 0;
        for (int ic = 0; ic < c; ++ic) {
            int64_t branch_sum = 0;
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                branch_sum += (int64_t)quantized_value(model->branches[branch][rad_index(
                    c, model->map_height, model->map_width, ic, target_y, target_x)],
                    input_scale, model->bits);
            }
            accumulator += (int64_t)kshira_quantize_symmetric(
                model->project_weights[(size_t)oc * (size_t)c + (size_t)ic],
                weight_scale, model->bits) * branch_sum;
        }
        model->fused[rad_index(c, model->map_height, model->map_width, oc, target_y, target_x)] =
            relu((float)accumulator * weight_scale * input_scale /
                 (float)RAD_BRANCHES + model->project_bias[oc]);
    }
}

static void rad_forward_target(kshira_rad_model *model, const kshira_image_f32 *image,
                               int target_y, int target_x) {
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        conv_stem_target_quant(model, image, target_y, target_x);
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            depthwise_branch_target_quant(model, branch, target_y, target_x);
        }
        project_fused_target_quant(model, target_y, target_x);
    } else {
        conv_stem_target(model, image, target_y, target_x);
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            depthwise_branch_target(model, branch, target_y, target_x);
        }
        project_fused_target(model, target_y, target_x);
    }
}

static void rad_forward(kshira_rad_model *model, const kshira_image_f32 *image) {
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        conv_stem_quant(model, image);
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            depthwise_branch_quant(model, branch);
        }
        project_fused_quant(model);
    } else {
        conv_stem(model, image);
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) depthwise_branch(model, branch);
        project_fused(model);
    }
}

static void head_forward_f32(const kshira_rad_model *model, const float *features,
                             float *output) {
    int c = model->spec.feature_channels;
    for (int o = 0; o < model->outputs; ++o) {
        float sum = model->head_bias[o];
        for (int ic = 0; ic < c; ++ic) {
            sum += model->head_weights[(size_t)o * (size_t)c + (size_t)ic] * features[ic];
        }
        output[o] = isfinite(sum) ? sum : (sum > 0.0f ? 1000000.0f : -1000000.0f);
    }
}

static void head_forward_quant(const kshira_rad_model *model, const float *features,
                               float *output) {
    int c = model->spec.feature_channels;
    float weight_scale = quant_scale_values(model->head_weights,
                                            (size_t)model->outputs * (size_t)c, model->bits);
    float feature_scale = quant_scale_values(features, (size_t)c, model->bits);
    for (int o = 0; o < model->outputs; ++o) {
        float value = quantized_dot(&model->head_weights[(size_t)o * (size_t)c], features,
                                    (size_t)c, weight_scale, feature_scale, model->bits) +
                      model->head_bias[o];
        output[o] = isfinite(value) ? value : (value > 0.0f ? 1000000.0f : -1000000.0f);
    }
}

static kshira_status encoder_delta(float current, float gradient, float learning_rate,
                                   float weight_scale, float input_scale, int bias,
                                   int quantized, float *delta) {
    kshira_status status;
    float updated;
    if (delta == NULL || !isfinite(current) || !isfinite(gradient) ||
        !isfinite(learning_rate) || learning_rate <= 0.0f) return KSHIRA_ERR_RANGE;
    if (bias) {
        status = kshira_apply_qas(NULL, 0U, &gradient, 1U, weight_scale, input_scale);
    } else {
        status = kshira_apply_qas(&gradient, 1U, NULL, 0U, weight_scale, input_scale);
    }
    if (status != KSHIRA_OK ||
        !normalize_qas_gradient(&gradient, quantized)) {
        return KSHIRA_ERR_RANGE;
    }
    *delta = learning_rate * gradient;
    updated = current - *delta;
    return isfinite(*delta) && isfinite(updated) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

/* Apply a single-target straight-through gradient through project, depthwise,
 * and stem operators. The first pass validates every aggregate update; the
 * second pass commits the same values, keeping oversized updates transactional. */
static kshira_status rad_update_encoder(kshira_rad_model *model,
                                         const kshira_image_f32 *image, int target_x,
                                         int target_y, const float *fused_gradient,
                                         const kshira_sparse_mask *channel_mask,
                                         float learning_rate,
                                         rad_encoder_delta_buffer *delta_buffer, int commit) {
    int c = model->spec.feature_channels;
    float project_weight_scale = 1.0f;
    float branch_weight_scale[RAD_BRANCHES];
    float stem_weight_scale = 1.0f;
    float image_scale = 1.0f;
    float branch_gradient[RAD_BRANCHES][32] = {{0.0f}};
    float branch_input_scale = 0.0f;
    float stem_input_scale = 1.0f;

    if (delta_buffer == NULL) return KSHIRA_ERR_ARGUMENT;
    if (commit) {
        for (int oc = 0; oc < c; ++oc) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)oc);
            if (!enabled) continue;
            model->project_bias[oc] -= delta_buffer->project_bias[oc];
            for (int ic = 0; ic < c; ++ic) {
                size_t index = (size_t)oc * (size_t)c + (size_t)ic;
                model->project_weights[index] -= delta_buffer->project_weights[index];
            }
        }
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            for (int channel = 0; channel < c; ++channel) {
                int enabled = channel_mask == NULL ||
                              kshira_sparse_mask_get(channel_mask, (size_t)channel);
                size_t base = (size_t)channel * 9U;
                if (!enabled) continue;
                model->branch_bias[branch][channel] -= delta_buffer->branch_bias[branch][channel];
                for (int i = 0; i < 9; ++i) {
                    model->branch_weights[branch][base + (size_t)i] -=
                        delta_buffer->branch_weights[branch][base + (size_t)i];
                }
            }
        }
        for (int channel = 0; channel < c; ++channel) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)channel);
            size_t base = (size_t)channel * (size_t)image->channels * 9U;
            if (!enabled) continue;
            model->stem_bias[channel] -= delta_buffer->stem_bias[channel];
            for (int i = 0; i < image->channels * 9; ++i) {
                model->stem_weights[base + (size_t)i] -=
                    delta_buffer->stem_weights[base + (size_t)i];
            }
        }
        return KSHIRA_OK;
    }

    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        project_weight_scale = quant_scale_values(
            model->project_weights, (size_t)c * (size_t)c, model->bits);
        stem_weight_scale = quant_scale_values(
            model->stem_weights, (size_t)c * (size_t)image->channels * 9U, model->bits);
        image_scale = model->calibration_samples > 0U ? model->calibration_input_scale :
            (model->transient_scales_valid ? model->transient_image_scale :
             quant_scale_values(image->data,
                                (size_t)image->channels * (size_t)image->height *
                                (size_t)image->width, model->bits));
        if (model->calibration_samples > 0U) {
            stem_input_scale = model->calibration_stem_scale;
        } else if (model->transient_scales_valid) {
            stem_input_scale = model->transient_stem_scale;
        } else {
            int y0, y1, x0, x1;
            target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
            stem_input_scale = quant_scale_region(
                model->stem, c, model->map_height, model->map_width,
                y0, y1, x0, x1, model->bits);
        }
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            branch_weight_scale[branch] = quant_scale_values(
                model->branch_weights[branch], (size_t)c * 9U, model->bits);
            {
                float scale;
                if (model->calibration_samples > 0U) {
                    scale = model->calibration_branch_scale[branch];
                } else {
                    float maximum = 0.0f;
                    for (int channel = 0; channel < c; ++channel) {
                        float value = model->branches[branch][rad_index(
                            c, model->map_height, model->map_width, channel,
                            target_y, target_x)];
                        if (isfinite(value) && fabsf(value) > maximum) maximum = fabsf(value);
                    }
                    scale = kshira_symmetric_scale(maximum, model->bits);
                }
                if (scale > branch_input_scale) branch_input_scale = scale;
            }
        }
        if (branch_input_scale <= 0.0f) branch_input_scale = 1.0f;
    } else {
        branch_input_scale = 1.0f;
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) branch_weight_scale[branch] = 1.0f;
    }
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        for (int ic = 0; ic < c; ++ic) {
            size_t branch_index = rad_index(c, model->map_height, model->map_width,
                                            ic, target_y, target_x);
            float gradient = 0.0f;
            if (model->branches[branch][branch_index] > 0.0f) {
                for (int oc = 0; oc < c; ++oc) {
                    size_t fused_index = rad_index(c, model->map_height, model->map_width,
                                                   oc, target_y, target_x);
                    if (model->fused[fused_index] > 0.0f) {
                        gradient += fused_gradient[oc] *
                                    model->project_weights[(size_t)oc * (size_t)c +
                                                           (size_t)ic] /
                                    (float)RAD_BRANCHES;
                    }
                }
            }
            branch_gradient[branch][ic] = gradient;
        }
    }

    /* Project gradients. A mask selects output channels for encoder updates. */
    for (int oc = 0; oc < c; ++oc) {
        int enabled = channel_mask == NULL || kshira_sparse_mask_get(channel_mask, (size_t)oc);
        float fused_index = model->fused[rad_index(c, model->map_height, model->map_width,
                                                   oc, target_y, target_x)];
        float fused_grad = fused_index > 0.0f ? fused_gradient[oc] : 0.0f;
        if (enabled) {
            float bias_delta;
            if (encoder_delta(model->project_bias[oc], fused_grad, learning_rate,
                              project_weight_scale, branch_input_scale, 1,
                              model->bits != KSHIRA_BITS_FLOAT,
                              &bias_delta) != KSHIRA_OK) return KSHIRA_ERR_RANGE;
            delta_buffer->project_bias[oc] = bias_delta;
        }
        for (int ic = 0; ic < c; ++ic) {
            float branches = 0.0f;
            float weight_gradient;
            float weight_delta;
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                branches += model->branches[branch][rad_index(
                    c, model->map_height, model->map_width, ic, target_y, target_x)];
            }
            weight_gradient = fused_grad * (branches / (float)RAD_BRANCHES);
            if (enabled && encoder_delta(model->project_weights[(size_t)oc * (size_t)c +
                                                                  (size_t)ic],
                                         weight_gradient, learning_rate,
                                         project_weight_scale, branch_input_scale, 0,
                                         model->bits != KSHIRA_BITS_FLOAT,
                                         &weight_delta) != KSHIRA_OK) return KSHIRA_ERR_RANGE;
            if (enabled) {
                delta_buffer->project_weights[(size_t)oc * (size_t)c + (size_t)ic] =
                    weight_delta;
            }
        }
    }

    /* Depthwise branch gradients at the target cell. */
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        int dilation = 1 << branch;
        for (int channel = 0; channel < c; ++channel) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)channel);
            size_t branch_index = rad_index(c, model->map_height, model->map_width,
                                            channel, target_y, target_x);
            float branch_grad = branch_gradient[branch][channel];
            if (model->branches[branch][branch_index] <= 0.0f) branch_grad = 0.0f;
            if (enabled && !commit) {
                float bias_delta;
                if (encoder_delta(model->branch_bias[branch][channel], branch_grad,
                                  learning_rate, branch_weight_scale[branch],
                                  stem_input_scale, 1, model->bits != KSHIRA_BITS_FLOAT,
                                  &bias_delta) != KSHIRA_OK) {
                    return KSHIRA_ERR_RANGE;
                }
                delta_buffer->branch_bias[branch][channel] = bias_delta;
            }
            for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                int iy = target_y + (ky - 1) * dilation;
                for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                    int ix = target_x + (kx - 1) * dilation;
                    float weight_gradient = 0.0f;
                    float weight_delta;
                    if (iy >= 0 && iy < model->map_height && ix >= 0 &&
                        ix < model->map_width) {
                        weight_gradient = branch_grad * model->stem[
                            rad_index(c, model->map_height, model->map_width,
                                      channel, iy, ix)];
                    }
                    if (enabled && !commit && encoder_delta(
                            model->branch_weights[branch][((size_t)channel * RAD_KERNEL +
                                                            (size_t)ky) * RAD_KERNEL +
                                                           (size_t)kx],
                            weight_gradient, learning_rate, branch_weight_scale[branch],
                            stem_input_scale, 0, model->bits != KSHIRA_BITS_FLOAT,
                            &weight_delta) != KSHIRA_OK) {
                        return KSHIRA_ERR_RANGE;
                    }
                    if (enabled) {
                        delta_buffer->branch_weights[branch][(size_t)channel * 9U +
                                                              (size_t)ky * 3U +
                                                              (size_t)kx] = weight_delta;
                    }
                }
            }
        }
    }

    /* Stem gradients aggregate all branch receptive-field contributions per
     * parameter before validation, avoiding partial updates on overlap. */
    for (int channel = 0; channel < c; ++channel) {
        int enabled = channel_mask == NULL ||
                      kshira_sparse_mask_get(channel_mask, (size_t)channel);
        float bias_gradient = 0.0f;
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            int dilation = 1 << branch;
            for (int ky = 0; ky < RAD_KERNEL; ++ky) {
                int iy = target_y + (ky - 1) * dilation;
                if (iy < 0 || iy >= model->map_height) continue;
                for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                    int ix = target_x + (kx - 1) * dilation;
                    size_t stem_index;
                    if (ix < 0 || ix >= model->map_width) continue;
                    stem_index = rad_index(c, model->map_height, model->map_width,
                                           channel, iy, ix);
                    if (model->stem[stem_index] > 0.0f) {
                        bias_gradient += branch_gradient[branch][channel] *
                                          model->branch_weights[branch][
                                              ((size_t)channel * RAD_KERNEL + (size_t)ky) *
                                              RAD_KERNEL + (size_t)kx];
                    }
                }
            }
        }
        if (enabled) {
            float bias_delta;
            if (encoder_delta(model->stem_bias[channel], bias_gradient, learning_rate,
                              stem_weight_scale, image_scale, 1,
                              model->bits != KSHIRA_BITS_FLOAT, &bias_delta) != KSHIRA_OK) {
                return KSHIRA_ERR_RANGE;
            }
            delta_buffer->stem_bias[channel] = bias_delta;
        }
        for (int input_channel = 0; input_channel < image->channels; ++input_channel) {
            for (int stem_ky = 0; stem_ky < RAD_KERNEL; ++stem_ky) {
                for (int stem_kx = 0; stem_kx < RAD_KERNEL; ++stem_kx) {
                    float weight_gradient = 0.0f;
                    float weight_delta;
                    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                        int dilation = 1 << branch;
                        for (int branch_ky = 0; branch_ky < RAD_KERNEL; ++branch_ky) {
                            int map_y = target_y + (branch_ky - 1) * dilation;
                            if (map_y < 0 || map_y >= model->map_height) continue;
                            for (int branch_kx = 0; branch_kx < RAD_KERNEL; ++branch_kx) {
                                int map_x = target_x + (branch_kx - 1) * dilation;
                                size_t stem_index;
                                int image_y;
                                int image_x;
                                if (map_x < 0 || map_x >= model->map_width) continue;
                                stem_index = rad_index(c, model->map_height, model->map_width,
                                                       channel, map_y, map_x);
                                if (model->stem[stem_index] <= 0.0f) continue;
                                image_y = map_y * RAD_STRIDE + stem_ky - 1;
                                image_x = map_x * RAD_STRIDE + stem_kx - 1;
                                if (image_y < 0 || image_y >= image->height ||
                                    image_x < 0 || image_x >= image->width) continue;
                                weight_gradient += branch_gradient[branch][channel] *
                                    model->branch_weights[branch][
                                        ((size_t)channel * RAD_KERNEL +
                                         (size_t)branch_ky) * RAD_KERNEL +
                                        (size_t)branch_kx] *
                                    image->data[((size_t)input_channel * (size_t)image->height +
                                                 (size_t)image_y) * (size_t)image->width +
                                                (size_t)image_x];
                            }
                        }
                    }
                    if (enabled && encoder_delta(
                            model->stem_weights[((size_t)channel * (size_t)image->channels +
                                                (size_t)input_channel) * RAD_KERNEL * RAD_KERNEL +
                                               (size_t)stem_ky * RAD_KERNEL +
                                               (size_t)stem_kx],
                            weight_gradient, learning_rate, stem_weight_scale, image_scale, 0,
                            model->bits != KSHIRA_BITS_FLOAT,
                            &weight_delta) != KSHIRA_OK) return KSHIRA_ERR_RANGE;
                    if (enabled) {
                        delta_buffer->stem_weights[
                            ((size_t)channel * (size_t)image->channels +
                             (size_t)input_channel) * 9U + (size_t)stem_ky * 3U +
                            (size_t)stem_kx] = weight_delta;
                    }
                }
            }
        }
    }
    return KSHIRA_OK;
}

static void insert_top_k(kshira_rad_detection *detections, int *count, int capacity,
                         int limit, const kshira_rad_detection *candidate) {
    int position;
    if (!isfinite(candidate->score) || candidate->score <= 0.0f || capacity <= 0 || limit <= 0) {
        return;
    }
    if (*count < capacity && *count < limit) {
        position = (*count)++;
    } else {
        if (*count == 0 || candidate->score <= detections[*count - 1].score) return;
        position = *count - 1;
    }
    while (position > 0 && detections[position - 1].score < candidate->score) {
        if (position < *count) detections[position] = detections[position - 1];
        --position;
    }
    detections[position] = *candidate;
}

kshira_status kshira_rad_build(kshira_arena *arena, const kshira_rad_spec *spec,
                                kshira_rad_model **out) {
    kshira_rad_model *model;
    size_t map_elements;
    size_t count;
    if (arena == NULL || out == NULL || !valid_spec(spec)) return KSHIRA_ERR_ARGUMENT;
    *out = NULL;
    model = (kshira_rad_model *)kshira_arena_alloc(arena, sizeof(*model), _Alignof(kshira_rad_model));
    if (model == NULL) return KSHIRA_ERR_MEMORY;
    model->spec = *spec;
    model->arena = arena;
    model->bits = KSHIRA_BITS_FLOAT;
    model->map_height = spec->height / RAD_STRIDE +
                        (spec->height % RAD_STRIDE != 0 ? 1 : 0);
    model->map_width = spec->width / RAD_STRIDE +
                       (spec->width % RAD_STRIDE != 0 ? 1 : 0);
    model->outputs = 5 + spec->classes;
    if (!checked_elements(spec->feature_channels, model->map_height, model->map_width,
                          &map_elements)) return KSHIRA_ERR_RANGE;
    count = (size_t)spec->feature_channels * (size_t)spec->channels * 9U;
    model->stem_weights = alloc_floats(arena, count);
    model->stem_bias = alloc_floats(arena, (size_t)spec->feature_channels);
    if (model->stem_weights == NULL || model->stem_bias == NULL) return KSHIRA_ERR_MEMORY;
    model->parameter_bytes = count * sizeof(float) + (size_t)spec->feature_channels * sizeof(float);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        model->branch_weights[branch] = alloc_floats(arena, (size_t)spec->feature_channels * 9U);
        model->branch_bias[branch] = alloc_floats(arena, (size_t)spec->feature_channels);
        if (model->branch_weights[branch] == NULL || model->branch_bias[branch] == NULL) {
            return KSHIRA_ERR_MEMORY;
        }
        model->parameter_bytes += ((size_t)spec->feature_channels * 9U +
                                   (size_t)spec->feature_channels) * sizeof(float);
    }
    model->project_weights = alloc_floats(arena, (size_t)spec->feature_channels *
                                           (size_t)spec->feature_channels);
    model->project_bias = alloc_floats(arena, (size_t)spec->feature_channels);
    model->head_weights = alloc_floats(arena, (size_t)model->outputs *
                                       (size_t)spec->feature_channels);
    model->head_bias = alloc_floats(arena, (size_t)model->outputs);
    if (model->project_weights == NULL || model->project_bias == NULL ||
        model->head_weights == NULL || model->head_bias == NULL) return KSHIRA_ERR_MEMORY;
    model->parameter_bytes += ((size_t)spec->feature_channels * (size_t)spec->feature_channels +
                               (size_t)spec->feature_channels +
                               (size_t)model->outputs * (size_t)spec->feature_channels +
                               (size_t)model->outputs) * sizeof(float);
    model->stem = alloc_floats(arena, map_elements);
    model->fused = alloc_floats(arena, map_elements);
    if (model->stem == NULL || model->fused == NULL) return KSHIRA_ERR_MEMORY;
    model->activation_bytes = 2U * map_elements * sizeof(float);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        model->branches[branch] = alloc_floats(arena, map_elements);
        if (model->branches[branch] == NULL) return KSHIRA_ERR_MEMORY;
        model->activation_bytes += map_elements * sizeof(float);
    }
    model->encoder_deltas = alloc_encoder_deltas(arena, spec);
    if (model->encoder_deltas == NULL) return KSHIRA_ERR_MEMORY;
    model->activation_bytes +=
        (sizeof(*model->encoder_deltas) +
         ((size_t)spec->feature_channels * (size_t)spec->feature_channels +
          (size_t)spec->feature_channels +
          (size_t)RAD_BRANCHES * (size_t)spec->feature_channels * 9U +
          (size_t)RAD_BRANCHES * (size_t)spec->feature_channels +
          (size_t)spec->feature_channels * (size_t)spec->channels * 9U +
          (size_t)spec->feature_channels) * sizeof(float));
    if (kshira_rad_reset(model, spec->seed) != KSHIRA_OK) return KSHIRA_ERR_ARGUMENT;
    *out = model;
    return KSHIRA_OK;
}

kshira_status kshira_rad_reset(kshira_rad_model *model, int seed) {
    uint32_t state;
    if (model == NULL) return KSHIRA_ERR_ARGUMENT;
    state = (uint32_t)seed + 0x9e3779b9U;
    fill_random(model->stem_weights, (size_t)model->spec.feature_channels *
                (size_t)model->spec.channels * 9U, &state, 0.05f);
    fill_zero(model->stem_bias, (size_t)model->spec.feature_channels);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        fill_random(model->branch_weights[branch], (size_t)model->spec.feature_channels * 9U,
                    &state, 0.05f);
        fill_zero(model->branch_bias[branch], (size_t)model->spec.feature_channels);
    }
    fill_random(model->project_weights, (size_t)model->spec.feature_channels *
                (size_t)model->spec.feature_channels, &state, 0.05f);
    fill_zero(model->project_bias, (size_t)model->spec.feature_channels);
    fill_random(model->head_weights, (size_t)model->outputs *
                (size_t)model->spec.feature_channels, &state, 0.05f);
    fill_zero(model->head_bias, (size_t)model->outputs);
    clear_calibration(model);
    clear_transient_scales(model);
    return KSHIRA_OK;
}

kshira_status kshira_rad_set_bits(kshira_rad_model *model, kshira_bit_mode bits) {
    if (model == NULL || (bits != KSHIRA_BITS_FLOAT && !kshira_bit_mode_valid(bits))) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (model->bits != bits) clear_calibration(model);
    model->bits = bits;
    return KSHIRA_OK;
}

kshira_bit_mode kshira_rad_bits(const kshira_rad_model *model) {
    return model == NULL ? (kshira_bit_mode)0 : model->bits;
}

kshira_status kshira_rad_calibrate(kshira_rad_model *model,
                                    const kshira_image_f32 *image) {
    size_t image_count;
    size_t map_count;
    float input_scale;
    float stem_scale;
    float branch_scale[RAD_BRANCHES];
    if (model == NULL || image == NULL || image->data == NULL ||
        !kshira_bit_mode_valid(model->bits) ||
        image->channels != model->spec.channels || image->height != model->spec.height ||
        image->width != model->spec.width) return KSHIRA_ERR_ARGUMENT;
    if ((size_t)image->channels > SIZE_MAX / (size_t)image->height ||
        (size_t)image->channels * (size_t)image->height > SIZE_MAX / (size_t)image->width) {
        return KSHIRA_ERR_RANGE;
    }
    image_count = (size_t)image->channels * (size_t)image->height * (size_t)image->width;
    if ((size_t)model->spec.feature_channels > SIZE_MAX / (size_t)model->map_height ||
        (size_t)model->spec.feature_channels * (size_t)model->map_height >
            SIZE_MAX / (size_t)model->map_width) return KSHIRA_ERR_RANGE;
    map_count = (size_t)model->spec.feature_channels * (size_t)model->map_height *
                (size_t)model->map_width;
    for (size_t i = 0U; i < image_count; ++i) {
        if (!isfinite(image->data[i])) return KSHIRA_ERR_ARGUMENT;
    }
    /* Calibration intentionally uses the full floating-point map. The resulting
     * ranges are then shared by local target training and deployment inference. */
    conv_stem(model, image);
    stem_scale = activation_scale(model->stem, map_count, model->bits);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        depthwise_branch(model, branch);
        branch_scale[branch] = activation_scale(model->branches[branch], map_count,
                                                model->bits);
    }
    project_fused(model);
    input_scale = activation_scale(image->data, image_count, model->bits);
    if (model->calibration_samples == SIZE_MAX) return KSHIRA_ERR_RANGE;
    if (model->calibration_samples == 0U) {
        model->calibration_input_scale = input_scale;
        model->calibration_stem_scale = stem_scale;
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            model->calibration_branch_scale[branch] = branch_scale[branch];
        }
    } else {
        if (input_scale > model->calibration_input_scale) {
            model->calibration_input_scale = input_scale;
        }
        if (stem_scale > model->calibration_stem_scale) {
            model->calibration_stem_scale = stem_scale;
        }
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            if (branch_scale[branch] > model->calibration_branch_scale[branch]) {
                model->calibration_branch_scale[branch] = branch_scale[branch];
            }
        }
    }
    ++model->calibration_samples;
    return KSHIRA_OK;
}

int kshira_rad_calibration_ready(const kshira_rad_model *model) {
    return model != NULL && model->calibration_samples > 0U;
}

kshira_status kshira_rad_predict(kshira_rad_model *model,
                                 const kshira_image_f32 *image, float threshold,
                                 kshira_rad_detection *detections, int capacity, int *count) {
    int c;
    if (model == NULL || image == NULL || image->data == NULL || detections == NULL ||
        count == NULL || capacity < 0 || !isfinite(threshold) || threshold < 0.0f ||
        threshold > 1.0f || image->channels != model->spec.channels ||
        image->height != model->spec.height || image->width != model->spec.width) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if ((size_t)image->channels > SIZE_MAX / (size_t)image->height ||
        (size_t)image->channels * (size_t)image->height > SIZE_MAX / (size_t)image->width) {
        return KSHIRA_ERR_RANGE;
    }
    {
        size_t elements = (size_t)image->channels * (size_t)image->height *
                          (size_t)image->width;
        for (size_t i = 0U; i < elements; ++i) {
            if (!isfinite(image->data[i])) return KSHIRA_ERR_ARGUMENT;
        }
    }
    *count = 0;
    rad_forward(model, image);
    c = model->spec.feature_channels;
    for (int y = 0; y < model->map_height; ++y) {
        for (int x = 0; x < model->map_width; ++x) {
            float output[5 + RAD_MAX_CLASSES] = {0.0f};
            int class_id = 0;
            float best_class;
            float quality;
            float features[32];
            for (int ic = 0; ic < c; ++ic) {
                features[ic] = model->fused[rad_index(c, model->map_height,
                                                       model->map_width, ic, y, x)];
            }
            if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
                head_forward_quant(model, features, output);
            } else {
                head_forward_f32(model, features, output);
            }
            quality = sigmoid(output[4]);
            best_class = output[5];
            for (int cls = 1; cls < model->spec.classes; ++cls) {
                if (output[5 + cls] > best_class) {
                    best_class = output[5 + cls];
                    class_id = cls;
                }
            }
            {
                float class_score = sigmoid(best_class);
                float score = quality * class_score;
                float cx = ((float)x + 0.5f) * (float)RAD_STRIDE;
                float cy = ((float)y + 0.5f) * (float)RAD_STRIDE;
                float left = fmaxf(0.0f, output[0]) * (float)RAD_STRIDE;
                float top = fmaxf(0.0f, output[1]) * (float)RAD_STRIDE;
                float right = fmaxf(0.0f, output[2]) * (float)RAD_STRIDE;
                float bottom = fmaxf(0.0f, output[3]) * (float)RAD_STRIDE;
                kshira_rad_detection candidate;
                if (score < threshold) continue;
                candidate.score = score;
                candidate.quality = quality;
                candidate.box.x1 = fminf((float)image->width, fmaxf(0.0f, cx - left));
                candidate.box.y1 = fminf((float)image->height, fmaxf(0.0f, cy - top));
                candidate.box.x2 = fminf((float)image->width, fmaxf(0.0f, cx + right));
                candidate.box.y2 = fminf((float)image->height, fmaxf(0.0f, cy + bottom));
                candidate.box.class_id = class_id;
                if (!isfinite(candidate.box.x1) || !isfinite(candidate.box.y1) ||
                    !isfinite(candidate.box.x2) || !isfinite(candidate.box.y2) ||
                    candidate.box.x2 <= candidate.box.x1 || candidate.box.y2 <= candidate.box.y1) {
                    continue;
                }
                insert_top_k(detections, count, capacity, model->spec.top_k, &candidate);
            }
        }
    }
    return KSHIRA_OK;
}

kshira_status kshira_rad_train_step(kshira_rad_model *model, const kshira_image_f32 *image,
                                     const kshira_rad_box *target,
                                     const kshira_rad_train_config *config, float *loss) {
    float features[32];
    float output[5 + RAD_MAX_CLASSES] = {0.0f};
    float gradients[5 + RAD_MAX_CLASSES];
    float fused_gradient[32] = {0.0f};
    int target_x;
    int target_y;
    int c;
    size_t image_count;
    float weight_scale;
    float feature_scale;
    rad_encoder_delta_buffer *encoder_deltas;
    float loss_sum = 0.0f;
    if (model == NULL || image == NULL || image->data == NULL || target == NULL ||
        config == NULL || loss == NULL ||
        (config->bits != KSHIRA_BITS_FLOAT && !kshira_bit_mode_valid(config->bits)) ||
        config->update_mode < KSHIRA_UPDATE_FREEZE ||
        config->update_mode > KSHIRA_UPDATE_FULL ||
        !isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        image->channels != model->spec.channels || image->height != model->spec.height ||
        image->width != model->spec.width || target->class_id < 0 ||
        target->class_id >= model->spec.classes || !isfinite(target->x1) ||
        !isfinite(target->y1) || !isfinite(target->x2) || !isfinite(target->y2) ||
        target->x1 < 0.0f || target->y1 < 0.0f || target->x2 > (float)image->width ||
        target->y2 > (float)image->height || target->x2 <= target->x1 ||
        target->y2 <= target->y1) {
        return KSHIRA_ERR_ARGUMENT;
    }
    c = model->spec.feature_channels;
    encoder_deltas = model->encoder_deltas;
    if (encoder_deltas == NULL) return KSHIRA_ERR_MEMORY;
    if (config->channel_mask != NULL && config->channel_mask->channel_count != (size_t)c) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if ((size_t)image->channels > SIZE_MAX / (size_t)image->height ||
        (size_t)image->channels * (size_t)image->height > SIZE_MAX / (size_t)image->width) {
        return KSHIRA_ERR_RANGE;
    }
    image_count = (size_t)image->channels * (size_t)image->height * (size_t)image->width;
    for (size_t i = 0U; i < image_count; ++i) {
        if (!isfinite(image->data[i])) return KSHIRA_ERR_ARGUMENT;
    }
    if (model->bits != config->bits) {
        clear_calibration(model);
    }
    model->bits = config->bits;
    target_x = (int)(((target->x1 + target->x2) * 0.5f) / (float)RAD_STRIDE);
    target_y = (int)(((target->y1 + target->y2) * 0.5f) / (float)RAD_STRIDE);
    if (target_x < 0) target_x = 0;
    if (target_y < 0) target_y = 0;
    if (target_x >= model->map_width) target_x = model->map_width - 1;
    if (target_y >= model->map_height) target_y = model->map_height - 1;
    /* A single-target training sample only needs a nine-by-nine map tile: the
     * largest dilation is four. This keeps PRE/TRAIN work proportional to the
     * supervised receptive field instead of materializing a full feature map. */
    rad_forward_target(model, image, target_y, target_x);
    for (int ic = 0; ic < c; ++ic) {
        features[ic] = model->fused[rad_index(c, model->map_height, model->map_width,
                                              ic, target_y, target_x)];
    }
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        head_forward_quant(model, features, output);
        weight_scale = quant_scale_values(model->head_weights,
                                          (size_t)model->outputs * (size_t)c, model->bits);
        feature_scale = quant_scale_values(features, (size_t)c, model->bits);
    } else {
        head_forward_f32(model, features, output);
        weight_scale = 1.0f;
        feature_scale = 1.0f;
    }
    for (int o = 0; o < model->outputs; ++o) {
        float gradient;
        float error;
        if (o < 4) {
            float center_x = ((float)target_x + 0.5f) * (float)RAD_STRIDE;
            float center_y = ((float)target_y + 0.5f) * (float)RAD_STRIDE;
            float target_value[4] = {
                fmaxf(0.0f, (center_x - target->x1) / (float)RAD_STRIDE),
                fmaxf(0.0f, (center_y - target->y1) / (float)RAD_STRIDE),
                fmaxf(0.0f, (target->x2 - center_x) / (float)RAD_STRIDE),
                fmaxf(0.0f, (target->y2 - center_y) / (float)RAD_STRIDE)
            };
            error = output[o] - target_value[o];
            gradient = error;
        } else {
            float probability = sigmoid(output[o]);
            float desired = o == 4 ? 1.0f : (o - 5 == target->class_id ? 1.0f : 0.0f);
            error = probability - desired;
            gradient = error;
        }
        if (!isfinite(error) || !isfinite(gradient)) return KSHIRA_ERR_RANGE;
        loss_sum += 0.5f * error * error;
        gradients[o] = gradient;
    }
    if (!isfinite(loss_sum)) return KSHIRA_ERR_RANGE;
    if (config->update_mode == KSHIRA_UPDATE_FULL) {
        for (int ic = 0; ic < c; ++ic) {
            float gradient = 0.0f;
            for (int o = 0; o < model->outputs; ++o) {
                gradient += gradients[o] * class_gradient_scale(model, o, target->class_id) *
                            model->head_weights[(size_t)o * (size_t)c + (size_t)ic];
            }
            if (!isfinite(gradient)) return KSHIRA_ERR_RANGE;
            fused_gradient[ic] = gradient;
        }
    }
    if (config->update_mode != KSHIRA_UPDATE_FREEZE) {
        /* Preflight every update so an oversized learning rate cannot partially
         * mutate the model before returning KSHIRA_ERR_RANGE. */
        for (int o = 0; o < model->outputs; ++o) {
            float gradient_bias = gradients[o];
            float delta;
            float updated;
            if (kshira_apply_qas(NULL, 0U, &gradient_bias, 1U,
                                 weight_scale, feature_scale) != KSHIRA_OK ||
                !normalize_qas_gradient(&gradient_bias, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            gradient_bias *= class_gradient_scale(model, o, target->class_id);
            delta = config->learning_rate * gradient_bias;
            updated = model->head_bias[o] - delta;
            if (!isfinite(delta) || !isfinite(updated)) return KSHIRA_ERR_RANGE;
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS ||
                config->update_mode == KSHIRA_UPDATE_FULL) {
                for (int ic = 0; ic < c; ++ic) {
                    float gradient_weight;
                    if (config->update_mode == KSHIRA_UPDATE_CHANNELS &&
                        config->channel_mask != NULL && !kshira_sparse_mask_get(
                            config->channel_mask, (size_t)ic)) continue;
                    gradient_weight = gradients[o] * features[ic];
                    if (!isfinite(gradient_weight) ||
                        kshira_apply_qas(&gradient_weight, 1U, NULL, 0U,
                                         weight_scale, feature_scale) != KSHIRA_OK ||
                        !normalize_qas_gradient(&gradient_weight,
                                                model->bits != KSHIRA_BITS_FLOAT)) {
                        return KSHIRA_ERR_RANGE;
                    }
                    gradient_weight *= class_gradient_scale(model, o, target->class_id);
                    delta = config->learning_rate * gradient_weight;
                    updated = model->head_weights[(size_t)o * (size_t)c + (size_t)ic] - delta;
                    if (!isfinite(delta) || !isfinite(updated)) return KSHIRA_ERR_RANGE;
                }
            }
        }
        if (config->update_mode == KSHIRA_UPDATE_FULL) {
            if (rad_update_encoder(model, image, target_x, target_y, fused_gradient,
                                   config->channel_mask, config->learning_rate,
                                   encoder_deltas, 0) != KSHIRA_OK ||
                rad_update_encoder(model, image, target_x, target_y, fused_gradient,
                                   config->channel_mask, config->learning_rate,
                                   encoder_deltas, 1) != KSHIRA_OK) {
                return KSHIRA_ERR_RANGE;
            }
        }
        for (int o = 0; o < model->outputs; ++o) {
            float gradient_bias = gradients[o];
            if (kshira_apply_qas(NULL, 0U, &gradient_bias, 1U,
                                 weight_scale, feature_scale) != KSHIRA_OK) {
                return KSHIRA_ERR_RANGE;
            }
            if (!normalize_qas_gradient(&gradient_bias, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            gradient_bias *= class_gradient_scale(model, o, target->class_id);
            model->head_bias[o] -= config->learning_rate * gradient_bias;
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS ||
                config->update_mode == KSHIRA_UPDATE_FULL) {
                for (int ic = 0; ic < c; ++ic) {
                    float gradient_weight;
                    if (config->update_mode == KSHIRA_UPDATE_CHANNELS &&
                        config->channel_mask != NULL && !kshira_sparse_mask_get(
                            config->channel_mask, (size_t)ic)) continue;
                    gradient_weight = gradients[o] * features[ic];
                    if (kshira_apply_qas(&gradient_weight, 1U, NULL, 0U,
                                         weight_scale, feature_scale) != KSHIRA_OK) {
                        return KSHIRA_ERR_RANGE;
                    }
                    if (!normalize_qas_gradient(&gradient_weight,
                                                model->bits != KSHIRA_BITS_FLOAT)) {
                        return KSHIRA_ERR_RANGE;
                    }
                    gradient_weight *= class_gradient_scale(model, o, target->class_id);
                    model->head_weights[(size_t)o * (size_t)c + (size_t)ic] -=
                        config->learning_rate * gradient_weight;
                }
            }
        }
    }
    if (config->update_mode != KSHIRA_UPDATE_FREEZE) clear_calibration(model);
    *loss = loss_sum / (float)model->outputs;
    return isfinite(*loss) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

int kshira_rad_map_height(const kshira_rad_model *model) {
    return model == NULL ? 0 : model->map_height;
}

int kshira_rad_map_width(const kshira_rad_model *model) {
    return model == NULL ? 0 : model->map_width;
}

size_t kshira_rad_parameter_bytes(const kshira_rad_model *model) {
    return model == NULL ? 0U : model->parameter_bytes;
}

size_t kshira_rad_activation_bytes(const kshira_rad_model *model) {
    return model == NULL ? 0U : model->activation_bytes;
}
