/* Purpose: static single-map RAD encoder and bounded top-K detector head.
 * Ownership: the model and every buffer live in the caller's KSHIRA arena.
 * Failure: invalid specs, exhausted arena, malformed images, or output capacity
 * return explicit status; prediction performs no allocation and no NMS. */
#include "kshira_rad_internal.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const float RAD_QAS_GRADIENT_LIMIT = 1.0f;

static int valid_spec(const kshira_rad_spec *spec) {
    return spec != NULL && spec->width >= 8 && spec->height >= 8 &&
           spec->width <= INT_MAX - (RAD_STRIDE - 1) &&
           spec->height <= INT_MAX - (RAD_STRIDE - 1) &&
           spec->channels >= 1 && spec->channels <= 4 && spec->classes >= 1 &&
           spec->classes <= RAD_MAX_CLASSES && spec->feature_channels >= 1 &&
           spec->feature_channels <= RAD_MAX_FEATURES && spec->top_k >= 1 &&
           spec->top_k <= 64 &&
           (spec->multiscale_heads == 0 || spec->multiscale_heads == 1);
}

/* Head input width: semantic channels + contrast channel (PLAN_UPDATED §19). */
static int rad_head_in(const kshira_rad_model *model) {
    return model != NULL ? model->head_in : 0;
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

static int configure_update_workspace(kshira_arena *arena,
                                      const kshira_rad_spec *spec, int outputs,
                                      int head_in, rad_encoder_delta_buffer *encoder,
                                      rad_head_delta_buffer *head, size_t *workspace_bytes) {
    size_t channels;
    size_t input_channels;
    size_t encoder_count;
    size_t head_count;
    size_t workspace_count;
    float *storage;
    size_t offset = 0U;
    /* Full encoder and multi-scale head updates never run concurrently. */
    if (arena == NULL || spec == NULL || outputs <= 0 || head_in <= 0 || encoder == NULL ||
        workspace_bytes == NULL) return 0;
    channels = (size_t)spec->feature_channels;
    input_channels = (size_t)spec->channels;
    encoder_count = channels * channels + channels + channels * input_channels * 9U + channels;
    encoder_count += (size_t)RAD_BRANCHES * (channels * 9U + channels);
    head_count = (size_t)outputs * (size_t)head_in + (size_t)outputs;
    workspace_count = encoder_count > head_count ? encoder_count : head_count;
    storage = alloc_floats(arena, workspace_count);
    if (storage == NULL) return 0;
    encoder->project_weights = storage + offset;
    offset += channels * channels;
    encoder->project_bias = storage + offset;
    offset += channels;
    encoder->stem_weights = storage + offset;
    offset += channels * input_channels * 9U;
    encoder->stem_bias = storage + offset;
    offset += channels;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        encoder->branch_weights[branch] = storage + offset;
        offset += channels * 9U;
        encoder->branch_bias[branch] = storage + offset;
        offset += channels;
    }
    if (head != NULL) {
        head->weights = storage;
        head->bias = storage + (size_t)outputs * (size_t)head_in;
    }
    *workspace_bytes = workspace_count * sizeof(float);
    return offset == encoder_count;
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

/* Softmax over quality logits (indices [4, 4+K)) for dual class CE. */
static int quality_softmax(const kshira_rad_model *model, const float *output,
                           float *probabilities) {
    float maximum;
    float sum = 0.0f;
    int K;
    if (model == NULL || output == NULL || probabilities == NULL) return 0;
    K = model->spec.classes;
    if (K < 1 || K > RAD_MAX_CLASSES) return 0;
    maximum = output[4];
    for (int k = 1; k < K; ++k) {
        if (output[4 + k] > maximum) maximum = output[4 + k];
    }
    if (!isfinite(maximum)) return 0;
    for (int k = 0; k < K; ++k) {
        float v = expf(output[4 + k] - maximum);
        if (!isfinite(v)) return 0;
        probabilities[k] = v;
        sum += v;
    }
    if (!isfinite(sum) || sum <= 0.0f) return 0;
    for (int k = 0; k < K; ++k) probabilities[k] /= sum;
    return 1;
}

/* Quality-class head: independent sigmoids at indices [4, 4+K). */
static float quality_logit_at(const kshira_rad_model *model, const float *features,
                              int class_id, float weight_scale, float feature_scale) {
    int hin = model->head_in;
    size_t row = (size_t)(4 + class_id) * (size_t)hin;
    float logit;
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        logit = quantized_dot(&model->head_weights[row], features, (size_t)hin,
                              weight_scale, feature_scale, model->bits) +
                model->head_bias[4 + class_id];
    } else {
        logit = model->head_bias[4 + class_id];
        for (int ic = 0; ic < hin; ++ic) {
            logit += model->head_weights[row + (size_t)ic] * features[ic];
        }
    }
    return logit;
}

static float max_quality_score(const kshira_rad_model *model, const float *output) {
    float best = 0.0f;
    if (model == NULL || output == NULL || model->spec.classes < 1) return 0.0f;
    best = sigmoid(output[4]);
    for (int k = 1; k < model->spec.classes; ++k) {
        float q = sigmoid(output[4 + k]);
        if (q > best) best = q;
    }
    return best;
}

/* VFL positive q>0: ∂/∂z = q(p−q). Negative q=0: α p^{γ+1} with γ=2. */
static void quality_vfl_grad(float logit, float q, float pos_scale, float neg_alpha,
                             float *gradient_out, float *loss_out) {
    float p = sigmoid(logit);
    float gradient;
    float loss_term;
    if (q < 0.0f) q = 0.0f;
    if (q > 1.0f) q = 1.0f;
    if (q > 0.0f) {
        float err = p - q;
        gradient = pos_scale * q * err;
        loss_term = pos_scale * q *
            (-(q * logf(fmaxf(p, 1.0e-6f)) +
               (1.0f - q) * logf(fmaxf(1.0f - p, 1.0e-6f))));
    } else {
        float focusing = p * p;
        gradient = neg_alpha * focusing * p;
        loss_term = neg_alpha * focusing * p;
    }
    *gradient_out = gradient;
    *loss_out = loss_term;
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

/* Pointwise mixer on fused (in-place, temp vector). fused must already hold
 * mean of depthwise branches. */
static void mix_fused_inplace(const kshira_rad_model *model) {
    int c = model->spec.feature_channels;
    int mh = model->map_height;
    int mw = model->map_width;
    for (int y = 0; y < mh; ++y) {
        for (int x = 0; x < mw; ++x) {
            float in_vec[RAD_MAX_FEATURES];
            float out_vec[RAD_MAX_FEATURES];
            for (int ic = 0; ic < c; ++ic) {
                in_vec[ic] = model->fused[rad_index(c, mh, mw, ic, y, x)];
            }
            for (int oc = 0; oc < c; ++oc) {
                float sum = model->project_bias[oc];
                for (int ic = 0; ic < c; ++ic) {
                    sum += model->project_weights[(size_t)oc * (size_t)c + (size_t)ic] *
                           in_vec[ic];
                }
                out_vec[oc] = relu(sum);
            }
            for (int oc = 0; oc < c; ++oc) {
                model->fused[rad_index(c, mh, mw, oc, y, x)] = out_vec[oc];
            }
        }
    }
}

/* Legacy path when three distinct branch maps are resident. */
static void project_fused(const kshira_rad_model *model) {
    int c = model->spec.feature_channels;
    if (model->branch_maps_shared) {
        /* fused already = mean(branches); only mixer remains. */
        mix_fused_inplace(model);
        return;
    }
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

/* Evaluate one depthwise branch at a single cell into out[c] (stack-friendly). */
static void depthwise_branch_cell(const kshira_rad_model *model, int branch,
                                  int cell_y, int cell_x, float *out) {
    int c = model->spec.feature_channels;
    int dilation = 1 << branch;
    for (int channel = 0; channel < c; ++channel) {
        float sum = model->branch_bias[branch][channel];
        for (int ky = 0; ky < RAD_KERNEL; ++ky) {
            int iy = cell_y + (ky - 1) * dilation;
            if (iy < 0 || iy >= model->map_height) continue;
            for (int kx = 0; kx < RAD_KERNEL; ++kx) {
                int ix = cell_x + (kx - 1) * dilation;
                size_t wi;
                size_t xi;
                if (ix < 0 || ix >= model->map_width) continue;
                wi = ((size_t)channel * RAD_KERNEL + (size_t)ky) * RAD_KERNEL +
                     (size_t)kx;
                xi = rad_index(c, model->map_height, model->map_width, channel, iy, ix);
                sum += model->branch_weights[branch][wi] * model->stem[xi];
            }
        }
        out[channel] = relu(sum);
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

/* Dependency tile for dilated branches (d≤4) plus contrast halo (r=2).
 * Stem uses RAD_DEP_RADIUS; fused/contrast fill uses RAD_CONTRAST_RADIUS. */
static void target_bounds(const kshira_rad_model *model, int target_y, int target_x,
                          int *y0, int *y1, int *x0, int *x1) {
    *y0 = target_y - RAD_DEP_RADIUS;
    *y1 = target_y + RAD_DEP_RADIUS + 1;
    *x0 = target_x - RAD_DEP_RADIUS;
    *x1 = target_x + RAD_DEP_RADIUS + 1;
    if (*y0 < 0) *y0 = 0;
    if (*x0 < 0) *x0 = 0;
    if (*y1 > model->map_height) *y1 = model->map_height;
    if (*x1 > model->map_width) *x1 = model->map_width;
}

/* Local feature contrast κ = Σ_c (M_c - mean_c)², C = log1p(κ). */
static float rad_contrast_at(const kshira_rad_model *model, int y, int x) {
    int c = model->spec.feature_channels;
    int r = RAD_CONTRAST_RADIUS;
    float mean[RAD_MAX_FEATURES];
    float kappa = 0.0f;
    int count = 0;
    if (y < 0 || x < 0 || y >= model->map_height || x >= model->map_width) {
        return 0.0f;
    }
    for (int ic = 0; ic < c; ++ic) mean[ic] = 0.0f;
    for (int ny = y - r; ny <= y + r; ++ny) {
        if (ny < 0 || ny >= model->map_height) continue;
        for (int nx = x - r; nx <= x + r; ++nx) {
            if (nx < 0 || nx >= model->map_width) continue;
            for (int ic = 0; ic < c; ++ic) {
                mean[ic] += model->fused[rad_index(c, model->map_height,
                                                   model->map_width, ic, ny, nx)];
            }
            ++count;
        }
    }
    if (count <= 0) return 0.0f;
    for (int ic = 0; ic < c; ++ic) mean[ic] /= (float)count;
    for (int ic = 0; ic < c; ++ic) {
        float d = model->fused[rad_index(c, model->map_height, model->map_width,
                                         ic, y, x)] - mean[ic];
        kappa += d * d;
    }
    if (!isfinite(kappa) || kappa < 0.0f) kappa = 0.0f;
    return log1pf(kappa);
}

/* Fill head feature vector: semantic channels + contrast (PLAN_UPDATED hybrid head).
 * Contrast is log1p(κ) scaled so it stays same order as ReLU features. The
 * transform is identical at train and predict (no stream-dependent gate) so
 * save/load predictions stay bit-stable. */
static void rad_head_features(const kshira_rad_model *model, int y, int x,
                              float *features) {
    int c = model->spec.feature_channels;
    for (int ic = 0; ic < c; ++ic) {
        features[ic] = model->fused[rad_index(c, model->map_height, model->map_width,
                                              ic, y, x)];
    }
    /* Soft scale keeps contrast as a side-channel; semantic features dominate. */
    features[c] = 0.10f * rad_contrast_at(model, y, x);
    if (model->contrast != NULL) {
        model->contrast[(size_t)y * (size_t)model->map_width + (size_t)x] = features[c];
    }
}

/* Accumulate ∂L/∂M at the supervised cell from contrast (center term of §47).
 * Uses the exact center coefficient (1 - 1/N); neighbor mean mass is absorbed
 * into this single-cell encoder update so the 13×13 tile remains the unit. */
static void rad_contrast_grad_center(const kshira_rad_model *model, int y, int x,
                                     float g_contrast, float *fused_grad_center) {
    int c = model->spec.feature_channels;
    int r = RAD_CONTRAST_RADIUS;
    float mean[RAD_MAX_FEATURES];
    float kappa = 0.0f;
    float scale;
    float inv_n;
    int count = 0;
    int mh = model->map_height;
    int mw = model->map_width;
    if (!isfinite(g_contrast) || g_contrast == 0.0f || fused_grad_center == NULL) return;
    for (int ic = 0; ic < c; ++ic) mean[ic] = 0.0f;
    for (int ny = y - r; ny <= y + r; ++ny) {
        if (ny < 0 || ny >= mh) continue;
        for (int nx = x - r; nx <= x + r; ++nx) {
            if (nx < 0 || nx >= mw) continue;
            for (int ic = 0; ic < c; ++ic) {
                mean[ic] += model->fused[rad_index(c, mh, mw, ic, ny, nx)];
            }
            ++count;
        }
    }
    if (count <= 0) return;
    inv_n = 1.0f / (float)count;
    for (int ic = 0; ic < c; ++ic) {
        float d;
        mean[ic] *= inv_n;
        d = model->fused[rad_index(c, mh, mw, ic, y, x)] - mean[ic];
        kappa += d * d;
    }
    if (!isfinite(kappa) || kappa < 0.0f) kappa = 0.0f;
    scale = g_contrast / (1.0f + kappa);
    if (!isfinite(scale)) return;
    for (int ic = 0; ic < c; ++ic) {
        float d = model->fused[rad_index(c, mh, mw, ic, y, x)] - mean[ic];
        fused_grad_center[ic] += scale * 2.0f * d * (1.0f - inv_n);
    }
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

static void conv_stem_region(kshira_rad_model *model, const kshira_image_f32 *image,
                             int y0, int y1, int x0, int x1) {
    int c = model->spec.feature_channels;
    model->transient_scales_valid = 0;
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

static void conv_stem_target(kshira_rad_model *model, const kshira_image_f32 *image,
                             int target_y, int target_x) {
    int y0, y1, x0, x1;
    target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
    conv_stem_region(model, image, y0, y1, x0, x1);
}

static void conv_stem_region_quant(kshira_rad_model *model,
                                   const kshira_image_f32 *image,
                                   int y0, int y1, int x0, int x1) {
    int c = model->spec.feature_channels;
    size_t weight_count = (size_t)c * (size_t)image->channels * 9U;
    size_t image_count = (size_t)image->channels * (size_t)image->height *
                         (size_t)image->width;
    float weight_scale = quant_scale_values(model->stem_weights, weight_count, model->bits);
    float input_scale = calibrated_or_dynamic(model->calibration_input_scale, image->data,
                                              image_count, model->bits,
                                              model->calibration_samples);
    model->transient_image_scale = input_scale;
    model->transient_scales_valid = 1;
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

static void conv_stem_target_quant(kshira_rad_model *model,
                                   const kshira_image_f32 *image,
                                   int target_y, int target_x) {
    int y0, y1, x0, x1;
    target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
    conv_stem_region_quant(model, image, y0, y1, x0, x1);
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

static void depthwise_branch_target_quant_scaled(const kshira_rad_model *model, int branch,
                                                 int target_y, int target_x,
                                                 float weight_scale, float input_scale,
                                                 const int8_t *quantized_weights) {
    int c = model->spec.feature_channels;
    int dilation = 1 << branch;
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
                accumulator += (int64_t)(quantized_weights != NULL ?
                    quantized_weights[wi] : kshira_quantize_symmetric(
                        model->branch_weights[branch][wi], weight_scale, model->bits)) *
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

static void depthwise_branch_target_quant(const kshira_rad_model *model, int branch,
                                          int target_y, int target_x) {
    int c = model->spec.feature_channels;
    int y0, y1, x0, x1;
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
    depthwise_branch_target_quant_scaled(model, branch, target_y, target_x,
                                         weight_scale, input_scale, NULL);
}

static void project_fused_target(const kshira_rad_model *model, int target_y, int target_x) {
    int c = model->spec.feature_channels;
    float mean_vec[RAD_MAX_FEATURES];
    float branch_vec[RAD_MAX_FEATURES];
    if (model->branch_maps_shared) {
        for (int ic = 0; ic < c; ++ic) mean_vec[ic] = 0.0f;
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            depthwise_branch_cell(model, branch, target_y, target_x, branch_vec);
            for (int ic = 0; ic < c; ++ic) {
                mean_vec[ic] += branch_vec[ic] / (float)RAD_BRANCHES;
                /* Keep last branch in workspace for optional debug; encoder
                 * re-evaluates via depthwise_branch_cell. */
                model->branches[0][rad_index(c, model->map_height, model->map_width,
                                             ic, target_y, target_x)] = branch_vec[ic];
            }
        }
        for (int oc = 0; oc < c; ++oc) {
            float sum = model->project_bias[oc];
            for (int ic = 0; ic < c; ++ic) {
                sum += model->project_weights[(size_t)oc * (size_t)c + (size_t)ic] *
                       mean_vec[ic];
            }
            model->fused[rad_index(c, model->map_height, model->map_width, oc,
                                   target_y, target_x)] = relu(sum);
        }
        return;
    }
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

static void project_fused_target_quant_scaled(const kshira_rad_model *model,
                                              int target_y, int target_x,
                                              float weight_scale,
                                              const int8_t *quantized_weights) {
    int c = model->spec.feature_channels;
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
            {
                size_t wi = (size_t)oc * (size_t)c + (size_t)ic;
                accumulator += (int64_t)(quantized_weights != NULL ?
                    quantized_weights[wi] : kshira_quantize_symmetric(
                        model->project_weights[wi], weight_scale, model->bits)) * branch_sum;
            }
        }
        model->fused[rad_index(c, model->map_height, model->map_width, oc, target_y, target_x)] =
            relu((float)accumulator * weight_scale * input_scale /
                 (float)RAD_BRANCHES + model->project_bias[oc]);
    }
}

static void project_fused_target_quant(const kshira_rad_model *model,
                                       int target_y, int target_x) {
    int c = model->spec.feature_channels;
    float weight_scale = quant_scale_values(model->project_weights,
                                            (size_t)c * (size_t)c, model->bits);
    project_fused_target_quant_scaled(model, target_y, target_x, weight_scale, NULL);
}

/* Forward the 13×13 dependency tile (stem) and 5×5 fused support for contrast.
 * PLAN_UPDATED §50: r_dep = contrast_r(2) + max_dilation(4). */
static void rad_forward_target(kshira_rad_model *model, const kshira_image_f32 *image,
                               int target_y, int target_x) {
    int y0, y1, x0, x1;
    int cy0, cy1, cx0, cx1;
    target_bounds(model, target_y, target_x, &y0, &y1, &x0, &x1);
    cy0 = target_y - RAD_CONTRAST_RADIUS;
    cy1 = target_y + RAD_CONTRAST_RADIUS + 1;
    cx0 = target_x - RAD_CONTRAST_RADIUS;
    cx1 = target_x + RAD_CONTRAST_RADIUS + 1;
    if (cy0 < 0) cy0 = 0;
    if (cx0 < 0) cx0 = 0;
    if (cy1 > model->map_height) cy1 = model->map_height;
    if (cx1 > model->map_width) cx1 = model->map_width;
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        conv_stem_region_quant(model, image, y0, y1, x0, x1);
        for (int y = cy0; y < cy1; ++y) {
            for (int x = cx0; x < cx1; ++x) {
                for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                    depthwise_branch_target_quant(model, branch, y, x);
                }
                project_fused_target_quant(model, y, x);
            }
        }
    } else {
        conv_stem_region(model, image, y0, y1, x0, x1);
        for (int y = cy0; y < cy1; ++y) {
            for (int x = cx0; x < cx1; ++x) {
                for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                    depthwise_branch_target(model, branch, y, x);
                }
                project_fused_target(model, y, x);
            }
        }
    }
}

static void rad_forward(kshira_rad_model *model, const kshira_image_f32 *image) {
    int c = model->spec.feature_channels;
    size_t map_count = (size_t)c * (size_t)model->map_height * (size_t)model->map_width;
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        conv_stem_quant(model, image);
        if (model->branch_maps_shared) {
            for (size_t i = 0U; i < map_count; ++i) model->fused[i] = 0.0f;
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                depthwise_branch_quant(model, branch);
                for (size_t i = 0U; i < map_count; ++i) {
                    model->fused[i] += model->branches[0][i] / (float)RAD_BRANCHES;
                }
            }
            mix_fused_inplace(model);
        } else {
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                depthwise_branch_quant(model, branch);
            }
            project_fused_quant(model);
        }
    } else {
        conv_stem(model, image);
        if (model->branch_maps_shared) {
            for (size_t i = 0U; i < map_count; ++i) model->fused[i] = 0.0f;
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                depthwise_branch(model, branch);
                for (size_t i = 0U; i < map_count; ++i) {
                    model->fused[i] += model->branches[0][i] / (float)RAD_BRANCHES;
                }
            }
            mix_fused_inplace(model);
        } else {
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                depthwise_branch(model, branch);
            }
            project_fused(model);
        }
    }
}

static void head_forward_f32(const kshira_rad_model *model, const float *head_weights,
                             const float *head_bias, const float *features, float *output) {
    int hin = rad_head_in(model);
    for (int o = 0; o < model->outputs; ++o) {
        float sum = head_bias[o];
        for (int ic = 0; ic < hin; ++ic) {
            sum += head_weights[(size_t)o * (size_t)hin + (size_t)ic] * features[ic];
        }
        output[o] = isfinite(sum) ? sum : (sum > 0.0f ? 1000000.0f : -1000000.0f);
    }
}

static void head_forward_quant(const kshira_rad_model *model, const float *head_weights,
                               const float *head_bias, const float *features, float *output) {
    int hin = rad_head_in(model);
    float weight_scale = quant_scale_values(head_weights,
                                            (size_t)model->outputs * (size_t)hin, model->bits);
    float feature_scale = quant_scale_values(features, (size_t)hin, model->bits);
    for (int o = 0; o < model->outputs; ++o) {
        float value = quantized_dot(&head_weights[(size_t)o * (size_t)hin], features,
                                    (size_t)hin, weight_scale, feature_scale, model->bits) +
                      head_bias[o];
        output[o] = isfinite(value) ? value : (value > 0.0f ? 1000000.0f : -1000000.0f);
    }
}

/* Build a pooled P4/P5 feature vector directly from the stride-4 map. No
 * additional feature-map storage is needed; the bounded head consumes one
 * cell at a time. Contrast is measured on the pooled semantic vector. */
static void pooled_features(const kshira_rad_model *model, int level, int y, int x,
                            float *features) {
    int span = 1 << level;
    int y0 = y * span;
    int x0 = x * span;
    int y1 = y0 + span;
    int x1 = x0 + span;
    int c = model->spec.feature_channels;
    float mean[RAD_MAX_FEATURES];
    float kappa = 0.0f;
    int pool_count = 0;
    if (y1 > model->map_height) y1 = model->map_height;
    if (x1 > model->map_width) x1 = model->map_width;
    for (int ic = 0; ic < c; ++ic) {
        float sum = 0.0f;
        int samples = 0;
        for (int iy = y0; iy < y1; ++iy) {
            for (int ix = x0; ix < x1; ++ix) {
                sum += model->fused[rad_index(c, model->map_height, model->map_width,
                                              ic, iy, ix)];
                ++samples;
            }
        }
        features[ic] = samples > 0 ? sum / (float)samples : 0.0f;
        mean[ic] = 0.0f;
        if (samples > 0) pool_count = samples;
    }
    /* Local contrast over pooled cell's supporting patch (scale-aware). */
    if (pool_count > 0) {
        for (int ic = 0; ic < c; ++ic) {
            for (int iy = y0; iy < y1; ++iy) {
                for (int ix = x0; ix < x1; ++ix) {
                    mean[ic] += model->fused[rad_index(c, model->map_height,
                                                       model->map_width, ic, iy, ix)];
                }
            }
            mean[ic] /= (float)pool_count;
            {
                float d = features[ic] - mean[ic];
                kappa += d * d;
            }
        }
    }
    if (!isfinite(kappa) || kappa < 0.0f) kappa = 0.0f;
    features[c] = log1pf(kappa);
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
                    float cell_act[RAD_MAX_FEATURES];
                    float maximum = 0.0f;
                    depthwise_branch_cell(model, branch, target_y, target_x, cell_act);
                    for (int channel = 0; channel < c; ++channel) {
                        if (isfinite(cell_act[channel]) &&
                            fabsf(cell_act[channel]) > maximum) {
                            maximum = fabsf(cell_act[channel]);
                        }
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
    /* Recompute each branch at target for correct multi-dilation activations. */
    {
        float branch_act[RAD_BRANCHES][RAD_MAX_FEATURES];
        if (model->branch_maps_shared) {
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                depthwise_branch_cell(model, branch, target_y, target_x,
                                      branch_act[branch]);
            }
        } else {
            for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                for (int ic = 0; ic < c; ++ic) {
                    branch_act[branch][ic] = model->branches[branch][rad_index(
                        c, model->map_height, model->map_width, ic, target_y, target_x)];
                }
            }
        }
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            for (int ic = 0; ic < c; ++ic) {
                float gradient = 0.0f;
                if (branch_act[branch][ic] > 0.0f) {
                    for (int oc = 0; oc < c; ++oc) {
                        size_t fused_index = rad_index(c, model->map_height,
                                                       model->map_width, oc,
                                                       target_y, target_x);
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
        /* Project weight grads use mean of branch activations. */
        for (int oc = 0; oc < c; ++oc) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)oc);
            float fused_val = model->fused[rad_index(c, model->map_height, model->map_width,
                                                     oc, target_y, target_x)];
            float fused_grad = fused_val > 0.0f ? fused_gradient[oc] : 0.0f;
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
                    branches += branch_act[branch][ic];
                }
                weight_gradient = fused_grad * (branches / (float)RAD_BRANCHES);
                if (enabled && encoder_delta(model->project_weights[(size_t)oc * (size_t)c +
                                                                      (size_t)ic],
                                             weight_gradient, learning_rate,
                                             project_weight_scale, branch_input_scale, 0,
                                             model->bits != KSHIRA_BITS_FLOAT,
                                             &weight_delta) != KSHIRA_OK) {
                    return KSHIRA_ERR_RANGE;
                }
                if (enabled) {
                    delta_buffer->project_weights[(size_t)oc * (size_t)c + (size_t)ic] =
                        weight_delta;
                }
            }
        }
    }

    /* Depthwise branch gradients at the target cell.
     * branch_gradient already includes ReLU gate from recomputed activations. */
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        int dilation = 1 << branch;
        for (int channel = 0; channel < c; ++channel) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)channel);
            float branch_grad = branch_gradient[branch][channel];
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

static float rad_box_iou(const kshira_rad_box *a, const kshira_rad_box *b) {
    float x1;
    float y1;
    float x2;
    float y2;
    float intersection;
    float area_a;
    float area_b;
    float denom;
    if (a == NULL || b == NULL) return 0.0f;
    x1 = fmaxf(a->x1, b->x1);
    y1 = fmaxf(a->y1, b->y1);
    x2 = fminf(a->x2, b->x2);
    y2 = fminf(a->y2, b->y2);
    intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    area_a = fmaxf(0.0f, a->x2 - a->x1) * fmaxf(0.0f, a->y2 - a->y1);
    area_b = fmaxf(0.0f, b->x2 - b->x1) * fmaxf(0.0f, b->y2 - b->y1);
    denom = area_a + area_b - intersection;
    return denom > 0.0f ? intersection / denom : 0.0f;
}

/* Class-aware soft-NMS-style suppress on the fixed top-K list.
 * Same-class IoU ≥ thr is suppressed (standard); also suppress lower-score
 * cross-class overlaps at a slightly higher IoU so multi-class FP twins die
 * without a heap-based NMS. */
static void suppress_duplicate_detections(kshira_rad_detection *detections, int *count,
                                          float iou_threshold) {
    int write = 0;
    float cross_thr;
    if (detections == NULL || count == NULL || *count <= 1) return;
    if (!isfinite(iou_threshold) || iou_threshold <= 0.0f) return;
    /* Cross-class: only kill near-duplicates (higher thr). */
    cross_thr = iou_threshold + 0.15f;
    if (cross_thr > 0.90f) cross_thr = 0.90f;
    for (int i = 0; i < *count; ++i) {
        int keep = 1;
        for (int j = 0; j < write; ++j) {
            float iou = rad_box_iou(&detections[i].box, &detections[j].box);
            if (detections[i].box.class_id == detections[j].box.class_id) {
                if (iou >= iou_threshold) {
                    keep = 0;
                    break;
                }
            } else if (iou >= cross_thr) {
                keep = 0;
                break;
            }
        }
        if (keep) {
            if (write != i) detections[write] = detections[i];
            ++write;
        }
    }
    *count = write;
}

kshira_status kshira_rad_build(kshira_arena *arena, const kshira_rad_spec *spec,
                                kshira_rad_model **out) {
    kshira_rad_model *model;
    size_t map_elements;
    size_t count;
    size_t start_offset;
    size_t start_high_water;
    kshira_status status;
    if (arena == NULL || out == NULL || !valid_spec(spec)) return KSHIRA_ERR_ARGUMENT;
    *out = NULL;
    start_offset = arena->offset;
    start_high_water = arena->high_water;
    model = (kshira_rad_model *)kshira_arena_alloc(arena, sizeof(*model), _Alignof(kshira_rad_model));
    if (model == NULL) {
        status = KSHIRA_ERR_MEMORY;
        goto fail;
    }
    model->spec = *spec;
    model->arena = arena;
    model->bits = KSHIRA_BITS_FLOAT;
    model->map_height = spec->height / RAD_STRIDE +
                        (spec->height % RAD_STRIDE != 0 ? 1 : 0);
    model->map_width = spec->width / RAD_STRIDE +
                       (spec->width % RAD_STRIDE != 0 ? 1 : 0);
    /* Quality-class head (reviewer): 4 box distances + K independent class
     * quality logits. Score = max_k σ(z_k). Removes objectness×softmax product. */
    model->outputs = 4 + spec->classes;
    model->head_in = spec->feature_channels + RAD_CONTRAST_CHANNELS;
    model->contrast = NULL;
    model->stream_samples = 0U;
    model->stream_horizon = 5000U;
    if (!checked_elements(spec->feature_channels, model->map_height, model->map_width,
                          &map_elements)) {
        status = KSHIRA_ERR_RANGE;
        goto fail;
    }
    count = (size_t)spec->feature_channels * (size_t)spec->channels * 9U;
    model->stem_weights = alloc_floats(arena, count);
    model->stem_bias = alloc_floats(arena, (size_t)spec->feature_channels);
    if (model->stem_weights == NULL || model->stem_bias == NULL) {
        status = KSHIRA_ERR_MEMORY;
        goto fail;
    }
    model->parameter_bytes = count * sizeof(float) + (size_t)spec->feature_channels * sizeof(float);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        model->branch_weights[branch] = alloc_floats(arena, (size_t)spec->feature_channels * 9U);
        model->branch_bias[branch] = alloc_floats(arena, (size_t)spec->feature_channels);
        if (model->branch_weights[branch] == NULL || model->branch_bias[branch] == NULL) {
            status = KSHIRA_ERR_MEMORY;
            goto fail;
        }
        model->parameter_bytes += ((size_t)spec->feature_channels * 9U +
                                   (size_t)spec->feature_channels) * sizeof(float);
    }
    model->project_weights = alloc_floats(arena, (size_t)spec->feature_channels *
                                           (size_t)spec->feature_channels);
    model->project_bias = alloc_floats(arena, (size_t)spec->feature_channels);
    model->head_weights = alloc_floats(arena, (size_t)model->outputs *
                                       (size_t)model->head_in);
    model->head_bias = alloc_floats(arena, (size_t)model->outputs);
    if (model->project_weights == NULL || model->project_bias == NULL ||
        model->head_weights == NULL || model->head_bias == NULL) {
        status = KSHIRA_ERR_MEMORY;
        goto fail;
    }
    model->parameter_bytes += ((size_t)spec->feature_channels * (size_t)spec->feature_channels +
                               (size_t)spec->feature_channels +
                               (size_t)model->outputs * (size_t)model->head_in +
                               (size_t)model->outputs) * sizeof(float);
    model->scale_heads_ready = 0;
    model->scale_head_trained_mask = 0;
    model->encoder_deltas = &model->encoder_delta_storage;
    model->scale_head_deltas = spec->multiscale_heads ? &model->scale_head_delta_storage : NULL;
    for (int level = 0; level < RAD_SCALES; ++level) {
        model->scale_head_weights[level] = NULL;
        model->scale_head_bias[level] = NULL;
    }
    if (spec->multiscale_heads) {
        size_t head_count = (size_t)model->outputs * (size_t)model->head_in;
        for (int level = 1; level < RAD_SCALES; ++level) {
            model->scale_head_weights[level] = alloc_floats(arena, head_count);
            model->scale_head_bias[level] = alloc_floats(arena, (size_t)model->outputs);
            if (model->scale_head_weights[level] == NULL ||
                model->scale_head_bias[level] == NULL) {
                status = KSHIRA_ERR_MEMORY;
                goto fail;
            }
        }
        model->scale_heads_ready = 1;
        model->parameter_bytes += (size_t)(RAD_SCALES - 1) *
                                  (head_count + (size_t)model->outputs) * sizeof(float);
    }
    model->stem = alloc_floats(arena, map_elements);
    model->fused = alloc_floats(arena, map_elements);
    if (model->stem == NULL || model->fused == NULL) {
        status = KSHIRA_ERR_MEMORY;
        goto fail;
    }
    model->activation_bytes = 2U * map_elements * sizeof(float);
    /* Sequential branch workspace (PLAN_UPDATED): one map for all dilations. */
    {
        float *branch_ws = alloc_floats(arena, map_elements);
        if (branch_ws == NULL) {
            status = KSHIRA_ERR_MEMORY;
            goto fail;
        }
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            model->branches[branch] = branch_ws;
        }
        model->branch_maps_shared = 1;
        model->activation_bytes += map_elements * sizeof(float);
    }
    {
        size_t update_workspace_bytes;
        if (!configure_update_workspace(arena, spec, model->outputs, model->head_in,
                                         &model->encoder_delta_storage,
                                         model->scale_head_deltas != NULL ?
                                             &model->scale_head_delta_storage : NULL,
                                         &update_workspace_bytes)) {
            status = KSHIRA_ERR_MEMORY;
            goto fail;
        }
        model->activation_bytes += update_workspace_bytes;
    }
    if (kshira_rad_reset(model, spec->seed) != KSHIRA_OK) {
        status = KSHIRA_ERR_ARGUMENT;
        goto fail;
    }
    *out = model;
    return KSHIRA_OK;

fail:
    arena->offset = start_offset;
    arena->high_water = start_high_water;
    *out = NULL;
    return status;
}

kshira_status kshira_rad_reset(kshira_rad_model *model, int seed) {
    uint32_t state;
    float encoder_limit;
    float head_limit;
    if (model == NULL) return KSHIRA_ERR_ARGUMENT;
    model->spec.seed = seed;
    state = (uint32_t)seed + 0x9e3779b9U;
    encoder_limit = sqrtf(6.0f /
                          ((float)model->spec.channels * (float)(RAD_KERNEL * RAD_KERNEL)));
    /* Xavier fan-in on semantic width so hybrid head matches pre-contrast scale. */
    head_limit = sqrtf(6.0f / (float)model->spec.feature_channels);
    fill_random(model->stem_weights, (size_t)model->spec.feature_channels *
                (size_t)model->spec.channels * 9U, &state, encoder_limit);
    fill_zero(model->stem_bias, (size_t)model->spec.feature_channels);
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        fill_random(model->branch_weights[branch], (size_t)model->spec.feature_channels * 9U,
                    &state, sqrtf(6.0f / 9.0f));
        fill_zero(model->branch_bias[branch], (size_t)model->spec.feature_channels);
    }
    /* Identity-init mixer (reviewer §6): W = I + ε so early training does not
     * scramble depthwise features before channel mixing is learned. */
    {
        int c = model->spec.feature_channels;
        fill_random(model->project_weights, (size_t)c * (size_t)c, &state, 0.02f);
        for (int i = 0; i < c; ++i) {
            model->project_weights[(size_t)i * (size_t)c + (size_t)i] += 1.0f;
        }
    }
    fill_zero(model->project_bias, (size_t)model->spec.feature_channels);
    fill_random(model->head_weights, (size_t)model->outputs * (size_t)model->head_in,
                &state, head_limit);
    fill_zero(model->head_bias, (size_t)model->outputs);
    /* Historical objectness prior −2 worked for TP formation. */
    {
        float quality_bias = -2.0f;
        for (int k = 0; k < model->spec.classes; ++k) {
            model->head_bias[4 + k] = quality_bias;
        }
    }
    /* Zero contrast column at init; grow from data. */
    if (model->head_in > model->spec.feature_channels) {
        for (int o = 0; o < model->outputs; ++o) {
            model->head_weights[(size_t)o * (size_t)model->head_in +
                                (size_t)model->spec.feature_channels] = 0.0f;
        }
    }
    if (model->scale_heads_ready) {
        size_t head_count = (size_t)model->outputs * (size_t)model->head_in;
        for (int level = 1; level < RAD_SCALES; ++level) {
            for (size_t i = 0U; i < head_count; ++i) {
                model->scale_head_weights[level][i] = model->head_weights[i];
            }
            for (int o = 0; o < model->outputs; ++o) {
                model->scale_head_bias[level][o] = model->head_bias[o];
            }
        }
        model->scale_head_trained_mask = 0;
    }
    clear_calibration(model);
    clear_transient_scales(model);
    for (int class_id = 0; class_id < RAD_MAX_CLASSES; ++class_id) {
        model->class_count[class_id] = 0.0f;
    }
    model->class_total = 0.0f;
    model->stream_samples = 0U;
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

int kshira_rad_multiscale_ready(const kshira_rad_model *model) {
    return model != NULL && model->scale_heads_ready;
}

kshira_status kshira_rad_predict(kshira_rad_model *model,
                                 const kshira_image_f32 *image, float threshold,
                                 kshira_rad_detection *detections, int capacity, int *count) {
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
    for (int level = 0; level < 3; ++level) {
        int span = 1 << level;
        int scale_height = (model->map_height + span - 1) / span;
        int scale_width = (model->map_width + span - 1) / span;
        float stride = (float)(RAD_STRIDE * span);
        const int use_scale_head = level > 0 && model->scale_heads_ready &&
                                   (model->scale_head_trained_mask & (1 << level));
        /* Untrained P4/P5 heads used to reuse the stride-4 head on pooled maps,
         * flooding near-duplicate boxes. Only emit coarser levels when that
         * scale head was actually trained (ODT path). */
        if (level > 0 && !use_scale_head) continue;
        for (int y = 0; y < scale_height; ++y) {
            for (int x = 0; x < scale_width; ++x) {
                int y0 = y * span;
                int x0 = x * span;
                int y1 = y0 + span;
                int x1 = x0 + span;
                float output[4 + RAD_MAX_CLASSES] = {0.0f};
                int class_id = 0;
                float quality;
                float score;
                float features[RAD_MAX_HEAD_IN];
                const float *head_weights = use_scale_head ?
                                             model->scale_head_weights[level] :
                                             model->head_weights;
                const float *head_bias = use_scale_head ?
                                         model->scale_head_bias[level] : model->head_bias;
                if (level == 0) {
                    rad_head_features(model, y, x, features);
                } else {
                    pooled_features(model, level, y, x, features);
                }
                if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
                    head_forward_quant(model, head_weights, head_bias, features, output);
                } else {
                    head_forward_f32(model, head_weights, head_bias, features, output);
                }
                /* Quality-class head: score = max_k σ(z_k) (reviewer §1).
                 * Mild temperature sharpen spreads mid-band scores for top-K
                 * without changing order (monotone). */
                quality = max_quality_score(model, output);
                class_id = 0;
                {
                    float best = sigmoid(output[4]);
                    for (int cls = 1; cls < model->spec.classes; ++cls) {
                        float qk = sigmoid(output[4 + cls]);
                        if (qk > best) {
                            best = qk;
                            class_id = cls;
                        }
                    }
                }
                {
                    float cx;
                    float cy;
                    float left = fmaxf(0.0f, output[0]) * stride;
                    float top = fmaxf(0.0f, output[1]) * stride;
                    float right = fmaxf(0.0f, output[2]) * stride;
                    float bottom = fmaxf(0.0f, output[3]) * stride;
                    kshira_rad_detection candidate;
                    float q_clamp;
                    float logit_q;
                    memset(&candidate, 0, sizeof(candidate));
                    q_clamp = quality;
                    if (q_clamp < 1.0e-4f) q_clamp = 1.0e-4f;
                    if (q_clamp > 1.0f - 1.0e-4f) q_clamp = 1.0f - 1.0e-4f;
                    logit_q = logf(q_clamp / (1.0f - q_clamp));
                    /* Sharper temperature spreads mid-band qualities so thr
                     * can cut FPs when ranking has opened even a small gap. */
                    score = sigmoid(logit_q / 0.42f);
                    if (y1 > model->map_height) y1 = model->map_height;
                    if (x1 > model->map_width) x1 = model->map_width;
                    cx = ((float)x0 + (float)x1) * 0.5f * (float)RAD_STRIDE;
                    cy = ((float)y0 + (float)y1) * 0.5f * (float)RAD_STRIDE;
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
    }
    /* Tighter class-aware suppress (0.40) + cross-class near-dup kill. */
    suppress_duplicate_detections(detections, count, 0.40f);
    return KSHIRA_OK;
}

/* Train head (+ optional encoder) at one positive map cell. Center sampling
 * neighbors call this with update_encoder=0 for cheap one-to-many coverage. */
/* train_scope: 0 = full (box+all quality), 1 = box + quality at assigned class
 * (and zero other qualities) for high-quality neighbors (reviewer §5).
 * map_ready: 1 if rad_forward already filled fused/stem/branches for this image. */
static kshira_status rad_train_positive_at_cell_ex(
    kshira_rad_model *model, const kshira_image_f32 *image,
    const kshira_rad_box *target, int cell_x, int cell_y,
    const kshira_rad_train_config *config, int update_encoder, float lr_scale,
    int train_scope, int map_ready, float *loss_out) {
    float features[RAD_MAX_HEAD_IN];
    float output[4 + RAD_MAX_CLASSES] = {0.0f};
    float gradients[4 + RAD_MAX_CLASSES];
    float bias_gradients[4 + RAD_MAX_CLASSES];
    float fused_gradient[RAD_MAX_FEATURES] = {0.0f};
    float weight_scale;
    float feature_scale;
    float loss_sum = 0.0f;
    float step_lr;
    float g_contrast = 0.0f;
    int c = model->spec.feature_channels;
    int hin = model->head_in;
    rad_encoder_delta_buffer *encoder_deltas = model->encoder_deltas;
    if (cell_x < 0 || cell_y < 0 || cell_x >= model->map_width ||
        cell_y >= model->map_height || encoder_deltas == NULL) {
        return KSHIRA_ERR_ARGUMENT;
    }
    step_lr = config->learning_rate * lr_scale;
    if (!(step_lr > 0.0f) || !isfinite(step_lr)) return KSHIRA_ERR_ARGUMENT;
    if (!map_ready) {
        rad_forward_target(model, image, cell_y, cell_x);
    }
    rad_head_features(model, cell_y, cell_x, features);
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        head_forward_quant(model, model->head_weights, model->head_bias, features, output);
        weight_scale = quant_scale_values(model->head_weights,
                                          (size_t)model->outputs * (size_t)hin, model->bits);
        feature_scale = quant_scale_values(features, (size_t)hin, model->bits);
    } else {
        head_forward_f32(model, model->head_weights, model->head_bias, features, output);
        weight_scale = 1.0f;
        feature_scale = 1.0f;
    }
    {
        /* Quality-class head (reviewer §1): outputs = 4 box + K independent
         * class-quality logits. Assigned class gets IoU-aware VFL target;
         * other classes get negative VFL (q=0). Score at deploy = max_k σ(z_k). */
        float center_x = ((float)cell_x + 0.5f) * (float)RAD_STRIDE;
        float center_y = ((float)cell_y + 0.5f) * (float)RAD_STRIDE;
        float target_value[4] = {
            fmaxf(0.0f, (center_x - target->x1) / (float)RAD_STRIDE),
            fmaxf(0.0f, (center_y - target->y1) / (float)RAD_STRIDE),
            fmaxf(0.0f, (target->x2 - center_x) / (float)RAD_STRIDE),
            fmaxf(0.0f, (target->y2 - center_y) / (float)RAD_STRIDE)
        };
        float pred_left = fmaxf(0.0f, output[0]);
        float pred_top = fmaxf(0.0f, output[1]);
        float pred_right = fmaxf(0.0f, output[2]);
        float pred_bottom = fmaxf(0.0f, output[3]);
        kshira_rad_box pred_box = {
            fmaxf(0.0f, center_x - pred_left * (float)RAD_STRIDE),
            fmaxf(0.0f, center_y - pred_top * (float)RAD_STRIDE),
            center_x + pred_right * (float)RAD_STRIDE,
            center_y + pred_bottom * (float)RAD_STRIDE,
            target->class_id
        };
        float iou = rad_box_iou(&pred_box, target);
        float class_weight = 1.0f;
        float q_star;
        float map_cx = ((target->x1 + target->x2) * 0.5f) / (float)RAD_STRIDE;
        float map_cy = ((target->y1 + target->y2) * 0.5f) / (float)RAD_STRIDE;
        float centre_weight = 1.0f;
        int k_star = target->class_id;
        if (!isfinite(iou) || iou < 0.0f) iou = 0.0f;
        if (iou > 1.0f) iou = 1.0f;
        {
            float progress = 0.0f;
            float eta;
            if (model->stream_horizon > 0U) {
                progress = (float)model->stream_samples / (float)model->stream_horizon;
                if (progress > 1.0f) progress = 1.0f;
            }
            eta = progress < 0.20f ? (1.0f - progress / 0.20f) : 0.0f;
            if (train_scope == 0) {
                q_star = eta * 1.0f + (1.0f - eta) * fmaxf(0.85f, iou);
                if (q_star < 0.85f) q_star = 0.85f;
            } else {
                float dx = (float)cell_x - map_cx;
                float dy = (float)cell_y - map_cy;
                float rx = fmaxf(1.0f, (target->x2 - target->x1) / (float)RAD_STRIDE * 0.25f);
                float ry = fmaxf(1.0f, (target->y2 - target->y1) / (float)RAD_STRIDE * 0.25f);
                if (rx > 3.0f) rx = 3.0f;
                if (ry > 3.0f) ry = 3.0f;
                centre_weight = expf(-(dx * dx) / (rx * rx) - (dy * dy) / (ry * ry));
                if (centre_weight < 0.10f) centre_weight = 0.10f;
                q_star = (eta * 0.6f + (1.0f - eta) * iou) * centre_weight;
                if (q_star < 0.05f) q_star = 0.05f;
            }
            if (q_star > 1.0f) q_star = 1.0f;
        }
        if (train_scope == 0 && k_star >= 0 && k_star < model->spec.classes) {
            float count;
            float prior;
            model->class_count[k_star] += 1.0f;
            model->class_total += 1.0f;
            if (model->class_total >= 64.0f) {
                count = model->class_count[k_star];
                prior = count / fmaxf(1.0f, model->class_total);
                class_weight = sqrtf(1.0f / fmaxf(0.2f, prior * (float)model->spec.classes));
                if (class_weight < 0.75f) class_weight = 0.75f;
                if (class_weight > 2.0f) class_weight = 2.0f;
            }
        }
        loss_sum += 1.0f * (1.0f - iou);
        for (int o = 0; o < model->outputs; ++o) {
            float gradient = 0.0f;
            float error = 0.0f;
            if (o < 4) {
                error = output[o] - target_value[o];
                if (!isfinite(error)) {
                    error = error > 0.0f ? 1.0e6f : -1.0e6f;
                } else if (error > 1.0e6f) {
                    error = 1.0e6f;
                } else if (error < -1.0e6f) {
                    error = -1.0e6f;
                }
                if (fabsf(error) < 1.0f) {
                    gradient = error;
                    loss_sum += 0.5f * error * error;
                } else {
                    gradient = error > 0.0f ? 1.0f : -1.0f;
                    loss_sum += fabsf(error) - 0.5f;
                }
                gradient *= (1.0f + 0.5f * (1.0f - iou));
            } else {
                int k = o - 4;
                float loss_term = 0.0f;
                if (k == k_star) {
                    float pos_scale = (train_scope == 0) ? 3.5f : 1.75f;
                    pos_scale *= class_weight;
                    quality_vfl_grad(output[o], q_star, pos_scale, 0.0f, &gradient,
                                     &loss_term);
                } else {
                    float p_wrong = sigmoid(output[o]);
                    float neg_w = (train_scope == 0) ? 1.5f : 0.75f;
                    gradient = neg_w * p_wrong;
                    loss_term = neg_w * (-logf(fmaxf(1.0f - p_wrong, 1.0e-6f)));
                }
                loss_sum += loss_term;
                error = gradient;
            }
            if (!isfinite(error) || !isfinite(gradient)) return KSHIRA_ERR_RANGE;
            gradients[o] = gradient;
            bias_gradients[o] = gradient;
        }
    }
    if (!isfinite(loss_sum)) return KSHIRA_ERR_RANGE;
    if (update_encoder && config->update_mode == KSHIRA_UPDATE_FULL) {
        for (int ic = 0; ic < c; ++ic) {
            float gradient = 0.0f;
            for (int o = 0; o < model->outputs; ++o) {
                gradient += gradients[o] *
                            model->head_weights[(size_t)o * (size_t)hin + (size_t)ic];
            }
            if (!isfinite(gradient)) return KSHIRA_ERR_RANGE;
            fused_gradient[ic] = gradient;
        }
        g_contrast = 0.0f;
        for (int o = 0; o < model->outputs; ++o) {
            g_contrast += gradients[o] *
                          model->head_weights[(size_t)o * (size_t)hin + (size_t)c];
        }
        if (isfinite(g_contrast)) {
            rad_contrast_grad_center(model, cell_y, cell_x, g_contrast, fused_gradient);
        }
    }
    if (config->update_mode != KSHIRA_UPDATE_FREEZE) {
        for (int o = 0; o < model->outputs; ++o) {
            float gradient_bias = bias_gradients[o];
            float delta;
            float updated;
            if (kshira_apply_qas(NULL, 0U, &gradient_bias, 1U,
                                 weight_scale, feature_scale) != KSHIRA_OK ||
                !normalize_qas_gradient(&gradient_bias, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            delta = step_lr * gradient_bias;
            updated = model->head_bias[o] - delta;
            if (!isfinite(delta) || !isfinite(updated)) return KSHIRA_ERR_RANGE;
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS ||
                config->update_mode == KSHIRA_UPDATE_FULL) {
                for (int ic = 0; ic < hin; ++ic) {
                    float gradient_weight;
                    /* Channel mask only gates semantic channels; contrast always trains. */
                    if (config->update_mode == KSHIRA_UPDATE_CHANNELS &&
                        ic < c && config->channel_mask != NULL &&
                        !kshira_sparse_mask_get(config->channel_mask, (size_t)ic)) continue;
                    gradient_weight = gradients[o] * features[ic];
                    if (!isfinite(gradient_weight) ||
                        kshira_apply_qas(&gradient_weight, 1U, NULL, 0U,
                                         weight_scale, feature_scale) != KSHIRA_OK ||
                        !normalize_qas_gradient(&gradient_weight,
                                                model->bits != KSHIRA_BITS_FLOAT)) {
                        return KSHIRA_ERR_RANGE;
                    }
                    delta = step_lr * gradient_weight;
                    updated = model->head_weights[(size_t)o * (size_t)hin + (size_t)ic] - delta;
                    if (!isfinite(delta) || !isfinite(updated)) return KSHIRA_ERR_RANGE;
                }
            }
        }
        if (update_encoder && config->update_mode == KSHIRA_UPDATE_FULL) {
            if (rad_update_encoder(model, image, cell_x, cell_y, fused_gradient,
                                   config->channel_mask, step_lr,
                                   encoder_deltas, 0) != KSHIRA_OK ||
                rad_update_encoder(model, image, cell_x, cell_y, fused_gradient,
                                   config->channel_mask, step_lr,
                                   encoder_deltas, 1) != KSHIRA_OK) {
                return KSHIRA_ERR_RANGE;
            }
        }
        for (int o = 0; o < model->outputs; ++o) {
            float gradient_bias = bias_gradients[o];
            if (kshira_apply_qas(NULL, 0U, &gradient_bias, 1U,
                                 weight_scale, feature_scale) != KSHIRA_OK) {
                return KSHIRA_ERR_RANGE;
            }
            if (!normalize_qas_gradient(&gradient_bias, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            model->head_bias[o] -= step_lr * gradient_bias;
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS ||
                config->update_mode == KSHIRA_UPDATE_FULL) {
                for (int ic = 0; ic < hin; ++ic) {
                    float gradient_weight;
                    if (config->update_mode == KSHIRA_UPDATE_CHANNELS &&
                        ic < c && config->channel_mask != NULL &&
                        !kshira_sparse_mask_get(config->channel_mask, (size_t)ic)) continue;
                    gradient_weight = gradients[o] * features[ic];
                    if (kshira_apply_qas(&gradient_weight, 1U, NULL, 0U,
                                         weight_scale, feature_scale) != KSHIRA_OK) {
                        return KSHIRA_ERR_RANGE;
                    }
                    if (!normalize_qas_gradient(&gradient_weight,
                                                model->bits != KSHIRA_BITS_FLOAT)) {
                        return KSHIRA_ERR_RANGE;
                    }
                    model->head_weights[(size_t)o * (size_t)hin + (size_t)ic] -=
                        step_lr * gradient_weight;
                }
            }
        }
    }
    *loss_out = loss_sum / (float)model->outputs;
    return isfinite(*loss_out) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

static kshira_status rad_train_positive_at_cell(
    kshira_rad_model *model, const kshira_image_f32 *image,
    const kshira_rad_box *target, int cell_x, int cell_y,
    const kshira_rad_train_config *config, int update_encoder, float lr_scale,
    int train_scope, float *loss_out) {
    return rad_train_positive_at_cell_ex(model, image, target, cell_x, cell_y,
                                         config, update_encoder, lr_scale,
                                         train_scope, 0, loss_out);
}

kshira_status kshira_rad_objectness_at(kshira_rad_model *model, const kshira_image_f32 *image,
                                        int cell_y, int cell_x, float *probability) {
    /* Returns max_k σ(quality_k) for HNM ranking (quality-class head). */
    float features[RAD_MAX_HEAD_IN];
    float best = 0.0f;
    float weight_scale = 1.0f;
    float feature_scale = 1.0f;
    int hin;
    if (model == NULL || image == NULL || image->data == NULL || probability == NULL ||
        cell_y < 0 || cell_x < 0 || cell_y >= model->map_height ||
        cell_x >= model->map_width || image->channels != model->spec.channels ||
        image->height != model->spec.height || image->width != model->spec.width) {
        return KSHIRA_ERR_ARGUMENT;
    }
    hin = model->head_in;
    rad_forward_target(model, image, cell_y, cell_x);
    rad_head_features(model, cell_y, cell_x, features);
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        weight_scale = quant_scale_values(model->head_weights,
                                          (size_t)model->outputs * (size_t)hin,
                                          model->bits);
        feature_scale = quant_scale_values(features, (size_t)hin, model->bits);
    }
    for (int k = 0; k < model->spec.classes; ++k) {
        float logit = quality_logit_at(model, features, k, weight_scale, feature_scale);
        float q;
        if (!isfinite(logit)) return KSHIRA_ERR_RANGE;
        q = sigmoid(logit);
        if (q > best) best = q;
    }
    *probability = best;
    return isfinite(*probability) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

kshira_status kshira_rad_train_step(kshira_rad_model *model, const kshira_image_f32 *image,
                                     const kshira_rad_box *target,
                                     const kshira_rad_train_config *config, float *loss) {
    int target_x;
    int target_y;
    int c;
    size_t image_count;
    float primary_loss = 0.0f;
    float neighbor_loss_sum = 0.0f;
    int neighbor_count = 0;
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
    /* Reject pathological learning rates before any weight mutation (tests and
     * safety). Matches historical session contract for FLT_MAX. */
    if (config->learning_rate > 10.0f) return KSHIRA_ERR_RANGE;
    c = model->spec.feature_channels;
    if (model->encoder_deltas == NULL) return KSHIRA_ERR_MEMORY;
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
    if (model->bits != config->bits) clear_calibration(model);
    model->bits = config->bits;
    target_x = (int)(((target->x1 + target->x2) * 0.5f) / (float)RAD_STRIDE);
    target_y = (int)(((target->y1 + target->y2) * 0.5f) / (float)RAD_STRIDE);
    if (target_x < 0) target_x = 0;
    if (target_y < 0) target_y = 0;
    if (target_x >= model->map_width) target_x = model->map_width - 1;
    if (target_y >= model->map_height) target_y = model->map_height - 1;
    /* Centre full encoder; budgeted head-only neighbours (dense train-only). */
    if (rad_train_positive_at_cell(model, image, target, target_x, target_y, config, 1,
                                   1.0f, 0, &primary_loss) != KSHIRA_OK) {
        return KSHIRA_ERR_RANGE;
    }
    if (config->update_mode != KSHIRA_UPDATE_FREEZE) {
        float box_w_cells = (target->x2 - target->x1) / (float)RAD_STRIDE;
        float box_h_cells = (target->y2 - target->y1) / (float)RAD_STRIDE;
        int radius_x = (int)(box_w_cells * 0.25f + 0.5f);
        int radius_y = (int)(box_h_cells * 0.25f + 0.5f);
        float map_cx = ((target->x1 + target->x2) * 0.5f) / (float)RAD_STRIDE;
        float map_cy = ((target->y1 + target->y2) * 0.5f) / (float)RAD_STRIDE;
        float r2;
        int aux_used = 0;
        if (radius_x < 1) radius_x = 1;
        if (radius_y < 1) radius_y = 1;
        if (radius_x > 3) radius_x = 3;
        if (radius_y > 3) radius_y = 3;
        r2 = (float)(radius_x * radius_x + radius_y * radius_y) + 1.0e-6f;
        for (int dy = -radius_y; dy <= radius_y; ++dy) {
            for (int dx = -radius_x; dx <= radius_x; ++dx) {
                int nx;
                int ny;
                float cell_cx;
                float cell_cy;
                float nloss = 0.0f;
                float dist2;
                float p_center;
                if (dx == 0 && dy == 0) continue;
                if (aux_used >= 5) break;
                nx = target_x + dx;
                ny = target_y + dy;
                if (nx < 0 || ny < 0 || nx >= model->map_width ||
                    ny >= model->map_height) continue;
                cell_cx = ((float)nx + 0.5f) * (float)RAD_STRIDE;
                cell_cy = ((float)ny + 0.5f) * (float)RAD_STRIDE;
                if (cell_cx < target->x1 || cell_cx > target->x2 ||
                    cell_cy < target->y1 || cell_cy > target->y2) continue;
                dist2 = ((float)nx - map_cx) * ((float)nx - map_cx) +
                        ((float)ny - map_cy) * ((float)ny - map_cy);
                p_center = 1.0f - dist2 / r2;
                if (p_center < 0.15f) p_center = 0.15f;
                if (rad_train_positive_at_cell(model, image, target, nx, ny, config, 0,
                                               0.5f * p_center, 1, &nloss) != KSHIRA_OK) {
                    return KSHIRA_ERR_RANGE;
                }
                neighbor_loss_sum += nloss;
                ++neighbor_count;
                ++aux_used;
            }
            if (aux_used >= 5) break;
        }
        if (model->stream_samples < SIZE_MAX) ++model->stream_samples;
    }
    if (config->update_mode != KSHIRA_UPDATE_FREEZE) clear_calibration(model);
    if (neighbor_count > 0) {
        *loss = 0.7f * primary_loss + 0.3f * (neighbor_loss_sum / (float)neighbor_count);
    } else {
        *loss = primary_loss;
    }
    return isfinite(*loss) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

kshira_status kshira_rad_train_background_step(
    kshira_rad_model *model, const kshira_image_f32 *image, int cell_y, int cell_x,
    const kshira_rad_train_config *config, float *loss) {
    /* Surgical FP background (ranking path):
     * 1) Score-gate: skip weight updates on easy BG (max_p < 0.05) — free
     *    compute so more epochs fit the same wall-clock. Init quality bias
     *    σ(−2)≈0.12 still trains; only near-zero scores are free skips.
     * 2) Only tweak the winning class head row (argmax quality) — the weights
     *    that produced the FP score. Other class rows stay untouched.
     * 3) Mid-band boost: if max_p ∈ [0.28, 0.55] (the dead band of FPs that
     *    survive thr≈0.1 but not thr≈0.2), amplify LR so those scores drop
     *    without hammering true positives elsewhere.
     * FULL still updates a 13×13 encoder tile, but only via the winning class
     * gradient (two-level HNM Level B, surgically sparse). */
    float features[RAD_MAX_HEAD_IN];
    float quality_grad[RAD_MAX_CLASSES];
    float fused_gradient[RAD_MAX_FEATURES] = {0.0f};
    float weight_scale = 1.0f;
    float feature_scale = 1.0f;
    float loss_sum = 0.0f;
    float max_p = 0.0f;
    float step_lr;
    size_t image_count;
    int c;
    int hin;
    int K;
    int k_star = 0;
    if (model == NULL || image == NULL || image->data == NULL || config == NULL ||
        loss == NULL || cell_y < 0 || cell_y >= model->map_height || cell_x < 0 ||
        cell_x >= model->map_width ||
        (config->bits != KSHIRA_BITS_FLOAT &&
         !kshira_bit_mode_valid(config->bits)) ||
        config->update_mode < KSHIRA_UPDATE_FREEZE ||
        config->update_mode > KSHIRA_UPDATE_FULL ||
        !isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        image->channels != model->spec.channels ||
        image->height != model->spec.height || image->width != model->spec.width) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (config->learning_rate > 10.0f) return KSHIRA_ERR_RANGE;
    c = model->spec.feature_channels;
    hin = model->head_in;
    K = model->spec.classes;
    if (K < 1 || K > RAD_MAX_CLASSES) return KSHIRA_ERR_RANGE;
    if (config->channel_mask != NULL &&
        config->channel_mask->channel_count != (size_t)c) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if ((size_t)image->channels > SIZE_MAX / (size_t)image->height ||
        (size_t)image->channels * (size_t)image->height >
            SIZE_MAX / (size_t)image->width) return KSHIRA_ERR_RANGE;
    image_count = (size_t)image->channels * (size_t)image->height *
                  (size_t)image->width;
    for (size_t i = 0U; i < image_count; ++i) {
        if (!isfinite(image->data[i])) return KSHIRA_ERR_ARGUMENT;
    }
    if (model->bits != config->bits) clear_calibration(model);
    model->bits = config->bits;
    rad_forward_target(model, image, cell_y, cell_x);
    rad_head_features(model, cell_y, cell_x, features);
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        weight_scale = quant_scale_values(model->head_weights,
                                          (size_t)model->outputs * (size_t)hin,
                                          model->bits);
        feature_scale = quant_scale_values(features, (size_t)hin, model->bits);
    }
    for (int k = 0; k < K; ++k) {
        float logit = quality_logit_at(model, features, k, weight_scale, feature_scale);
        float p;
        float grad;
        float loss_term;
        if (!isfinite(logit)) return KSHIRA_ERR_RANGE;
        p = sigmoid(logit);
        if (!isfinite(p)) return KSHIRA_ERR_RANGE;
        if (p > max_p) {
            max_p = p;
            k_star = k;
        }
        quality_vfl_grad(logit, 0.0f, 0.0f, 3.0f, &grad, &loss_term);
        quality_grad[k] = grad;
        loss_sum += loss_term;
    }
    /* Easy BG: report loss but do not mutate weights (speed for more epochs). */
    if (config->update_mode != KSHIRA_UPDATE_FREEZE && max_p >= 0.05f) {
        step_lr = config->learning_rate;
        /* Mild mid-band focus (1.35×). Stronger boosts (1.75×) crushed TP by
         * sharing the winning-class head row with true objects. */
        if (max_p >= 0.30f && max_p <= 0.52f) {
            step_lr *= 1.35f;
        }
        if (step_lr > 10.0f) step_lr = 10.0f;
        {
            /* Only the winning class head row — weights that caused the FP. */
            int k = k_star;
            float gradient_bias = quality_grad[k];
            size_t row = (size_t)(4 + k) * (size_t)hin;
            if (kshira_apply_qas(NULL, 0U, &gradient_bias, 1U,
                                 weight_scale, feature_scale) != KSHIRA_OK ||
                !normalize_qas_gradient(&gradient_bias,
                                        model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            model->head_bias[4 + k] -= step_lr * gradient_bias;
            if (!isfinite(model->head_bias[4 + k])) return KSHIRA_ERR_RANGE;
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS ||
                config->update_mode == KSHIRA_UPDATE_FULL) {
                for (int ic = 0; ic < hin; ++ic) {
                    float gradient_weight = quality_grad[k] * features[ic];
                    if (config->update_mode == KSHIRA_UPDATE_CHANNELS &&
                        ic < c && config->channel_mask != NULL &&
                        !kshira_sparse_mask_get(config->channel_mask, (size_t)ic)) {
                        continue;
                    }
                    if (kshira_apply_qas(&gradient_weight, 1U, NULL, 0U,
                                         weight_scale, feature_scale) != KSHIRA_OK ||
                        !normalize_qas_gradient(&gradient_weight,
                                                model->bits != KSHIRA_BITS_FLOAT)) {
                        return KSHIRA_ERR_RANGE;
                    }
                    model->head_weights[row + (size_t)ic] -=
                        step_lr * gradient_weight;
                    if (!isfinite(model->head_weights[row + (size_t)ic])) {
                        return KSHIRA_ERR_RANGE;
                    }
                }
            }
        }
        /* Level-B encoder: FULL mode, gradient only from winning class. */
        if (config->update_mode == KSHIRA_UPDATE_FULL && model->encoder_deltas != NULL) {
            float g_contrast = 0.0f;
            size_t row = (size_t)(4 + k_star) * (size_t)hin;
            for (int ic = 0; ic < c; ++ic) {
                fused_gradient[ic] =
                    quality_grad[k_star] * model->head_weights[row + (size_t)ic];
            }
            g_contrast = quality_grad[k_star] *
                         model->head_weights[row + (size_t)c];
            if (isfinite(g_contrast)) {
                rad_contrast_grad_center(model, cell_y, cell_x, g_contrast,
                                         fused_gradient);
            }
            if (rad_update_encoder(model, image, cell_x, cell_y, fused_gradient,
                                   config->channel_mask, step_lr,
                                   model->encoder_deltas, 0) != KSHIRA_OK ||
                rad_update_encoder(model, image, cell_x, cell_y, fused_gradient,
                                   config->channel_mask, step_lr,
                                   model->encoder_deltas, 1) != KSHIRA_OK) {
                return KSHIRA_ERR_RANGE;
            }
        }
        clear_calibration(model);
    }
    *loss = loss_sum / (float)K + 2.0f * max_p * max_p * max_p;
    return isfinite(*loss) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

kshira_status kshira_rad_train_multiscale_step(
    kshira_rad_model *model, const kshira_image_f32 *image,
    const kshira_rad_box *target, int level,
    const kshira_rad_train_config *config, float *loss) {
    float features[RAD_MAX_HEAD_IN];
    float output[4 + RAD_MAX_CLASSES] = {0.0f};
    float gradients[4 + RAD_MAX_CLASSES];
    float bias_gradients[4 + RAD_MAX_CLASSES];
    int target_x;
    int target_y;
    int span;
    int cell_x;
    int cell_y;
    int source_x;
    int source_y;
    int source_x1;
    int source_y1;
    int region_x0;
    int region_y0;
    int region_x1;
    int region_y1;
    int c;
    size_t image_count;
    float weight_scale;
    float feature_scale;
    float branch_weight_scales[RAD_BRANCHES] = {0.0f};
    float project_weight_scale = 0.0f;
    float stem_input_scale = 0.0f;
    /* Fixed-size quantized caches avoid requantizing the same weights for each
     * pooled cell; their bounds follow valid_spec's feature-channel ceiling. */
    int8_t branch_quantized[32 * 9] = {0};
    int8_t project_quantized[32 * 32] = {0};
    float loss_sum = 0.0f;
    float *head_weights;
    float *head_bias;
    int seed_scale_head;
    if (model == NULL || image == NULL || image->data == NULL || target == NULL ||
        config == NULL || loss == NULL || !model->scale_heads_ready ||
        level < 1 || level >= RAD_SCALES ||
        (config->bits != KSHIRA_BITS_FLOAT && !kshira_bit_mode_valid(config->bits)) ||
        config->update_mode < KSHIRA_UPDATE_FREEZE ||
        config->update_mode > KSHIRA_UPDATE_CHANNELS ||
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
    if (config->channel_mask != NULL &&
        config->channel_mask->channel_count != (size_t)model->spec.feature_channels) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if ((size_t)image->channels > SIZE_MAX / (size_t)image->height ||
        (size_t)image->channels * (size_t)image->height > SIZE_MAX / (size_t)image->width) {
        return KSHIRA_ERR_RANGE;
    }
    image_count = (size_t)image->channels * (size_t)image->height *
                  (size_t)image->width;
    for (size_t i = 0U; i < image_count; ++i) {
        if (!isfinite(image->data[i])) return KSHIRA_ERR_ARGUMENT;
    }
    if (model->bits != config->bits) clear_calibration(model);
    model->bits = config->bits;
    c = model->spec.feature_channels;
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            branch_weight_scales[branch] = quant_scale_values(
                model->branch_weights[branch], (size_t)c * 9U, model->bits);
        }
        project_weight_scale = quant_scale_values(
            model->project_weights, (size_t)c * (size_t)c, model->bits);
        for (int i = 0; i < c * c; ++i) {
            project_quantized[i] = (int8_t)kshira_quantize_symmetric(
                model->project_weights[i], project_weight_scale, model->bits);
        }
    }
    target_x = (int)(((target->x1 + target->x2) * 0.5f) / (float)RAD_STRIDE);
    target_y = (int)(((target->y1 + target->y2) * 0.5f) / (float)RAD_STRIDE);
    if (target_x < 0) target_x = 0;
    if (target_y < 0) target_y = 0;
    if (target_x >= model->map_width) target_x = model->map_width - 1;
    if (target_y >= model->map_height) target_y = model->map_height - 1;
    span = 1 << level;
    cell_x = target_x / span;
    cell_y = target_y / span;
    source_x = cell_x * span;
    source_y = cell_y * span;
    source_x1 = source_x + span;
    source_y1 = source_y + span;
    if (source_x1 > model->map_width) source_x1 = model->map_width;
    if (source_y1 > model->map_height) source_y1 = model->map_height;
    region_x0 = source_x - 4;
    region_y0 = source_y - 4;
    region_x1 = source_x1 + 4;
    region_y1 = source_y1 + 4;
    if (region_x0 < 0) region_x0 = 0;
    if (region_y0 < 0) region_y0 = 0;
    if (region_x1 > model->map_width) region_x1 = model->map_width;
    if (region_y1 > model->map_height) region_y1 = model->map_height;
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        conv_stem_region_quant(model, image, region_y0, region_y1, region_x0, region_x1);
        stem_input_scale = model->calibration_samples > 0U ?
                           model->calibration_stem_scale : model->transient_stem_scale;
    } else {
        conv_stem_region(model, image, region_y0, region_y1, region_x0, region_x1);
    }
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            for (int i = 0; i < c * 9; ++i) {
                branch_quantized[i] = (int8_t)kshira_quantize_symmetric(
                    model->branch_weights[branch][i], branch_weight_scales[branch],
                    model->bits);
            }
            for (int y = source_y; y < source_y1; ++y) {
                for (int x = source_x; x < source_x1; ++x) {
                    depthwise_branch_target_quant_scaled(model, branch, y, x,
                                                         branch_weight_scales[branch],
                                                         stem_input_scale,
                                                         branch_quantized);
                }
            }
        }
    } else {
        for (int y = source_y; y < source_y1; ++y) {
            for (int x = source_x; x < source_x1; ++x) {
                for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
                    depthwise_branch_target(model, branch, y, x);
                }
            }
        }
    }
    for (int y = source_y; y < source_y1; ++y) {
        for (int x = source_x; x < source_x1; ++x) {
            if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
                project_fused_target_quant_scaled(model, y, x, project_weight_scale,
                                                  project_quantized);
            } else {
                project_fused_target(model, y, x);
            }
        }
    }
    pooled_features(model, level, cell_y, cell_x, features);
    seed_scale_head = (model->scale_head_trained_mask & (1 << level)) == 0;
    head_weights = seed_scale_head ? model->head_weights : model->scale_head_weights[level];
    head_bias = seed_scale_head ? model->head_bias : model->scale_head_bias[level];
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        head_forward_quant(model, head_weights, head_bias, features, output);
        weight_scale = quant_scale_values(head_weights,
                                          (size_t)model->outputs * (size_t)model->head_in,
                                          model->bits);
        feature_scale = quant_scale_values(features, (size_t)model->head_in, model->bits);
    } else {
        head_forward_f32(model, head_weights, head_bias, features, output);
        weight_scale = 1.0f;
        feature_scale = 1.0f;
    }
    for (int o = 0; o < model->outputs; ++o) {
        float center_x = ((float)source_x + (float)source_x1) *
                         0.5f * (float)RAD_STRIDE;
        float center_y = ((float)source_y + (float)source_y1) *
                         0.5f * (float)RAD_STRIDE;
        float stride = (float)(RAD_STRIDE * span);
        if (o < 4) {
            float values[4] = {
                fmaxf(0.0f, (center_x - target->x1) / stride),
                fmaxf(0.0f, (center_y - target->y1) / stride),
                fmaxf(0.0f, (target->x2 - center_x) / stride),
                fmaxf(0.0f, (target->y2 - center_y) / stride)
            };
            gradients[o] = output[o] - values[o];
            if (!isfinite(gradients[o])) return KSHIRA_ERR_RANGE;
            loss_sum += 0.5f * gradients[o] * gradients[o];
        } else {
            int k = o - 4;
            float q = (k == target->class_id) ? 0.85f : 0.0f;
            float grad = 0.0f;
            float loss_term = 0.0f;
            quality_vfl_grad(output[o], q, 2.0f, 2.0f, &grad, &loss_term);
            gradients[o] = grad;
            if (!isfinite(gradients[o])) return KSHIRA_ERR_RANGE;
            loss_sum += loss_term;
        }
        bias_gradients[o] = gradients[o];
    }
    if (!isfinite(loss_sum)) return KSHIRA_ERR_RANGE;
    if (config->update_mode != KSHIRA_UPDATE_FREEZE) {
        rad_head_delta_buffer *deltas = model->scale_head_deltas;
        if (deltas == NULL) return KSHIRA_ERR_MEMORY;
        for (int o = 0; o < model->outputs; ++o) {
            float gradient_bias = bias_gradients[o];
            float delta;
            float updated;
            if (kshira_apply_qas(NULL, 0U, &gradient_bias, 1U,
                                 weight_scale, feature_scale) != KSHIRA_OK ||
                !normalize_qas_gradient(&gradient_bias, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            delta = config->learning_rate * gradient_bias;
            updated = head_bias[o] - delta;
            if (!isfinite(delta) || !isfinite(updated)) return KSHIRA_ERR_RANGE;
            deltas->bias[o] = delta;
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS) {
                int hin = model->head_in;
                for (int ic = 0; ic < hin; ++ic) {
                    float gradient_weight;
                    if (ic < c && config->channel_mask != NULL &&
                        !kshira_sparse_mask_get(config->channel_mask, (size_t)ic)) {
                        continue;
                    }
                    gradient_weight = gradients[o] * features[ic];
                    if (!isfinite(gradient_weight) ||
                        kshira_apply_qas(&gradient_weight, 1U, NULL, 0U,
                                         weight_scale, feature_scale) != KSHIRA_OK ||
                        !normalize_qas_gradient(&gradient_weight,
                                                model->bits != KSHIRA_BITS_FLOAT)) {
                        return KSHIRA_ERR_RANGE;
                    }
                    delta = config->learning_rate * gradient_weight;
                    updated = head_weights[(size_t)o * (size_t)hin + (size_t)ic] - delta;
                    if (!isfinite(delta) || !isfinite(updated)) return KSHIRA_ERR_RANGE;
                    deltas->weights[(size_t)o * (size_t)hin + (size_t)ic] = delta;
                }
            }
        }
        if (seed_scale_head) {
            size_t head_count = (size_t)model->outputs * (size_t)model->head_in;
            for (size_t i = 0U; i < head_count; ++i) {
                model->scale_head_weights[level][i] = model->head_weights[i];
            }
            for (int o = 0; o < model->outputs; ++o) {
                model->scale_head_bias[level][o] = model->head_bias[o];
            }
            head_weights = model->scale_head_weights[level];
            head_bias = model->scale_head_bias[level];
        }
        for (int o = 0; o < model->outputs; ++o) {
            int hin = model->head_in;
            head_bias[o] -= deltas->bias[o];
            if (config->update_mode == KSHIRA_UPDATE_CHANNELS) {
                for (int ic = 0; ic < hin; ++ic) {
                    if (ic < c && config->channel_mask != NULL &&
                        !kshira_sparse_mask_get(config->channel_mask, (size_t)ic)) {
                        continue;
                    }
                    head_weights[(size_t)o * (size_t)hin + (size_t)ic] -=
                        deltas->weights[(size_t)o * (size_t)hin + (size_t)ic];
                }
            }
        }
        model->scale_head_trained_mask |= 1 << level;
    }
    *loss = loss_sum / (float)model->outputs;
    return isfinite(*loss) ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

kshira_status kshira_rad_train_rank_pair(
    kshira_rad_model *model, const kshira_image_f32 *image,
    int pos_y, int pos_x, int class_id, int neg_y, int neg_x, float margin,
    const kshira_rad_train_config *config, float *loss) {
    /* Head-only hinge: L = max(0, m + s_n − s_p) on class-quality scores.
     * ∂L/∂z_p = −s_p(1−s_p), ∂L/∂z_n = +s_n(1−s_n) when active.
     * INT8/INT4: keep separate pos/neg feature scales so QAS never applies
     * the negative cell scale to positive weight updates. */
    float pos_feat[RAD_MAX_HEAD_IN];
    float neg_feat[RAD_MAX_HEAD_IN];
    float weight_scale = 1.0f;
    float pos_feature_scale = 1.0f;
    float neg_feature_scale = 1.0f;
    float s_pos;
    float s_neg;
    float hinge;
    int hin;
    int K;
    if (model == NULL || image == NULL || image->data == NULL || config == NULL ||
        loss == NULL || class_id < 0 || class_id >= model->spec.classes ||
        pos_y < 0 || pos_x < 0 || neg_y < 0 || neg_x < 0 ||
        pos_y >= model->map_height || pos_x >= model->map_width ||
        neg_y >= model->map_height || neg_x >= model->map_width ||
        !isfinite(margin) || margin < 0.0f ||
        !isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        config->learning_rate > 10.0f ||
        config->update_mode == KSHIRA_UPDATE_FREEZE) {
        return KSHIRA_ERR_ARGUMENT;
    }
    hin = model->head_in;
    K = model->spec.classes;
    rad_forward_target(model, image, pos_y, pos_x);
    rad_head_features(model, pos_y, pos_x, pos_feat);
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        weight_scale = quant_scale_values(model->head_weights,
                                          (size_t)model->outputs * (size_t)hin,
                                          model->bits);
        pos_feature_scale = quant_scale_values(pos_feat, (size_t)hin, model->bits);
    }
    {
        float logit_p = quality_logit_at(model, pos_feat, class_id, weight_scale,
                                         pos_feature_scale);
        if (!isfinite(logit_p)) return KSHIRA_ERR_RANGE;
        s_pos = sigmoid(logit_p);
    }
    rad_forward_target(model, image, neg_y, neg_x);
    rad_head_features(model, neg_y, neg_x, neg_feat);
    if (model->bits == KSHIRA_BITS_INT4 || model->bits == KSHIRA_BITS_INT8) {
        neg_feature_scale = quant_scale_values(neg_feat, (size_t)hin, model->bits);
    }
    s_neg = 0.0f;
    for (int k = 0; k < K; ++k) {
        float logit_n = quality_logit_at(model, neg_feat, k, weight_scale,
                                         neg_feature_scale);
        float sn;
        if (!isfinite(logit_n)) return KSHIRA_ERR_RANGE;
        sn = sigmoid(logit_n);
        if (sn > s_neg) s_neg = sn;
    }
    hinge = margin + s_neg - s_pos;
    if (hinge <= 0.0f) {
        *loss = 0.0f;
        return KSHIRA_OK;
    }
    /* Positive: pull assigned class quality up (grad on logit = −s(1−s)).
     * Asymmetric vs neg: lift TP a bit more so ranking opens a gap.
     * QAS uses pos_feature_scale only — never the neg-cell scale. */
    {
        float g = -1.75f * (s_pos * (1.0f - s_pos));
        size_t row = (size_t)(4 + class_id) * (size_t)hin;
        float gb = g;
        float pos_lr = config->learning_rate;
        if (s_pos >= 0.20f && s_pos <= 0.50f) pos_lr *= 1.20f;
        if (kshira_apply_qas(NULL, 0U, &gb, 1U, weight_scale, pos_feature_scale) !=
                KSHIRA_OK ||
            !normalize_qas_gradient(&gb, model->bits != KSHIRA_BITS_FLOAT)) {
            return KSHIRA_ERR_RANGE;
        }
        model->head_bias[4 + class_id] -= pos_lr * gb;
        for (int ic = 0; ic < hin; ++ic) {
            float gw = g * pos_feat[ic];
            if (kshira_apply_qas(&gw, 1U, NULL, 0U, weight_scale, pos_feature_scale) !=
                    KSHIRA_OK ||
                !normalize_qas_gradient(&gw, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            model->head_weights[row + (size_t)ic] -= pos_lr * gw;
        }
    }
    /* Surgical negative: only the winning class on the FP cell — the head row
     * that produced s_neg. Other class rows stay frozen (sparse FP tweak). */
    {
        int k_neg = 0;
        float sn_best = 0.0f;
        float g;
        size_t row;
        float gb;
        float neg_lr = config->learning_rate;
        for (int k = 0; k < K; ++k) {
            float logit_n = quality_logit_at(model, neg_feat, k, weight_scale,
                                             neg_feature_scale);
            float sn = sigmoid(logit_n);
            if (sn > sn_best) {
                sn_best = sn;
                k_neg = k;
            }
        }
        g = 1.75f * (sn_best * (1.0f - sn_best));
        if (sn_best >= 0.30f && sn_best <= 0.52f) neg_lr *= 1.30f;
        row = (size_t)(4 + k_neg) * (size_t)hin;
        gb = g;
        if (kshira_apply_qas(NULL, 0U, &gb, 1U, weight_scale, neg_feature_scale) !=
                KSHIRA_OK ||
            !normalize_qas_gradient(&gb, model->bits != KSHIRA_BITS_FLOAT)) {
            return KSHIRA_ERR_RANGE;
        }
        model->head_bias[4 + k_neg] -= neg_lr * gb;
        for (int ic = 0; ic < hin; ++ic) {
            float gw = g * neg_feat[ic];
            if (kshira_apply_qas(&gw, 1U, NULL, 0U, weight_scale, neg_feature_scale) !=
                    KSHIRA_OK ||
                !normalize_qas_gradient(&gw, model->bits != KSHIRA_BITS_FLOAT)) {
                return KSHIRA_ERR_RANGE;
            }
            model->head_weights[row + (size_t)ic] -= neg_lr * gw;
        }
    }
    *loss = hinge;
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
