/* Trainable, on-demand P3/P4/P5 feature construction for KSHIRA. */
#include "kshira_rad_internal.h"

#include <math.h>

static size_t pyramid_index(int channels, int height, int width, int channel,
                            int y, int x) {
    (void)channels;
    return (((size_t)channel * (size_t)height + (size_t)y) *
            (size_t)width) + (size_t)x;
}

static const float *pyramid_projection_weights(const kshira_rad_model *model,
                                               int level) {
    size_t stride = (size_t)model->spec.feature_channels *
                    (size_t)model->spec.feature_channels;
    return model->pyramid_projection_weights + (size_t)level * stride;
}

static float *pyramid_projection_weights_mut(kshira_rad_model *model, int level) {
    size_t stride = (size_t)model->spec.feature_channels *
                    (size_t)model->spec.feature_channels;
    return model->pyramid_projection_weights + (size_t)level * stride;
}

static const float *pyramid_projection_bias(const kshira_rad_model *model,
                                            int level) {
    return model->pyramid_projection_bias +
           (size_t)level * (size_t)model->spec.feature_channels;
}

static float *pyramid_projection_bias_mut(kshira_rad_model *model, int level) {
    return model->pyramid_projection_bias +
           (size_t)level * (size_t)model->spec.feature_channels;
}

static void pyramid_topdown_features(const kshira_rad_model *model,
                                     const float *p5, float *topdown);

static void pyramid_raw_features(const kshira_rad_model *model, int level, int y,
                                 int x, float *raw) {
    int span = 1 << level;
    int y0 = y * span;
    int x0 = x * span;
    int y1 = y0 + span;
    int x1 = x0 + span;
    int channels = model->spec.feature_channels;
    if (y1 > model->map_height) y1 = model->map_height;
    if (x1 > model->map_width) x1 = model->map_width;
    for (int channel = 0; channel < channels; ++channel) {
        float sum = 0.0f;
        int count = 0;
        for (int iy = y0; iy < y1; ++iy) {
            for (int ix = x0; ix < x1; ++ix) {
                sum += model->fused[pyramid_index(
                    channels, model->map_height, model->map_width,
                    channel, iy, ix)];
                ++count;
            }
        }
        raw[channel] = count > 0 ? sum / (float)count : 0.0f;
    }
}

static void pyramid_raw_input_features(const kshira_rad_model *model, int level, int y,
                                       int x, float *raw) {
    int span = 1 << level;
    int y0 = y * span;
    int x0 = x * span;
    int y1 = y0 + span;
    int x1 = x0 + span;
    int count = 0;
    if (y1 > model->map_height) y1 = model->map_height;
    if (x1 > model->map_width) x1 = model->map_width;
    for (int channel = 0; channel < RAD_RAW_CHANNELS; ++channel) raw[channel] = 0.0f;
    for (int iy = y0; iy < y1; ++iy) {
        for (int ix = x0; ix < x1; ++ix) {
            for (int channel = 0; channel < RAD_RAW_CHANNELS; ++channel) {
                raw[channel] += model->raw_features[pyramid_index(
                    RAD_RAW_CHANNELS, model->map_height, model->map_width,
                    channel, iy, ix)];
            }
            ++count;
        }
    }
    if (count > 0) {
        float inverse = 1.0f / (float)count;
        for (int channel = 0; channel < RAD_RAW_CHANNELS; ++channel) raw[channel] *= inverse;
    }
}

static void pyramid_projected_features(const kshira_rad_model *model, int level,
                                       int y, int x, float *projected) {
    float raw[RAD_MAX_FEATURES] = {0.0f};
    int channels = model->spec.feature_channels;
    const float *weights = pyramid_projection_weights(model, level);
    const float *bias = pyramid_projection_bias(model, level);
    pyramid_raw_features(model, level, y, x, raw);
    for (int output = 0; output < channels; ++output) {
        float value = bias[output];
        for (int input = 0; input < channels; ++input) {
            value += weights[(size_t)output * (size_t)channels + (size_t)input] *
                     raw[input];
        }
        projected[output] = value;
    }
}

static void pyramid_level_features_local(const kshira_rad_model *model, int level, int y,
                                   int x, float *features) {
    float projected[RAD_MAX_FEATURES] = {0.0f};
    int channels = model->spec.feature_channels;
    int span = 1 << level;
    int scale_height = (model->map_height + span - 1) / span;
    int scale_width = (model->map_width + span - 1) / span;
    pyramid_projected_features(model, level, y, x, projected);
    for (int output = 0; output < channels; ++output) {
        float residual = model->pyramid_refine_bias[(size_t)level *
                                                     (size_t)channels +
                                                     (size_t)output];
        const float *refine = model->pyramid_refine_weights +
                              ((size_t)level * (size_t)channels +
                               (size_t)output) * 9U;
        int kernel = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx, ++kernel) {
                int ny = y + ky;
                int nx = x + kx;
                float neighbor[RAD_MAX_FEATURES] = {0.0f};
                if (ny < 0 || nx < 0 || ny >= scale_height || nx >= scale_width) {
                    continue;
                }
                pyramid_projected_features(model, level, ny, nx, neighbor);
                residual += refine[kernel] * neighbor[output];
            }
        }
        features[output] =
            (projected[output] + residual) *
            model->pyramid_feature_gain[level][output] +
            model->pyramid_feature_bias[level][output];
    }
}

static size_t pyramid_cache_index(const kshira_rad_model *model, int level,
                                  int channel, int y, int x) {
    int span = 1 << level;
    int width = (model->map_width + span - 1) / span;
    return (((size_t)channel * (size_t)((model->map_height + span - 1) / span) +
             (size_t)y) * (size_t)width) + (size_t)x;
}

void kshira_rad_pyramid_prepare(kshira_rad_model *model) {
    if (model == NULL || !model->spec.context_fusion || model->spec.p3_only_deployment ||
        model->pyramid_cache[1] == NULL || model->pyramid_cache[2] == NULL) return;
    model->pyramid_cache_valid = 0;
    for (int level = RAD_SCALES - 1; level >= 1; --level) {
        int span = 1 << level;
        int height = (model->map_height + span - 1) / span;
        int width = (model->map_width + span - 1) / span;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float values[RAD_MAX_FEATURES] = {0.0f};
                pyramid_level_features_local(model, level, y, x, values);
                if (level == 1) {
                    float p5[RAD_MAX_FEATURES] = {0.0f};
                    float topdown[RAD_MAX_FEATURES] = {0.0f};
                    for (int channel = 0; channel < model->spec.feature_channels; ++channel) {
                        p5[channel] = model->pyramid_cache[2][pyramid_cache_index(
                            model, 2, channel, y / 2, x / 2)];
                    }
                    pyramid_topdown_features(model, p5, topdown);
                    for (int channel = 0; channel < model->spec.feature_channels; ++channel) {
                        values[channel] += topdown[channel];
                    }
                }
                for (int channel = 0; channel < model->spec.feature_channels; ++channel) {
                    model->pyramid_cache[level][pyramid_cache_index(
                        model, level, channel, y, x)] = values[channel];
                }
            }
        }
    }
    model->pyramid_cache_valid = 1;
}

static void pyramid_topdown_features(const kshira_rad_model *model,
                                     const float *p5, float *topdown) {
    int channels = model->spec.feature_channels;
    for (int output = 0; output < channels; ++output) {
        float value = 0.0f;
        for (int input = 0; input < channels; ++input) {
            value += model->pyramid_topdown_weights[(size_t)output *
                                                    (size_t)channels +
                                                    (size_t)input] * p5[input];
        }
        topdown[output] = value;
    }
}

void kshira_rad_pyramid_features(const kshira_rad_model *model, int level, int y,
                                  int x, float *features) {
    float raw_input[RAD_RAW_CHANNELS] = {0.0f};
    if (model == NULL || features == NULL || level < 1 || level >= RAD_SCALES) {
        return;
    }
    for (int channel = model->spec.feature_channels; channel < RAD_MAX_HEAD_IN; ++channel) {
        features[channel] = 0.0f;
    }
    if (model->spec.raw_input_features) {
        pyramid_raw_input_features(model, level, y, x, raw_input);
        for (int channel = 0; channel < RAD_RAW_CHANNELS; ++channel) {
            features[model->spec.feature_channels + RAD_CONTRAST_CHANNELS + channel] =
                raw_input[channel];
        }
    }
    if (!model->spec.context_fusion) {
        pyramid_raw_features(model, level, y, x, features);
        return;
    }
    if (model->pyramid_cache_valid && model->pyramid_cache[level] != NULL) {
        for (int channel = 0; channel < model->spec.feature_channels; ++channel) {
            features[channel] = model->pyramid_cache[level][pyramid_cache_index(
                model, level, channel, y, x)];
        }
        return;
    }
    pyramid_level_features_local(model, level, y, x, features);
    if (model->spec.context_fusion && level == 1) {
        float p5[RAD_MAX_FEATURES] = {0.0f};
        float topdown[RAD_MAX_FEATURES] = {0.0f};
        int p5_y = y / 2;
        int p5_x = x / 2;
        pyramid_level_features_local(model, 2, p5_y, p5_x, p5);
        pyramid_topdown_features(model, p5, topdown);
        for (int channel = 0; channel < model->spec.feature_channels; ++channel) {
            features[channel] += topdown[channel];
        }
    }
}

void kshira_rad_pyramid_gradients(
    const kshira_rad_model *model, int level, int y, int x,
    const float *feature_gradient, const kshira_sparse_mask *channel_mask,
    float *topdown_gradient, float *gain_gradient, float *bias_gradient) {
    float raw_level[RAD_MAX_FEATURES] = {0.0f};
    float p5[RAD_MAX_FEATURES] = {0.0f};
    float topdown[RAD_MAX_FEATURES] = {0.0f};
    int channels;
    if (topdown_gradient != NULL) *topdown_gradient = 0.0f;
    if (model == NULL || feature_gradient == NULL || gain_gradient == NULL ||
        bias_gradient == NULL || level < 1 || level >= RAD_SCALES) return;
    channels = model->spec.feature_channels;
    if (model->pyramid_projection_weights == NULL ||
        model->pyramid_projection_bias == NULL ||
        model->pyramid_topdown_weights == NULL) return;
    pyramid_raw_features(model, level, y, x, raw_level);
    for (int channel = 0; channel < channels; ++channel) {
        gain_gradient[channel] = 0.0f;
        bias_gradient[channel] = 0.0f;
    }
    if (model->spec.context_fusion && level == 1) {
        pyramid_level_features_local(model, 2, y / 2, x / 2, p5);
        pyramid_topdown_features(model, p5, topdown);
    }
    for (int channel = 0; channel < channels; ++channel) {
        float gradient;
        if (channel_mask != NULL &&
            !kshira_sparse_mask_get(channel_mask, (size_t)channel)) {
            continue;
        }
        gradient = feature_gradient[channel];
        gain_gradient[channel] = gradient * raw_level[channel];
        bias_gradient[channel] = gradient;
        if (model->spec.context_fusion && level == 1) {
            if (topdown_gradient != NULL) {
                *topdown_gradient += gradient * topdown[channel];
            }
        }
    }
    if (topdown_gradient != NULL && !isfinite(*topdown_gradient)) {
        *topdown_gradient = 0.0f;
    }
}

void kshira_rad_pyramid_update(
    kshira_rad_model *model, int level, int y, int x,
    const float *feature_gradient, const kshira_sparse_mask *channel_mask,
    float learning_rate) {
    float raw_level[RAD_MAX_FEATURES] = {0.0f};
    float raw_p5[RAD_MAX_FEATURES] = {0.0f};
    float p5[RAD_MAX_FEATURES] = {0.0f};
    float projected_gradient[RAD_MAX_FEATURES] = {0.0f};
    int channels;
    if (model == NULL || feature_gradient == NULL || !model->spec.context_fusion ||
        level < 1 || level >= RAD_SCALES || !isfinite(learning_rate) ||
        learning_rate <= 0.0f) return;
    channels = model->spec.feature_channels;
    pyramid_raw_features(model, level, y, x, raw_level);
    if (level == 1) {
        pyramid_raw_features(model, 2, y / 2, x / 2, raw_p5);
        pyramid_level_features_local(model, 2, y / 2, x / 2, p5);
    }
    for (int output = 0; output < channels; ++output) {
        int enabled = channel_mask == NULL ||
                      kshira_sparse_mask_get(channel_mask, (size_t)output);
        if (!enabled) continue;
        projected_gradient[output] =
            feature_gradient[output] * model->pyramid_feature_gain[level][output];
        {
            float *refine = model->pyramid_refine_weights +
                            ((size_t)level * (size_t)channels +
                             (size_t)output) * 9U;
            int kernel = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx, ++kernel) {
                    int ny = y + ky;
                    int nx = x + kx;
                    float neighbor[RAD_MAX_FEATURES] = {0.0f};
                    if (ny < 0 || nx < 0 ||
                        ny >= (model->map_height + (1 << level) - 1) / (1 << level) ||
                        nx >= (model->map_width + (1 << level) - 1) / (1 << level)) {
                        continue;
                    }
                    pyramid_projected_features(model, level, ny, nx, neighbor);
                    refine[kernel] -= learning_rate * projected_gradient[output] *
                                      neighbor[output];
                }
            }
            model->pyramid_refine_bias[(size_t)level * (size_t)channels +
                                       (size_t)output] -=
                learning_rate * projected_gradient[output];
        }
        float *weights = pyramid_projection_weights_mut(model, level);
        float *bias = pyramid_projection_bias_mut(model, level);
        bias[output] -=
            learning_rate * projected_gradient[output];
        for (int input = 0; input < channels; ++input) {
            weights[(size_t)output * (size_t)channels + (size_t)input] -=
                learning_rate * projected_gradient[output] * raw_level[input];
        }
    }
    if (level == 1) {
        float p5_gradient[RAD_MAX_FEATURES] = {0.0f};
        for (int output = 0; output < channels; ++output) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)output);
            if (!enabled) continue;
            for (int input = 0; input < channels; ++input) {
                size_t index = (size_t)output * (size_t)channels + (size_t)input;
                float old_weight = model->pyramid_topdown_weights[index];
                model->pyramid_topdown_weights[index] -=
                    learning_rate * projected_gradient[output] * p5[input];
                p5_gradient[input] += projected_gradient[output] *
                                      old_weight;
            }
        }
        for (int output = 0; output < channels; ++output) {
            int enabled = channel_mask == NULL ||
                          kshira_sparse_mask_get(channel_mask, (size_t)output);
            if (!enabled) continue;
            p5_gradient[output] *= model->pyramid_feature_gain[2][output];
            {
                float *p5_weights = pyramid_projection_weights_mut(model, 2);
                float *p5_bias = pyramid_projection_bias_mut(model, 2);
                p5_bias[output] -=
                learning_rate * p5_gradient[output];
                for (int input = 0; input < channels; ++input) {
                    p5_weights[(size_t)output * (size_t)channels + (size_t)input] -=
                        learning_rate * p5_gradient[output] * raw_p5[input];
                }
            }
        }
    }
    model->pyramid_cache_valid = 0;
}
