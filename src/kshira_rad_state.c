/* Purpose: canonical pointer-free RAD state for the detector model format.
 * Ownership: callers own the byte buffer and live model arena.
 * Failure: import validates the complete payload before changing the model. */
#include "kshira_rad_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t classes;
    uint32_t feature_channels;
    uint32_t top_k;
    int32_t seed;
    uint32_t multiscale_heads;
    uint32_t stem_mode;
    uint32_t one_to_one_head;
    uint32_t shared_multiscale_head;
    uint32_t context_fusion;
    uint32_t raw_input_features;
    uint32_t p3_only_deployment;
    uint32_t smooth_box_decode;
    uint32_t bits;
    uint32_t scale_head_trained_mask;
    uint64_t calibration_samples;
    float calibration_input_scale;
    float calibration_stem_scale;
    float calibration_branch_scale[RAD_BRANCHES];
    float transient_image_scale;
    float transient_stem_scale;
    uint32_t transient_scales_valid;
    float scale_head_gain[RAD_SCALES];
    float scale_head_offset[RAD_SCALES];
    uint64_t parameter_floats;
} kshira_rad_state_header;

static int add_count(size_t *total, size_t count) {
    if (total == NULL || *total > SIZE_MAX - count) return 0;
    *total += count;
    return 1;
}

static int rad_parameter_float_count(const kshira_rad_model *model,
                                     size_t *out) {
    size_t channels;
    size_t outputs;
    size_t head_in;
    size_t total = 0U;
    size_t head_count;
    if (model == NULL || out == NULL) return 0;
    channels = (size_t)model->spec.feature_channels;
    outputs = (size_t)model->outputs;
    head_in = (size_t)model->head_in;
    if (head_in == 0U) head_in = channels + (size_t)RAD_CONTRAST_CHANNELS +
                                      (size_t)RAD_RAW_CHANNELS;
    head_count = outputs * head_in + outputs;
    if (!add_count(&total, channels * (size_t)model->stem_input_channels * 9U) ||
        !add_count(&total, channels) ||
        !add_count(&total, (size_t)RAD_BRANCHES * (channels * 9U + channels)) ||
        !add_count(&total, channels * channels + channels) ||
        !add_count(&total, head_count)) return 0;
    if (model->scale_heads_ready && !model->spec.shared_multiscale_head &&
        !add_count(&total, (size_t)(RAD_SCALES - 1) * head_count)) return 0;
    if (model->spec.one_to_one_head &&
        !add_count(&total, head_count)) return 0;
    if (!add_count(&total, RAD_SCALES) ||
        !add_count(&total, 2U * RAD_SCALES * channels)) return 0;
    if (model->spec.context_fusion &&
        (!add_count(&total, RAD_SCALES * (channels * channels + channels)) ||
         !add_count(&total, channels * channels) ||
         !add_count(&total, RAD_SCALES * (channels * 9U + channels)))) return 0;
    if (model->spec.stem_mode == 2 &&
        (!add_count(&total, channels * (size_t)model->spec.channels * 9U) ||
         !add_count(&total, channels))) return 0;
    *out = total;
    return 1;
}

static int finite_buffer(const float *values, size_t count) {
    if (values == NULL) return 0;
    for (size_t i = 0U; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static int rad_parameters_finite(const kshira_rad_model *model) {
    size_t channels = (size_t)model->spec.feature_channels;
    size_t outputs = (size_t)model->outputs;
    size_t head_in = (size_t)model->head_in;
    size_t head_weights;
    if (head_in == 0U) head_in = channels + (size_t)RAD_CONTRAST_CHANNELS +
                                      (size_t)RAD_RAW_CHANNELS;
    head_weights = outputs * head_in;
    if (!finite_buffer(model->stem_weights,
                       channels * (size_t)model->stem_input_channels * 9U) ||
        !finite_buffer(model->stem_bias, channels)) return 0;
    if (model->spec.stem_mode == 2 &&
        (!finite_buffer(model->stem_pre_weights,
                        channels * (size_t)model->spec.channels * 9U) ||
         !finite_buffer(model->stem_pre_bias, channels))) return 0;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        if (!finite_buffer(model->branch_weights[branch], channels * 9U) ||
            !finite_buffer(model->branch_bias[branch], channels)) return 0;
    }
    if (!finite_buffer(model->project_weights, channels * channels) ||
        !finite_buffer(model->project_bias, channels) ||
        !finite_buffer(model->head_weights, head_weights) ||
        !finite_buffer(model->head_bias, outputs)) return 0;
    if (model->spec.one_to_one_head &&
        (!finite_buffer(model->one_to_one_head_weights, head_weights) ||
         !finite_buffer(model->one_to_one_head_bias, outputs))) return 0;
    if (model->scale_heads_ready && !model->spec.shared_multiscale_head) {
        for (int level = 1; level < RAD_SCALES; ++level) {
            if (!finite_buffer(model->scale_head_weights[level], head_weights) ||
                !finite_buffer(model->scale_head_bias[level], outputs)) return 0;
        }
    }
    for (int level = 0; level < RAD_SCALES; ++level) {
        if (!isfinite(model->scale_head_gain[level]) ||
            model->scale_head_gain[level] <= 0.0f ||
            !isfinite(model->scale_head_offset[level]) ||
            !isfinite(model->pyramid_topdown_gain[level]) ||
            model->pyramid_topdown_gain[level] < 0.0f ||
            model->pyramid_topdown_gain[level] > 1.0f ||
            !finite_buffer(model->pyramid_feature_gain[level], channels) ||
            !finite_buffer(model->pyramid_feature_bias[level], channels)) return 0;
        if (model->spec.context_fusion &&
            (!finite_buffer(model->pyramid_projection_weights +
                                (size_t)level * channels * channels,
                            channels * channels) ||
             !finite_buffer(model->pyramid_projection_bias +
                                (size_t)level * channels, channels))) {
            return 0;
        }
    }
    if (model->spec.context_fusion &&
        (!finite_buffer(model->pyramid_topdown_weights, channels * channels) ||
         !finite_buffer(model->pyramid_refine_weights,
                        RAD_SCALES * channels * 9U) ||
         !finite_buffer(model->pyramid_refine_bias, RAD_SCALES * channels))) {
        return 0;
    }
    return 1;
}

static void export_floats(uint8_t **cursor, const float *values, size_t count) {
    size_t bytes = count * sizeof(float);
    memcpy(*cursor, values, bytes);
    *cursor += bytes;
}

static void import_floats(const uint8_t **cursor, float *values, size_t count) {
    size_t bytes = count * sizeof(float);
    memcpy(values, *cursor, bytes);
    *cursor += bytes;
}

static int encoded_floats_finite(const uint8_t *cursor, size_t count) {
    for (size_t i = 0U; i < count; ++i) {
        float value;
        memcpy(&value, cursor + i * sizeof(float), sizeof(value));
        if (!isfinite(value)) return 0;
    }
    return 1;
}

size_t kshira_rad_state_bytes(const kshira_rad_model *model) {
    size_t count;
    size_t payload_bytes;
    if (!rad_parameter_float_count(model, &count) ||
        count > (SIZE_MAX - sizeof(kshira_rad_state_header)) / sizeof(float)) return 0U;
    payload_bytes = count * sizeof(float);
    return sizeof(kshira_rad_state_header) + payload_bytes;
}

kshira_status kshira_rad_export_state(const kshira_rad_model *model, void *buffer,
                                       size_t capacity, size_t *written) {
    kshira_rad_state_header header;
    size_t count;
    size_t required;
    size_t channels;
    size_t outputs;
    uint8_t *cursor;
    if (written != NULL) *written = 0U;
    if (model == NULL || buffer == NULL || written == NULL ||
        !rad_parameter_float_count(model, &count)) return KSHIRA_ERR_ARGUMENT;
    required = kshira_rad_state_bytes(model);
    if (required == 0U) return KSHIRA_ERR_RANGE;
    if (capacity < required) return KSHIRA_ERR_MEMORY;
    if (!rad_parameters_finite(model) ||
        !isfinite(model->calibration_input_scale) ||
        !isfinite(model->calibration_stem_scale) ||
        model->calibration_input_scale <= 0.0f ||
        model->calibration_stem_scale <= 0.0f ||
        !isfinite(model->transient_image_scale) ||
        !isfinite(model->transient_stem_scale) ||
        model->transient_image_scale <= 0.0f ||
        model->transient_stem_scale <= 0.0f ||
        (model->transient_scales_valid != 0 &&
         model->transient_scales_valid != 1)) return KSHIRA_ERR_RANGE;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        if (!isfinite(model->calibration_branch_scale[branch]) ||
            model->calibration_branch_scale[branch] <= 0.0f) return KSHIRA_ERR_RANGE;
    }
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "KRAD", 4U);
    /* v15: smooth positive box-side parameterization. */
    header.version = 15U;
    header.width = (uint32_t)model->spec.width;
    header.height = (uint32_t)model->spec.height;
    header.channels = (uint32_t)model->spec.channels;
    header.classes = (uint32_t)model->spec.classes;
    header.feature_channels = (uint32_t)model->spec.feature_channels;
    header.top_k = (uint32_t)model->spec.top_k;
    header.seed = (int32_t)model->spec.seed;
    header.multiscale_heads = (uint32_t)model->spec.multiscale_heads;
    header.stem_mode = (uint32_t)model->spec.stem_mode;
    header.one_to_one_head = (uint32_t)model->spec.one_to_one_head;
    header.shared_multiscale_head = (uint32_t)model->spec.shared_multiscale_head;
    header.context_fusion = (uint32_t)model->spec.context_fusion;
    header.raw_input_features = (uint32_t)model->spec.raw_input_features;
    header.p3_only_deployment = (uint32_t)model->spec.p3_only_deployment;
    header.smooth_box_decode = (uint32_t)model->spec.smooth_box_decode;
    header.bits = (uint32_t)model->bits;
    header.scale_head_trained_mask = model->spec.shared_multiscale_head ?
                                     0U : (uint32_t)model->scale_head_trained_mask;
    header.calibration_samples = (uint64_t)model->calibration_samples;
    header.calibration_input_scale = model->calibration_input_scale;
    header.calibration_stem_scale = model->calibration_stem_scale;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        header.calibration_branch_scale[branch] =
            model->calibration_branch_scale[branch];
    }
    header.transient_image_scale = model->transient_image_scale;
    header.transient_stem_scale = model->transient_stem_scale;
    header.transient_scales_valid = (uint32_t)model->transient_scales_valid;
    for (int level = 0; level < RAD_SCALES; ++level) {
        header.scale_head_gain[level] = model->scale_head_gain[level];
        header.scale_head_offset[level] = model->scale_head_offset[level];
    }
    header.parameter_floats = (uint64_t)count;
    memcpy(buffer, &header, sizeof(header));
    cursor = (uint8_t *)buffer + sizeof(header);
    channels = (size_t)model->spec.feature_channels;
    outputs = (size_t)model->outputs;
    {
        size_t head_in = (size_t)model->head_in;
        if (head_in == 0U) head_in = channels + (size_t)RAD_CONTRAST_CHANNELS +
                                          (size_t)RAD_RAW_CHANNELS;
        export_floats(&cursor, model->stem_weights,
                      channels * (size_t)model->stem_input_channels * 9U);
        export_floats(&cursor, model->stem_bias, channels);
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            export_floats(&cursor, model->branch_weights[branch], channels * 9U);
            export_floats(&cursor, model->branch_bias[branch], channels);
        }
        export_floats(&cursor, model->project_weights, channels * channels);
        export_floats(&cursor, model->project_bias, channels);
        export_floats(&cursor, model->head_weights, outputs * head_in);
        export_floats(&cursor, model->head_bias, outputs);
        if (model->spec.one_to_one_head) {
            export_floats(&cursor, model->one_to_one_head_weights, outputs * head_in);
            export_floats(&cursor, model->one_to_one_head_bias, outputs);
        }
        if (model->scale_heads_ready && !model->spec.shared_multiscale_head) {
            for (int level = 1; level < RAD_SCALES; ++level) {
                export_floats(&cursor, model->scale_head_weights[level],
                              outputs * head_in);
                export_floats(&cursor, model->scale_head_bias[level], outputs);
            }
        }
        export_floats(&cursor, model->pyramid_topdown_gain, RAD_SCALES);
        for (int level = 0; level < RAD_SCALES; ++level) {
            export_floats(&cursor, model->pyramid_feature_gain[level], channels);
            export_floats(&cursor, model->pyramid_feature_bias[level], channels);
        }
        if (model->spec.context_fusion) {
            for (int level = 0; level < RAD_SCALES; ++level) {
                export_floats(&cursor, model->pyramid_projection_weights +
                              (size_t)level * channels * channels,
                              channels * channels);
                export_floats(&cursor, model->pyramid_projection_bias +
                              (size_t)level * channels, channels);
            }
            export_floats(&cursor, model->pyramid_topdown_weights,
                          channels * channels);
            export_floats(&cursor, model->pyramid_refine_weights,
                          RAD_SCALES * channels * 9U);
            export_floats(&cursor, model->pyramid_refine_bias,
                          RAD_SCALES * channels);
        }
        if (model->spec.stem_mode == 2) {
            export_floats(&cursor, model->stem_pre_weights,
                          channels * (size_t)model->spec.channels * 9U);
            export_floats(&cursor, model->stem_pre_bias, channels);
        }
    }
    *written = (size_t)(cursor - (uint8_t *)buffer);
    return *written == required ? KSHIRA_OK : KSHIRA_ERR_RANGE;
}

kshira_status kshira_rad_import_state(kshira_rad_model *model, const void *buffer,
                                       size_t bytes) {
    kshira_rad_state_header header;
    size_t count;
    size_t required;
    size_t channels;
    size_t outputs;
    const uint8_t *cursor;
    if (model == NULL || buffer == NULL || bytes < sizeof(header) ||
        !rad_parameter_float_count(model, &count)) return KSHIRA_ERR_ARGUMENT;
    memcpy(&header, buffer, sizeof(header));
    required = kshira_rad_state_bytes(model);
    if (required == 0U || bytes != required || memcmp(header.magic, "KRAD", 4U) != 0 ||
         header.version != 15U ||
        header.width != (uint32_t)model->spec.width ||
        header.height != (uint32_t)model->spec.height ||
        header.channels != (uint32_t)model->spec.channels ||
        header.classes != (uint32_t)model->spec.classes ||
        header.feature_channels != (uint32_t)model->spec.feature_channels ||
        header.top_k != (uint32_t)model->spec.top_k ||
        header.seed != (int32_t)model->spec.seed ||
        header.multiscale_heads != (uint32_t)model->spec.multiscale_heads ||
        header.stem_mode != (uint32_t)model->spec.stem_mode ||
        header.one_to_one_head != (uint32_t)model->spec.one_to_one_head ||
        header.shared_multiscale_head !=
            (uint32_t)model->spec.shared_multiscale_head ||
        header.context_fusion != (uint32_t)model->spec.context_fusion ||
        header.raw_input_features != (uint32_t)model->spec.raw_input_features ||
        header.p3_only_deployment != (uint32_t)model->spec.p3_only_deployment ||
        header.smooth_box_decode != (uint32_t)model->spec.smooth_box_decode ||
        header.parameter_floats != (uint64_t)count ||
        (header.bits != (uint32_t)KSHIRA_BITS_FLOAT &&
         header.bits != (uint32_t)KSHIRA_BITS_INT4 &&
         header.bits != (uint32_t)KSHIRA_BITS_INT8) ||
        (header.scale_head_trained_mask & ~((1U << RAD_SCALES) - 2U)) != 0U ||
        (model->spec.shared_multiscale_head && header.scale_head_trained_mask != 0U) ||
        (!model->scale_heads_ready && header.scale_head_trained_mask != 0U) ||
        header.calibration_samples > (uint64_t)SIZE_MAX ||
        !isfinite(header.calibration_input_scale) ||
        !isfinite(header.calibration_stem_scale) ||
        header.calibration_input_scale <= 0.0f ||
        header.calibration_stem_scale <= 0.0f ||
        !isfinite(header.transient_image_scale) ||
        !isfinite(header.transient_stem_scale) ||
        header.transient_image_scale <= 0.0f ||
        header.transient_stem_scale <= 0.0f ||
        header.transient_scales_valid > 1U) return KSHIRA_ERR_RANGE;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        if (!isfinite(header.calibration_branch_scale[branch]) ||
            header.calibration_branch_scale[branch] <= 0.0f) return KSHIRA_ERR_RANGE;
    }
    for (int level = 0; level < RAD_SCALES; ++level) {
        if (!isfinite(header.scale_head_gain[level]) ||
            header.scale_head_gain[level] <= 0.0f ||
            !isfinite(header.scale_head_offset[level])) return KSHIRA_ERR_RANGE;
    }
    cursor = (const uint8_t *)buffer + sizeof(header);
    if (!encoded_floats_finite(cursor, count)) return KSHIRA_ERR_RANGE;
    channels = (size_t)model->spec.feature_channels;
    outputs = (size_t)model->outputs;
    {
        size_t head_in = (size_t)model->head_in;
        if (head_in == 0U) head_in = channels + (size_t)RAD_CONTRAST_CHANNELS;
        import_floats(&cursor, model->stem_weights,
                      channels * (size_t)model->stem_input_channels * 9U);
        import_floats(&cursor, model->stem_bias, channels);
        for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
            import_floats(&cursor, model->branch_weights[branch], channels * 9U);
            import_floats(&cursor, model->branch_bias[branch], channels);
        }
        import_floats(&cursor, model->project_weights, channels * channels);
        import_floats(&cursor, model->project_bias, channels);
        import_floats(&cursor, model->head_weights, outputs * head_in);
        import_floats(&cursor, model->head_bias, outputs);
        if (model->spec.one_to_one_head) {
            import_floats(&cursor, model->one_to_one_head_weights, outputs * head_in);
            import_floats(&cursor, model->one_to_one_head_bias, outputs);
        }
        if (model->scale_heads_ready && !model->spec.shared_multiscale_head) {
            for (int level = 1; level < RAD_SCALES; ++level) {
                import_floats(&cursor, model->scale_head_weights[level],
                              outputs * head_in);
                import_floats(&cursor, model->scale_head_bias[level], outputs);
            }
        }
        import_floats(&cursor, model->pyramid_topdown_gain, RAD_SCALES);
        for (int level = 0; level < RAD_SCALES; ++level) {
            import_floats(&cursor, model->pyramid_feature_gain[level], channels);
            import_floats(&cursor, model->pyramid_feature_bias[level], channels);
            if (!isfinite(model->pyramid_topdown_gain[level]) ||
                model->pyramid_topdown_gain[level] < 0.0f ||
                model->pyramid_topdown_gain[level] > 1.0f ||
                !finite_buffer(model->pyramid_feature_gain[level], channels) ||
                !finite_buffer(model->pyramid_feature_bias[level], channels)) {
                return KSHIRA_ERR_RANGE;
            }
        }
        if (model->spec.context_fusion) {
            for (int level = 0; level < RAD_SCALES; ++level) {
                import_floats(&cursor, model->pyramid_projection_weights +
                              (size_t)level * channels * channels,
                              channels * channels);
                import_floats(&cursor, model->pyramid_projection_bias +
                              (size_t)level * channels, channels);
                if (!finite_buffer(model->pyramid_projection_weights +
                                       (size_t)level * channels * channels,
                                   channels * channels) ||
                    !finite_buffer(model->pyramid_projection_bias +
                                       (size_t)level * channels, channels)) {
                    return KSHIRA_ERR_RANGE;
                }
            }
            import_floats(&cursor, model->pyramid_topdown_weights,
                          channels * channels);
            if (!finite_buffer(model->pyramid_topdown_weights,
                               channels * channels)) return KSHIRA_ERR_RANGE;
            import_floats(&cursor, model->pyramid_refine_weights,
                          RAD_SCALES * channels * 9U);
            import_floats(&cursor, model->pyramid_refine_bias,
                          RAD_SCALES * channels);
            if (!finite_buffer(model->pyramid_refine_weights,
                               RAD_SCALES * channels * 9U) ||
                !finite_buffer(model->pyramid_refine_bias,
                               RAD_SCALES * channels)) return KSHIRA_ERR_RANGE;
        }
        if (model->spec.stem_mode == 2) {
            import_floats(&cursor, model->stem_pre_weights,
                          channels * (size_t)model->spec.channels * 9U);
            import_floats(&cursor, model->stem_pre_bias, channels);
        }
    }
    model->bits = (kshira_bit_mode)header.bits;
    model->scale_head_trained_mask = model->spec.shared_multiscale_head ?
                                     0 : (int)header.scale_head_trained_mask;
    model->calibration_samples = (size_t)header.calibration_samples;
    model->calibration_input_scale = header.calibration_input_scale;
    model->calibration_stem_scale = header.calibration_stem_scale;
    for (int branch = 0; branch < RAD_BRANCHES; ++branch) {
        model->calibration_branch_scale[branch] = header.calibration_branch_scale[branch];
    }
    for (int level = 0; level < RAD_SCALES; ++level) {
        model->scale_head_gain[level] = header.scale_head_gain[level];
        model->scale_head_offset[level] = header.scale_head_offset[level];
    }
    model->transient_image_scale = header.transient_image_scale;
    model->transient_stem_scale = header.transient_stem_scale;
    model->transient_scales_valid = (int)header.transient_scales_valid;
    return (size_t)(cursor - (const uint8_t *)buffer) == bytes ?
           KSHIRA_OK : KSHIRA_ERR_RANGE;
}
