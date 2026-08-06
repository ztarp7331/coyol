/* Model allocation, reset, precision selection, and quantized cache rebuild. */
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
        spec->num_classes > DET_MAX_CLASSES || spec->max_detections <= 0 ||
        (spec->architecture != DET_ARCH_CDET &&
         spec->architecture != DET_ARCH_KSHIRA) || spec->feature_channels < 0 ||
        spec->feature_channels > 32 ||
        (spec->stem_mode != DET_STEM_LEGACY &&
         spec->stem_mode != DET_STEM_SPACE_TO_DEPTH &&
         spec->stem_mode != DET_STEM_TWO_STAGE) ||
        (spec->one_to_one_head != 0 && spec->one_to_one_head != 1) ||
        (spec->shared_multiscale_head != 0 && spec->shared_multiscale_head != 1) ||
        (spec->context_fusion != 0 && spec->context_fusion != 1) ||
        (spec->raw_input_features != 0 && spec->raw_input_features != 1) ||
        (spec->p3_only_deployment != 0 && spec->p3_only_deployment != 1) ||
        (spec->smooth_box_decode != 0 && spec->smooth_box_decode != 1) ||
        spec->research_mode < DET_RESEARCH_SCRATCH ||
        spec->research_mode > DET_RESEARCH_FULL ||
        (spec->architecture == DET_ARCH_CDET && spec->feature_channels != 0)) {
        return DET_ERR_ARGUMENT;
    }
    *out = NULL;
    det_model *model = (det_model *)calloc(1U, sizeof(*model));
    if (model == NULL) return DET_ERR_MEMORY;
    model->ctx = ctx;
    model->spec = *spec;
    model->architecture = spec->architecture;
    if (model->architecture == DET_ARCH_KSHIRA) {
        det_status status = det_kshira_build(model);
        if (status != DET_OK) {
            free(model);
            return status;
        }
        if (spec->native_graph_dir != NULL) {
            status = det_native_graph_attach(model, spec->native_graph_dir);
            if (status != DET_OK) {
                det_kshira_release(model);
                free(model);
                return status;
            }
        }
        *out = model;
        return DET_OK;
    }
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
    for (int i = 0; i < DET_NECK_STAGES; ++i) {
        const det_stage *backbone = &model->stages[i + 2];
        det_status status = alloc_stage(&model->neck[i], DET_GRAPH_CHANNELS,
                                        backbone->output_height, backbone->output_width,
                                        DET_GRAPH_CHANNELS, 1, 1, 0, 0);
        if (status != DET_OK) {
            det_model_destroy(model);
            return status;
        }
        size_t count = (size_t)DET_GRAPH_CHANNELS * (size_t)backbone->output_height *
                       (size_t)backbone->output_width;
        model->neck_fused[i] = (float *)calloc(count, sizeof(float));
        model->neck_gradient[i] = (float *)calloc(count, sizeof(float));
        model->neck_topdown[i] = (float *)calloc(count, sizeof(float));
        if (model->neck_fused[i] == NULL || model->neck_gradient[i] == NULL ||
            model->neck_topdown[i] == NULL) {
            det_model_destroy(model);
            return DET_ERR_MEMORY;
        }
    }
    for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
        const det_stage *input = &model->neck[i];
        const det_stage *target = &model->neck[i + 1];
        det_status status = alloc_stage(&model->bottomup[i], DET_GRAPH_CHANNELS,
                                        input->output_height, input->output_width,
                                        DET_GRAPH_CHANNELS, 3, 2, 1, 1);
        if (status != DET_OK || model->bottomup[i].output_height != target->output_height ||
            model->bottomup[i].output_width != target->output_width) {
            det_model_destroy(model);
            return status != DET_OK ? status : DET_ERR_SHAPE;
        }
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
    if (model == NULL) return DET_ERR_ARGUMENT;
    if (model->architecture == DET_ARCH_KSHIRA) {
        return det_kshira_set_precision(model, precision);
    }
    if (!valid_precision(precision)) return DET_ERR_ARGUMENT;
    if (precision != DET_PRECISION_F32) {
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            const det_stage *stage = &model->stages[i];
            if (!finite_values(stage->weights, stage_weight_count(stage)) ||
                !finite_values(stage->bias, (size_t)stage->output_channels)) {
                return DET_ERR_FORMAT;
            }
        }
        for (int i = 0; i < DET_NECK_STAGES; ++i) {
            const det_stage *stage = &model->neck[i];
            if (!finite_values(stage->weights, stage_weight_count(stage)) ||
                !finite_values(stage->bias, (size_t)stage->output_channels)) {
                return DET_ERR_FORMAT;
            }
        }
        for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
            const det_stage *stage = &model->bottomup[i];
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
        for (int i = 0; i < DET_NECK_STAGES; ++i) {
            det_status status = quantize_stage(&model->neck[i], precision);
            if (status != DET_OK) return status;
        }
        for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
            det_status status = quantize_stage(&model->bottomup[i], precision);
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
        for (int i = 0; i < DET_NECK_STAGES; ++i) {
            det_stage *stage = &model->neck[i];
            size_t wc = stage_weight_count(stage);
            if (precision == DET_PRECISION_INT8) {
                memset(stage->packed_weights, 0,
                       stage_packed_weight_count(stage) * sizeof(*stage->packed_weights));
            } else {
                memset(stage->quant_weights, 0, wc * sizeof(*stage->quant_weights));
            }
        }
        for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
            det_stage *stage = &model->bottomup[i];
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
        for (int i = 0; i < DET_NECK_STAGES; ++i) {
            det_stage *stage = &model->neck[i];
            memset(stage->quant_weights, 0,
                   stage_weight_count(stage) * sizeof(*stage->quant_weights));
            memset(stage->packed_weights, 0,
                   stage_packed_weight_count(stage) * sizeof(*stage->packed_weights));
            memset(stage->quant_scales, 0,
                   (size_t)stage->output_channels * sizeof(*stage->quant_scales));
        }
        for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
            det_stage *stage = &model->bottomup[i];
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

det_status det_model_set_native_graph(det_model *model, const char *directory) {
    if (model == NULL || model->architecture != DET_ARCH_KSHIRA || directory == NULL) {
        return DET_ERR_ARGUMENT;
    }
    det_native_graph_release(model);
    return det_native_graph_attach(model, directory);
}

det_status det_model_native_graph_calibration(const det_model *model,
                                             float *gain, float *bias) {
    if (model == NULL || gain == NULL || bias == NULL ||
        !model->native_graph_enabled) return DET_ERR_ARGUMENT;
    *gain = model->native_score_gain;
    *bias = model->native_score_bias;
    return DET_OK;
}

det_precision det_model_precision(const det_model *model) {
    return model == NULL ? DET_PRECISION_F32 : model->precision;
}

det_architecture det_model_architecture(const det_model *model) {
    return model == NULL ? DET_ARCH_CDET : model->architecture;
}

int det_model_one_to_one_head(const det_model *model) {
    return model != NULL && model->architecture == DET_ARCH_KSHIRA ?
           model->spec.one_to_one_head : 0;
}

int det_model_shared_multiscale(const det_model *model) {
    return model != NULL && model->architecture == DET_ARCH_KSHIRA ?
           model->spec.shared_multiscale_head : 0;
}

int det_model_context_fusion(const det_model *model) {
    return model != NULL && model->architecture == DET_ARCH_KSHIRA ?
           model->spec.context_fusion : 0;
}

int det_model_p3_only_deployment(const det_model *model) {
    return model != NULL && model->architecture == DET_ARCH_KSHIRA ?
           model->spec.p3_only_deployment : 0;
}

det_research_mode det_model_research_mode(const det_model *model) {
    return model == NULL ? DET_RESEARCH_SCRATCH : model->spec.research_mode;
}

static int add_memory_bytes(size_t *total, size_t elements,
                            size_t element_bytes) {
    size_t bytes;
    if (total == NULL || element_bytes == 0U ||
        elements > SIZE_MAX / element_bytes) return 0;
    bytes = elements * element_bytes;
    return checked_add_size(*total, bytes, total);
}

static int add_stage_memory(const det_stage *stage,
                            det_memory_report *report) {
    size_t weights = stage_weight_count(stage);
    size_t biases = (size_t)stage->output_channels;
    size_t outputs;
    size_t inputs;
    if (!checked_size3(stage->output_channels, stage->output_height,
                       stage->output_width, &outputs) ||
        !checked_size3(stage->input_channels, stage->input_height,
                       stage->input_width, &inputs) ||
        !add_memory_bytes(&report->parameter_bytes, weights + biases,
                          sizeof(float)) ||
        !add_memory_bytes(&report->optimizer_bytes, 2U * (weights + biases),
                          sizeof(float)) ||
        !add_memory_bytes(&report->quant_cache_bytes, weights,
                          sizeof(int8_t)) ||
        !add_memory_bytes(&report->quant_cache_bytes,
                          stage_packed_weight_count(stage), sizeof(uint8_t)) ||
        !add_memory_bytes(&report->quant_cache_bytes, biases, sizeof(float)) ||
        !add_memory_bytes(&report->activation_workspace_bytes,
                          2U * outputs + inputs, sizeof(float))) return 0;
    return 1;
}

static int add_head_memory(const det_head *head,
                           det_memory_report *report) {
    size_t weights = (size_t)head->channels * (size_t)head->outputs;
    size_t biases = (size_t)head->outputs;
    size_t packed = biases * (((size_t)head->channels + 1U) / 2U);
    return add_memory_bytes(&report->parameter_bytes, weights + biases,
                            sizeof(float)) &&
           add_memory_bytes(&report->optimizer_bytes, weights + biases,
                            sizeof(float)) &&
           add_memory_bytes(&report->quant_cache_bytes, weights,
                            sizeof(int8_t)) &&
           add_memory_bytes(&report->quant_cache_bytes, packed,
                            sizeof(uint8_t)) &&
           add_memory_bytes(&report->quant_cache_bytes, biases, sizeof(float));
}

det_status det_model_memory(const det_model *model, det_memory_report *report) {
    if (model == NULL || report == NULL) return DET_ERR_ARGUMENT;
    memset(report, 0, sizeof(*report));
    if (model->architecture == DET_ARCH_KSHIRA) {
        report->parameter_bytes = kshira_rad_parameter_bytes(model->kshira.rad);
        report->activation_workspace_bytes =
            kshira_rad_activation_bytes(model->kshira.rad);
        report->arena_high_water_bytes =
            kshira_session_arena_high_water(&model->kshira);
        if (model->native_graph_arena_high_water > report->arena_high_water_bytes) {
            report->arena_high_water_bytes = model->native_graph_arena_high_water;
        }
        report->arena_capacity_bytes = model->kshira_memory_bytes;
        if (model->native_graph_enabled) {
            if (model->native_graph_arena_bytes > report->arena_capacity_bytes) {
                report->arena_capacity_bytes = model->native_graph_arena_bytes;
            }
            if (!checked_add_size(report->parameter_bytes,
                                  model->native_graph_weights.weight_blob_size +
                                  model->native_graph_weights.bias_count * sizeof(int32_t) +
                                  model->native_graph_weights.mult_count * sizeof(int32_t) +
                                  model->native_graph_weights.shift_count * sizeof(uint8_t),
                                  &report->parameter_bytes) ||
                !checked_add_size(report->activation_workspace_bytes,
                                  model->native_graph_arena_bytes,
                                  &report->activation_workspace_bytes) ||
                !checked_add_size(report->activation_workspace_bytes,
                                  (size_t)model->native_graph_weights.input_channels *
                                  (size_t)model->spec.width * (size_t)model->spec.height,
                                  &report->activation_workspace_bytes)) return DET_ERR_MEMORY;
        }
        return DET_OK;
    }
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
        if (!add_stage_memory(&model->stages[i], report)) return DET_ERR_SHAPE;
    }
    for (int i = 0; i < DET_NECK_STAGES; ++i) {
        size_t elements;
        if (!add_stage_memory(&model->neck[i], report) ||
            !checked_size3(model->neck[i].output_channels,
                           model->neck[i].output_height,
                           model->neck[i].output_width, &elements) ||
            !add_memory_bytes(&report->activation_workspace_bytes,
                              3U * elements, sizeof(float))) return DET_ERR_SHAPE;
    }
    for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
        if (!add_stage_memory(&model->bottomup[i], report)) return DET_ERR_SHAPE;
    }
    for (int i = 0; i < DET_MAX_SCALES; ++i) {
        if (!add_head_memory(&model->heads[i], report) ||
            !add_head_memory(&model->aux_heads[i], report)) return DET_ERR_SHAPE;
    }
    return DET_OK;
}

void det_model_destroy(det_model *model) {
    if (model == NULL) return;
    if (model->architecture == DET_ARCH_KSHIRA) {
        det_kshira_release(model);
        det_native_graph_release(model);
        free(model);
        return;
    }
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) free_stage(&model->stages[i]);
    for (int i = 0; i < DET_NECK_STAGES; ++i) {
        free_stage(&model->neck[i]);
        free(model->neck_topdown[i]);
        free(model->neck_fused[i]);
        free(model->neck_gradient[i]);
    }
    for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) free_stage(&model->bottomup[i]);
    for (int i = 0; i < DET_MAX_SCALES; ++i) free_head(&model->heads[i]);
    for (int i = 0; i < DET_MAX_SCALES; ++i) free_head(&model->aux_heads[i]);
    free(model);
}

det_status det_model_reset(det_model *model, int seed) {
    if (model == NULL) return DET_ERR_ARGUMENT;
    if (model->architecture == DET_ARCH_KSHIRA) {
        return det_kshira_reset(model, seed);
    }
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
    for (int s = 0; s < DET_NECK_STAGES; ++s) {
        det_stage *stage = &model->neck[s];
        size_t wc = stage_weight_count(stage);
        for (size_t i = 0U; i < wc; ++i) {
            stage->weights[i] = 0.0f;
            stage->velocity_w[i] = 0.0f;
            stage->gradient_w[i] = 0.0f;
        }
        for (int c = 0; c < stage->output_channels; ++c) {
            stage->weights[(size_t)c * (size_t)stage->input_channels + (size_t)c] = 1.0f;
            stage->bias[c] = 0.0f;
            stage->velocity_b[c] = 0.0f;
            stage->gradient_b[c] = 0.0f;
        }
        size_t out_count = (size_t)stage->output_channels * (size_t)stage->output_height *
                           (size_t)stage->output_width;
        memset(stage->activation, 0, out_count * sizeof(float));
        memset(stage->gradient_output, 0, out_count * sizeof(float));
        memset(stage->gradient_input, 0,
               (size_t)stage->input_channels * (size_t)stage->input_height *
               (size_t)stage->input_width * sizeof(float));
        memset(model->neck_fused[s], 0, out_count * sizeof(float));
        memset(model->neck_gradient[s], 0, out_count * sizeof(float));
    }
    for (int s = 0; s < DET_BOTTOMUP_STAGES; ++s) {
        det_stage *stage = &model->bottomup[s];
        size_t wc = stage_weight_count(stage);
        memset(stage->weights, 0, wc * sizeof(float));
        memset(stage->velocity_w, 0, wc * sizeof(float));
        memset(stage->gradient_w, 0, wc * sizeof(float));
        memset(stage->bias, 0, (size_t)stage->output_channels * sizeof(float));
        memset(stage->velocity_b, 0, (size_t)stage->output_channels * sizeof(float));
        memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
        size_t out_count = (size_t)stage->output_channels * (size_t)stage->output_height *
                           (size_t)stage->output_width;
        memset(stage->activation, 0, out_count * sizeof(float));
        memset(stage->gradient_output, 0, out_count * sizeof(float));
        memset(stage->gradient_input, 0,
               (size_t)stage->input_channels * (size_t)stage->input_height *
               (size_t)stage->input_width * sizeof(float));
    }
    model->train_updates = 0U;
    return det_model_set_precision(model, DET_PRECISION_F32);
}
