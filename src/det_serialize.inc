/* Versioned CDET serialization, CRC validation, and canonical cache loading. */
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
    input_height = (int)header->height;
    input_width = (int)header->width;
    for (int s = 0; s < DET_GRAPH_STAGES; ++s) {
        int output_height = 0;
        int output_width = 0;
        if (!convolution_output_dim(input_height, 3, 2, 1, &output_height) ||
            !convolution_output_dim(input_width, 3, 2, 1, &output_width)) return 0;
        input_height = output_height;
        input_width = output_width;
        if (s < 2) continue;
        size_t output_channels = DET_GRAPH_CHANNELS;
        size_t wc = output_channels * output_channels;
        size_t packed = output_channels * ((output_channels + 1U) / 2U);
        size_t bytes = 10U * sizeof(int);
        if (!checked_add_size(bytes, (2U * wc + 2U * output_channels) * sizeof(float), &bytes) ||
            !checked_add_size(bytes, wc * sizeof(int8_t), &bytes) ||
            !checked_add_size(bytes, packed * sizeof(uint8_t), &bytes) ||
            !checked_add_size(bytes, output_channels * sizeof(float), &bytes) ||
            !checked_add_size(total, bytes, &total)) return 0;
    }
    input_height = (int)header->height;
    input_width = (int)header->width;
    for (int s = 0; s < DET_GRAPH_STAGES; ++s) {
        int output_height = 0;
        int output_width = 0;
        if (!convolution_output_dim(input_height, 3, 2, 1, &output_height) ||
            !convolution_output_dim(input_width, 3, 2, 1, &output_width)) return 0;
        input_height = output_height;
        input_width = output_width;
        if (s < 2) continue;
        if (s >= DET_GRAPH_STAGES - 1) break;
        size_t output_channels = DET_GRAPH_CHANNELS;
        size_t wc = output_channels * 9U;
        size_t packed = output_channels * ((9U + 1U) / 2U);
        size_t bytes = 10U * sizeof(int);
        if (!checked_add_size(bytes, (2U * wc + 2U * output_channels) * sizeof(float), &bytes) ||
            !checked_add_size(bytes, wc * sizeof(int8_t), &bytes) ||
            !checked_add_size(bytes, packed * sizeof(uint8_t), &bytes) ||
            !checked_add_size(bytes, output_channels * sizeof(float), &bytes) ||
            !checked_add_size(total, bytes, &total)) return 0;
    }
    *out = total;
    return 1;
}

det_status det_save(const det_model *model, const char *path) {
    if (model == NULL || path == NULL) return DET_ERR_ARGUMENT;
    if (model->architecture == DET_ARCH_KSHIRA) return DET_ERR_UNSUPPORTED;
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
    for (int s = 0; s < DET_NECK_STAGES; ++s) {
        const det_stage *stage = &model->neck[s];
        if (!finite_values(stage->weights, stage_weight_count(stage)) ||
            !finite_values(stage->bias, (size_t)stage->output_channels) ||
            !finite_values(stage->velocity_w, stage_weight_count(stage)) ||
            !finite_values(stage->velocity_b, (size_t)stage->output_channels) ||
            (model->precision != DET_PRECISION_F32 &&
             !positive_finite_values(stage->quant_scales, (size_t)stage->output_channels))) {
            return DET_ERR_FORMAT;
        }
    }
    for (int s = 0; s < DET_BOTTOMUP_STAGES; ++s) {
        const det_stage *stage = &model->bottomup[s];
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
    det_file_header header = {{'C', 'D', 'E', 'T'}, 8U, (uint32_t)model->spec.width,
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
    for (int s = 0; ok && s < DET_NECK_STAGES; ++s) {
        const det_stage *stage = &model->neck[s];
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
    for (int s = 0; ok && s < DET_BOTTOMUP_STAGES; ++s) {
        const det_stage *stage = &model->bottomup[s];
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
        memcmp(header.magic, "CDET", 4U) != 0 || header.version != 8U ||
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
                           (int)header.classes, (int)header.max_detections, 1,
                           DET_ARCH_CDET, 0};
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
    for (int s = 0; ok && s < DET_NECK_STAGES; ++s) {
        det_stage *stage = &model->neck[s];
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
    for (int s = 0; ok && s < DET_BOTTOMUP_STAGES; ++s) {
        det_stage *stage = &model->bottomup[s];
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
