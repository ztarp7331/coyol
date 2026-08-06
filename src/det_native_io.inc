/* Portable native graph profile checkpoint.  This is separate from the
 * compact KSHIRA state format because the accuracy profile owns a complete
 * exported graph rather than the trainable RAD weights. */

#define DET_NATIVE_NGM_LUT_COUNT 4U
#define DET_NATIVE_CALIBRATION_MAGIC 0x314C4143U
#define DET_NATIVE_F32_MAGIC 0x324E4633U
#define DET_NATIVE_CALIBRATION_FLOATS (DET_MAX_CLASSES + DET_MAX_SCALES)
#define DET_NATIVE_CALIBRATION_OLD_BYTES \
    (sizeof(uint32_t) + DET_NATIVE_CALIBRATION_FLOATS * sizeof(float))
#define DET_NATIVE_CALIBRATION_BYTES \
    (DET_NATIVE_CALIBRATION_OLD_BYTES + DET_MAX_SCALES * 4U * sizeof(float))
#define DET_NATIVE_CALIBRATION_V2_BYTES \
    (DET_NATIVE_CALIBRATION_BYTES + DET_MAX_SCALES * DET_MAX_CLASSES * 4U * sizeof(float))

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t classes;
    uint32_t max_detections;
    uint32_t feature_channels;
    uint32_t stem_mode;
    uint32_t research_mode;
    uint32_t precision;
    uint32_t input_channels;
    uint32_t class_count;
    uint32_t conv_count;
    uint32_t weight_quant;
    uint32_t scale;
    float act_scale;
    float score_gain;
    float score_bias;
    uint64_t train_updates;
    uint64_t weight_blob_size;
    uint64_t bias_count;
    uint64_t mult_count;
    uint64_t shift_count;
} det_native_file_header;

typedef struct {
    int32_t in_channels;
    int32_t out_channels;
    int32_t kernel;
    int32_t stride;
    int32_t padding;
    int32_t groups;
    uint32_t activation;
    uint32_t weight_quant;
    uint64_t weight_offset;
    uint64_t bias_index;
} det_native_conv_record;

static int det_native_u64_size(uint64_t value, size_t *out) {
    if (out == NULL || value > (uint64_t)SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int det_native_conv_bytes(const det_native_conv_record *record,
                                 size_t *out) {
    size_t weights;
    if (record == NULL || out == NULL || record->in_channels <= 0 ||
        record->out_channels <= 0 || record->kernel <= 0 || record->groups <= 0 ||
        record->in_channels % record->groups != 0 ||
        record->out_channels % record->groups != 0 ||
        !checked_size3(record->out_channels,
                       record->in_channels / record->groups,
                       record->kernel * record->kernel, &weights)) return 0;
    if (record->weight_quant == NG_QUANT_INT4) {
        *out = (weights + 1U) / 2U;
    } else if (record->weight_quant == NG_QUANT_INT8) {
        *out = weights;
    } else {
        return 0;
    }
    return 1;
}

static int det_native_conv_f32_bytes(const det_native_conv_record *record,
                                     size_t *out) {
    if (record == NULL || out == NULL || record->in_channels <= 0 ||
        record->out_channels <= 0 || record->kernel <= 0 || record->groups <= 0 ||
        record->in_channels % record->groups != 0 ||
        record->out_channels % record->groups != 0) return 0;
    return checked_size3(record->out_channels,
                          record->in_channels / record->groups,
                          record->kernel * record->kernel * (int)sizeof(float), out);
}

static int det_native_header_valid(const det_native_file_header *header,
                                   size_t file_size) {
    size_t weight_bytes;
    size_t bias_count;
    size_t mult_count;
    size_t shift_count;
    size_t records_bytes;
    size_t graph_extra;
    size_t total;
    if (header == NULL || memcmp(header->magic, "CY8N", 4U) != 0 ||
        (header->version != 1U && header->version != 2U) || header->width == 0U || header->height == 0U ||
        header->width > 4096U || header->height > 4096U || header->channels == 0U ||
        header->channels > 4U || header->classes == 0U ||
        header->classes > DET_MAX_CLASSES || header->max_detections == 0U ||
        header->max_detections > 100U || header->feature_channels == 0U ||
        header->feature_channels > 32U || header->conv_count == 0U ||
        header->conv_count > NG_LOADER_MAX_CONVS || header->input_channels == 0U ||
        header->input_channels > 3U || header->class_count != header->classes ||
        (header->weight_quant != NG_QUANT_INT8 && header->weight_quant != NG_QUANT_INT4) ||
        (header->scale > (uint32_t)NG_SCALE_X &&
         header->scale != (uint32_t)NG_GRAPH_MULTISCALE) ||
        !isfinite(header->act_scale) || header->act_scale <= 0.0f ||
        !isfinite(header->score_gain) || header->score_gain <= 0.0f ||
        !isfinite(header->score_bias) || !det_native_u64_size(header->weight_blob_size,
                                                               &weight_bytes) ||
        !det_native_u64_size(header->bias_count, &bias_count) ||
        !det_native_u64_size(header->mult_count, &mult_count) ||
        !det_native_u64_size(header->shift_count, &shift_count) ||
        header->bias_count == 0U || header->bias_count != header->mult_count ||
        header->bias_count != header->shift_count ||
        !checked_add_size((size_t)header->conv_count * sizeof(det_native_conv_record),
                          weight_bytes, &records_bytes) ||
        !checked_add_size(records_bytes, bias_count * sizeof(int32_t), &records_bytes) ||
        !checked_add_size(records_bytes, mult_count * sizeof(int32_t), &records_bytes) ||
        !checked_add_size(records_bytes, shift_count, &records_bytes) ||
        !checked_add_size(records_bytes, 256U + 512U, &total) ||
        ((header->scale == (uint32_t)NG_GRAPH_MULTISCALE) &&
         (!checked_size3((int)header->conv_count, 2, (int)sizeof(float), &graph_extra) ||
          !checked_add_size(graph_extra, (size_t)header->conv_count, &graph_extra) ||
          !checked_add_size(graph_extra, DET_NATIVE_NGM_LUT_COUNT * 256U, &graph_extra) ||
          !checked_add_size(total, graph_extra, &total))) ||
        !checked_add_size(sizeof(*header), total, &total) ||
        total > file_size || file_size > 256U * 1024U * 1024U) {
        return 0;
    }
    return 1;
}

static int det_native_fill_conv_records(const det_model *model,
                                        det_native_conv_record *records) {
    const NGLoadedWeights *loaded;
    if (model == NULL || records == NULL || !model->native_graph_enabled) return 0;
    loaded = &model->native_graph_weights;
    for (int32_t i = 0; i < loaded->conv_count; ++i) {
        const NGConv2D *conv = &loaded->convs[i];
        size_t weight_offset;
        size_t bias_index;
        if ((uintptr_t)conv->weights < (uintptr_t)loaded->weight_blob ||
            (uintptr_t)conv->bias < (uintptr_t)loaded->bias_blob ||
            (uintptr_t)conv->multiplier < (uintptr_t)loaded->mult_blob ||
            (uintptr_t)conv->shift < (uintptr_t)loaded->shift_blob) return 0;
        weight_offset = (size_t)((const uint8_t *)conv->weights - loaded->weight_blob);
        bias_index = (size_t)(conv->bias - loaded->bias_blob);
        records[i] = (det_native_conv_record){conv->in_channels, conv->out_channels,
                                               conv->kernel, conv->stride, conv->padding,
                                               conv->groups, (uint32_t)conv->activation,
                                               (uint32_t)conv->weight_quant,
                                               (uint64_t)weight_offset,
                                               (uint64_t)bias_index};
    }
    return 1;
}

static det_status det_native_graph_save_file(const det_model *model, const char *path) {
    const NGLoadedWeights *loaded;
    det_native_file_header header;
    det_native_conv_record records[NG_LOADER_MAX_CONVS];
    size_t payload_size;
    size_t graph_extra = 0U;
    size_t f32_extra = 0U;
    size_t cursor;
    uint8_t *payload;
    uint8_t ngm_luts[DET_NATIVE_NGM_LUT_COUNT * 256U];
    uint32_t checksum;
    FILE *file;
    int ok;
    if (model == NULL || path == NULL || !model->native_graph_enabled) return DET_ERR_ARGUMENT;
    loaded = &model->native_graph_weights;
    if (loaded->conv_count <= 0 || loaded->conv_count > NG_LOADER_MAX_CONVS ||
        !det_native_fill_conv_records(model, records) ||
        loaded->bias_count != loaded->mult_count ||
        loaded->bias_count != loaded->shift_count ||
        loaded->bias_count == 0U || loaded->weight_blob == NULL ||
        loaded->bias_blob == NULL || loaded->mult_blob == NULL ||
        loaded->shift_blob == NULL) return DET_ERR_FORMAT;
    if (loaded->architecture == NG_GRAPH_MULTISCALE &&
        (loaded->activation_scale_blob == NULL ||
         loaded->activation_scale_count != (size_t)loaded->conv_count * 2U ||
         loaded->silu_index_blob == NULL ||
         loaded->silu_index_count < (size_t)loaded->conv_count ||
         loaded->silu_blob == NULL || loaded->silu_blob_size < 256U ||
         !checked_size3(loaded->conv_count, 2, (int)sizeof(float), &graph_extra) ||
         !checked_add_size(graph_extra, (size_t)loaded->conv_count, &graph_extra) ||
         !checked_add_size(graph_extra, DET_NATIVE_NGM_LUT_COUNT * 256U, &graph_extra))) {
        return DET_ERR_FORMAT;
    }
    if (loaded->weight_f32_blob != NULL && loaded->bias_f32_blob != NULL &&
        loaded->weight_f32_blob_size > 0U &&
        loaded->bias_f32_count == loaded->bias_count) {
        if (!checked_add_size(sizeof(uint32_t), loaded->weight_f32_blob_size,
                              &f32_extra) ||
            !checked_add_size(f32_extra, loaded->bias_f32_count * sizeof(float),
                              &f32_extra)) return DET_ERR_MEMORY;
    }
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "CY8N", 4U);
    header.version = 2U;
    header.width = (uint32_t)model->spec.width;
    header.height = (uint32_t)model->spec.height;
    header.channels = (uint32_t)model->spec.channels;
    header.classes = (uint32_t)model->spec.num_classes;
    header.max_detections = (uint32_t)model->spec.max_detections;
    header.feature_channels = (uint32_t)model->spec.feature_channels;
    header.stem_mode = (uint32_t)model->spec.stem_mode;
    header.research_mode = (uint32_t)model->spec.research_mode;
    header.precision = (uint32_t)model->precision;
    header.input_channels = (uint32_t)loaded->input_channels;
    header.class_count = (uint32_t)loaded->class_count;
    header.conv_count = (uint32_t)loaded->conv_count;
    header.weight_quant = (uint32_t)loaded->weight_quant;
    header.scale = loaded->architecture == NG_GRAPH_MULTISCALE ?
                   (uint32_t)NG_GRAPH_MULTISCALE : (uint32_t)loaded->scale;
    header.act_scale = loaded->act_scale;
    header.score_gain = model->native_score_gain;
    header.score_bias = model->native_score_bias;
    header.train_updates = (uint64_t)model->train_updates;
    header.weight_blob_size = (uint64_t)loaded->weight_blob_size;
    header.bias_count = (uint64_t)loaded->bias_count;
    header.mult_count = (uint64_t)loaded->mult_count;
    header.shift_count = (uint64_t)loaded->shift_count;
    if (!det_native_u64_size(header.weight_blob_size, &cursor) ||
        !det_native_u64_size(header.bias_count, &payload_size) ||
        !det_native_header_valid(&header, 256U * 1024U * 1024U) ||
        !checked_add_size(sizeof(header),
                          (size_t)header.conv_count * sizeof(records[0]), &payload_size) ||
        !checked_add_size(payload_size, loaded->weight_blob_size, &payload_size) ||
        !checked_add_size(payload_size, loaded->bias_count * sizeof(int32_t), &payload_size) ||
        !checked_add_size(payload_size, loaded->mult_count * sizeof(int32_t), &payload_size) ||
        !checked_add_size(payload_size, loaded->shift_count + 256U + 512U + graph_extra,
                          &payload_size) ||
        !checked_add_size(payload_size, DET_NATIVE_CALIBRATION_V2_BYTES, &payload_size) ||
        !checked_add_size(payload_size, f32_extra, &payload_size) ||
        payload_size > 256U * 1024U * 1024U) {
        return DET_ERR_MEMORY;
    }
    payload = (uint8_t *)malloc(payload_size);
    if (payload == NULL) return DET_ERR_MEMORY;
    cursor = 0U;
    memcpy(payload + cursor, &header, sizeof(header)); cursor += sizeof(header);
    memcpy(payload + cursor, records, (size_t)header.conv_count * sizeof(records[0]));
    cursor += (size_t)header.conv_count * sizeof(records[0]);
    memcpy(payload + cursor, loaded->weight_blob, loaded->weight_blob_size);
    cursor += loaded->weight_blob_size;
    memcpy(payload + cursor, loaded->bias_blob, loaded->bias_count * sizeof(int32_t));
    cursor += loaded->bias_count * sizeof(int32_t);
    memcpy(payload + cursor, loaded->mult_blob, loaded->mult_count * sizeof(int32_t));
    cursor += loaded->mult_count * sizeof(int32_t);
    memcpy(payload + cursor, loaded->shift_blob, loaded->shift_count); cursor += loaded->shift_count;
    memcpy(payload + cursor, loaded->silu_lut, 256U); cursor += 256U;
    memcpy(payload + cursor, loaded->dfl_exp_lut, 512U); cursor += 512U;
    if (loaded->architecture == NG_GRAPH_MULTISCALE) {
        memcpy(payload + cursor, loaded->activation_scale_blob,
               (size_t)loaded->conv_count * 2U * sizeof(float));
        cursor += (size_t)loaded->conv_count * 2U * sizeof(float);
        memcpy(payload + cursor, loaded->silu_index_blob, (size_t)loaded->conv_count);
        cursor += (size_t)loaded->conv_count;
        memset(ngm_luts, 0, sizeof(ngm_luts));
        memcpy(ngm_luts, loaded->silu_blob,
               loaded->silu_blob_size < sizeof(ngm_luts) ?
                   loaded->silu_blob_size : sizeof(ngm_luts));
        memcpy(payload + cursor, ngm_luts, sizeof(ngm_luts));
        cursor += sizeof(ngm_luts);
    }
    {
        uint32_t calibration_magic = DET_NATIVE_CALIBRATION_MAGIC;
        memcpy(payload + cursor, &calibration_magic, sizeof(calibration_magic));
        cursor += sizeof(calibration_magic);
        memcpy(payload + cursor, model->native_class_bias,
               sizeof(model->native_class_bias));
        cursor += sizeof(model->native_class_bias);
        memcpy(payload + cursor, model->native_scale_bias,
               sizeof(model->native_scale_bias));
        cursor += sizeof(model->native_scale_bias);
        memcpy(payload + cursor, model->native_box_bias,
               sizeof(model->native_box_bias));
        cursor += sizeof(model->native_box_bias);
        memcpy(payload + cursor, model->native_class_box_bias,
               sizeof(model->native_class_box_bias));
        cursor += sizeof(model->native_class_box_bias);
    }
    if (f32_extra > 0U) {
        uint32_t f32_magic = DET_NATIVE_F32_MAGIC;
        memcpy(payload + cursor, &f32_magic, sizeof(f32_magic));
        cursor += sizeof(f32_magic);
        memcpy(payload + cursor, loaded->weight_f32_blob,
               loaded->weight_f32_blob_size);
        cursor += loaded->weight_f32_blob_size;
        memcpy(payload + cursor, loaded->bias_f32_blob,
               loaded->bias_f32_count * sizeof(float));
    }
    checksum = crc32_bytes(payload, payload_size);
    file = fopen(path, "wb");
    if (file == NULL) { free(payload); return DET_ERR_IO; }
    ok = fwrite(payload, 1U, payload_size, file) == payload_size &&
         fwrite(&checksum, sizeof(checksum), 1U, file) == 1U;
    free(payload);
    if (fclose(file) != 0) ok = 0;
    return ok ? DET_OK : DET_ERR_IO;
}

static det_status det_native_graph_load_file(det_context *ctx, const char *path,
                                             det_model **out) {
    FILE *file;
    long length;
    size_t encoded_size;
    size_t payload_size;
    uint8_t *encoded = NULL;
    uint32_t checksum;
    det_native_file_header header;
    det_native_conv_record records[NG_LOADER_MAX_CONVS];
    det_model_spec spec;
    det_model *model = NULL;
    const uint8_t *cursor;
    uint32_t calibration_magic;
    uint32_t f32_magic;
    size_t remaining;
    size_t weight_size, bias_count, mult_count, shift_count, f32_weight_bytes = 0U;
    det_status status;
    if (ctx == NULL || path == NULL || out == NULL) return DET_ERR_ARGUMENT;
    *out = NULL;
    file = fopen(path, "rb");
    if (file == NULL) return DET_ERR_IO;
    if (fseek(file, 0L, SEEK_END) != 0 || (length = ftell(file)) < 0L ||
        (unsigned long)length > 256UL * 1024UL * 1024UL ||
        (size_t)length < sizeof(header) + sizeof(checksum)) {
        fclose(file); return DET_ERR_FORMAT;
    }
    encoded_size = (size_t)length;
    payload_size = encoded_size - sizeof(checksum);
    encoded = (uint8_t *)malloc(encoded_size);
    if (encoded == NULL || fseek(file, 0L, SEEK_SET) != 0 ||
        fread(encoded, 1U, encoded_size, file) != encoded_size ||
        fclose(file) != 0) { free(encoded); return DET_ERR_FORMAT; }
    memcpy(&checksum, encoded + payload_size, sizeof(checksum));
    memcpy(&header, encoded, sizeof(header));
    if (crc32_bytes(encoded, payload_size) != checksum ||
        !det_native_header_valid(&header, encoded_size)) { free(encoded); return DET_ERR_FORMAT; }
    if (!det_native_u64_size(header.weight_blob_size, &weight_size) ||
        !det_native_u64_size(header.bias_count, &bias_count) ||
        !det_native_u64_size(header.mult_count, &mult_count) ||
        !det_native_u64_size(header.shift_count, &shift_count)) { free(encoded); return DET_ERR_FORMAT; }
    cursor = encoded + sizeof(header);
    memcpy(records, cursor, (size_t)header.conv_count * sizeof(records[0]));
    cursor += (size_t)header.conv_count * sizeof(records[0]);
    for (uint32_t i = 0U; i < header.conv_count; ++i) {
        size_t conv_bytes;
        size_t weight_offset;
        size_t bias_index;
        if (!det_native_conv_bytes(&records[i], &conv_bytes) ||
            !det_native_u64_size(records[i].weight_offset, &weight_offset) ||
            !det_native_u64_size(records[i].bias_index, &bias_index) ||
            weight_offset > weight_size || conv_bytes > weight_size - weight_offset ||
            bias_index > bias_count ||
            (size_t)records[i].out_channels > bias_count - bias_index ||
            (size_t)records[i].out_channels > mult_count - bias_index ||
            (size_t)records[i].out_channels > shift_count - bias_index) {
            free(encoded); return DET_ERR_FORMAT;
        }
        {
            size_t f32_bytes;
            if (!det_native_conv_f32_bytes(&records[i], &f32_bytes) ||
                !checked_add_size(f32_weight_bytes, f32_bytes, &f32_weight_bytes)) {
                free(encoded); return DET_ERR_FORMAT;
            }
        }
    }
    spec = (det_model_spec){(int)header.width, (int)header.height, (int)header.channels,
                            (int)header.classes,
                            header.max_detections > DET_KSHIRA_MAX_DETECTIONS ?
                                DET_KSHIRA_MAX_DETECTIONS : (int)header.max_detections,
                            1,
                            DET_ARCH_KSHIRA, (int)header.feature_channels,
                            (det_stem_mode)header.stem_mode, 0, 0,
                            (det_research_mode)header.research_mode, 0, 0, 0, 0, NULL};
    status = det_model_build(ctx, &spec, &model);
    if (status != DET_OK) { free(encoded); return status; }
    model->native_graph_weights.weight_blob = (uint8_t *)malloc(weight_size);
    model->native_graph_weights.bias_blob = (int32_t *)malloc(bias_count * sizeof(int32_t));
    model->native_graph_weights.mult_blob = (int32_t *)malloc(mult_count * sizeof(int32_t));
    model->native_graph_weights.shift_blob = (uint8_t *)malloc(shift_count);
    if (model->native_graph_weights.weight_blob == NULL ||
        model->native_graph_weights.bias_blob == NULL ||
        model->native_graph_weights.mult_blob == NULL ||
        model->native_graph_weights.shift_blob == NULL) { status = DET_ERR_MEMORY; goto fail; }
    memcpy(model->native_graph_weights.weight_blob, cursor, weight_size); cursor += weight_size;
    memcpy(model->native_graph_weights.bias_blob, cursor, bias_count * sizeof(int32_t));
    cursor += bias_count * sizeof(int32_t);
    memcpy(model->native_graph_weights.mult_blob, cursor, mult_count * sizeof(int32_t));
    cursor += mult_count * sizeof(int32_t);
    memcpy(model->native_graph_weights.shift_blob, cursor, shift_count); cursor += shift_count;
    memcpy(model->native_graph_weights.silu_lut, cursor, 256U); cursor += 256U;
    memcpy(model->native_graph_weights.dfl_exp_lut, cursor, 512U); cursor += 512U;
    if (header.scale == (uint32_t)NG_GRAPH_MULTISCALE) {
        size_t scale_bytes = (size_t)header.conv_count * 2U * sizeof(float);
        model->native_graph_weights.activation_scale_blob =
            (float *)malloc(scale_bytes);
        model->native_graph_weights.silu_index_blob =
            (uint8_t *)malloc((size_t)header.conv_count);
        model->native_graph_weights.silu_blob =
            (uint8_t *)malloc(DET_NATIVE_NGM_LUT_COUNT * 256U);
        if (model->native_graph_weights.activation_scale_blob == NULL ||
            model->native_graph_weights.silu_index_blob == NULL ||
            model->native_graph_weights.silu_blob == NULL) {
            status = DET_ERR_MEMORY;
            goto fail;
        }
        memcpy(model->native_graph_weights.activation_scale_blob, cursor, scale_bytes);
        cursor += scale_bytes;
        memcpy(model->native_graph_weights.silu_index_blob, cursor,
               (size_t)header.conv_count);
        cursor += (size_t)header.conv_count;
        memcpy(model->native_graph_weights.silu_blob, cursor,
               DET_NATIVE_NGM_LUT_COUNT * 256U);
        cursor += DET_NATIVE_NGM_LUT_COUNT * 256U;
        model->native_graph_weights.activation_scale_count = scale_bytes / sizeof(float);
        model->native_graph_weights.silu_index_count = (size_t)header.conv_count;
        model->native_graph_weights.silu_blob_size = DET_NATIVE_NGM_LUT_COUNT * 256U;
    }
    model->native_graph_weights.weight_blob_size = weight_size;
    model->native_graph_weights.bias_count = bias_count;
    model->native_graph_weights.mult_count = mult_count;
    model->native_graph_weights.shift_count = shift_count;
    model->native_graph_weights.conv_count = (int32_t)header.conv_count;
    model->native_graph_weights.weight_quant = (NGQuantMode)header.weight_quant;
    model->native_graph_weights.input_channels = (int32_t)header.input_channels;
    model->native_graph_weights.class_count = (int32_t)header.class_count;
    model->native_graph_weights.scale = (NGModelScale)header.scale;
    model->native_graph_weights.architecture = header.scale == NG_GRAPH_MULTISCALE ?
                                              NG_GRAPH_MULTISCALE : NG_GRAPH_BASE;
    if (model->native_graph_weights.architecture == NG_GRAPH_MULTISCALE)
        model->native_graph_weights.scale = NG_SCALE_N;
    model->native_graph_weights.act_scale = header.act_scale;
    model->native_graph_weights.dfl.exp_lut_q15 = model->native_graph_weights.dfl_exp_lut;
    for (uint32_t i = 0U; i < header.conv_count; ++i) {
        NGConv2D *conv = &model->native_graph_weights.convs[i];
        size_t weight_offset;
        size_t bias_index;
        if (!det_native_u64_size(records[i].weight_offset, &weight_offset) ||
            !det_native_u64_size(records[i].bias_index, &bias_index) ||
            weight_offset >= weight_size || bias_index >= bias_count) { status = DET_ERR_FORMAT; goto fail; }
        conv->weights = model->native_graph_weights.weight_blob + weight_offset;
        conv->bias = model->native_graph_weights.bias_blob + bias_index;
        conv->multiplier = model->native_graph_weights.mult_blob + bias_index;
        conv->shift = model->native_graph_weights.shift_blob + bias_index;
        conv->silu_lut = model->native_graph_weights.silu_lut;
        conv->in_channels = records[i].in_channels;
        conv->out_channels = records[i].out_channels;
        conv->kernel = records[i].kernel;
        conv->stride = records[i].stride;
        conv->padding = records[i].padding;
        conv->groups = records[i].groups;
        conv->activation = (NGActivation)records[i].activation;
        conv->weight_quant = (NGQuantMode)records[i].weight_quant;
        conv->input_scale = 0.0f;
        conv->output_scale = 0.0f;
        if (model->native_graph_weights.activation_scale_blob != NULL) {
            conv->input_scale = model->native_graph_weights.activation_scale_blob[2U * i];
            conv->output_scale = model->native_graph_weights.activation_scale_blob[2U * i + 1U];
        }
        if (model->native_graph_weights.silu_index_blob != NULL) {
            size_t lut_offset = (size_t)model->native_graph_weights.silu_index_blob[i] * 256U;
            if (lut_offset + 256U <= model->native_graph_weights.silu_blob_size)
                conv->silu_lut = (const int8_t *)(model->native_graph_weights.silu_blob + lut_offset);
        }
    }
    status = det_native_graph_prepare(model);
    if (status != DET_OK) goto fail;
    model->spec.max_detections = (int)header.max_detections;
    model->native_score_gain = header.score_gain;
    model->native_score_bias = header.score_bias;
    remaining = payload_size - (size_t)(cursor - encoded);
    if (remaining >= DET_NATIVE_CALIBRATION_OLD_BYTES) {
        memcpy(&calibration_magic, cursor, sizeof(calibration_magic));
        if (calibration_magic == DET_NATIVE_CALIBRATION_MAGIC) {
            cursor += sizeof(calibration_magic);
            memcpy(model->native_class_bias, cursor, sizeof(model->native_class_bias));
            cursor += sizeof(model->native_class_bias);
            memcpy(model->native_scale_bias, cursor, sizeof(model->native_scale_bias));
            cursor += sizeof(model->native_scale_bias);
            if (remaining >= DET_NATIVE_CALIBRATION_BYTES) {
                memcpy(model->native_box_bias, cursor, sizeof(model->native_box_bias));
                cursor += sizeof(model->native_box_bias);
            }
            remaining = payload_size - (size_t)(cursor - encoded);
            if (header.version >= 2U && remaining >= sizeof(model->native_class_box_bias)) {
                memcpy(model->native_class_box_bias, cursor,
                       sizeof(model->native_class_box_bias));
                cursor += sizeof(model->native_class_box_bias);
            }
        }
    }
    remaining = payload_size - (size_t)(cursor - encoded);
    if (remaining >= sizeof(f32_magic) + f32_weight_bytes +
        bias_count * sizeof(float)) {
        memcpy(&f32_magic, cursor, sizeof(f32_magic));
        if (f32_magic == DET_NATIVE_F32_MAGIC) {
            cursor += sizeof(f32_magic);
            model->native_graph_weights.weight_f32_blob =
                (float *)malloc(f32_weight_bytes);
            model->native_graph_weights.bias_f32_blob =
                (float *)malloc(bias_count * sizeof(float));
            if (model->native_graph_weights.weight_f32_blob == NULL ||
                model->native_graph_weights.bias_f32_blob == NULL) {
                status = DET_ERR_MEMORY;
                goto fail;
            }
            memcpy(model->native_graph_weights.weight_f32_blob, cursor,
                   f32_weight_bytes);
            cursor += f32_weight_bytes;
            memcpy(model->native_graph_weights.bias_f32_blob, cursor,
                   bias_count * sizeof(float));
            model->native_graph_weights.weight_f32_blob_size = f32_weight_bytes;
            model->native_graph_weights.bias_f32_count = bias_count;
        }
    }
    if (model->native_graph_weights.weight_f32_blob != NULL &&
        model->native_graph_weights.bias_f32_blob != NULL) {
        size_t f32_offset = 0U;
        for (uint32_t i = 0U; i < header.conv_count; ++i) {
            size_t f32_bytes;
            size_t bias_index;
            if (!det_native_conv_f32_bytes(&records[i], &f32_bytes) ||
                !det_native_u64_size(records[i].bias_index, &bias_index)) {
                status = DET_ERR_FORMAT;
                goto fail;
            }
            model->native_graph_weights.convs[i].weights_f32 =
                model->native_graph_weights.weight_f32_blob + f32_offset / sizeof(float);
            model->native_graph_weights.convs[i].bias_f32 =
                model->native_graph_weights.bias_f32_blob + bias_index;
            f32_offset += f32_bytes;
        }
        if (model->native_graph_input_f32 == NULL) {
            size_t input_bytes;
            if (!checked_size3(model->native_graph_weights.input_channels,
                               model->spec.height, model->spec.width,
                               &input_bytes)) {
                status = DET_ERR_FORMAT;
                goto fail;
            }
            model->native_graph_input_f32 =
                (float *)calloc(input_bytes, sizeof(float));
            if (model->native_graph_input_f32 == NULL) {
                status = DET_ERR_MEMORY;
                goto fail;
            }
        }
        if (model->native_graph_architecture != NG_GRAPH_MULTISCALE &&
            model->native_graph_arena_bytes < 8U * 1024U * 1024U) {
            int8_t *expanded_arena = (int8_t *)realloc(
                model->native_graph_arena, 8U * 1024U * 1024U);
            if (expanded_arena == NULL) {
                status = DET_ERR_MEMORY;
                goto fail;
            }
            model->native_graph_arena = expanded_arena;
            model->native_graph_arena_bytes = 8U * 1024U * 1024U;
        }
    }
    model->precision = (det_precision)header.precision;
    model->train_updates = (size_t)header.train_updates;
    free(encoded);
    *out = model;
    return DET_OK;
fail:
    free(encoded);
    det_model_destroy(model);
    return status;
}
