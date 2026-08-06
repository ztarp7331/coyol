/* Version-22 CDET wrapper for integrated KSHIRA model/session state. */
typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t architecture;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t classes;
    uint32_t max_detections;
    int32_t seed;
    uint32_t feature_channels;
    uint32_t stem_mode;
    uint32_t one_to_one_head;
    uint32_t shared_multiscale_head;
    uint32_t research_mode;
    uint32_t context_fusion;
    uint32_t raw_input_features;
    uint32_t p3_only_deployment;
    uint32_t smooth_box_decode;
    uint32_t precision;
    uint64_t train_updates;
    uint32_t phase;
    uint32_t phase_bits;
    uint32_t update_mode;
    uint32_t qas_enabled;
    uint64_t phase_steps;
    uint64_t phase_transitions;
    uint32_t mask_bytes;
    uint64_t rad_state_bytes;
} det_kshira_file_header;

static size_t det_kshira_mask_bytes(size_t channels) {
    return channels > SIZE_MAX - 7U ? SIZE_MAX : (channels + 7U) / 8U;
}

static int det_kshira_payload_size(const det_kshira_file_header *header,
                                   size_t *out) {
    size_t total;
    if (header == NULL || out == NULL || header->rad_state_bytes > (uint64_t)SIZE_MAX) {
        return 0;
    }
    total = sizeof(*header);
    if (!checked_add_size(total, (size_t)header->mask_bytes, &total) ||
        !checked_add_size(total, (size_t)header->rad_state_bytes, &total)) return 0;
    *out = total;
    return 1;
}

static int det_kshira_header_valid(const det_kshira_file_header *header) {
    kshira_bit_mode precision_bits;
    size_t expected_mask;
    if (header == NULL || memcmp(header->magic, "CDET", 4U) != 0 ||
        header->version != 22U || header->architecture != (uint32_t)DET_ARCH_KSHIRA ||
        header->width < 8U || header->width > 4096U ||
        header->height < 8U || header->height > 4096U ||
        header->channels == 0U || header->channels > 4U ||
        header->classes == 0U || header->classes > DET_MAX_CLASSES ||
        header->max_detections == 0U ||
        header->max_detections > DET_KSHIRA_MAX_DETECTIONS ||
        header->feature_channels == 0U || header->feature_channels > 32U ||
        (header->stem_mode != (uint32_t)DET_STEM_LEGACY &&
         header->stem_mode != (uint32_t)DET_STEM_SPACE_TO_DEPTH &&
         header->stem_mode != (uint32_t)DET_STEM_TWO_STAGE) ||
        header->one_to_one_head > 1U ||
        header->shared_multiscale_head > 1U ||
        header->context_fusion > 1U ||
        header->raw_input_features > 1U ||
        header->p3_only_deployment > 1U ||
        header->smooth_box_decode > 1U ||
        header->research_mode > (uint32_t)DET_RESEARCH_FULL ||
        !det_kshira_precision_bits((det_precision)header->precision,
                                    &precision_bits) ||
        header->phase > (uint32_t)KSHIRA_PHASE_ODT ||
        header->phase_transitions != (uint64_t)header->phase ||
        header->phase_steps > (uint64_t)SIZE_MAX ||
        header->train_updates > (uint64_t)SIZE_MAX ||
        header->rad_state_bytes == 0U) return 0;
    expected_mask = det_kshira_mask_bytes((size_t)header->feature_channels);
    return expected_mask != SIZE_MAX && header->mask_bytes == (uint32_t)expected_mask;
}

static det_status det_kshira_save_file(const det_model *model, const char *path) {
    det_kshira_file_header header;
    const kshira_phase_contract *contract;
    size_t mask_bytes;
    size_t rad_bytes;
    size_t payload_bytes;
    size_t written = 0U;
    kshira_bit_mode precision_bits;
    uint8_t *payload;
    uint8_t *cursor;
    uint32_t checksum;
    FILE *file;
    int ok;
    if (model == NULL || path == NULL || model->architecture != DET_ARCH_KSHIRA) {
        return DET_ERR_ARGUMENT;
    }
    contract = kshira_session_contract(&model->kshira);
    mask_bytes = det_kshira_mask_bytes(model->kshira.channel_mask.channel_count);
    rad_bytes = kshira_rad_state_bytes(model->kshira.rad);
    if (contract == NULL || kshira_phase_validate(contract) != KSHIRA_OK ||
        mask_bytes == SIZE_MAX || rad_bytes == 0U ||
        model->kshira.phase.transitions != (size_t)contract->phase ||
        !det_kshira_precision_bits(model->precision, &precision_bits) ||
        kshira_rad_bits(model->kshira.rad) != precision_bits) return DET_ERR_FORMAT;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "CDET", 4U);
    header.version = 22U;
    header.architecture = (uint32_t)DET_ARCH_KSHIRA;
    header.width = (uint32_t)model->spec.width;
    header.height = (uint32_t)model->spec.height;
    header.channels = (uint32_t)model->spec.channels;
    header.classes = (uint32_t)model->spec.num_classes;
    header.max_detections = (uint32_t)model->spec.max_detections;
    header.seed = (int32_t)model->spec.seed;
    header.feature_channels = (uint32_t)model->spec.feature_channels;
    header.stem_mode = (uint32_t)model->spec.stem_mode;
    header.one_to_one_head = (uint32_t)model->spec.one_to_one_head;
    header.shared_multiscale_head = (uint32_t)model->spec.shared_multiscale_head;
    header.research_mode = (uint32_t)model->spec.research_mode;
    header.context_fusion = (uint32_t)model->spec.context_fusion;
    header.raw_input_features = (uint32_t)model->spec.raw_input_features;
    header.p3_only_deployment = (uint32_t)model->spec.p3_only_deployment;
    header.smooth_box_decode = (uint32_t)model->spec.smooth_box_decode;
    header.precision = (uint32_t)model->precision;
    header.train_updates = (uint64_t)model->train_updates;
    header.phase = (uint32_t)contract->phase;
    header.phase_bits = (uint32_t)contract->bits;
    header.update_mode = (uint32_t)contract->update_mode;
    header.qas_enabled = (uint32_t)contract->qas_enabled;
    header.phase_steps = (uint64_t)model->kshira.phase.steps;
    header.phase_transitions = (uint64_t)model->kshira.phase.transitions;
    header.mask_bytes = (uint32_t)mask_bytes;
    header.rad_state_bytes = (uint64_t)rad_bytes;
    if (!det_kshira_payload_size(&header, &payload_bytes) ||
        payload_bytes > 256U * 1024U * 1024U) return DET_ERR_MEMORY;
    payload = (uint8_t *)malloc(payload_bytes);
    if (payload == NULL) return DET_ERR_MEMORY;
    memcpy(payload, &header, sizeof(header));
    cursor = payload + sizeof(header);
    memcpy(cursor, model->kshira.channel_mask.channel_bits, mask_bytes);
    cursor += mask_bytes;
    if (kshira_rad_export_state(model->kshira.rad, cursor, rad_bytes, &written) !=
            KSHIRA_OK || written != rad_bytes) {
        free(payload);
        return DET_ERR_FORMAT;
    }
    checksum = crc32_bytes(payload, payload_bytes);
    file = fopen(path, "wb");
    if (file == NULL) {
        free(payload);
        return DET_ERR_IO;
    }
    ok = fwrite(payload, 1U, payload_bytes, file) == payload_bytes &&
         fwrite(&checksum, sizeof(checksum), 1U, file) == 1U;
    free(payload);
    if (fclose(file) != 0) ok = 0;
    return ok ? DET_OK : DET_ERR_IO;
}

static det_status det_kshira_load_file(det_context *ctx, const char *path,
                                       det_model **out) {
    FILE *file;
    long length;
    size_t encoded_bytes;
    size_t payload_bytes;
    uint8_t *encoded;
    uint32_t stored_checksum;
    det_kshira_file_header header;
    size_t expected_payload;
    det_model_spec spec;
    det_model *model = NULL;
    const uint8_t *mask;
    const uint8_t *rad_state;
    kshira_phase_contract contract;
    kshira_bit_mode precision_bits;
    det_status status;
    if (ctx == NULL || path == NULL || out == NULL) return DET_ERR_ARGUMENT;
    *out = NULL;
    file = fopen(path, "rb");
    if (file == NULL) return DET_ERR_IO;
    if (fseek(file, 0L, SEEK_END) != 0 || (length = ftell(file)) < 0L ||
        (unsigned long)length > 256UL * 1024UL * 1024UL ||
        (size_t)length < sizeof(header) + sizeof(uint32_t)) {
        fclose(file);
        return DET_ERR_FORMAT;
    }
    encoded_bytes = (size_t)length;
    payload_bytes = encoded_bytes - sizeof(uint32_t);
    encoded = (uint8_t *)malloc(encoded_bytes);
    if (encoded == NULL) {
        fclose(file);
        return DET_ERR_MEMORY;
    }
    if (fseek(file, 0L, SEEK_SET) != 0 ||
        fread(encoded, 1U, encoded_bytes, file) != encoded_bytes) {
        free(encoded);
        fclose(file);
        return DET_ERR_FORMAT;
    }
    if (fclose(file) != 0) {
        free(encoded);
        return DET_ERR_IO;
    }
    memcpy(&stored_checksum, encoded + payload_bytes, sizeof(stored_checksum));
    memcpy(&header, encoded, sizeof(header));
    if (crc32_bytes(encoded, payload_bytes) != stored_checksum ||
        !det_kshira_header_valid(&header) ||
        !det_kshira_payload_size(&header, &expected_payload) ||
        expected_payload != payload_bytes) {
        free(encoded);
        return DET_ERR_FORMAT;
    }
    spec = (det_model_spec){(int)header.width, (int)header.height,
                            (int)header.channels, (int)header.classes,
                            (int)header.max_detections, (int)header.seed,
                            DET_ARCH_KSHIRA, (int)header.feature_channels,
                            (det_stem_mode)header.stem_mode,
                            (int)header.one_to_one_head,
                             (int)header.shared_multiscale_head,
                             (det_research_mode)header.research_mode,
                             (int)header.context_fusion,
                             (int)header.raw_input_features,
                             (int)header.p3_only_deployment,
                             (int)header.smooth_box_decode, NULL};
    status = det_model_build(ctx, &spec, &model);
    if (status != DET_OK) {
        free(encoded);
        return status;
    }
    mask = encoded + sizeof(header);
    rad_state = mask + header.mask_bytes;
    if ((header.feature_channels & 7U) != 0U) {
        uint8_t valid_tail = (uint8_t)((1U << (header.feature_channels & 7U)) - 1U);
        if ((mask[header.mask_bytes - 1U] & (uint8_t)~valid_tail) != 0U) {
            status = DET_ERR_FORMAT;
            goto fail;
        }
    }
    contract = (kshira_phase_contract){
        (kshira_phase)header.phase, (kshira_bit_mode)header.phase_bits,
        (kshira_update_mode)header.update_mode, model->kshira.phase.contract.arena_cap,
        (int)header.qas_enabled
    };
    if (kshira_phase_validate(&contract) != KSHIRA_OK ||
        !det_kshira_precision_bits((det_precision)header.precision,
                                    &precision_bits) ||
        kshira_rad_import_state(model->kshira.rad, rad_state,
                                (size_t)header.rad_state_bytes) != KSHIRA_OK ||
        kshira_rad_bits(model->kshira.rad) != precision_bits) {
        status = DET_ERR_FORMAT;
        goto fail;
    }
    memcpy(model->kshira.channel_mask.channel_bits, mask, header.mask_bytes);
    model->kshira.phase.contract = contract;
    model->kshira.phase.steps = (size_t)header.phase_steps;
    model->kshira.phase.transitions = (size_t)header.phase_transitions;
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
