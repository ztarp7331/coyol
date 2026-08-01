/* KSHIRA architecture adapter behind the public detector lifecycle. */

enum { DET_KSHIRA_DEFAULT_CHANNELS = 8, DET_KSHIRA_MAX_DETECTIONS = 64 };

static det_status det_status_from_kshira(kshira_status status) {
    switch (status) {
        case KSHIRA_OK: return DET_OK;
        case KSHIRA_ERR_ARGUMENT: return DET_ERR_ARGUMENT;
        case KSHIRA_ERR_MEMORY: return DET_ERR_MEMORY;
        case KSHIRA_ERR_RANGE: return DET_ERR_SHAPE;
        case KSHIRA_ERR_UNSUPPORTED: return DET_ERR_UNSUPPORTED;
        default: return DET_ERR_FORMAT;
    }
}

static int det_kshira_feature_channels(const det_model_spec *spec) {
    return spec->feature_channels == 0 ? DET_KSHIRA_DEFAULT_CHANNELS :
                                        spec->feature_channels;
}

static int det_kshira_precision_bits(det_precision precision,
                                     kshira_bit_mode *bits) {
    if (bits == NULL) return 0;
    if (precision == DET_PRECISION_F32) {
        *bits = KSHIRA_BITS_FLOAT;
        return 1;
    }
    if (precision == DET_PRECISION_INT8) {
        *bits = KSHIRA_BITS_INT8;
        return 1;
    }
    if (precision == DET_PRECISION_INT4) {
        *bits = KSHIRA_BITS_INT4;
        return 1;
    }
    return 0;
}

static det_status det_kshira_initialize(det_model *model, int seed) {
    int features;
    kshira_session_spec session_spec;
    kshira_status status;
    if (model == NULL || model->ctx == NULL) return DET_ERR_ARGUMENT;
    features = det_kshira_feature_channels(&model->spec);
    session_spec.rad = (kshira_rad_spec){
        model->spec.width,
        model->spec.height,
        model->spec.channels,
        model->spec.num_classes,
        features,
        model->spec.max_detections,
        seed,
        1
    };
    session_spec.arena_cap = model->kshira_memory_bytes;
    status = kshira_session_init(&model->kshira, model->kshira_memory,
                                 model->kshira_memory_bytes, &session_spec);
    if (status != KSHIRA_OK) return det_status_from_kshira(status);
    for (int channel = 0; channel < features; ++channel) {
        status = kshira_session_set_channel(&model->kshira, (size_t)channel, 1);
        if (status != KSHIRA_OK) return det_status_from_kshira(status);
    }
    model->spec.seed = seed;
    model->spec.feature_channels = features;
    model->precision = DET_PRECISION_F32;
    model->train_updates = 0U;
    return DET_OK;
}

static det_status det_kshira_build(det_model *model) {
    int features;
    det_status status;
    if (model == NULL || model->ctx == NULL) return DET_ERR_ARGUMENT;
    features = det_kshira_feature_channels(&model->spec);
    if (features < 1 || features > 32 || model->spec.max_detections >
        DET_KSHIRA_MAX_DETECTIONS) return DET_ERR_ARGUMENT;
    model->kshira_memory = (uint8_t *)malloc(model->ctx->arena_bytes);
    if (model->kshira_memory == NULL) return DET_ERR_MEMORY;
    model->kshira_memory_bytes = model->ctx->arena_bytes;
    status = det_kshira_initialize(model, model->spec.seed);
    if (status != DET_OK) {
        free(model->kshira_memory);
        model->kshira_memory = NULL;
        model->kshira_memory_bytes = 0U;
        return status;
    }
    model->architecture = DET_ARCH_KSHIRA;
    return DET_OK;
}

static void det_kshira_release(det_model *model) {
    if (model == NULL || model->architecture != DET_ARCH_KSHIRA) return;
    memset(&model->kshira, 0, sizeof(model->kshira));
    free(model->kshira_memory);
    model->kshira_memory = NULL;
    model->kshira_memory_bytes = 0U;
}

static det_status det_kshira_reset(det_model *model, int seed) {
    if (model == NULL || model->architecture != DET_ARCH_KSHIRA) {
        return DET_ERR_ARGUMENT;
    }
    return det_kshira_initialize(model, seed);
}

static det_status det_kshira_set_precision(det_model *model,
                                            det_precision precision) {
    kshira_bit_mode bits;
    kshira_status status;
    if (model == NULL || model->architecture != DET_ARCH_KSHIRA) {
        return DET_ERR_ARGUMENT;
    }
    if (!det_kshira_precision_bits(precision, &bits)) return DET_ERR_UNSUPPORTED;
    status = kshira_rad_set_bits(model->kshira.rad, bits);
    if (status != KSHIRA_OK) return det_status_from_kshira(status);
    model->precision = precision;
    return DET_OK;
}

static int det_kshira_sample_valid(const det_model *model,
                                   const det_sample *sample) {
    size_t pixels;
    if (model == NULL || sample == NULL || sample->image.data == NULL ||
        sample->image.channels != model->spec.channels ||
        sample->image.height != model->spec.height ||
        sample->image.width != model->spec.width || sample->box_count < 0 ||
        (sample->box_count > 0 && sample->boxes == NULL)) return 0;
    for (int i = 0; i < sample->box_count; ++i) {
        const det_box *box = &sample->boxes[i];
        if (box->class_id < 0 || box->class_id >= model->spec.num_classes ||
            !isfinite(box->x1) || !isfinite(box->y1) || !isfinite(box->x2) ||
            !isfinite(box->y2) || box->x1 < 0.0f || box->y1 < 0.0f ||
            box->x2 > (float)model->spec.width ||
            box->y2 > (float)model->spec.height || box->x2 <= box->x1 ||
            box->y2 <= box->y1) return 0;
    }
    if (!checked_size3(sample->image.channels, sample->image.height,
                       sample->image.width, &pixels)) return 0;
    return finite_values(sample->image.data, pixels);
}

static int det_kshira_target_level(const det_model *model, const det_box *box) {
    float box_area = (box->x2 - box->x1) * (box->y2 - box->y1);
    float image_area = (float)model->spec.width * (float)model->spec.height;
    return box_area / image_area < (1.0f / 16.0f) ? 1 : 2;
}

static det_status det_kshira_prepare_training(det_model *model,
                                               const det_train_config *config) {
    const kshira_phase_contract *contract;
    kshira_bit_mode bits;
    kshira_status status;
    if (config->momentum != 0.0f ||
        !det_kshira_precision_bits(config->precision, &bits) ||
        config->mode == DET_TRAIN_GLOBAL_BP) return DET_ERR_UNSUPPORTED;
    contract = kshira_session_contract(&model->kshira);
    if (contract == NULL) return DET_ERR_FORMAT;
    if (config->mode == DET_TRAIN_ODT) {
        if (bits == KSHIRA_BITS_FLOAT || contract->phase == KSHIRA_PHASE_PRE) {
            return DET_ERR_UNSUPPORTED;
        }
        if (contract->phase == KSHIRA_PHASE_TRAIN) {
            status = kshira_session_transition(&model->kshira, bits,
                                               KSHIRA_UPDATE_CHANNELS, 1);
            if (status != KSHIRA_OK) return det_status_from_kshira(status);
            contract = kshira_session_contract(&model->kshira);
        }
        if (contract->phase != KSHIRA_PHASE_ODT || contract->bits != bits) {
            return DET_ERR_UNSUPPORTED;
        }
    } else if (bits == KSHIRA_BITS_FLOAT) {
        if (contract->phase != KSHIRA_PHASE_PRE) return DET_ERR_UNSUPPORTED;
    } else {
        if (contract->phase == KSHIRA_PHASE_PRE) {
            status = kshira_session_transition(&model->kshira, bits,
                                               KSHIRA_UPDATE_FULL, 1);
            if (status != KSHIRA_OK) return det_status_from_kshira(status);
            contract = kshira_session_contract(&model->kshira);
        }
        if (contract->phase != KSHIRA_PHASE_TRAIN || contract->bits != bits) {
            return DET_ERR_UNSUPPORTED;
        }
    }
    model->precision = config->precision;
    return DET_OK;
}

static det_status det_kshira_train(det_model *model, const det_dataset *dataset,
                                    const det_train_config *config,
                                    det_train_report *report) {
    double start;
    float loss_sum = 0.0f;
    size_t updates = 0U;
    det_status status;
    if (config->reset_weights != 0) {
        status = det_kshira_reset(model, config->seed);
        if (status != DET_OK) return status;
    }
    status = det_kshira_prepare_training(model, config);
    if (status != DET_OK) return status;
    memset(report, 0, sizeof(*report));
    start = now_ms();
    for (int epoch = 0; epoch < config->epochs; ++epoch) {
        size_t seen = 0U;
        if (dataset->reset != NULL) dataset->reset(dataset->user);
        while (config->max_samples <= 0 || seen < (size_t)config->max_samples) {
            det_sample sample;
            float sample_loss = 0.0f;
            int next_status = dataset->next(dataset->user, &sample);
            if (next_status < 0) return DET_ERR_IO;
            if (next_status == 0) break;
            if (!det_kshira_sample_valid(model, &sample)) return DET_ERR_ARGUMENT;
            if (sample.box_count == 0) return DET_ERR_UNSUPPORTED;
            for (int box_index = 0; box_index < sample.box_count; ++box_index) {
                const det_box *box = &sample.boxes[box_index];
                kshira_image_f32 image = {sample.image.data, sample.image.channels,
                                          sample.image.height, sample.image.width};
                kshira_rad_box target = {box->x1, box->y1, box->x2, box->y2,
                                         box->class_id};
                float loss = 0.0f;
                kshira_status step_status;
                if (config->mode == DET_TRAIN_ODT) {
                    step_status = kshira_session_multiscale_step(
                        &model->kshira, &image, &target,
                        det_kshira_target_level(model, box), config->learning_rate,
                        &loss);
                } else {
                    step_status = kshira_session_step(&model->kshira, &image, &target,
                                                      config->learning_rate, &loss);
                }
                if (step_status != KSHIRA_OK) return det_status_from_kshira(step_status);
                sample_loss += loss;
                ++updates;
            }
            if (sample.box_count > 0) sample_loss /= (float)sample.box_count;
            loss_sum += sample_loss;
            ++seen;
            ++report->samples_seen;
        }
        if (seen == 0U) return DET_ERR_ARGUMENT;
    }
    model->train_updates += updates;
    report->updates = updates;
    report->elapsed_ms = now_ms() - start;
    report->mean_loss = report->samples_seen == 0U ? 0.0f :
                        loss_sum / (float)report->samples_seen;
    report->used_global_backward = 0;
    return DET_OK;
}

static det_status det_kshira_predict(det_model *model, const det_image *image,
                                      float score_threshold,
                                      det_detection *detections, int capacity,
                                      int *count) {
    kshira_rad_detection raw[DET_KSHIRA_MAX_DETECTIONS];
    kshira_image_f32 input;
    int raw_capacity;
    int raw_count = 0;
    kshira_status status;
    if (model == NULL || image == NULL || detections == NULL || count == NULL) {
        return DET_ERR_ARGUMENT;
    }
    input = (kshira_image_f32){image->data, image->channels, image->height,
                               image->width};
    raw_capacity = capacity < model->spec.max_detections ? capacity :
                                                          model->spec.max_detections;
    status = kshira_session_predict(&model->kshira, &input, score_threshold, raw,
                                    raw_capacity, &raw_count);
    if (status != KSHIRA_OK) return det_status_from_kshira(status);
    for (int i = 0; i < raw_count; ++i) {
        detections[i].box = (det_box){raw[i].box.x1, raw[i].box.y1,
                                      raw[i].box.x2, raw[i].box.y2,
                                      raw[i].box.class_id};
        detections[i].score = raw[i].score;
    }
    *count = raw_count;
    return DET_OK;
}
