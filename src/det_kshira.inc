/* KSHIRA architecture adapter behind the public detector lifecycle. */

enum {
    DET_KSHIRA_DEFAULT_CHANNELS = 8,
    DET_KSHIRA_MAX_DETECTIONS = 64,
    DET_KSHIRA_NATIVE_MAX_DETECTIONS = 100,
    DET_KSHIRA_READOUT_RESERVE_MS = 3000,
    /* Outside-box objectness negatives per sample. Two is enough to cut empty
     * clutter without drowning the single positive update. */
    DET_KSHIRA_BACKGROUND_SAMPLES = 2,
    DET_KSHIRA_READOUT_IN = 38,
    DET_KSHIRA_READOUT_MAX_BOXES = 128
};

enum { DET_NATIVE_FEATURE_READOUT_MAX = 64 };

typedef struct {
    int channels[DET_MAX_SCALES];
    int terms[DET_MAX_SCALES];
    double gram[DET_MAX_SCALES][DET_NATIVE_FEATURE_READOUT_MAX + 1]
        [DET_NATIVE_FEATURE_READOUT_MAX + 1];
    double rhs[DET_MAX_SCALES][DET_MAX_CLASSES]
        [DET_NATIVE_FEATURE_READOUT_MAX + 1];
    double target_square[DET_MAX_SCALES][DET_MAX_CLASSES];
    size_t examples;
} det_native_feature_fit;

static det_status det_kshira_bootstrap_pass(
    det_model *model, const det_dataset *dataset, double deadline,
    float learning_rate, det_train_report *report);
static det_status det_native_bootstrap_pass(
    det_model *model, const det_dataset *dataset, double deadline,
    float learning_rate, det_train_report *report);
static int det_native_scale_bin(float x1, float y1, float x2, float y2,
                                int width, int height);
static float det_native_calibrated_score(const det_model *model, float raw_score,
                                         int class_id, float x1, float y1,
                                         float x2, float y2, int width, int height);
static float det_native_iou(const det_box *left, const det_box *right);
static void det_native_adjust_box(const det_model *model, int scale_bin,
                                  int width, int height, float *x1, float *y1,
                                  float *x2, float *y2);
static void det_native_adjust_class_box(const det_model *model, int scale_bin,
                                        int class_id, int width, int height,
                                        float *x1, float *y1, float *x2, float *y2);
static void det_native_update_head_bias(det_model *model, int scale_bin,
                                        int class_id, float delta_logit);
static int det_native_update_candidate_head(det_model *model, int scale_bin,
                                            int class_id, int cell_x, int cell_y,
                                            float target, float target_weight,
                                            float rate);
typedef struct {
    float raw_logit;
    uint8_t class_id;
    uint8_t scale_bin;
    uint8_t target;
} det_native_readout_example;
static det_status det_native_fit_readout(
    det_model *model, const det_native_readout_example *examples,
    size_t example_count, det_train_report *report);
static det_status det_native_fit_feature_readout(
    det_model *model, const det_sample *sample, size_t sample_index,
    det_native_feature_fit *fit);
static det_status det_native_apply_feature_readout(
    det_model *model, det_native_feature_fit *fit, det_train_report *report);

static void det_native_cache_features_f32(det_model *model,
                                          const NGF32ModelOutput *raw) {
    if (model == NULL || raw == NULL) return;
    for (int level = 0; level < DET_MAX_SCALES; ++level) {
        const NGF32Tensor *source = &raw->class_feature[level];
        size_t count;
        if (model->native_feature_cache[level] == NULL || source->data == NULL ||
            source->n != 1 || source->c != model->native_feature_channels[level] ||
            source->h != model->native_feature_height[level] ||
            source->w != model->native_feature_width[level] ||
            !checked_size3(source->c, source->h, source->w, &count) ||
            count != model->native_feature_cache_count[level]) continue;
        memcpy(model->native_feature_cache[level], source->data,
               count * sizeof(float));
    }
}

static void det_native_cache_features_s8(det_model *model,
                                         const NGModelOutput *raw) {
    if (model == NULL || raw == NULL) return;
    for (int level = 0; level < DET_MAX_SCALES; ++level) {
        const NGTensor *source = &raw->class_feature[level];
        size_t count;
        if (model->native_feature_cache[level] == NULL || source->data == NULL ||
            source->n != 1 || source->c != model->native_feature_channels[level] ||
            source->h != model->native_feature_height[level] ||
            source->w != model->native_feature_width[level] ||
            !checked_size3(source->c, source->h, source->w, &count) ||
            count != model->native_feature_cache_count[level]) continue;
        for (size_t i = 0U; i < count; ++i)
            model->native_feature_cache[level][i] = (float)source->data[i];
    }
}

static det_status det_native_graph_predict(det_model *model, const det_image *image,
                                          float score_threshold,
                                          det_detection *detections, int capacity,
                                          int *count);
static det_status det_native_graph_predict_f32(det_model *model, const det_image *image,
                                               float score_threshold,
                                               det_detection *detections, int capacity,
                                               int *count);
static det_status det_native_graph_train(det_model *model, const det_dataset *dataset,
                                        const det_train_config *config,
                                        det_train_report *report);

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
    int rad_max_detections;
    kshira_session_spec session_spec;
    kshira_status status;
    if (model == NULL || model->ctx == NULL) return DET_ERR_ARGUMENT;
    features = det_kshira_feature_channels(&model->spec);
    rad_max_detections = model->spec.native_graph_dir != NULL ?
                         DET_KSHIRA_MAX_DETECTIONS : model->spec.max_detections;
    session_spec.rad = (kshira_rad_spec){
        model->spec.width,
        model->spec.height,
        model->spec.channels,
        model->spec.num_classes,
        features,
        rad_max_detections,
        seed,
        1,
        (int)model->spec.stem_mode,
        model->spec.one_to_one_head,
        model->spec.shared_multiscale_head,
        model->spec.context_fusion,
        model->spec.raw_input_features,
        model->spec.p3_only_deployment,
        model->spec.smooth_box_decode
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
    if (features < 1 || features > 32 ||
        model->spec.max_detections >
        (model->spec.native_graph_dir != NULL ?
             DET_KSHIRA_NATIVE_MAX_DETECTIONS : DET_KSHIRA_MAX_DETECTIONS)) {
        return DET_ERR_ARGUMENT;
    }
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
            box->y2 <= box->y1 || !isfinite(box->target_weight) ||
            box->target_weight < 0.0f || box->target_weight > 1.0f) return 0;
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

static int det_kshira_background_cell(const det_model *model,
                                       const det_sample *sample, size_t start,
                                       size_t *cell_index) {
    int map_height;
    int map_width;
    size_t cells;
    if (model == NULL || sample == NULL || cell_index == NULL) return 0;
    map_height = kshira_rad_map_height(model->kshira.rad);
    map_width = kshira_rad_map_width(model->kshira.rad);
    if (map_height <= 0 || map_width <= 0 ||
        (size_t)map_height > SIZE_MAX / (size_t)map_width) return 0;
    cells = (size_t)map_height * (size_t)map_width;
    for (size_t attempt = 0U; attempt < cells; ++attempt) {
        size_t index = (start % cells + attempt) % cells;
        int y = (int)(index / (size_t)map_width);
        int x = (int)(index % (size_t)map_width);
        /* Cell center in image pixels. Treat any center that lands inside a GT
         * box as positive territory so background steps do not punch holes in
         * objects and so interior object cells are not used as negatives. */
        float cell_cx = ((float)x + 0.5f) * (float)sample->image.width /
                        (float)map_width;
        float cell_cy = ((float)y + 0.5f) * (float)sample->image.height /
                        (float)map_height;
        int on_object = 0;
        for (int box_index = 0; box_index < sample->box_count; ++box_index) {
            const det_box *box = &sample->boxes[box_index];
            if (cell_cx >= box->x1 && cell_cx <= box->x2 &&
                cell_cy >= box->y1 && cell_cy <= box->y2) {
                on_object = 1;
                break;
            }
        }
        if (!on_object) {
            *cell_index = index;
            return 1;
        }
    }
    return 0;
}

static int det_kshira_cell_inside_sample(const det_model *model,
                                         const det_sample *sample,
                                         size_t cell_index) {
    int map_height;
    int map_width;
    int cell_x;
    int cell_y;
    float cell_cx;
    float cell_cy;
    if (model == NULL || sample == NULL) return 1;
    map_height = kshira_rad_map_height(model->kshira.rad);
    map_width = kshira_rad_map_width(model->kshira.rad);
    if (map_height <= 0 || map_width <= 0 ||
        cell_index >= (size_t)map_height * (size_t)map_width) return 1;
    cell_x = (int)(cell_index % (size_t)map_width);
    cell_y = (int)(cell_index / (size_t)map_width);
    cell_cx = ((float)cell_x + 0.5f) * (float)sample->image.width /
              (float)map_width;
    cell_cy = ((float)cell_y + 0.5f) * (float)sample->image.height /
              (float)map_height;
    for (int box_index = 0; box_index < sample->box_count; ++box_index) {
        const det_box *box = &sample->boxes[box_index];
        if (cell_cx >= box->x1 && cell_cx <= box->x2 &&
            cell_cy >= box->y1 && cell_cy <= box->y2) return 1;
    }
    return 0;
}

static det_status det_kshira_readout(det_model *model, const det_dataset *dataset,
                                      double deadline, det_train_report *report) {
    double gram[DET_KSHIRA_READOUT_IN * DET_KSHIRA_READOUT_IN] = {0.0};
    double rhs[DET_KSHIRA_READOUT_IN * DET_MAX_CLASSES] = {0.0};
    double target_square_sum[DET_MAX_CLASSES] = {0.0};
    size_t readout_feature_count = (size_t)model->spec.feature_channels +
                                   (model->spec.raw_input_features ? 6U : 2U);
    kshira_rad_readout_stats stats = {
        gram, rhs, target_square_sum, DET_KSHIRA_READOUT_IN,
        (size_t)model->spec.num_classes, 0U, 0U, 0U
    };
    stats.feature_count = readout_feature_count;
    if (dataset->reset != NULL) dataset->reset(dataset->user);
    for (;;) {
        det_sample sample;
        kshira_rad_box targets[DET_KSHIRA_READOUT_MAX_BOXES];
        int next_status;
        if (deadline > 0.0 && now_ms() >= deadline) {
            report->stopped_by_budget = 1;
            break;
        }
        next_status = dataset->next(dataset->user, &sample);
        if (next_status < 0) return DET_ERR_IO;
        if (next_status == 0) break;
        if (!det_kshira_sample_valid(model, &sample) ||
            sample.box_count > DET_KSHIRA_READOUT_MAX_BOXES) return DET_ERR_ARGUMENT;
        for (int i = 0; i < sample.box_count; ++i) {
            targets[i] = (kshira_rad_box){sample.boxes[i].x1, sample.boxes[i].y1,
                                          sample.boxes[i].x2, sample.boxes[i].y2,
                                          sample.boxes[i].class_id};
        }
        {
            kshira_image_f32 image = {sample.image.data, sample.image.channels,
                                       sample.image.height, sample.image.width};
            kshira_status status = kshira_rad_readout_accumulate(
                model->kshira.rad, &image, targets, sample.box_count, &stats, 64);
            if (status != KSHIRA_OK) return det_status_from_kshira(status);
        }
        report->readout_examples = stats.examples;
        report->readout_positives = stats.positives;
        report->readout_negatives = stats.negatives;
    }
    if (stats.examples > 0U) {
        kshira_rad_readout_report readout_report;
        kshira_status status = kshira_rad_readout_solve(
            model->kshira.rad, &stats, 1.0e-2, &readout_report);
        if (status != KSHIRA_OK) return det_status_from_kshira(status);
        report->analytic_readout_applied = readout_report.applied;
        report->readout_examples = readout_report.examples;
        report->readout_positives = readout_report.positives;
        report->readout_negatives = readout_report.negatives;
        report->readout_pre_objective = readout_report.pre_objective;
        report->readout_post_objective = readout_report.post_objective;
    }
    return DET_OK;
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
    double deadline = 0.0;
    int budget_stop = 0;
    kshira_budget budget;
    double training_budget_ms;
    size_t total_seen = 0U;
    size_t total_cap = config->max_total_samples > 0 ?
                       (size_t)config->max_total_samples : 0U;
    if (model != NULL && model->native_graph_enabled) {
        return det_native_graph_train(model, dataset, config, report);
    }
    if (!isfinite(config->time_budget_ms) || config->time_budget_ms < 0.0 ||
        !isfinite(config->bootstrap_budget_ms) || config->bootstrap_budget_ms < 0.0 ||
        (config->residual_budget != 0 && config->residual_budget != 1) ||
        (config->quality_aligned_assignment != 0 &&
         config->quality_aligned_assignment != 1)) {
        return DET_ERR_ARGUMENT;
    }
    training_budget_ms = config->time_budget_ms;
    if (config->analytic_readout && config->precision == DET_PRECISION_F32 &&
        training_budget_ms > (double)DET_KSHIRA_READOUT_RESERVE_MS) {
        training_budget_ms -= (double)DET_KSHIRA_READOUT_RESERVE_MS;
    }
    if (config->reset_weights != 0) {
        status = det_kshira_reset(model, config->seed);
        if (status != DET_OK) return status;
    }
    status = det_kshira_prepare_training(model, config);
    if (status != DET_OK) return status;
    if (config->dense_aux_budget > 0) {
        status = det_status_from_kshira(kshira_session_set_dense_aux_budget(
            &model->kshira, config->dense_aux_budget));
        if (status != DET_OK) return status;
    }
    status = det_status_from_kshira(kshira_session_set_quality_aligned_assignment(
        &model->kshira, config->quality_aligned_assignment));
    if (status != DET_OK) return status;
    if (config->reset_schedule) {
        status = det_status_from_kshira(
            kshira_rad_reset_training_state(model->kshira.rad));
        if (status != DET_OK) return status;
        model->train_updates = 0U;
    }
    if (config->freeze_encoder) {
        int features = det_kshira_feature_channels(&model->spec);
        for (int channel = 0; channel < features; ++channel) {
            kshira_status mask_status = kshira_session_set_channel(
                &model->kshira, (size_t)channel, 0);
            if (mask_status != KSHIRA_OK) return det_status_from_kshira(mask_status);
        }
    }
    memset(report, 0, sizeof(*report));
    start = now_ms();
    kshira_budget_init(&budget, config->adaptive_budget ?
                       training_budget_ms : 0.0, 1000.0, start);
    if (training_budget_ms > 0.0) {
        if (budget.active) deadline = start + budget.work_ms;
        else {
            double work_budget = training_budget_ms - 1000.0;
            if (work_budget < 1.0) work_budget = training_budget_ms;
            deadline = start + work_budget;
        }
    }
    if (config->bootstrap_enabled) {
        double bootstrap_budget = config->bootstrap_budget_ms > 0.0 ?
                                  config->bootstrap_budget_ms : 5000.0;
        double bootstrap_deadline = start + bootstrap_budget;
        if (deadline > 0.0 && bootstrap_deadline > deadline) bootstrap_deadline = deadline;
        if (config->precision != DET_PRECISION_F32) return DET_ERR_UNSUPPORTED;
        status = det_kshira_bootstrap_pass(model, dataset, bootstrap_deadline,
                                           config->learning_rate, report);
        if (status != DET_OK) return status;
    }
    for (int epoch = 0; epoch < config->epochs; ++epoch) {
        if (kshira_budget_should_stop(&budget, now_ms()) ||
            (deadline > 0.0 && now_ms() >= deadline)) {
            budget_stop = 1;
            break;
        }
        size_t seen = 0U;
        /* Milder inverse-time decay: late epochs still move mid-band ranking
         * weights. Surgical FP updates keep per-epoch cost close to the old
         * 0.35 schedule, so more epochs fit similar wall-clock. */
        float epoch_lr = config->learning_rate / (1.0f + 0.20f * (float)epoch);
        if (!(epoch_lr > 0.0f) || !isfinite(epoch_lr)) return DET_ERR_ARGUMENT;
        if (dataset->reset != NULL) dataset->reset(dataset->user);
        while ((config->max_samples <= 0 || seen < (size_t)config->max_samples) &&
               (total_cap == 0U || total_seen < total_cap)) {
            det_sample sample;
            float sample_loss = 0.0f;
            int sample_terms = 0;
            int next_status = dataset->next(dataset->user, &sample);
            if (next_status < 0) return DET_ERR_IO;
            if (next_status == 0) break;
            if (!det_kshira_sample_valid(model, &sample)) return DET_ERR_ARGUMENT;
            if (sample.box_count == 0) {
                kshira_image_f32 image = {sample.image.data, sample.image.channels,
                                          sample.image.height, sample.image.width};
                float loss = 0.0f;
                kshira_status step_status;
                size_t cell_index;
                if (model->train_updates > SIZE_MAX - updates) return DET_ERR_SHAPE;
                if (!det_kshira_background_cell(model, &sample,
                                                 model->train_updates + updates,
                                                 &cell_index)) return DET_ERR_SHAPE;
                step_status = kshira_session_background_step(
                    &model->kshira, &image, cell_index, epoch_lr, &loss);
                if (step_status != KSHIRA_OK) {
                    return det_status_from_kshira(step_status);
                }
                sample_loss = loss;
                ++updates;
                ++sample_terms;
            }
            for (int box_index = 0; box_index < sample.box_count; ++box_index) {
                const det_box *box = &sample.boxes[box_index];
                float target_weight = box->target_weight > 0.0f ? box->target_weight : 1.0f;
                kshira_image_f32 image = {sample.image.data, sample.image.channels,
                                          sample.image.height, sample.image.width};
                kshira_rad_box target = {box->x1, box->y1, box->x2, box->y2,
                                         box->class_id};
                float loss = 0.0f;
                kshira_status step_status;
                if (box->target_weight > 0.0f && box->target_weight < 1.0f) {
                    /* Cached teacher boxes are trusted as assignment hints,
                     * not as representation labels. Keep their update
                     * head-only so uncertain pseudo boxes cannot consume the
                     * expensive encoder budget reserved for ground truth. */
                    kshira_rad_train_config teacher_cfg = {
                        kshira_rad_bits(model->kshira.rad),
                        KSHIRA_UPDATE_CHANNELS,
                        &model->kshira.channel_mask,
                        epoch_lr * target_weight,
                        1,
                        config->quality_aligned_assignment
                    };
                    step_status = kshira_rad_train_step(
                        model->kshira.rad, &image, &target, &teacher_cfg, &loss);
                } else if (config->mode == DET_TRAIN_ODT) {
                    step_status = kshira_session_multiscale_step(
                        &model->kshira, &image, &target,
                        det_kshira_target_level(model, box),
                        epoch_lr * target_weight, &loss);
                } else {
                    step_status = kshira_session_step(&model->kshira, &image, &target,
                                                      epoch_lr * target_weight, &loss);
                }
                if (step_status != KSHIRA_OK) return det_status_from_kshira(step_status);
                sample_loss += loss;
                ++updates;
                ++sample_terms;
                /* The shared head is first warmed on the native P3 assignment.
                 * Feeding P4/P5 positives from sample zero lets their broader,
                 * less calibrated features dominate the same rows and was
                 * observed to collapse the score ordering.  Keep the legacy
                 * schedule unchanged; the context profile unlocks its
                 * training-only scale path only after the P3 warm-up. */
                int multiscale_aux_active =
                    config->multiscale_aux && config->mode != DET_TRAIN_ODT &&
                    (!model->spec.context_fusion ||
                     (budget.active ? kshira_budget_stage(&budget, now_ms()) >= 2 :
                                      epoch >= 3));
                if (multiscale_aux_active) {
                    kshira_rad_train_config scale_cfg = {
                        kshira_rad_bits(model->kshira.rad),
                        KSHIRA_UPDATE_CHANNELS,
                        &model->kshira.channel_mask,
                        epoch_lr * (model->spec.context_fusion ? 0.10f : 0.25f),
                        config->dense_aux_budget,
                        config->quality_aligned_assignment
                    };
                    int level_begin = 1;
                    int level_end = 3;
                    if (model->spec.context_fusion) {
                        level_begin = det_kshira_target_level(model, box);
                        level_end = level_begin + 1;
                    }
                    for (int level = level_begin; level < level_end; ++level) {
                        float scale_loss = 0.0f;
                        kshira_status scale_status = kshira_rad_train_multiscale_step(
                            model->kshira.rad, &image, &target, level,
                            &scale_cfg, &scale_loss);
                        if (scale_status != KSHIRA_OK) {
                            return det_status_from_kshira(scale_status);
                        }
                        sample_loss += scale_loss;
                        ++updates;
                        ++sample_terms;
                    }
                }
            }
            if (sample.box_count > 0) {
                /* Two-level HNM + surgical FP ranking:
                 * Level A: head-only on mid/high score BG (score-gated).
                 * Level B: top-1 FULL encoder tile (late epochs only).
                 * Ranking pairs: GT centres vs hardest negs (early + late).
                 * Surgical path only tweaks winning-class head weights, so
                 * more epochs cost ~same wall-clock as denser all-class BG. */
                kshira_image_f32 image = {
                    sample.image.data, sample.image.channels,
                    sample.image.height, sample.image.width
                };
                enum { DET_KSHIRA_HNM_HEAD = 6, DET_KSHIRA_HNM_ENC = 1,
                       DET_KSHIRA_HNM_CANDIDATES = 64,
                       DET_KSHIRA_RANK_PAIRS = 4 };
                int hard_xy[DET_KSHIRA_HNM_HEAD][2];
                size_t candidate_cells[DET_KSHIRA_HNM_CANDIDATES];
                float candidate_scores[DET_KSHIRA_HNM_CANDIDATES];
                int hard_count = 0;
                int map_width = kshira_rad_map_width(model->kshira.rad);
                int map_height = kshira_rad_map_height(model->kshira.rad);
                /* Staged head HNM (1→2→3) + late encoder hard-neg. With a
                 * wall budget, stage transitions use measured elapsed time;
                 * the epoch schedule remains the zero-budget fallback. */
                int schedule_stage = budget.active ?
                    kshira_budget_stage(&budget, now_ms()) :
                    (epoch < 1 ? 0 : (epoch < 2 ? 1 : 2));
                int head_budget = schedule_stage == 0 ? 1 :
                                  (schedule_stage == 1 ? 2 : 3);
                if (config->residual_budget) head_budget *= 2;
                int enc_budget = schedule_stage >= 2 ? DET_KSHIRA_HNM_ENC : 0;
                if (config->residual_budget) {
                    int candidate_count = 0;
                    if (kshira_rad_hard_negative_candidates(
                            model->kshira.rad, &image, candidate_cells,
                            candidate_scores, DET_KSHIRA_HNM_CANDIDATES,
                            &candidate_count) != KSHIRA_OK) {
                        return DET_ERR_SHAPE;
                    }
                    for (int candidate = 0; candidate < candidate_count; ++candidate) {
                        size_t probe_cell = candidate_cells[candidate];
                        float probe_score = candidate_scores[candidate];
                        int probe_x;
                        int probe_y;
                        int too_close = 0;
                        if (map_width <= 0 || map_height <= 0 || probe_score < 0.15f ||
                            det_kshira_cell_inside_sample(model, &sample, probe_cell)) {
                            continue;
                        }
                        probe_x = (int)(probe_cell % (size_t)map_width);
                        probe_y = (int)(probe_cell / (size_t)map_width);
                        for (int i = 0; i < hard_count; ++i) {
                            int dx = probe_x - hard_xy[i][0];
                            int dy = probe_y - hard_xy[i][1];
                            if (dx * dx + dy * dy <= 4) {
                                too_close = 1;
                                break;
                            }
                        }
                        if (too_close) continue;
                        if (hard_count >= head_budget) break;
                        hard_xy[hard_count][0] = probe_x;
                        hard_xy[hard_count][1] = probe_y;
                        ++hard_count;
                    }
                } else {
                    for (int probe = 0; probe < 20; ++probe) {
                        size_t cell_index;
                        float quality = 0.0f;
                        int cell_x;
                        int cell_y;
                        int too_close = 0;
                        if (map_width <= 0 || map_height <= 0 ||
                            !det_kshira_background_cell(
                                model, &sample,
                                model->train_updates + updates +
                                    (size_t)probe * 110U + (size_t)epoch * 31U,
                                &cell_index)) {
                            continue;
                        }
                        cell_x = (int)(cell_index % (size_t)map_width);
                        cell_y = (int)(cell_index / (size_t)map_width);
                        for (int i = 0; i < hard_count; ++i) {
                            int dx = cell_x - hard_xy[i][0];
                            int dy = cell_y - hard_xy[i][1];
                            if (dx * dx + dy * dy <= 4) {
                                too_close = 1;
                                break;
                            }
                        }
                        if (too_close || hard_count >= head_budget) continue;
                        if (kshira_rad_objectness_at(model->kshira.rad, &image,
                                                     cell_y, cell_x,
                                                     &quality) != KSHIRA_OK) {
                            return DET_ERR_SHAPE;
                        }
                        if (quality < 0.15f) continue;
                        hard_xy[hard_count][0] = cell_x;
                        hard_xy[hard_count][1] = cell_y;
                        ++hard_count;
                    }
                }
                for (int background = 0; background < hard_count; ++background) {
                    int background_index = config->residual_budget ?
                                           hard_count - 1 - background : background;
                    float loss = 0.0f;
                    kshira_status step_status;
                    kshira_rad_train_config bg_cfg;
                    kshira_bit_mode bits = kshira_rad_bits(model->kshira.rad);
                    if (model->train_updates > SIZE_MAX - updates) {
                        return DET_ERR_SHAPE;
                    }
                    bg_cfg.bits = bits;
                    /* Level B: top enc_budget use FULL (encoder tile); rest head. */
                    bg_cfg.update_mode = (background_index < enc_budget) ?
                                         KSHIRA_UPDATE_FULL : KSHIRA_UPDATE_CHANNELS;
                    bg_cfg.channel_mask = &model->kshira.channel_mask;
                    bg_cfg.learning_rate = epoch_lr;
                    if (config->residual_budget &&
                        bg_cfg.update_mode == KSHIRA_UPDATE_CHANNELS) {
                        step_status = kshira_rad_train_cached_background_step(
                            model->kshira.rad, &image, hard_xy[background_index][1],
                            hard_xy[background_index][0], &bg_cfg, &loss);
                    } else {
                        step_status = kshira_rad_train_background_step(
                            model->kshira.rad, &image, hard_xy[background_index][1],
                            hard_xy[background_index][0], &bg_cfg, &loss);
                    }
                    if (step_status != KSHIRA_OK) {
                        return det_status_from_kshira(step_status);
                    }
                    sample_loss += loss;
                    ++updates;
                    ++sample_terms;
                    if (background_index == 0 && model->spec.context_fusion &&
                        model->spec.shared_multiscale_head) {
                        for (int level = 1; level < 3; ++level) {
                            int scale = 1 << level;
                            float scale_loss = 0.0f;
                            kshira_status scale_status =
                                kshira_rad_train_multiscale_background_step(
                                    model->kshira.rad, &image, level,
                                    hard_xy[background_index][1] / scale,
                                    hard_xy[background_index][0] / scale,
                                    &bg_cfg, &scale_loss);
                            if (scale_status != KSHIRA_OK) {
                                return det_status_from_kshira(scale_status);
                            }
                            sample_loss += scale_loss;
                            ++sample_terms;
                        }
                    }
                }
                /* Ranking after warm-up (epoch≥2). Head-only surgical pairs
                 * stay cheap; INT8 ep5 still ranks on epochs 2..4. */
                if (hard_count > 0 && epoch >= 2) {
                    kshira_rad_train_config rank_cfg;
                    int rank_done = 0;
                    int neg_slots = hard_count < 2 ? hard_count : 2;
                    rank_cfg.bits = kshira_rad_bits(model->kshira.rad);
                    rank_cfg.update_mode = KSHIRA_UPDATE_CHANNELS;
                    rank_cfg.channel_mask = &model->kshira.channel_mask;
                    rank_cfg.learning_rate = epoch_lr * 0.30f;
                    for (int box_index = 0;
                         box_index < sample.box_count &&
                         rank_done < DET_KSHIRA_RANK_PAIRS;
                         ++box_index) {
                        const det_box *box = &sample.boxes[box_index];
                        if (box->target_weight > 0.0f && box->target_weight < 1.0f) {
                            continue;
                        }
                        int pos_x = (int)(((box->x1 + box->x2) * 0.5f) / 4.0f);
                        int pos_y = (int)(((box->y1 + box->y2) * 0.5f) / 4.0f);
                        if (map_width <= 0 || map_height <= 0) break;
                        if (pos_x < 0) pos_x = 0;
                        if (pos_y < 0) pos_y = 0;
                        if (pos_x >= map_width) pos_x = map_width - 1;
                        if (pos_y >= map_height) pos_y = map_height - 1;
                        for (int ni = 0; ni < neg_slots &&
                             rank_done < DET_KSHIRA_RANK_PAIRS; ++ni) {
                            float rloss = 0.0f;
                            kshira_status rs = kshira_rad_train_rank_pair(
                                model->kshira.rad, &image, pos_y, pos_x,
                                box->class_id, hard_xy[ni][1], hard_xy[ni][0],
                                0.12f, &rank_cfg, &rloss);
                            if (rs != KSHIRA_OK) {
                                return det_status_from_kshira(rs);
                            }
                            sample_loss += rloss;
                            ++sample_terms;
                            ++rank_done;
                        }
                    }
                }
            }
            if (sample_terms > 0) sample_loss /= (float)sample_terms;
            loss_sum += sample_loss;
            ++seen;
            ++total_seen;
            ++report->samples_seen;
            {
                double sample_now = now_ms();
                kshira_budget_observe(&budget, sample_now);
                if (kshira_budget_should_stop(&budget, sample_now) ||
                    (deadline > 0.0 && sample_now >= deadline)) {
                    budget_stop = 1;
                    break;
                }
            }
        }
        if (seen == 0U && !budget_stop) return DET_ERR_ARGUMENT;
        if (total_cap > 0U && total_seen >= total_cap) break;
        if (budget_stop) break;
    }
    model->train_updates += updates;
    report->updates = updates;
    report->elapsed_ms = now_ms() - start;
    report->mean_loss = report->samples_seen == 0U ? 0.0f :
                        loss_sum / (float)report->samples_seen;
    report->used_global_backward = 0;
    report->stopped_by_budget = budget_stop;
    if (config->analytic_readout && config->precision == DET_PRECISION_F32) {
        status = det_kshira_readout(model, dataset,
                                    start + config->time_budget_ms, report);
        if (status != DET_OK) return status;
    }
    report->elapsed_ms = now_ms() - start;
    return DET_OK;
}

static det_status det_native_graph_predict(det_model *model, const det_image *image,
                                          float score_threshold,
                                          det_detection *detections, int capacity,
                                          int *count) {
    NGModelConfig config;
    NGModelWeights weights = {0};
    NGModelOutput raw;
    NGTensor input;
    NGArena arena;
    Y8DetectPostConfig post;
    Y8DetectResult result;
    size_t pixels;
    float scale;
    if (model == NULL || image == NULL || detections == NULL || count == NULL ||
        !model->native_graph_enabled || model->native_graph_input == NULL ||
        model->native_graph_arena == NULL || image->channels != model->spec.channels ||
        image->width != model->spec.width || image->height != model->spec.height ||
        image->data == NULL ||
        !checked_size3(image->channels, image->height, image->width, &pixels)) {
        return DET_ERR_ARGUMENT;
    }
    if (capacity <= 0) {
        *count = 0;
        return DET_OK;
    }
    if (model->native_graph_architecture == NG_GRAPH_MULTISCALE) {
        NGMDetection native[300];
        det_detection ranked[300];
        int32_t native_count = 0;
        int native_capacity = 300;
        int ranked_count = 0;
        scale = model->native_graph_weights.act_scale > 0.0f ?
                model->native_graph_weights.act_scale : 1.0f / 127.0f;
        for (int c = 0; c < model->native_graph_weights.input_channels; ++c) {
            int source_channel = c < image->channels ? c : 0;
            for (int y = 0; y < image->height; ++y) {
                for (int x = 0; x < image->width; ++x) {
                    size_t source = ((size_t)source_channel * (size_t)image->height +
                                     (size_t)y) * (size_t)image->width + (size_t)x;
                    float value = image->data[source];
                    int quantized;
                    if (value < 0.0f) value = 0.0f;
                    if (value > 1.0f) value = 1.0f;
                    if (model->native_graph_input_f32 != NULL) {
                        model->native_graph_input_f32[((size_t)c * (size_t)image->height +
                                                      (size_t)y) * (size_t)image->width +
                                                     (size_t)x] = value;
                    }
                    quantized = (int)(value / scale + 0.5f);
                    if (quantized > 127) quantized = 127;
                    model->native_graph_input[((size_t)c * (size_t)image->height +
                                              (size_t)y) * (size_t)image->width +
                                             (size_t)x] = (int8_t)quantized;
                }
            }
        }
        ng_arena_init(&arena, model->native_graph_arena,
                      (int32_t)model->native_graph_arena_bytes);
        if ((model->precision == DET_PRECISION_F32 &&
             model->native_graph_input_f32 != NULL &&
             model->native_graph_weights.weight_f32_blob != NULL &&
             model->native_graph_weights.bias_f32_blob != NULL) ?
            ngm_detect_f32(&model->native_graph_weights, model->native_graph_input_f32,
                           model->native_graph_weights.input_channels,
                           image->height, image->width, 0.0f, native_capacity,
                           &arena, native, &native_count) :
            ngm_detect_s8(&model->native_graph_weights, model->native_graph_input,
                          model->native_graph_weights.input_channels,
                          image->height, image->width, 0.0f, native_capacity,
                          &arena, native, &native_count)) {
            return DET_ERR_SHAPE;
        }
        if ((size_t)arena.used > model->native_graph_arena_high_water) {
            model->native_graph_arena_high_water = (size_t)arena.used;
        }
        for (int32_t i = 0; i < native_count; ++i) {
            float raw_score = native[i].score;
            float calibrated = det_native_calibrated_score(
                model, raw_score, native[i].class_id, native[i].x1, native[i].y1,
                native[i].x2, native[i].y2, image->width, image->height);
            if (calibrated < score_threshold) continue;
            {
                float x1 = fmaxf(0.0f, fminf((float)image->width, native[i].x1));
                float y1 = fmaxf(0.0f, fminf((float)image->height, native[i].y1));
                float x2 = fmaxf(0.0f, fminf((float)image->width, native[i].x2));
                float y2 = fmaxf(0.0f, fminf((float)image->height, native[i].y2));
                int scale_bin = det_native_scale_bin(x1, y1, x2, y2,
                                                     image->width, image->height);
                det_detection candidate;
                det_native_adjust_box(model, scale_bin, image->width, image->height,
                                      &x1, &y1, &x2, &y2);
                det_native_adjust_class_box(model, scale_bin, native[i].class_id,
                                            image->width, image->height,
                                            &x1, &y1, &x2, &y2);
                candidate = (det_detection){{x1, y1, x2, y2,
                                             native[i].class_id, 0.0f}, calibrated};
                {
                    int position = ranked_count < 300 ? ranked_count : 299;
                    if (ranked_count < 300) ++ranked_count;
                    else if (candidate.score <= ranked[299].score) continue;
                    while (position > 0 && ranked[position - 1].score < candidate.score) {
                        ranked[position] = ranked[position - 1];
                        --position;
                    }
                    ranked[position] = candidate;
                }
            }
        }
        *count = 0;
        for (int i = 0; i < ranked_count && *count < capacity; ++i) {
            int suppressed = 0;
            for (int j = 0; j < *count; ++j) {
                if (detections[j].box.class_id == ranked[i].box.class_id &&
                    det_native_iou(&detections[j].box, &ranked[i].box) > 0.45f) {
                    suppressed = 1;
                    break;
                }
            }
            if (!suppressed) detections[(*count)++] = ranked[i];
        }
        (void)pixels;
        return DET_OK;
    }
    if (model->precision == DET_PRECISION_F32 &&
        model->native_graph_input_f32 != NULL &&
        model->native_graph_weights.weight_f32_blob != NULL &&
        model->native_graph_weights.bias_f32_blob != NULL) {
        return det_native_graph_predict_f32(model, image, score_threshold,
                                            detections, capacity, count);
    }
    scale = model->native_graph_weights.act_scale > 0.0f ?
            model->native_graph_weights.act_scale : 1.0f / 127.0f;
    for (int c = 0; c < model->native_graph_weights.input_channels; ++c) {
        int source_channel = c < image->channels ? c : 0;
        for (int y = 0; y < image->height; ++y) {
            for (int x = 0; x < image->width; ++x) {
                size_t source = ((size_t)source_channel * (size_t)image->height +
                                 (size_t)y) * (size_t)image->width + (size_t)x;
                float value = image->data[source];
                int quantized;
                if (value < 0.0f) value = 0.0f;
                if (value > 1.0f) value = 1.0f;
                quantized = (int)(value / scale + 0.5f);
                if (quantized > 127) quantized = 127;
                model->native_graph_input[((size_t)c * (size_t)image->height +
                                          (size_t)y) * (size_t)image->width +
                                         (size_t)x] = (int8_t)quantized;
            }
        }
    }
    config = (NGModelConfig){model->native_graph_weights.scale,
                              model->native_graph_weights.input_channels,
                              model->native_graph_weights.class_count};
    weights.convs = model->native_graph_weights.convs;
    weights.conv_count = model->native_graph_weights.conv_count;
    input = (NGTensor){model->native_graph_input, 1, config.input_channels,
                       image->height, image->width};
    ng_arena_init(&arena, model->native_graph_arena,
                  (int32_t)model->native_graph_arena_bytes);
    if (ng_native_graph_forward_s8(&config, &weights, &input, &arena, &raw) != 0) {
        return DET_ERR_SHAPE;
    }
    det_native_cache_features_s8(model, &raw);
    if ((size_t)arena.used > model->native_graph_arena_high_water) {
        model->native_graph_arena_high_water = (size_t)arena.used;
    }
    post = (Y8DetectPostConfig){0.0f, 0.45f, scale,
                                NG_MAX_CANDIDATES,
                                capacity < NG_MAX_KEEP ? capacity : NG_MAX_KEEP,
                                &model->native_graph_weights.dfl};
    if (ng_detect_post_s8(&raw, config.class_count, &post, &result) != 0) {
        return DET_ERR_FORMAT;
    }
    *count = 0;
    for (int i = 0; i < result.count; ++i) {
        float x1 = (float)result.dets[i].x1_q12 / 4096.0f;
        float y1 = (float)result.dets[i].y1_q12 / 4096.0f;
        float x2 = (float)result.dets[i].x2_q12 / 4096.0f;
        float y2 = (float)result.dets[i].y2_q12 / 4096.0f;
        float raw_score = (float)result.dets[i].score_q15 / 32767.0f;
        int scale_bin = det_native_scale_bin(x1, y1, x2, y2,
                                             image->width, image->height);
        det_native_adjust_box(model, scale_bin, image->width, image->height,
                              &x1, &y1, &x2, &y2);
        float calibrated = det_native_calibrated_score(
            model, raw_score, result.dets[i].class_id, x1, y1, x2, y2,
            image->width, image->height);
        det_native_adjust_class_box(model, scale_bin, result.dets[i].class_id,
                                    image->width, image->height,
                                    &x1, &y1, &x2, &y2);
        if (calibrated < score_threshold || *count >= capacity) continue;
        int output = *count;
        detections[output].box.x1 = fmaxf(0.0f, fminf((float)image->width, x1));
        detections[output].box.y1 = fmaxf(0.0f, fminf((float)image->height, y1));
        detections[output].box.x2 = fmaxf(0.0f, fminf((float)image->width, x2));
        detections[output].box.y2 = fmaxf(0.0f, fminf((float)image->height, y2));
        detections[output].box.class_id = result.dets[i].class_id;
        detections[output].box.target_weight = 0.0f;
        detections[output].score = calibrated;
        *count = output + 1;
    }
    (void)pixels;
    return DET_OK;
}

static float det_native_iou(const det_box *left, const det_box *right) {
    float x1 = fmaxf(left->x1, right->x1);
    float y1 = fmaxf(left->y1, right->y1);
    float x2 = fminf(left->x2, right->x2);
    float y2 = fminf(left->y2, right->y2);
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_left = fmaxf(0.0f, left->x2 - left->x1) *
                      fmaxf(0.0f, left->y2 - left->y1);
    float area_right = fmaxf(0.0f, right->x2 - right->x1) *
                       fmaxf(0.0f, right->y2 - right->y1);
    float union_area = area_left + area_right - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static det_status det_native_graph_predict_f32(det_model *model,
                                               const det_image *image,
                                               float score_threshold,
                                               det_detection *detections,
                                               int capacity, int *count) {
    NGModelConfig config;
    NGModelWeights weights = {0};
    NGF32ModelOutput raw;
    Y8F32DetectPostConfig post;
    Y8F32DetectResult result;
    NGArena arena;
    if (model == NULL || image == NULL || detections == NULL || count == NULL ||
        model->native_graph_input_f32 == NULL ||
        model->native_graph_weights.weight_f32_blob == NULL ||
        model->native_graph_weights.bias_f32_blob == NULL || image->data == NULL ||
        image->channels != model->spec.channels || image->width != model->spec.width ||
        image->height != model->spec.height) return DET_ERR_ARGUMENT;
    for (int c = 0; c < model->native_graph_weights.input_channels; ++c) {
        int source_channel = c < image->channels ? c : 0;
        for (int y = 0; y < image->height; ++y) {
            for (int x = 0; x < image->width; ++x) {
                size_t source = ((size_t)source_channel * (size_t)image->height +
                                 (size_t)y) * (size_t)image->width + (size_t)x;
                float value = image->data[source];
                if (value < 0.0f) value = 0.0f;
                if (value > 1.0f) value = 1.0f;
                model->native_graph_input_f32[((size_t)c * (size_t)image->height +
                                               (size_t)y) * (size_t)image->width +
                                              (size_t)x] = value;
            }
        }
    }
    config = (NGModelConfig){model->native_graph_weights.scale,
                              model->native_graph_weights.input_channels,
                              model->native_graph_weights.class_count};
    weights.convs = model->native_graph_weights.convs;
    weights.conv_count = model->native_graph_weights.conv_count;
    ng_arena_init(&arena, model->native_graph_arena,
                  (int32_t)model->native_graph_arena_bytes);
    if (ng_native_graph_forward_f32(&config, &weights, model->native_graph_input_f32,
                                    config.input_channels, image->height, image->width,
                                    &arena, &raw) != 0) return DET_ERR_SHAPE;
    det_native_cache_features_f32(model, &raw);
    if ((size_t)arena.used > model->native_graph_arena_high_water)
        model->native_graph_arena_high_water = (size_t)arena.used;
    post = (Y8F32DetectPostConfig){score_threshold, 0.45f,
                                   NG_MAX_CANDIDATES,
                                   capacity < NG_MAX_KEEP ? capacity : NG_MAX_KEEP};
    if (ng_detect_post_f32(&raw, config.class_count, &post, &result) != 0)
        return DET_ERR_FORMAT;
    *count = 0;
    for (int i = 0; i < result.count && *count < capacity; ++i) {
        float x1 = result.dets[i].x1;
        float y1 = result.dets[i].y1;
        float x2 = result.dets[i].x2;
        float y2 = result.dets[i].y2;
        int scale_bin = det_native_scale_bin(x1, y1, x2, y2,
                                             image->width, image->height);
        float calibrated;
        det_native_adjust_box(model, scale_bin, image->width, image->height,
                              &x1, &y1, &x2, &y2);
        calibrated = det_native_calibrated_score(
            model, result.dets[i].score, result.dets[i].class_id,
            x1, y1, x2, y2, image->width, image->height);
        det_native_adjust_class_box(model, scale_bin, result.dets[i].class_id,
                                    image->width, image->height,
                                    &x1, &y1, &x2, &y2);
        if (calibrated < score_threshold) continue;
        detections[*count].box.x1 = fmaxf(0.0f, fminf((float)image->width, x1));
        detections[*count].box.y1 = fmaxf(0.0f, fminf((float)image->height, y1));
        detections[*count].box.x2 = fmaxf(0.0f, fminf((float)image->width, x2));
        detections[*count].box.y2 = fmaxf(0.0f, fminf((float)image->height, y2));
        detections[*count].box.class_id = result.dets[i].class_id;
        detections[*count].box.target_weight = 0.0f;
        detections[*count].score = calibrated;
        ++*count;
    }
    return DET_OK;
}

static int det_native_scale_bin(float x1, float y1, float x2, float y2,
                                int width, int height) {
    float area = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float image_area = (float)width * (float)height;
    float fraction = image_area > 0.0f ? area / image_area : 0.0f;
    if (fraction < 0.04f) return 0;
    if (fraction < 0.16f) return 1;
    return 2;
}

static float det_native_calibrated_score(const det_model *model, float raw_score,
                                         int class_id, float x1, float y1,
                                         float x2, float y2, int width, int height) {
    float raw_logit = logf(fmaxf(1.0e-5f, raw_score) /
                           fmaxf(1.0e-5f, 1.0f - raw_score));
    int scale_bin = det_native_scale_bin(x1, y1, x2, y2, width, height);
    float class_bias = class_id >= 0 && class_id < DET_MAX_CLASSES ?
                       model->native_class_bias[class_id] : 0.0f;
    return det_sigmoid(model->native_score_gain * raw_logit +
                       model->native_score_bias + class_bias +
                       model->native_scale_bias[scale_bin]);
}

static void det_native_adjust_box(const det_model *model, int scale_bin,
                                  int width, int height, float *x1, float *y1,
                                  float *x2, float *y2) {
    float center_x;
    float center_y;
    float box_width;
    float box_height;
    float width_scale;
    float height_scale;
    if (model == NULL || x1 == NULL || y1 == NULL || x2 == NULL || y2 == NULL ||
        scale_bin < 0 || scale_bin >= DET_MAX_SCALES) return;
    center_x = 0.5f * (*x1 + *x2) + model->native_box_bias[scale_bin][0] * (float)width;
    center_y = 0.5f * (*y1 + *y2) + model->native_box_bias[scale_bin][1] * (float)height;
    box_width = fmaxf(1.0f, *x2 - *x1);
    box_height = fmaxf(1.0f, *y2 - *y1);
    width_scale = expf(fmaxf(-0.5f, fminf(0.5f, model->native_box_bias[scale_bin][2])));
    height_scale = expf(fmaxf(-0.5f, fminf(0.5f, model->native_box_bias[scale_bin][3])));
    box_width *= width_scale;
    box_height *= height_scale;
    *x1 = fmaxf(0.0f, fminf((float)width, center_x - 0.5f * box_width));
    *y1 = fmaxf(0.0f, fminf((float)height, center_y - 0.5f * box_height));
    *x2 = fmaxf(0.0f, fminf((float)width, center_x + 0.5f * box_width));
    *y2 = fmaxf(0.0f, fminf((float)height, center_y + 0.5f * box_height));
}

static void det_native_adjust_class_box(const det_model *model, int scale_bin,
                                        int class_id, int width, int height,
                                        float *x1, float *y1, float *x2, float *y2) {
    float center_x;
    float center_y;
    float box_width;
    float box_height;
    float width_scale;
    float height_scale;
    if (model == NULL || x1 == NULL || y1 == NULL || x2 == NULL || y2 == NULL ||
        scale_bin < 0 || scale_bin >= DET_MAX_SCALES || class_id < 0 ||
        class_id >= DET_MAX_CLASSES) return;
    center_x = 0.5f * (*x1 + *x2) +
               model->native_class_box_bias[scale_bin][class_id][0] * (float)width;
    center_y = 0.5f * (*y1 + *y2) +
               model->native_class_box_bias[scale_bin][class_id][1] * (float)height;
    box_width = fmaxf(1.0f, *x2 - *x1);
    box_height = fmaxf(1.0f, *y2 - *y1);
    width_scale = expf(fmaxf(-0.5f, fminf(0.5f,
        model->native_class_box_bias[scale_bin][class_id][2])));
    height_scale = expf(fmaxf(-0.5f, fminf(0.5f,
        model->native_class_box_bias[scale_bin][class_id][3])));
    box_width *= width_scale;
    box_height *= height_scale;
    *x1 = fmaxf(0.0f, fminf((float)width, center_x - 0.5f * box_width));
    *y1 = fmaxf(0.0f, fminf((float)height, center_y - 0.5f * box_height));
    *x2 = fmaxf(0.0f, fminf((float)width, center_x + 0.5f * box_width));
    *y2 = fmaxf(0.0f, fminf((float)height, center_y + 0.5f * box_height));
}

static NGConv2D *det_native_head_conv(det_model *model, int scale_bin) {
    int seen = 0;
    if (model == NULL || scale_bin < 0 || scale_bin >= DET_MAX_SCALES ||
        model->native_graph_weights.class_count <= 0) return NULL;
    for (int i = 0; i < model->native_graph_weights.conv_count; ++i) {
        NGConv2D *conv = &model->native_graph_weights.convs[i];
        if (conv->out_channels == model->native_graph_weights.class_count &&
            conv->kernel == 1 && conv->stride == 1) {
            if (seen++ == scale_bin) return conv;
        }
    }
    return NULL;
}

static int det_native_update_candidate_head(det_model *model, int scale_bin,
                                            int class_id, int cell_x, int cell_y,
                                            float target, float target_weight,
                                            float rate) {
    NGConv2D *conv;
    const float *features;
    float logit;
    float error;
    int f32;
    size_t cell;
    size_t plane;
    int channels;
    int height;
    int width;
    if (model == NULL || !model->native_graph_enabled ||
        model->native_graph_architecture == NG_GRAPH_MULTISCALE ||
        (model->precision != DET_PRECISION_F32 &&
         model->precision != DET_PRECISION_INT8) || scale_bin < 0 ||
        scale_bin >= DET_MAX_SCALES || class_id < 0 ||
        class_id >= model->native_graph_weights.class_count ||
        class_id >= DET_MAX_CLASSES || !isfinite(target) ||
        !isfinite(target_weight) || !isfinite(rate) || rate <= 0.0f) return 0;
    conv = det_native_head_conv(model, scale_bin);
    features = model->native_feature_cache[scale_bin];
    channels = model->native_feature_channels[scale_bin];
    height = model->native_feature_height[scale_bin];
    width = model->native_feature_width[scale_bin];
    f32 = model->precision == DET_PRECISION_F32;
    if (conv == NULL || features == NULL || channels <= 0 || height <= 0 ||
        width <= 0 || cell_x < 0 || cell_x >= width || cell_y < 0 ||
        cell_y >= height || conv->in_channels != channels) return 0;
    cell = (size_t)cell_y * (size_t)width + (size_t)cell_x;
    plane = (size_t)height * (size_t)width;
    if (f32) {
        const float *weights = conv->weights_f32;
        const float *bias = conv->bias_f32;
        if (weights == NULL || bias == NULL) return 0;
        logit = bias[class_id];
        for (int channel = 0; channel < channels; ++channel)
            logit += weights[(size_t)class_id * (size_t)channels +
                             (size_t)channel] *
                     features[(size_t)channel * plane + cell];
    } else {
        const int8_t *weights;
        float output_scale;
        if (conv->weight_quant != NG_QUANT_INT8 || conv->weights == NULL ||
            conv->bias == NULL || conv->multiplier == NULL ||
            conv->shift == NULL || model->native_head_weight_accum[scale_bin] == NULL ||
            model->native_head_weight_accum_count[scale_bin] <
                (size_t)model->native_graph_weights.class_count *
                (size_t)channels) return 0;
        weights = (const int8_t *)conv->weights;
        output_scale = conv->output_scale > 0.0f ? conv->output_scale :
                       (model->native_graph_weights.act_scale > 0.0f ?
                        model->native_graph_weights.act_scale : 1.0f / 127.0f);
        logit = (float)conv->bias[class_id] * output_scale;
        for (int channel = 0; channel < channels; ++channel)
            logit += (float)weights[(size_t)class_id * (size_t)channels +
                                    (size_t)channel] *
                     features[(size_t)channel * plane + cell] * output_scale;
    }
    error = (det_sigmoid(logit) - target) * target_weight;
    if (!isfinite(error)) return 0;
    if (f32) {
        float *weights = (float *)(uintptr_t)conv->weights_f32;
        float *bias = (float *)(uintptr_t)conv->bias_f32;
        for (int channel = 0; channel < channels; ++channel) {
            size_t index = (size_t)class_id * (size_t)channels +
                           (size_t)channel;
            weights[index] -= rate * error * features[(size_t)channel * plane + cell];
        }
        bias[class_id] -= rate * error;
    } else {
        int8_t *weights = (int8_t *)(uintptr_t)conv->weights;
        float *accum = model->native_head_weight_accum[scale_bin];
        float output_scale = conv->output_scale > 0.0f ? conv->output_scale :
                             (model->native_graph_weights.act_scale > 0.0f ?
                              model->native_graph_weights.act_scale : 1.0f / 127.0f);
        float input_scale = conv->input_scale > 0.0f ? conv->input_scale :
                            (model->native_graph_weights.act_scale > 0.0f ?
                             model->native_graph_weights.act_scale : 1.0f / 127.0f);
        float weight_scale = ldexpf((float)conv->multiplier[class_id],
                                    -(int)conv->shift[class_id]) *
                             output_scale / input_scale;
        float *sidecar = conv->weights_f32 != NULL ?
                         (float *)(uintptr_t)conv->weights_f32 : NULL;
        if (!(weight_scale > 0.0f) || !isfinite(weight_scale)) return 0;
        for (int channel = 0; channel < channels; ++channel) {
            size_t index = (size_t)class_id * (size_t)channels +
                           (size_t)channel;
            int64_t whole;
            float delta = -8.0f * rate * error *
                          features[(size_t)channel * plane + cell] * output_scale;
            accum[index] += delta;
            whole = accum[index] >= 0.0f ? (int64_t)floorf(accum[index]) :
                                          (int64_t)ceilf(accum[index]);
            if (whole != 0) {
                int value = (int)weights[index] +
                            (whole > INT_MAX ? INT_MAX :
                             (whole < INT_MIN ? INT_MIN : (int)whole));
                if (value > 127) value = 127;
                if (value < -127) value = -127;
                weights[index] = (int8_t)value;
                if (sidecar != NULL) sidecar[index] = (float)value * weight_scale;
                accum[index] -= (float)whole;
            }
        }
    }
    return 1;
}

static void det_native_update_head_bias(det_model *model, int scale_bin,
                                        int class_id, float delta_logit) {
    NGConv2D *conv;
    float output_scale;
    double accumulator_delta;
    int64_t whole;
    int32_t updated_bias;
    if (model == NULL || class_id < 0 ||
        class_id >= model->native_graph_weights.class_count ||
        class_id >= DET_MAX_CLASSES || !isfinite(delta_logit) ||
        delta_logit == 0.0f) return;
    conv = det_native_head_conv(model, scale_bin);
    if (conv == NULL) return;
    if (conv->bias_f32 != NULL)
        ((float *)(uintptr_t)conv->bias_f32)[class_id] += delta_logit;
    if (conv->bias == NULL || conv->multiplier == NULL || conv->shift == NULL) return;
    output_scale = model->native_graph_architecture == NG_GRAPH_MULTISCALE ?
                   (conv->output_scale > 0.0f ? conv->output_scale : 0.0625f) :
                   (model->native_graph_weights.act_scale > 0.0f ?
                    model->native_graph_weights.act_scale : 1.0f / 127.0f);
    accumulator_delta = ((double)delta_logit / (double)output_scale) *
                        ldexp(1.0, (int)conv->shift[class_id]) /
                        (double)conv->multiplier[class_id];
    model->native_head_accum[scale_bin][class_id] += (float)accumulator_delta;
    if (model->native_head_accum[scale_bin][class_id] >= 0.0f)
        whole = (int64_t)floorf(model->native_head_accum[scale_bin][class_id]);
    else
        whole = (int64_t)ceilf(model->native_head_accum[scale_bin][class_id]);
    if (whole == 0) return;
    updated_bias = (int32_t)conv->bias[class_id];
    if (whole > (int64_t)INT32_MAX - updated_bias) updated_bias = INT32_MAX;
    else if (whole < (int64_t)INT32_MIN - updated_bias) updated_bias = INT32_MIN;
    else updated_bias += (int32_t)whole;
    ((int32_t *)(uintptr_t)conv->bias)[class_id] = updated_bias;
    model->native_head_accum[scale_bin][class_id] -= (float)whole;
}

static det_status det_native_fit_readout(
    det_model *model, const det_native_readout_example *examples,
    size_t example_count, det_train_report *report) {
    enum { dimension = 2 + DET_MAX_CLASSES + DET_MAX_SCALES };
    float theta[dimension];
    float anchor[dimension];
    float features[dimension];
    float system[dimension][dimension + 1];
    double pre_objective = 0.0;
    double post_objective = 0.0;
    size_t positives = 0U;
    size_t negatives = 0U;
    const float regularization = 0.05f;
    if (model == NULL || examples == NULL || report == NULL || example_count == 0U) {
        return DET_ERR_ARGUMENT;
    }
    theta[0] = model->native_score_gain;
    theta[1] = model->native_score_bias;
    for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
        theta[2 + class_id] = model->native_class_bias[class_id];
    }
    for (int scale_bin = 0; scale_bin < DET_MAX_SCALES; ++scale_bin) {
        theta[2 + DET_MAX_CLASSES + scale_bin] = model->native_scale_bias[scale_bin];
    }
    memcpy(anchor, theta, sizeof(anchor));
    for (size_t i = 0U; i < example_count; ++i) {
        if (examples[i].target != 0U) ++positives;
        else ++negatives;
    }
    for (size_t i = 0U; i < example_count; ++i) {
        float logit;
        float probability;
        memset(features, 0, sizeof(features));
        features[0] = fmaxf(-12.0f, fminf(12.0f, examples[i].raw_logit));
        features[1] = 1.0f;
        if (examples[i].class_id < DET_MAX_CLASSES)
            features[2 + examples[i].class_id] = 1.0f;
        if (examples[i].scale_bin < DET_MAX_SCALES)
            features[2 + DET_MAX_CLASSES + examples[i].scale_bin] = 1.0f;
        logit = 0.0f;
        for (int j = 0; j < dimension; ++j) logit += theta[j] * features[j];
        probability = det_sigmoid(logit);
        pre_objective += -(examples[i].target != 0U ?
            logf(fmaxf(probability, 1.0e-5f)) :
            logf(fmaxf(1.0f - probability, 1.0e-5f)));
    }
    for (int iteration = 0; iteration < 6; ++iteration) {
        memset(system, 0, sizeof(system));
        for (int j = 0; j < dimension; ++j) {
            system[j][j] = regularization;
            system[j][dimension] = regularization * (theta[j] - anchor[j]);
        }
        for (size_t i = 0U; i < example_count; ++i) {
            float logit;
            float probability;
            float curvature;
            float residual;
            float sample_weight = examples[i].target != 0U ? 1.5f : 1.0f;
            memset(features, 0, sizeof(features));
            features[0] = fmaxf(-12.0f, fminf(12.0f, examples[i].raw_logit));
            features[1] = 1.0f;
            if (examples[i].class_id < DET_MAX_CLASSES)
                features[2 + examples[i].class_id] = 1.0f;
            if (examples[i].scale_bin < DET_MAX_SCALES)
                features[2 + DET_MAX_CLASSES + examples[i].scale_bin] = 1.0f;
            logit = 0.0f;
            for (int j = 0; j < dimension; ++j) logit += theta[j] * features[j];
            probability = det_sigmoid(logit);
            curvature = fmaxf(1.0e-4f, probability * (1.0f - probability)) *
                        sample_weight;
            residual = (probability - (examples[i].target != 0U ? 1.0f : 0.0f)) *
                       sample_weight;
            for (int row = 0; row < dimension; ++row) {
                system[row][dimension] += residual * features[row];
                for (int column = 0; column < dimension; ++column)
                    system[row][column] += curvature * features[row] * features[column];
            }
        }
        for (int pivot = 0; pivot < dimension; ++pivot) {
            int best = pivot;
            float magnitude = fabsf(system[pivot][pivot]);
            for (int row = pivot + 1; row < dimension; ++row) {
                float candidate = fabsf(system[row][pivot]);
                if (candidate > magnitude) {
                    magnitude = candidate;
                    best = row;
                }
            }
            if (magnitude < 1.0e-8f) continue;
            if (best != pivot) {
                for (int column = pivot; column <= dimension; ++column) {
                    float swap = system[pivot][column];
                    system[pivot][column] = system[best][column];
                    system[best][column] = swap;
                }
            }
            {
                float divisor = system[pivot][pivot];
                for (int column = pivot; column <= dimension; ++column)
                    system[pivot][column] /= divisor;
            }
            for (int row = 0; row < dimension; ++row) {
                if (row == pivot) continue;
                float factor = system[row][pivot];
                for (int column = pivot; column <= dimension; ++column)
                    system[row][column] -= factor * system[pivot][column];
            }
        }
        {
            float maximum_step = 0.0f;
            for (int j = 0; j < dimension; ++j) {
                float step = system[j][dimension];
                if (!isfinite(step)) step = 0.0f;
                if (step > 0.5f) step = 0.5f;
                if (step < -0.5f) step = -0.5f;
                theta[j] -= step;
                maximum_step = fmaxf(maximum_step, fabsf(step));
            }
            if (maximum_step < 1.0e-3f) break;
        }
    }
    for (size_t i = 0U; i < example_count; ++i) {
        float logit;
        float probability;
        memset(features, 0, sizeof(features));
        features[0] = fmaxf(-12.0f, fminf(12.0f, examples[i].raw_logit));
        features[1] = 1.0f;
        if (examples[i].class_id < DET_MAX_CLASSES)
            features[2 + examples[i].class_id] = 1.0f;
        if (examples[i].scale_bin < DET_MAX_SCALES)
            features[2 + DET_MAX_CLASSES + examples[i].scale_bin] = 1.0f;
        logit = 0.0f;
        for (int j = 0; j < dimension; ++j) logit += theta[j] * features[j];
        probability = det_sigmoid(logit);
        post_objective += -(examples[i].target != 0U ?
            logf(fmaxf(probability, 1.0e-5f)) :
            logf(fmaxf(1.0f - probability, 1.0e-5f)));
    }
    if (post_objective >= pre_objective) memcpy(theta, anchor, sizeof(theta));
    model->native_score_gain = fmaxf(0.25f, fminf(4.0f, theta[0]));
    model->native_score_bias = fmaxf(-4.0f, fminf(4.0f, theta[1]));
    for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
        model->native_class_bias[class_id] = fmaxf(-3.0f, fminf(3.0f,
            theta[2 + class_id]));
    }
    for (int scale_bin = 0; scale_bin < DET_MAX_SCALES; ++scale_bin) {
        model->native_scale_bias[scale_bin] = fmaxf(-3.0f, fminf(3.0f,
            theta[2 + DET_MAX_CLASSES + scale_bin]));
    }
    report->analytic_readout_applied = post_objective < pre_objective ? 1 : 0;
    report->readout_examples = example_count;
    report->readout_positives = positives;
    report->readout_negatives = negatives;
    report->readout_pre_objective = pre_objective / (double)example_count;
    report->readout_post_objective = (post_objective < pre_objective ?
                                      post_objective : pre_objective) /
                                     (double)example_count;
    return DET_OK;
}

static int det_native_feature_target(const det_sample *sample, int level,
                                     int cell_x, int cell_y) {
    int stride = 8 << level;
    if (sample == NULL || level < 0 || level >= DET_MAX_SCALES) return -1;
    for (int box_index = 0; box_index < sample->box_count; ++box_index) {
        const det_box *box = &sample->boxes[box_index];
        int box_level;
        int target_x;
        int target_y;
        if (box->target_weight > 0.0f && box->target_weight < 1.0f) continue;
        box_level = det_native_scale_bin(box->x1, box->y1, box->x2, box->y2,
                                         sample->image.width, sample->image.height);
        target_x = (int)(0.5f * (box->x1 + box->x2) / (float)stride);
        target_y = (int)(0.5f * (box->y1 + box->y2) / (float)stride);
        if (box_level == level && target_x == cell_x && target_y == cell_y)
            return box->class_id;
    }
    return -1;
}

static det_status det_native_fit_feature_readout(
    det_model *model, const det_sample *sample, size_t sample_index,
    det_native_feature_fit *fit) {
    if (model == NULL || sample == NULL || fit == NULL ||
        !model->native_graph_enabled || model->native_graph_architecture == NG_GRAPH_MULTISCALE)
        return DET_ERR_ARGUMENT;
    for (int level = 0; level < DET_MAX_SCALES; ++level) {
        const float *cache = model->native_feature_cache[level];
        int channels = model->native_feature_channels[level];
        int height = model->native_feature_height[level];
        int width = model->native_feature_width[level];
        int terms;
        if (cache == NULL || channels <= 0 || channels > DET_NATIVE_FEATURE_READOUT_MAX ||
            height <= 0 || width <= 0) return DET_ERR_UNSUPPORTED;
        terms = channels + 1;
        fit->channels[level] = channels;
        fit->terms[level] = terms;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int positive = det_native_feature_target(sample, level, x, y);
                size_t cell = (size_t)y * (size_t)width + (size_t)x;
                double weight;
                double vector[DET_NATIVE_FEATURE_READOUT_MAX + 1];
                if (positive < 0 && ((cell + sample_index * 17U +
                                      (size_t)level * 31U) & 7U) != 0U) continue;
                weight = positive >= 0 ? 4.0 : 0.25;
                for (int channel = 0; channel < channels; ++channel) {
                    vector[channel] = (double)cache[(size_t)channel *
                                                    (size_t)height * (size_t)width + cell];
                }
                vector[channels] = 1.0;
                for (int row = 0; row < terms; ++row) {
                    for (int column = 0; column < terms; ++column) {
                        fit->gram[level][row][column] +=
                            weight * vector[row] * vector[column];
                    }
                }
                for (int class_id = 0; class_id < model->native_graph_weights.class_count &&
                     class_id < DET_MAX_CLASSES; ++class_id) {
                    double target = class_id == positive ? 4.0 : -4.0;
                    fit->target_square[level][class_id] += weight * target * target;
                    for (int row = 0; row < terms; ++row)
                        fit->rhs[level][class_id][row] += weight * vector[row] * target;
                }
                ++fit->examples;
            }
        }
    }
    return DET_OK;
}

static double det_native_feature_objective(
    const det_native_feature_fit *fit, int level, int class_id,
    const double *coeff) {
    double value;
    int terms;
    if (fit == NULL || coeff == NULL || level < 0 || level >= DET_MAX_SCALES ||
        class_id < 0 || class_id >= DET_MAX_CLASSES) return HUGE_VAL;
    terms = fit->terms[level];
    value = fit->target_square[level][class_id];
    for (int row = 0; row < terms; ++row) {
        value -= 2.0 * coeff[row] * fit->rhs[level][class_id][row];
        for (int column = 0; column < terms; ++column) {
            value += coeff[row] * fit->gram[level][row][column] * coeff[column];
        }
    }
    return value;
}

static int det_native_feature_solve(const det_native_feature_fit *fit, int level,
                                    int class_id, double *solution) {
    double system[DET_NATIVE_FEATURE_READOUT_MAX + 1]
                 [DET_NATIVE_FEATURE_READOUT_MAX + 2];
    int terms;
    if (fit == NULL || solution == NULL || level < 0 || level >= DET_MAX_SCALES ||
        class_id < 0 || class_id >= DET_MAX_CLASSES) return 0;
    terms = fit->terms[level];
    if (terms <= 0 || terms > DET_NATIVE_FEATURE_READOUT_MAX + 1) return 0;
    for (int row = 0; row < terms; ++row) {
        for (int column = 0; column < terms; ++column)
            system[row][column] = fit->gram[level][row][column];
        system[row][terms] = fit->rhs[level][class_id][row];
        system[row][row] += 0.01;
    }
    for (int pivot = 0; pivot < terms; ++pivot) {
        int best = pivot;
        double magnitude = fabs(system[pivot][pivot]);
        for (int row = pivot + 1; row < terms; ++row) {
            double candidate = fabs(system[row][pivot]);
            if (candidate > magnitude) {
                magnitude = candidate;
                best = row;
            }
        }
        if (!(magnitude > 1.0e-9) || !isfinite(magnitude)) return 0;
        if (best != pivot) {
            for (int column = pivot; column <= terms; ++column) {
                double swap = system[pivot][column];
                system[pivot][column] = system[best][column];
                system[best][column] = swap;
            }
        }
        for (int row = pivot + 1; row < terms; ++row) {
            double factor = system[row][pivot] / system[pivot][pivot];
            for (int column = pivot; column <= terms; ++column)
                system[row][column] -= factor * system[pivot][column];
        }
    }
    for (int row = terms - 1; row >= 0; --row) {
        double value = system[row][terms];
        for (int column = row + 1; column < terms; ++column)
            value -= system[row][column] * solution[column];
        if (!isfinite(value) || !isfinite(system[row][row]) ||
            fabs(system[row][row]) <= 1.0e-9) return 0;
        solution[row] = value / system[row][row];
        if (!isfinite(solution[row])) return 0;
    }
    return 1;
}

static det_status det_native_apply_feature_readout(
    det_model *model, det_native_feature_fit *fit, det_train_report *report) {
    int applied = 0;
    const double blend = 0.25;
    double old_total = 0.0;
    double new_total = 0.0;
    if (model == NULL || fit == NULL || report == NULL) return DET_ERR_ARGUMENT;
    report->feature_readout_examples = fit->examples;
    for (int level = 0; level < DET_MAX_SCALES; ++level) {
        NGConv2D *conv = det_native_head_conv(model, level);
        if (fit->terms[level] <= 0 || conv == NULL || conv->weights_f32 == NULL ||
            conv->bias_f32 == NULL || conv->in_channels != fit->channels[level])
            return DET_ERR_UNSUPPORTED;
        for (int class_id = 0; class_id < model->native_graph_weights.class_count &&
             class_id < DET_MAX_CLASSES; ++class_id) {
            double solution[DET_NATIVE_FEATURE_READOUT_MAX + 1] = {0.0};
            double old[DET_NATIVE_FEATURE_READOUT_MAX + 1] = {0.0};
            float *weights = (float *)(uintptr_t)conv->weights_f32;
            float *bias = (float *)(uintptr_t)conv->bias_f32;
            if (!det_native_feature_solve(fit, level, class_id, solution)) continue;
            for (int channel = 0; channel < fit->channels[level]; ++channel)
                old[channel] = weights[(size_t)class_id *
                                        (size_t)conv->in_channels + (size_t)channel];
            old[fit->channels[level]] = bias[class_id];
            old_total += det_native_feature_objective(fit, level, class_id, old);
            new_total += det_native_feature_objective(fit, level, class_id, solution);
            if (det_native_feature_objective(fit, level, class_id, solution) + 1.0e-6 <
                det_native_feature_objective(fit, level, class_id, old)) {
                for (int channel = 0; channel < fit->channels[level]; ++channel)
                    weights[(size_t)class_id * (size_t)conv->in_channels +
                            (size_t)channel] = (float)(old[channel] +
                                blend * (solution[channel] - old[channel]));
                bias[class_id] = (float)(old[fit->channels[level]] + blend *
                                         (solution[fit->channels[level]] -
                                          old[fit->channels[level]]));
                ++applied;
            } else {
                new_total -= det_native_feature_objective(fit, level, class_id, solution);
                new_total += det_native_feature_objective(fit, level, class_id, old);
            }
        }
    }
    (void)old_total;
    (void)new_total;
    report->feature_readout_applied = applied > 0 ? 1 : 0;
    return DET_OK;
}

static det_status det_native_graph_train(det_model *model, const det_dataset *dataset,
                                        const det_train_config *config,
                                        det_train_report *report) {
    det_detection detections[100];
    unsigned char matched[100];
    double start;
    double deadline = 0.0;
    float loss_sum = 0.0f;
    size_t seen = 0U;
    size_t updates = 0U;
    det_native_readout_example *readout_examples = NULL;
    size_t readout_count = 0U;
    const size_t readout_capacity = 200000U;
    det_status status;
    det_native_feature_fit feature_fit;
    int stopped = 0;
    if (model == NULL || dataset == NULL || config == NULL || report == NULL ||
        dataset->next == NULL || config->epochs <= 0 || config->max_samples < 0 ||
        config->max_total_samples < 0 || config->time_budget_ms < 0.0 ||
        config->bootstrap_budget_ms < 0.0 || !isfinite(config->time_budget_ms) ||
        !isfinite(config->bootstrap_budget_ms) ||
        (config->feature_readout != 0 && config->feature_readout != 1) ||
        (config->candidate_head_adapt != 0 && config->candidate_head_adapt != 1)) return DET_ERR_ARGUMENT;
    memset(report, 0, sizeof(*report));
    memset(&feature_fit, 0, sizeof(feature_fit));
    if (config->feature_readout) {
        if (config->precision != DET_PRECISION_F32 ||
            model->native_graph_architecture == NG_GRAPH_MULTISCALE) {
            return DET_ERR_UNSUPPORTED;
        }
        for (int level = 0; level < DET_MAX_SCALES; ++level) {
            if (model->native_feature_channels[level] <= 0 ||
                model->native_feature_channels[level] > DET_NATIVE_FEATURE_READOUT_MAX) {
                return DET_ERR_UNSUPPORTED;
            }
        }
    }
    if (config->candidate_head_adapt) {
        if ((config->precision != DET_PRECISION_F32 &&
             config->precision != DET_PRECISION_INT8) ||
            model->native_graph_architecture == NG_GRAPH_MULTISCALE) {
            return DET_ERR_UNSUPPORTED;
        }
        for (int level = 0; level < DET_MAX_SCALES; ++level) {
            if (model->native_feature_cache[level] == NULL ||
                model->native_feature_channels[level] <= 0) return DET_ERR_UNSUPPORTED;
        }
    }
    start = now_ms();
    /* A native scalar forward is indivisible, so reserve 2.5 seconds for the
       final sample and report before the caller's hard wall-clock gate. */
    if (config->time_budget_ms > 2500.0) {
        deadline = start + config->time_budget_ms - 2500.0;
    } else if (config->time_budget_ms > 0.0) {
        deadline = start + config->time_budget_ms;
    }
    if (config->bootstrap_enabled) {
        double bootstrap_budget = config->bootstrap_budget_ms > 0.0 ?
                                  config->bootstrap_budget_ms : 5000.0;
        double bootstrap_deadline = start + bootstrap_budget;
        if (deadline > 0.0 && bootstrap_deadline > deadline) bootstrap_deadline = deadline;
        det_status bootstrap_status = det_native_bootstrap_pass(
            model, dataset, bootstrap_deadline, config->learning_rate, report);
        if (bootstrap_status != DET_OK) return bootstrap_status;
    }
    if (config->analytic_readout) {
        readout_examples = (det_native_readout_example *)calloc(
            readout_capacity, sizeof(*readout_examples));
        if (readout_examples == NULL) return DET_ERR_MEMORY;
    }
    for (int epoch = 0; epoch < config->epochs && !stopped; ++epoch) {
        size_t epoch_seen = 0U;
        if (dataset->reset != NULL) dataset->reset(dataset->user);
        for (;;) {
            det_sample sample;
            int count = 0;
            float grad_gain = 0.0f;
            float grad_bias = 0.0f;
            float grad_class[DET_MAX_CLASSES] = {0.0f};
            float grad_scale[DET_MAX_SCALES] = {0.0f};
            float grad_head[DET_MAX_SCALES][DET_MAX_CLASSES] = {{0.0f}};
            float grad_box[DET_MAX_SCALES][4] = {{0.0f}};
            float grad_class_box[DET_MAX_SCALES][DET_MAX_CLASSES][4] = {{{0.0f}}};
            float box_terms[DET_MAX_SCALES] = {0.0f};
            float class_box_terms[DET_MAX_SCALES][DET_MAX_CLASSES] = {{0.0f}};
            float sample_loss = 0.0f;
            float terms = 0.0f;
            int next = dataset->next(dataset->user, &sample);
            if (next < 0) {
                free(readout_examples);
                return DET_ERR_IO;
            }
            if (next == 0) break;
            if (!det_kshira_sample_valid(model, &sample)) {
                free(readout_examples);
                return DET_ERR_ARGUMENT;
            }
            status = det_native_graph_predict(model, &sample.image, 0.0f, detections,
                                              100, &count);
            if (status != DET_OK) {
                free(readout_examples);
                return status == DET_ERR_ARGUMENT ? status : DET_ERR_SHAPE;
            }
            if (config->feature_readout) {
                status = det_native_fit_feature_readout(model, &sample, seen,
                                                        &feature_fit);
                if (status != DET_OK) {
                    free(readout_examples);
                    return status;
                }
            }
            memset(matched, 0, sizeof(matched));
            for (int i = 0; i < count; ++i) {
                int best = -1;
                int scale_bin = det_native_scale_bin(
                    detections[i].box.x1, detections[i].box.y1,
                    detections[i].box.x2, detections[i].box.y2,
                    sample.image.width, sample.image.height);
                float best_iou = 0.5f;
                for (int b = 0; b < sample.box_count; ++b) {
                    float overlap;
                    if (matched[b] != 0U || detections[i].box.class_id !=
                        sample.boxes[b].class_id) continue;
                    overlap = det_native_iou(&detections[i].box, &sample.boxes[b]);
                    if (overlap >= best_iou) {
                        best_iou = overlap;
                        best = b;
                    }
                }
                if (best >= 0) matched[best] = 1U;
                float target = best >= 0 ? 1.0f : 0.0f;
                float target_weight = 1.0f;
                if (best >= 0) {
                    const det_box *target_box = &sample.boxes[best];
                    target_weight = target_box->target_weight > 0.0f ?
                                     target_box->target_weight : 1.0f;
                    float predicted_center_x = 0.5f *
                        (detections[i].box.x1 + detections[i].box.x2);
                    float predicted_center_y = 0.5f *
                        (detections[i].box.y1 + detections[i].box.y2);
                    float predicted_width = fmaxf(1.0f,
                        detections[i].box.x2 - detections[i].box.x1);
                    float predicted_height = fmaxf(1.0f,
                        detections[i].box.y2 - detections[i].box.y1);
                    float target_center_x = 0.5f * (target_box->x1 + target_box->x2);
                    float target_center_y = 0.5f * (target_box->y1 + target_box->y2);
                    float target_width = fmaxf(1.0f, target_box->x2 - target_box->x1);
                    float target_height = fmaxf(1.0f, target_box->y2 - target_box->y1);
                    grad_box[scale_bin][0] +=
                        target_weight * (target_center_x - predicted_center_x) /
                        (float)sample.image.width;
                    grad_box[scale_bin][1] +=
                        target_weight * (target_center_y - predicted_center_y) /
                        (float)sample.image.height;
                    grad_box[scale_bin][2] += target_weight *
                                              logf(target_width / predicted_width);
                    grad_box[scale_bin][3] += target_weight *
                                              logf(target_height / predicted_height);
                    box_terms[scale_bin] += target_weight;
                    if (target_box->class_id >= 0 &&
                        target_box->class_id < DET_MAX_CLASSES) {
                        int matched_class = target_box->class_id;
                        grad_class_box[scale_bin][matched_class][0] +=
                            target_weight * (target_center_x - predicted_center_x) /
                            (float)sample.image.width;
                        grad_class_box[scale_bin][matched_class][1] +=
                            target_weight * (target_center_y - predicted_center_y) /
                            (float)sample.image.height;
                        grad_class_box[scale_bin][matched_class][2] += target_weight *
                            logf(target_width / predicted_width);
                        grad_class_box[scale_bin][matched_class][3] += target_weight *
                            logf(target_height / predicted_height);
                        class_box_terms[scale_bin][matched_class] += target_weight;
                    }
                }
                float calibrated_logit = logf(fmaxf(1.0e-5f, detections[i].score) /
                                              fmaxf(1.0e-5f, 1.0f - detections[i].score));
                int class_id = detections[i].box.class_id;
                float class_bias = class_id >= 0 && class_id < DET_MAX_CLASSES ?
                                   model->native_class_bias[class_id] : 0.0f;
                float raw_logit = (calibrated_logit - model->native_score_bias -
                                   class_bias - model->native_scale_bias[scale_bin]) /
                                  fmaxf(0.25f, model->native_score_gain);
                float probability = det_sigmoid(model->native_score_gain * raw_logit +
                                                model->native_score_bias + class_bias +
                                                model->native_scale_bias[scale_bin]);
                float error = (probability - target) * target_weight;
                if (config->candidate_head_adapt && class_id >= 0 &&
                    class_id < model->native_graph_weights.class_count) {
                    int stride = 8 << scale_bin;
                    int cell_x = (int)floorf((0.5f * (detections[i].box.x1 +
                                                         detections[i].box.x2)) /
                                             (float)stride);
                    int cell_y = (int)floorf((0.5f * (detections[i].box.y1 +
                                                         detections[i].box.y2)) /
                                             (float)stride);
                    int feature_width = model->native_feature_width[scale_bin];
                    int feature_height = model->native_feature_height[scale_bin];
                    float head_rate = (config->learning_rate > 0.0f ?
                                       config->learning_rate : 0.05f) * 0.025f;
                    if (feature_width > 0 && feature_height > 0) {
                        cell_x = cell_x < 0 ? 0 :
                                 (cell_x >= feature_width ? feature_width - 1 : cell_x);
                        cell_y = cell_y < 0 ? 0 :
                                 (cell_y >= feature_height ? feature_height - 1 : cell_y);
                        if (det_native_update_candidate_head(
                                model, scale_bin, class_id, cell_x, cell_y,
                                target, target_weight, head_rate / fmaxf(1.0f, terms)))
                            ++report->candidate_head_updates;
                    }
                }
                if (readout_examples != NULL && readout_count < readout_capacity) {
                    readout_examples[readout_count].raw_logit = raw_logit;
                    readout_examples[readout_count].class_id =
                        (uint8_t)(class_id >= 0 ? class_id : 0);
                    readout_examples[readout_count].scale_bin = (uint8_t)scale_bin;
                    readout_examples[readout_count].target = (uint8_t)(best >= 0);
                    ++readout_count;
                }
                grad_gain += error * raw_logit;
                grad_bias += error;
                if (class_id >= 0 && class_id < DET_MAX_CLASSES)
                    grad_class[class_id] += error;
                if (class_id >= 0 && class_id < DET_MAX_CLASSES)
                    grad_head[scale_bin][class_id] += error;
                grad_scale[scale_bin] += error;
                sample_loss += target_weight *
                    -(target * logf(fmaxf(probability, 1.0e-5f)) +
                      (1.0f - target) * logf(fmaxf(1.0f - probability, 1.0e-5f)));
                terms += target_weight;
            }
            if (terms > 0U) {
                float rate = config->learning_rate > 0.0f ? config->learning_rate : 0.05f;
                float inverse = 1.0f / (float)terms;
                model->native_score_gain -= rate * grad_gain * inverse;
                model->native_score_bias -= rate * grad_bias * inverse;
                model->native_score_gain = fmaxf(0.25f, fminf(4.0f,
                                                               model->native_score_gain));
                model->native_score_bias = fmaxf(-4.0f, fminf(4.0f,
                                                               model->native_score_bias));
                for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
                    model->native_class_bias[class_id] -= rate * grad_class[class_id] * inverse;
                    model->native_class_bias[class_id] = fmaxf(-3.0f, fminf(3.0f,
                                                                             model->native_class_bias[class_id]));
                }
                for (int scale_bin = 0; scale_bin < DET_MAX_SCALES; ++scale_bin) {
                    for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
                        det_native_update_head_bias(
                            model, scale_bin, class_id,
                            -0.5f * rate * grad_head[scale_bin][class_id] * inverse);
                    }
                }
                for (int scale_bin = 0; scale_bin < DET_MAX_SCALES; ++scale_bin) {
                    model->native_scale_bias[scale_bin] -= rate * grad_scale[scale_bin] * inverse;
                    model->native_scale_bias[scale_bin] = fmaxf(-3.0f, fminf(3.0f,
                                                                             model->native_scale_bias[scale_bin]));
                    if (box_terms[scale_bin] > 0U) {
                        float box_rate = rate / (float)box_terms[scale_bin];
                        for (int component = 0; component < 4; ++component) {
                            model->native_box_bias[scale_bin][component] +=
                                box_rate * grad_box[scale_bin][component];
                            model->native_box_bias[scale_bin][component] =
                                fmaxf(-0.5f, fminf(0.5f,
                                    model->native_box_bias[scale_bin][component]));
                        }
                    }
                }
                for (int scale_bin = 0; scale_bin < DET_MAX_SCALES; ++scale_bin) {
                    for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
                        if (class_box_terms[scale_bin][class_id] > 0U) {
                            float class_box_rate = 0.5f * rate /
                                (float)class_box_terms[scale_bin][class_id];
                            for (int component = 0; component < 4; ++component) {
                                model->native_class_box_bias[scale_bin][class_id][component] +=
                                    class_box_rate * grad_class_box[scale_bin][class_id][component];
                                model->native_class_box_bias[scale_bin][class_id][component] =
                                    fmaxf(-0.5f, fminf(0.5f,
                                        model->native_class_box_bias[scale_bin][class_id][component]));
                            }
                        }
                    }
                }
                loss_sum += sample_loss * inverse;
                ++updates;
            }
            ++seen;
            ++epoch_seen;
            if ((config->max_samples > 0 && epoch_seen >= (size_t)config->max_samples) ||
                (config->max_total_samples > 0 && seen >= (size_t)config->max_total_samples) ||
                (deadline > 0.0 && now_ms() >= deadline)) {
                stopped = deadline > 0.0 && now_ms() >= deadline;
                break;
            }
        }
    }
    if (config->feature_readout) {
        status = det_native_apply_feature_readout(model, &feature_fit, report);
        if (status != DET_OK) {
            free(readout_examples);
            return status;
        }
    }
    if (readout_examples != NULL) {
        status = det_native_fit_readout(model, readout_examples, readout_count, report);
        free(readout_examples);
        if (status != DET_OK && readout_count > 0U) return status;
    }
    report->samples_seen = seen;
    report->updates = updates;
    report->elapsed_ms = now_ms() - start;
    report->mean_loss = updates > 0U ? loss_sum / (float)updates : 0.0f;
    report->stopped_by_budget = stopped;
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
    if (model->native_graph_enabled) {
        return det_native_graph_predict(model, image, score_threshold, detections,
                                       capacity, count);
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
                                      raw[i].box.class_id, 0.0f};
        detections[i].score = raw[i].score;
    }
    *count = raw_count;
    return DET_OK;
}
