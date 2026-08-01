    /* Local/global training schedules, assignment banks, and streamed training. */
    size_t weight_count = stage_weight_count(stage);
    for (size_t i = 0U; i < weight_count; ++i) {
        float gradient = stage->gradient_w[i] * scale;
        stage->velocity_w[i] = momentum * stage->velocity_w[i] + gradient;
        stage->weights[i] -= lr * stage->velocity_w[i];
    }
    for (int i = 0; i < stage->output_channels; ++i) {
        float gradient = stage->gradient_b[i] * scale;
        stage->velocity_b[i] = momentum * stage->velocity_b[i] + gradient;
        stage->bias[i] -= lr * stage->velocity_b[i];
    }
}

static int stage_box_target(const det_sample *sample, int stage_index, int x, int y,
                            float target[4]) {
    for (int i = 0; i < 4; ++i) target[i] = 0.0f;
    int stride = 1 << (stage_index + 1);
    for (int b = 0; b < sample->box_count; ++b) {
        const det_box *box = &sample->boxes[b];
        if (box->class_id < 0) continue;
        int tx = (int)(((box->x1 + box->x2) * 0.5f) / (float)stride);
        int ty = (int)(((box->y1 + box->y2) * 0.5f) / (float)stride);
        if (tx != x || ty != y) continue;
        target[0] = 1.0f;
        target[1] = ((box->x1 + box->x2) * 0.5f) / (float)sample->image.width;
        target[2] = ((box->y1 + box->y2) * 0.5f) / (float)sample->image.height;
        target[3] = fmaxf((box->x2 - box->x1) / (float)sample->image.width,
                          (box->y2 - box->y1) / (float)sample->image.height);
        return 1;
    }
    return 0;
}

static void local_update_stage_cell(det_stage *stage, const float *input_data,
                                    int y, int x, const float target[4],
                                    float lr, float momentum, size_t *updates) {
    for (int oc = 0; oc < stage->output_channels; ++oc) {
        size_t output_index = ((size_t)oc * (size_t)stage->output_height + (size_t)y) *
                              (size_t)stage->output_width + (size_t)x;
        /* LOCAL_FAST uses a straight-through ReLU surrogate so a dead stage
           can receive a positive local target and recover. */
        float error = stage->activation[output_index] - target[oc];
        size_t bias_index = (size_t)oc;
        stage->velocity_b[bias_index] = momentum * stage->velocity_b[bias_index] + error;
        stage->bias[bias_index] -= lr * stage->velocity_b[bias_index];
        for (int ic = 0; ic < stage->input_channels; ++ic) {
            if (stage->depthwise && ic != oc) continue;
            for (int ky = 0; ky < stage->kernel; ++ky) {
                int iy = y * stage->stride + ky - stage->padding;
                if (iy < 0 || iy >= stage->input_height) continue;
                for (int kx = 0; kx < stage->kernel; ++kx) {
                    int ix = x * stage->stride + kx - stage->padding;
                    if (ix < 0 || ix >= stage->input_width) continue;
                    size_t input_index = ((size_t)ic * (size_t)stage->input_height +
                                          (size_t)iy) * (size_t)stage->input_width +
                                         (size_t)ix;
                    size_t weight_index = stage_weight_index(stage, oc, ic, ky, kx);
                    float gradient = error * input_data[input_index];
                    stage->velocity_w[weight_index] = momentum * stage->velocity_w[weight_index] + gradient;
                    stage->weights[weight_index] -= lr * stage->velocity_w[weight_index];
                }
            }
        }
    }
    ++(*updates);
}

static void local_update_stage_data(det_stage *stage, const float *input_data,
                                    const det_sample *sample, int target_stage_index,
                                    size_t sample_index, float lr, float momentum,
                                    size_t *updates) {
    if (sample->box_count == 1) {
        const det_box *box = &sample->boxes[0];
        int stride = 1 << (target_stage_index + 1);
        int target_x = (int)(((box->x1 + box->x2) * 0.5f) / (float)stride);
        int target_y = (int)(((box->y1 + box->y2) * 0.5f) / (float)stride);
        float positive_target[4] = {
            1.0f,
            ((box->x1 + box->x2) * 0.5f) / (float)sample->image.width,
            ((box->y1 + box->y2) * 0.5f) / (float)sample->image.height,
            fmaxf((box->x2 - box->x1) / (float)sample->image.width,
                  (box->y2 - box->y1) / (float)sample->image.height)
        };
        for (int y = 0; y < stage->output_height; ++y) {
            for (int x = 0; x < stage->output_width; ++x) {
                int positive = x == target_x && y == target_y;
                if (!positive && ((x + y + (int)sample_index + target_stage_index) % 64 != 0)) {
                    continue;
                }
                float target[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (positive) {
                    for (int i = 0; i < 4; ++i) target[i] = positive_target[i];
                }
                local_update_stage_cell(stage, input_data, y, x, target,
                                        lr, momentum, updates);
            }
        }
        return;
    }
    /* Preserve the original target ordering for multi-box samples. */
    for (int y = 0; y < stage->output_height; ++y) {
        for (int x = 0; x < stage->output_width; ++x) {
            float target[4];
            int positive = stage_box_target(sample, target_stage_index, x, y, target);
            if (!positive && ((x + y + (int)sample_index + target_stage_index) % 64 != 0)) {
                continue;
            }
            local_update_stage_cell(stage, input_data, y, x, target,
                                    lr, momentum, updates);
        }
    }
}

static void local_update_stage(det_model *model, const det_sample *sample, int stage_index,
                               size_t sample_index, float lr, float momentum,
                               size_t *updates) {
    const float *input_data = stage_index == 0 ? sample->image.data :
                              model->stages[stage_index - 1].activation;
    local_update_stage_data(&model->stages[stage_index], input_data, sample, stage_index,
                            sample_index, lr, momentum, updates);
}

static void local_update_neck(det_model *model, const det_sample *sample, int scale,
                              size_t sample_index, float lr, float momentum,
                              size_t *updates) {
    const det_stage *backbone = &model->stages[scale + 2];
    local_update_stage_data(&model->neck[scale], backbone->activation, sample, scale + 2,
                            sample_index, lr, momentum, updates);
}

static void local_update_bottomup(det_model *model, const det_sample *sample, int scale,
                                  size_t sample_index, float lr, float momentum,
                                  size_t *updates) {
    if (sample_index < 1024U || (sample_index & 15U) != 0U) return;
    const float *input_data = scale == 0 ? model->neck_topdown[0] :
                              model->neck_fused[scale];
    local_update_stage_data(&model->bottomup[scale], input_data, sample, scale + 3,
                            sample_index, lr * 0.01f, momentum, updates);
}

static int validate_sample(const det_model *model, const det_sample *sample) {
    if (model == NULL || sample == NULL || sample->image.data == NULL ||
        sample->image.channels != model->spec.channels ||
        sample->image.height != model->spec.height || sample->image.width != model->spec.width ||
        sample->box_count < 0 || (sample->box_count > 0 && sample->boxes == NULL)) {
        return 0;
    }
    for (int i = 0; i < sample->box_count; ++i) {
        const det_box *box = &sample->boxes[i];
        if (box->class_id < 0 || box->class_id >= model->spec.num_classes ||
            !isfinite(box->x1) || !isfinite(box->y1) || !isfinite(box->x2) ||
            !isfinite(box->y2) || box->x1 < 0.0f || box->y1 < 0.0f ||
            box->x2 > (float)model->spec.width || box->y2 > (float)model->spec.height ||
            box->x2 <= box->x1 || box->y2 <= box->y1) {
            return 0;
        }
    }
    size_t pixels = (size_t)sample->image.channels * (size_t)sample->image.height *
                    (size_t)sample->image.width;
    return finite_values(sample->image.data, pixels);
}

static const det_box *assigned_training_box(const det_sample *sample, int scale,
                                             int cell_x, int cell_y, int radius,
                                             int image_width, int image_height) {
    const det_box *best = NULL;
    int best_distance = INT_MAX;
    for (int b = 0; b < sample->box_count; ++b) {
        const det_box *box = &sample->boxes[b];
        if (choose_scale(box, image_width, image_height) != scale) continue;
        int target_x = (int)(((box->x1 + box->x2) * 0.5f) /
                             (float)k_strides[scale]);
        int target_y = (int)(((box->y1 + box->y2) * 0.5f) /
                             (float)k_strides[scale]);
        int distance = abs(cell_x - target_x) + abs(cell_y - target_y);
        if (abs(cell_x - target_x) > radius || abs(cell_y - target_y) > radius ||
            distance >= best_distance) continue;
        best = box;
        best_distance = distance;
    }
    return best;
}

static int fill_head_target(const det_box *box, int scale, int cell_x, int cell_y,
                            int classes, float *target) {
    for (int o = 0; o < 4 + DET_MAX_CLASSES; ++o) target[o] = 0.0f;
    if (box == NULL || box->class_id < 0 || box->class_id >= classes) return -1;
    target[box->class_id + 4] = 1.0f;
    float center_x = ((float)cell_x + 0.5f) * (float)k_strides[scale];
    float center_y = ((float)cell_y + 0.5f) * (float)k_strides[scale];
    target[0] = fmaxf(0.0f, (center_x - box->x1) / (float)k_strides[scale]);
    target[1] = fmaxf(0.0f, (center_y - box->y1) / (float)k_strides[scale]);
    target[2] = fmaxf(0.0f, (box->x2 - center_x) / (float)k_strides[scale]);
    target[3] = fmaxf(0.0f, (box->y2 - center_y) / (float)k_strides[scale]);
    return box->class_id;
}

static float train_head_cell(det_model *model, const det_sample *sample,
                             det_head *head, int scale, int cell_x, int cell_y,
                             const det_box *assigned, size_t sample_index,
                             int regression_enabled, int force_update, int use_global,
                             float learning_rate, float momentum,
                             float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4],
                             float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES],
                             size_t *terms, size_t *updates) {
    float features[4];
    float target[4 + DET_MAX_CLASSES];
    float prediction[4 + DET_MAX_CLASSES] = {0.0f};
    extract_feature(model, &sample->image, scale, cell_y, cell_x, features, NULL);
    int positive = fill_head_target(assigned, scale, cell_x, cell_y,
                                    model->spec.num_classes, target);
    head_forward(head, features, prediction);
    float prediction_gradient[4 + DET_MAX_CLASSES];
    float loss = prediction_loss_gradient(prediction, target, head->outputs,
                                          positive, regression_enabled, prediction_gradient);
    *terms += (size_t)head->outputs;
    if (use_global) {
        for (int o = 0; o < head->outputs; ++o) {
            gradient_b[scale][o] += prediction_gradient[o];
            for (int c = 0; c < 4; ++c) {
                gradient_w[scale][o][c] += prediction_gradient[o] * features[c];
            }
        }
        float feature_gradient[4];
        feature_gradient_from_error(head, prediction_gradient, feature_gradient);
        add_feature_gradient_to_backbone(model, scale, cell_y, cell_x, feature_gradient);
        ++(*updates);
    } else if (positive >= 0 || force_update ||
               ((cell_x + cell_y + (int)sample_index + scale) % 16 == 0)) {
        update_head(head, features, target, positive, regression_enabled,
                    learning_rate, momentum);
        ++(*updates);
    }
    return loss;
}

static float train_head_bank(det_model *model, const det_sample *sample,
                             det_head *heads, int assignment_radius,
                             size_t sample_index, int use_global,
                             float learning_rate, float momentum,
                             float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4],
                             float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES],
                             size_t *terms, size_t *updates) {
    float loss = 0.0f;
    float target[4 + DET_MAX_CLASSES];
    float prediction[4 + DET_MAX_CLASSES] = {0.0f};
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head = &heads[s];
        *terms += (size_t)head->height * (size_t)head->width * (size_t)head->outputs;
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                float features[4];
                extract_feature(model, &sample->image, s, y, x, features, NULL);
                for (int o = 0; o < head->outputs; ++o) target[o] = 0.0f;
                int positive = -1;
                const det_box *assigned = NULL;
                for (int b = 0; b < sample->box_count; ++b) {
                    if (sample->boxes[b].class_id < 0 ||
                        sample->boxes[b].class_id >= model->spec.num_classes ||
                        choose_scale(&sample->boxes[b], sample->image.width,
                                     sample->image.height) != s) continue;
                    int tx = (int)(((sample->boxes[b].x1 + sample->boxes[b].x2) * 0.5f) /
                                   (float)k_strides[s]);
                    int ty = (int)(((sample->boxes[b].y1 + sample->boxes[b].y2) * 0.5f) /
                                   (float)k_strides[s]);
                    if (abs(x - tx) <= assignment_radius &&
                        abs(y - ty) <= assignment_radius) {
                        positive = sample->boxes[b].class_id;
                        assigned = &sample->boxes[b];
                        break;
                    }
                }
                if (positive >= 0 && assigned != NULL) {
                    float center_x = ((float)x + 0.5f) * (float)k_strides[s];
                    float center_y = ((float)y + 0.5f) * (float)k_strides[s];
                    target[positive + 4] = 1.0f;
                    target[0] = fmaxf(0.0f, (center_x - assigned->x1) /
                                      (float)k_strides[s]);
                    target[1] = fmaxf(0.0f, (center_y - assigned->y1) /
                                      (float)k_strides[s]);
                    target[2] = fmaxf(0.0f, (assigned->x2 - center_x) /
                                      (float)k_strides[s]);
                    target[3] = fmaxf(0.0f, (assigned->y2 - center_y) /
                                      (float)k_strides[s]);
                }
                head_forward(head, features, prediction);
                float prediction_gradient[4 + DET_MAX_CLASSES];
                loss += prediction_loss_gradient(prediction, target, head->outputs,
                                                  positive, 1, prediction_gradient);
                if (use_global) {
                    for (int o = 0; o < head->outputs; ++o) {
                        gradient_b[s][o] += prediction_gradient[o];
                        for (int c = 0; c < 4; ++c) {
                            gradient_w[s][o][c] += prediction_gradient[o] * features[c];
                        }
                    }
                    float feature_gradient[4];
                    feature_gradient_from_error(head, prediction_gradient, feature_gradient);
                    add_feature_gradient_to_backbone(model, s, y, x, feature_gradient);
                    ++(*updates);
                } else if (positive >= 0 ||
                           ((x + y + (int)sample_index) % 16 == 0)) {
                    update_head(head, features, target, positive, 1,
                                learning_rate, momentum);
                    ++(*updates);
                }
            }
        }
    }
    return loss;
}

static float train_auxiliary_head_bank(det_model *model, const det_sample *sample,
                                       size_t sample_index, int use_global,
                                       float learning_rate, float momentum,
                                       float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4],
                                       float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES],
                                       size_t *terms, size_t *updates) {
    float loss = 0.0f;
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        det_head *head = &model->aux_heads[s];
        for (int b = 0; b < sample->box_count; ++b) {
            const det_box *box = &sample->boxes[b];
            if (choose_scale(box, sample->image.width, sample->image.height) != s) continue;
            int target_x = (int)(((box->x1 + box->x2) * 0.5f) /
                                 (float)k_strides[s]);
            int target_y = (int)(((box->y1 + box->y2) * 0.5f) /
                                 (float)k_strides[s]);
            for (int y = target_y - 1; y <= target_y + 1; ++y) {
                if (y < 0 || y >= head->height) continue;
                for (int x = target_x - 1; x <= target_x + 1; ++x) {
                    if (x < 0 || x >= head->width) continue;
                    const det_box *assigned = assigned_training_box(
                        sample, s, x, y, 1,
                        sample->image.width, sample->image.height);
                    if (assigned != box) continue;
                    float center_x = ((float)x + 0.5f) * (float)k_strides[s];
                    float center_y = ((float)y + 0.5f) * (float)k_strides[s];
                    int center_inside = center_x >= assigned->x1 && center_x <= assigned->x2 &&
                                        center_y >= assigned->y1 && center_y <= assigned->y2;
                    if (!center_inside && (x != target_x || y != target_y)) continue;
                    loss += train_head_cell(model, sample, head, s, x, y, assigned,
                                            sample_index, center_inside, 0, use_global,
                                            learning_rate, momentum,
                                            gradient_w, gradient_b, terms, updates);
                }
            }
        }
        int cell_count = head->height * head->width;
        int negatives = 0;
        size_t start = sample_index * 17U + (size_t)s * 313U;
        for (int step = 0; step < cell_count && negatives < 4; ++step) {
            int flat = (int)((start + (size_t)step) % (size_t)cell_count);
            int x = flat % head->width;
            int y = flat / head->width;
            if (assigned_training_box(sample, s, x, y, 1,
                                      sample->image.width, sample->image.height) != NULL) {
                continue;
            }
            loss += train_head_cell(model, sample, head, s, x, y, NULL,
                                    sample_index, 0, 1, use_global,
                                    learning_rate, momentum,
                                    gradient_w, gradient_b, terms, updates);
            ++negatives;
        }
    }
    return loss;
}

static float train_sample(det_model *model, const det_sample *sample,
                           const det_train_config *config, size_t sample_index,
                           size_t global_sample_index,
                           size_t *updates) {
    backbone_forward(model, &sample->image);
    float loss = 0.0f;
    float gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4] = {{{0.0f}}};
    float gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES] = {{0.0f}};
    float aux_gradient_w[DET_MAX_SCALES][4 + DET_MAX_CLASSES][4] = {{{0.0f}}};
    float aux_gradient_b[DET_MAX_SCALES][4 + DET_MAX_CLASSES] = {{0.0f}};
    int use_global = config->mode == DET_TRAIN_GLOBAL_BP;
    int aux_sampled = (sample_index & 15U) == 0U;
    if (use_global) clear_backbone_gradients(model);
    size_t terms = 0U;
    loss += train_head_bank(model, sample, model->heads, 0, sample_index, use_global,
                            config->learning_rate, config->momentum,
                            gradient_w, gradient_b, &terms, updates);
    /* The auxiliary one-to-many bank is a training-only regularizer. Updating
       it every sixteenth sample bounds its CPU cost while preserving a full
       one-to-many neighborhood whenever the bank is sampled. */
    if (aux_sampled) {
        loss += train_auxiliary_head_bank(model, sample, sample_index, use_global,
                                          config->learning_rate, config->momentum,
                                          aux_gradient_w, aux_gradient_b, &terms, updates);
    }
    if (use_global && terms > 0U) {
        float scale = 1.0f / (float)terms;
        for (int s = 0; s < DET_MAX_SCALES; ++s) {
            apply_head_gradient(&model->heads[s], gradient_w[s], gradient_b[s],
                                config->learning_rate, config->momentum, scale);
            if (aux_sampled) {
                apply_head_gradient(&model->aux_heads[s], aux_gradient_w[s], aux_gradient_b[s],
                                    config->learning_rate, config->momentum, scale);
            }
        }
        neck_backward(model);
        backbone_backward(model, &sample->image);
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            apply_stage_gradient(&model->stages[i], config->learning_rate,
                                  config->momentum, scale);
        }
        for (int i = 0; i < DET_NECK_STAGES; ++i) {
            apply_stage_gradient(&model->neck[i], config->learning_rate,
                                 config->momentum, scale);
        }
        for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
            apply_stage_gradient(&model->bottomup[i], config->learning_rate,
                                 config->momentum, scale);
        }
    } else {
        for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
            local_update_stage(model, sample, i, sample_index, config->learning_rate,
                               config->momentum, updates);
        }
        for (int i = 0; i < DET_NECK_STAGES; ++i) {
            local_update_neck(model, sample, i, sample_index, config->learning_rate,
                              config->momentum, updates);
        }
        for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
            local_update_bottomup(model, sample, i, global_sample_index, config->learning_rate,
                                  config->momentum, updates);
        }
    }
    return terms > 0U ? loss / (float)terms : 0.0f;
}

det_status det_train(det_model *model, const det_dataset *dataset,
                     const det_train_config *config, det_train_report *report) {
    if (model == NULL || dataset == NULL || config == NULL || report == NULL ||
        dataset->next == NULL || config->epochs <= 0 || config->learning_rate <= 0.0f ||
        config->momentum < 0.0f || config->momentum >= 1.0f ||
        config->reset_weights < 0 || config->reset_weights > 1 ||
        (config->mode != DET_TRAIN_LOCAL_FAST && config->mode != DET_TRAIN_GLOBAL_BP &&
         config->mode != DET_TRAIN_ODT) ||
        (config->precision != DET_PRECISION_F32 && config->precision != DET_PRECISION_INT8 &&
         config->precision != DET_PRECISION_W4A8 &&
         config->precision != DET_PRECISION_INT4) || !isfinite(config->learning_rate) ||
         !isfinite(config->momentum) || !isfinite(config->score_threshold)) {
        return DET_ERR_ARGUMENT;
    }
    if (model->architecture == DET_ARCH_KSHIRA) {
        if (config->momentum != 0.0f) return DET_ERR_UNSUPPORTED;
        return det_kshira_train(model, dataset, config, report);
    }
    if (config->mode == DET_TRAIN_ODT || config->precision == DET_PRECISION_INT4) {
        return DET_ERR_UNSUPPORTED;
    }
    if (config->precision != DET_PRECISION_F32) {
        return DET_ERR_UNSUPPORTED;
    }
    if (model->precision != DET_PRECISION_F32) return DET_ERR_UNSUPPORTED;
    if (config->reset_weights != 0 && det_model_reset(model, config->seed) != DET_OK) {
        return DET_ERR_MEMORY;
    }
    memset(report, 0, sizeof(*report));
    double start = now_ms();
    float loss_sum = 0.0f;
    size_t updates = 0U;
    for (int epoch = 0; epoch < config->epochs; ++epoch) {
        if (dataset->reset != NULL) dataset->reset(dataset->user);
        det_sample sample;
        size_t seen = 0U;
        while (config->max_samples <= 0 || seen < (size_t)config->max_samples) {
            int next_status = dataset->next(dataset->user, &sample);
            if (next_status < 0) return DET_ERR_IO;
            if (next_status == 0) break;
            if (!validate_sample(model, &sample)) return DET_ERR_ARGUMENT;
            loss_sum += train_sample(model, &sample, config, seen,
                                     report->samples_seen, &updates);
            ++seen;
            ++report->samples_seen;
        }
        if (seen == 0U) return DET_ERR_ARGUMENT;
    }
    model->train_updates += updates;
    report->updates = updates;
    report->elapsed_ms = now_ms() - start;
    report->mean_loss = report->samples_seen > 0U ? loss_sum / (float)report->samples_seen : 0.0f;
    report->used_global_backward = config->mode == DET_TRAIN_GLOBAL_BP ? 1 : 0;
    return DET_OK;
}

static void insert_detection(det_detection *detections, int *count, int capacity,
                             const det_detection *candidate) {
    if (capacity <= 0 || count == NULL || candidate == NULL) return;
    if (*count < capacity) {
        detections[*count] = *candidate;
        ++(*count);
        return;
    }
    int lowest = 0;
    for (int i = 1; i < *count; ++i) {
        if (detections[i].score < detections[lowest].score) lowest = i;
    }
    if (candidate->score > detections[lowest].score) detections[lowest] = *candidate;
}

static void sort_detections_descending(det_detection *detections, int count) {
    for (int i = 1; i < count; ++i) {
        det_detection value = detections[i];
        int j = i;
        while (j > 0 && detections[j - 1].score < value.score) {
            detections[j] = detections[j - 1];
            --j;
        }
        detections[j] = value;
    }
}
