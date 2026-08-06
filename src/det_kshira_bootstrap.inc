/* Target-domain bootstrap for the accuracy-oriented KSHIRA path.  This stays
 * above the detector kernels: it only transforms streamed samples and routes
 * the resulting crops through the existing public session step. */

static det_status det_kshira_make_bootstrap_crop(
    const det_model *model, const det_sample *source, const det_box *box,
    int variant, float *output, det_box *target) {
    float box_width;
    float box_height;
    float context = variant == 0 ? 0.10f : 0.35f;
    float crop_width;
    float crop_height;
    float center_x;
    float center_y;
    float crop_x0;
    float crop_y0;
    size_t output_plane;
    if (model == NULL || source == NULL || box == NULL || output == NULL || target == NULL ||
        variant < 0 || variant > 1 || box->class_id < 0 ||
        box->class_id >= model->spec.num_classes || !isfinite(box->x1) ||
        !isfinite(box->y1) || !isfinite(box->x2) || !isfinite(box->y2) ||
        box->x2 <= box->x1 || box->y2 <= box->y1) {
        return DET_ERR_ARGUMENT;
    }
    box_width = box->x2 - box->x1;
    box_height = box->y2 - box->y1;
    crop_width = box_width * (1.0f + 2.0f * context);
    crop_height = box_height * (1.0f + 2.0f * context);
    if (crop_width > (float)source->image.width) crop_width = (float)source->image.width;
    if (crop_height > (float)source->image.height) crop_height = (float)source->image.height;
    if (crop_width < 2.0f || crop_height < 2.0f) return DET_ERR_SHAPE;
    center_x = (box->x1 + box->x2) * 0.5f;
    center_y = (box->y1 + box->y2) * 0.5f;
    if (variant == 1) {
        center_x += (box->class_id & 1) != 0 ? 0.12f * box_width : -0.12f * box_width;
        center_y += (box->class_id & 2) != 0 ? 0.08f * box_height : -0.08f * box_height;
    }
    crop_x0 = center_x - 0.5f * crop_width;
    crop_y0 = center_y - 0.5f * crop_height;
    if (crop_x0 < 0.0f) crop_x0 = 0.0f;
    if (crop_y0 < 0.0f) crop_y0 = 0.0f;
    if (crop_x0 + crop_width > (float)source->image.width) {
        crop_x0 = (float)source->image.width - crop_width;
    }
    if (crop_y0 + crop_height > (float)source->image.height) {
        crop_y0 = (float)source->image.height - crop_height;
    }
    output_plane = (size_t)model->spec.width * (size_t)model->spec.height;
    for (int channel = 0; channel < model->spec.channels; ++channel) {
        const float *input_channel = source->image.data +
            (size_t)channel * (size_t)source->image.height * (size_t)source->image.width;
        float *output_channel = output + (size_t)channel * output_plane;
        for (int y = 0; y < model->spec.height; ++y) {
            float source_y = crop_y0 + ((float)y + 0.5f) * crop_height /
                             (float)model->spec.height - 0.5f;
            int y0 = (int)floorf(source_y);
            int y1 = y0 + 1;
            float fy = source_y - (float)y0;
            if (y0 < 0) y0 = 0;
            if (y1 < 0) y1 = 0;
            if (y0 >= source->image.height) y0 = source->image.height - 1;
            if (y1 >= source->image.height) y1 = source->image.height - 1;
            for (int x = 0; x < model->spec.width; ++x) {
                float source_x = crop_x0 + ((float)x + 0.5f) * crop_width /
                                 (float)model->spec.width - 0.5f;
                int x0 = (int)floorf(source_x);
                int x1 = x0 + 1;
                float fx = source_x - (float)x0;
                float top;
                float bottom;
                if (x0 < 0) x0 = 0;
                if (x1 < 0) x1 = 0;
                if (x0 >= source->image.width) x0 = source->image.width - 1;
                if (x1 >= source->image.width) x1 = source->image.width - 1;
                top = input_channel[(size_t)y0 * (size_t)source->image.width +
                                     (size_t)x0] * (1.0f - fx) +
                      input_channel[(size_t)y0 * (size_t)source->image.width +
                                    (size_t)x1] * fx;
                bottom = input_channel[(size_t)y1 * (size_t)source->image.width +
                                       (size_t)x0] * (1.0f - fx) +
                         input_channel[(size_t)y1 * (size_t)source->image.width +
                                       (size_t)x1] * fx;
                output_channel[(size_t)y * (size_t)model->spec.width + (size_t)x] =
                    top * (1.0f - fy) + bottom * fy;
            }
        }
    }
    target->x1 = (box->x1 - crop_x0) * (float)model->spec.width / crop_width;
    target->y1 = (box->y1 - crop_y0) * (float)model->spec.height / crop_height;
    target->x2 = (box->x2 - crop_x0) * (float)model->spec.width / crop_width;
    target->y2 = (box->y2 - crop_y0) * (float)model->spec.height / crop_height;
    target->x1 = fmaxf(0.0f, fminf((float)model->spec.width, target->x1));
    target->y1 = fmaxf(0.0f, fminf((float)model->spec.height, target->y1));
    target->x2 = fmaxf(0.0f, fminf((float)model->spec.width, target->x2));
    target->y2 = fmaxf(0.0f, fminf((float)model->spec.height, target->y2));
    target->class_id = box->class_id;
    return target->x2 > target->x1 && target->y2 > target->y1 ? DET_OK : DET_ERR_SHAPE;
}

static det_status det_kshira_bootstrap_pass(
    det_model *model, const det_dataset *dataset, double deadline,
    float learning_rate, det_train_report *report) {
    size_t pixel_count;
    float *crop_pixels;
    double start = now_ms();
    if (model == NULL || dataset == NULL || dataset->next == NULL || report == NULL ||
        !isfinite(learning_rate) || learning_rate <= 0.0f ||
        !checked_size3(model->spec.channels, model->spec.height, model->spec.width,
                       &pixel_count) || pixel_count > SIZE_MAX / sizeof(float)) {
        return DET_ERR_ARGUMENT;
    }
    crop_pixels = (float *)malloc(pixel_count * sizeof(*crop_pixels));
    if (crop_pixels == NULL) return DET_ERR_MEMORY;
    if (dataset->reset != NULL) dataset->reset(dataset->user);
    for (;;) {
        det_sample sample;
        int next_status;
        if (deadline > 0.0 && now_ms() >= deadline) break;
        next_status = dataset->next(dataset->user, &sample);
        if (next_status < 0) {
            free(crop_pixels);
            return DET_ERR_IO;
        }
        if (next_status == 0) break;
        if (!det_kshira_sample_valid(model, &sample)) {
            free(crop_pixels);
            return DET_ERR_ARGUMENT;
        }
        for (int box_index = 0; box_index < sample.box_count; ++box_index) {
            for (int variant = 0; variant < 2; ++variant) {
                det_box target;
                kshira_image_f32 image;
                kshira_status status;
                float loss = 0.0f;
                if (deadline > 0.0 && now_ms() >= deadline) break;
                if (det_kshira_make_bootstrap_crop(model, &sample,
                                                   &sample.boxes[box_index], variant,
                                                   crop_pixels, &target) != DET_OK) {
                    free(crop_pixels);
                    return DET_ERR_SHAPE;
                }
                image = (kshira_image_f32){crop_pixels, model->spec.channels,
                                           model->spec.height, model->spec.width};
                status = kshira_session_step(&model->kshira, &image,
                                             &(kshira_rad_box){target.x1, target.y1,
                                                               target.x2, target.y2,
                                                               target.class_id},
                                             learning_rate * (variant == 0 ? 1.0f : 0.75f),
                                             &loss);
                if (status != KSHIRA_OK) {
                    free(crop_pixels);
                    return det_status_from_kshira(status);
                }
                ++report->bootstrap_samples;
            }
        }
    }
    free(crop_pixels);
    report->bootstrap_elapsed_ms = now_ms() - start;
    return DET_OK;
}

/* Native adaptation uses the same deterministic crops, but trains the
 * deployment-side score and localization adapters rather than the unrelated
 * session model.  Crops increase positive density while still exposing the
 * background candidates produced by the complete native graph. */
static det_status det_native_bootstrap_pass(
    det_model *model, const det_dataset *dataset, double deadline,
    float learning_rate, det_train_report *report) {
    size_t pixel_count;
    float *crop_pixels;
    double start = now_ms();
    if (model == NULL || dataset == NULL || dataset->next == NULL || report == NULL ||
        !isfinite(learning_rate) || learning_rate <= 0.0f ||
        !checked_size3(model->spec.channels, model->spec.height, model->spec.width,
                       &pixel_count) || pixel_count > SIZE_MAX / sizeof(float)) {
        return DET_ERR_ARGUMENT;
    }
    crop_pixels = (float *)malloc(pixel_count * sizeof(*crop_pixels));
    if (crop_pixels == NULL) return DET_ERR_MEMORY;
    if (dataset->reset != NULL) dataset->reset(dataset->user);
    for (;;) {
        det_sample sample;
        int next_status;
        if (deadline > 0.0 && now_ms() >= deadline) break;
        next_status = dataset->next(dataset->user, &sample);
        if (next_status < 0) {
            free(crop_pixels);
            return DET_ERR_IO;
        }
        if (next_status == 0) break;
        if (!det_kshira_sample_valid(model, &sample)) {
            free(crop_pixels);
            return DET_ERR_ARGUMENT;
        }
        for (int box_index = 0; box_index < sample.box_count; ++box_index) {
            for (int variant = 0; variant < 2; ++variant) {
                det_box target_box;
                det_detection detections[DET_KSHIRA_NATIVE_MAX_DETECTIONS];
                det_image crop_image;
                int count = 0;
                float grad_class[DET_MAX_CLASSES] = {0.0f};
                float grad_scale[DET_MAX_SCALES] = {0.0f};
                float grad_head[DET_MAX_SCALES][DET_MAX_CLASSES] = {{0.0f}};
                float grad_box[DET_MAX_SCALES][4] = {{0.0f}};
                float grad_class_box[DET_MAX_SCALES][DET_MAX_CLASSES][4] = {{{0.0f}}};
                size_t box_terms[DET_MAX_SCALES] = {0U};
                size_t class_box_terms[DET_MAX_SCALES][DET_MAX_CLASSES] = {{0U}};
                size_t terms = 0U;
                float loss = 0.0f;
                if (deadline > 0.0 && now_ms() >= deadline) break;
                if (det_kshira_make_bootstrap_crop(model, &sample,
                                                   &sample.boxes[box_index], variant,
                                                   crop_pixels, &target_box) != DET_OK) {
                    free(crop_pixels);
                    return DET_ERR_SHAPE;
                }
                crop_image = (det_image){crop_pixels, model->spec.channels,
                                         model->spec.height, model->spec.width};
                if (det_native_graph_predict(model, &crop_image, 0.0f, detections,
                                             DET_KSHIRA_NATIVE_MAX_DETECTIONS,
                                             &count) != DET_OK) {
                    free(crop_pixels);
                    return DET_ERR_SHAPE;
                }
                for (int i = 0; i < count; ++i) {
                    int scale_bin = det_native_scale_bin(
                        detections[i].box.x1, detections[i].box.y1,
                        detections[i].box.x2, detections[i].box.y2,
                        crop_image.width, crop_image.height);
                    int class_id = detections[i].box.class_id;
                    int positive = class_id == target_box.class_id &&
                                   det_native_iou(&detections[i].box, &target_box) >= 0.5f;
                    float raw_logit = logf(fmaxf(1.0e-5f, detections[i].score) /
                                           fmaxf(1.0e-5f, 1.0f - detections[i].score));
                    float class_bias = class_id >= 0 && class_id < DET_MAX_CLASSES ?
                                       model->native_class_bias[class_id] : 0.0f;
                    float calibrated_logit = raw_logit - model->native_score_bias -
                                             class_bias - model->native_scale_bias[scale_bin];
                    float probability = det_sigmoid(model->native_score_gain *
                                                     calibrated_logit +
                                                     model->native_score_bias + class_bias +
                                                     model->native_scale_bias[scale_bin]);
                    float error = probability - (positive ? 1.0f : 0.0f);
                    if (class_id >= 0 && class_id < DET_MAX_CLASSES) {
                        grad_class[class_id] += error;
                        grad_head[scale_bin][class_id] += error;
                    }
                    grad_scale[scale_bin] += error;
                    loss += -(positive ? logf(fmaxf(probability, 1.0e-5f)) :
                              logf(fmaxf(1.0f - probability, 1.0e-5f)));
                    ++terms;
                    if (positive) {
                        float predicted_center_x = 0.5f *
                            (detections[i].box.x1 + detections[i].box.x2);
                        float predicted_center_y = 0.5f *
                            (detections[i].box.y1 + detections[i].box.y2);
                        float predicted_width = fmaxf(1.0f,
                            detections[i].box.x2 - detections[i].box.x1);
                        float predicted_height = fmaxf(1.0f,
                            detections[i].box.y2 - detections[i].box.y1);
                        float target_center_x = 0.5f * (target_box.x1 + target_box.x2);
                        float target_center_y = 0.5f * (target_box.y1 + target_box.y2);
                        float target_width = fmaxf(1.0f, target_box.x2 - target_box.x1);
                        float target_height = fmaxf(1.0f, target_box.y2 - target_box.y1);
                        float box_delta[4] = {
                            (target_center_x - predicted_center_x) /
                                (float)crop_image.width,
                            (target_center_y - predicted_center_y) /
                                (float)crop_image.height,
                            logf(target_width / predicted_width),
                            logf(target_height / predicted_height)
                        };
                        for (int component = 0; component < 4; ++component) {
                            grad_box[scale_bin][component] += box_delta[component];
                        }
                        ++box_terms[scale_bin];
                        for (int component = 0; component < 4; ++component) {
                            grad_class_box[scale_bin][class_id][component] +=
                                box_delta[component];
                        }
                        ++class_box_terms[scale_bin][class_id];
                    }
                }
                if (terms > 0U) {
                    float rate = learning_rate * (variant == 0 ? 0.35f : 0.20f);
                    float inverse = 1.0f / (float)terms;
                    for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
                        model->native_class_bias[class_id] -=
                            rate * grad_class[class_id] * inverse;
                        model->native_class_bias[class_id] = fmaxf(-3.0f, fminf(3.0f,
                            model->native_class_bias[class_id]));
                    }
                    for (int scale_bin = 0; scale_bin < DET_MAX_SCALES; ++scale_bin) {
                        model->native_scale_bias[scale_bin] -=
                            rate * grad_scale[scale_bin] * inverse;
                        model->native_scale_bias[scale_bin] = fmaxf(-3.0f, fminf(3.0f,
                            model->native_scale_bias[scale_bin]));
                        for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
                            det_native_update_head_bias(model, scale_bin, class_id,
                                -0.25f * rate * grad_head[scale_bin][class_id] * inverse);
                        }
                        if (box_terms[scale_bin] > 0U) {
                            float box_rate = 0.5f * rate /
                                             (float)box_terms[scale_bin];
                            for (int component = 0; component < 4; ++component) {
                                model->native_box_bias[scale_bin][component] +=
                                    box_rate * grad_box[scale_bin][component];
                                model->native_box_bias[scale_bin][component] =
                                    fmaxf(-0.5f, fminf(0.5f,
                                        model->native_box_bias[scale_bin][component]));
                            }
                        }
                        for (int class_id = 0; class_id < DET_MAX_CLASSES; ++class_id) {
                            if (class_box_terms[scale_bin][class_id] == 0U) continue;
                            float box_rate = 0.25f * rate /
                                             (float)class_box_terms[scale_bin][class_id];
                            for (int component = 0; component < 4; ++component) {
                                model->native_class_box_bias[scale_bin][class_id][component] +=
                                    box_rate * grad_class_box[scale_bin][class_id][component];
                                model->native_class_box_bias[scale_bin][class_id][component] =
                                    fmaxf(-0.5f, fminf(0.5f,
                                        model->native_class_box_bias[scale_bin][class_id][component]));
                            }
                        }
                    }
                    report->mean_loss += loss * inverse;
                }
                ++report->bootstrap_samples;
            }
        }
    }
    free(crop_pixels);
    report->bootstrap_elapsed_ms = now_ms() - start;
    return DET_OK;
}
