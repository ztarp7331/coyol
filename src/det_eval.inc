typedef struct {
    det_detection detection;
    size_t sample_index;
    size_t ordinal;
} det_eval_prediction;

typedef struct {
    det_box box;
    size_t sample_index;
} det_eval_ground_truth;

static int append_eval_prediction(det_eval_prediction **items, size_t *count,
                                  size_t *capacity, const det_eval_prediction *item) {
    if (*count == *capacity) {
        size_t next = *capacity == 0U ? 64U : *capacity * 2U;
        if (next < *capacity || next > SIZE_MAX / sizeof(**items)) return 0;
        det_eval_prediction *grown = (det_eval_prediction *)realloc(
            *items, next * sizeof(**items));
        if (grown == NULL) return 0;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = *item;
    return 1;
}

static int append_eval_ground_truth(det_eval_ground_truth **items, size_t *count,
                                    size_t *capacity, const det_eval_ground_truth *item) {
    if (*count == *capacity) {
        size_t next = *capacity == 0U ? 64U : *capacity * 2U;
        if (next < *capacity || next > SIZE_MAX / sizeof(**items)) return 0;
        det_eval_ground_truth *grown = (det_eval_ground_truth *)realloc(
            *items, next * sizeof(**items));
        if (grown == NULL) return 0;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = *item;
    return 1;
}

static int compare_eval_predictions(const void *left, const void *right) {
    const det_eval_prediction *a = (const det_eval_prediction *)left;
    const det_eval_prediction *b = (const det_eval_prediction *)right;
    if (a->detection.score < b->detection.score) return 1;
    if (a->detection.score > b->detection.score) return -1;
    if (a->sample_index < b->sample_index) return -1;
    if (a->sample_index > b->sample_index) return 1;
    if (a->ordinal < b->ordinal) return -1;
    if (a->ordinal > b->ordinal) return 1;
    return 0;
}

typedef struct {
    size_t gt_start;
    size_t gt_count;
} det_eval_sample_range;

static int append_eval_range(det_eval_sample_range **items, size_t *count,
                             size_t *capacity, det_eval_sample_range item) {
    if (*count == *capacity) {
        size_t next = *capacity == 0U ? 64U : *capacity * 2U;
        if (next < *capacity || next > SIZE_MAX / sizeof(**items)) return 0;
        det_eval_sample_range *grown = (det_eval_sample_range *)realloc(
            *items, next * sizeof(**items));
        if (grown == NULL) return 0;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = item;
    return 1;
}

static int eval_box_size_group(const det_box *box) {
    float width = fmaxf(0.0f, box->x2 - box->x1);
    float height = fmaxf(0.0f, box->y2 - box->y1);
    float area = width * height;
    return area < 1024.0f ? 0 : (area < 9216.0f ? 1 : 2);
}

static float compute_ranked_ap(const det_eval_prediction *predictions, size_t prediction_count,
                               const size_t *prediction_indices, size_t prediction_index_count,
                               const det_eval_ground_truth *ground_truths,
                               size_t ground_truth_count,
                               const det_eval_sample_range *ranges, size_t range_count,
                               int class_filter, int size_filter, float iou_threshold,
                               unsigned char *matched, double *precisions, double *recalls,
                               size_t *out_ground_truths) {
    size_t ground_truth_total = 0U;
    for (size_t g = 0U; g < ground_truth_count; ++g) {
        if ((class_filter < 0 || ground_truths[g].box.class_id == class_filter) &&
            (size_filter < 0 || eval_box_size_group(&ground_truths[g].box) == size_filter)) {
            ++ground_truth_total;
        }
    }
    if (out_ground_truths != NULL) *out_ground_truths = ground_truth_total;
    if (ground_truth_total == 0U) return 0.0f;
    memset(matched, 0, ground_truth_count);
    size_t used = 0U;
    size_t true_positives = 0U;
    size_t candidate_count = prediction_indices != NULL ? prediction_index_count : prediction_count;
    for (size_t candidate = 0U; candidate < candidate_count; ++candidate) {
        size_t i = prediction_indices != NULL ? prediction_indices[candidate] : candidate;
        if (class_filter >= 0 && predictions[i].detection.box.class_id != class_filter) {
            continue;
        }
        if (size_filter >= 0 &&
            eval_box_size_group(&predictions[i].detection.box) != size_filter) {
            continue;
        }
        size_t best_index = SIZE_MAX;
        float best_iou = 0.0f;
        size_t sample_index = predictions[i].sample_index;
        if (sample_index < range_count) {
            det_eval_sample_range range = ranges[sample_index];
            for (size_t offset = 0U; offset < range.gt_count; ++offset) {
                size_t g = range.gt_start + offset;
                if (matched[g] != 0U ||
                    (class_filter >= 0 && ground_truths[g].box.class_id != class_filter) ||
                    (size_filter >= 0 && eval_box_size_group(&ground_truths[g].box) != size_filter)) {
                    continue;
                }
                float iou = det_iou(&predictions[i].detection.box, &ground_truths[g].box);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_index = g;
                }
            }
        }
        if (best_index != SIZE_MAX && best_iou >= iou_threshold) {
            matched[best_index] = 1U;
            ++true_positives;
        }
        precisions[used] = (double)true_positives / (double)(used + 1U);
        recalls[used] = (double)true_positives / (double)ground_truth_total;
        ++used;
    }
    double envelope = 0.0;
    for (size_t i = used; i > 0U; --i) {
        size_t index = i - 1U;
        if (precisions[index] > envelope) envelope = precisions[index];
        precisions[index] = envelope;
    }
    double ap = 0.0;
    size_t recall_index = 0U;
    for (int sample = 0; sample <= 100; ++sample) {
        double recall_threshold = (double)sample / 100.0;
        while (recall_index < used && recalls[recall_index] < recall_threshold) ++recall_index;
        double best_precision = recall_index < used ? precisions[recall_index] : 0.0;
        ap += best_precision / 101.0;
    }
    return (float)ap;
}

static det_status finalize_eval_metrics(const det_model *model,
                                        det_eval_prediction *predictions,
                                        size_t prediction_count,
                                        det_eval_ground_truth *ground_truths,
                                        size_t ground_truth_count,
                                        det_eval_sample_range *ranges,
                                        size_t range_count, det_eval_report *report) {
    if (prediction_count > SIZE_MAX / sizeof(double)) return DET_ERR_MEMORY;
    if (prediction_count > 1U) {
        qsort(predictions, prediction_count, sizeof(*predictions), compare_eval_predictions);
    }
    size_t class_counts[DET_MAX_CLASSES] = {0U};
    size_t class_offsets[DET_MAX_CLASSES] = {0U};
    size_t class_cursors[DET_MAX_CLASSES] = {0U};
    for (size_t i = 0U; i < prediction_count; ++i) {
        int class_id = predictions[i].detection.box.class_id;
        if (class_id >= 0 && class_id < DET_MAX_CLASSES) ++class_counts[class_id];
    }
    for (int class_id = 1; class_id < DET_MAX_CLASSES; ++class_id) {
        class_offsets[class_id] = class_offsets[class_id - 1] + class_counts[class_id - 1];
    }
    memcpy(class_cursors, class_offsets, sizeof(class_cursors));
    if (prediction_count > SIZE_MAX / sizeof(size_t)) return DET_ERR_MEMORY;
    size_t *class_indices = prediction_count > 0U ?
        (size_t *)malloc(prediction_count * sizeof(*class_indices)) : NULL;
    if (prediction_count > 0U && class_indices == NULL) return DET_ERR_MEMORY;
    for (size_t i = 0U; i < prediction_count; ++i) {
        int class_id = predictions[i].detection.box.class_id;
        if (class_id >= 0 && class_id < DET_MAX_CLASSES) {
            class_indices[class_cursors[class_id]++] = i;
        }
    }
    unsigned char *matched = ground_truth_count > 0U ?
        (unsigned char *)malloc(ground_truth_count) : NULL;
    double *precisions = prediction_count > 0U ?
        (double *)malloc(prediction_count * sizeof(*precisions)) : NULL;
    double *recalls = prediction_count > 0U ?
        (double *)malloc(prediction_count * sizeof(*recalls)) : NULL;
    if ((ground_truth_count > 0U && matched == NULL) ||
        (prediction_count > 0U && (precisions == NULL || recalls == NULL))) {
        free(matched);
        free(precisions);
        free(recalls);
        free(class_indices);
        return DET_ERR_MEMORY;
    }
    const float iou_thresholds[10] =
        {0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f};
    double map_sum = 0.0;
    size_t map_terms = 0U;
    double ap50_sum = 0.0;
    double ap75_sum = 0.0;
    size_t classes_with_ground_truth = 0U;
    for (int class_id = 0; class_id < model->spec.num_classes; ++class_id) {
        size_t class_ground_truths = 0U;
        float class_aps[10] = {0.0f};
        for (int t = 0; t < 10; ++t) {
            size_t ground_truth_total = 0U;
            const size_t *indices = class_indices == NULL ? NULL :
                class_indices + class_offsets[class_id];
            class_aps[t] = compute_ranked_ap(
                predictions, prediction_count, indices, class_counts[class_id],
                ground_truths, ground_truth_count,
                ranges, range_count, class_id, -1, iou_thresholds[t], matched,
                precisions, recalls, &ground_truth_total);
            if (t == 0) class_ground_truths = ground_truth_total;
        }
        if (class_ground_truths == 0U) continue;
        ++classes_with_ground_truth;
        report->class_ap50[class_id] = class_aps[0];
        ap50_sum += class_aps[0];
        ap75_sum += class_aps[5];
        for (int t = 0; t < 10; ++t) map_sum += class_aps[t];
        map_terms += 10U;
    }
    report->ap50 = classes_with_ground_truth > 0U ?
        (float)(ap50_sum / (double)classes_with_ground_truth) : 0.0f;
    report->ap75 = classes_with_ground_truth > 0U ?
        (float)(ap75_sum / (double)classes_with_ground_truth) : 0.0f;
    report->map50_95 = map_terms > 0U ? (float)(map_sum / (double)map_terms) : 0.0f;
    for (int size_group = 0; size_group < 3; ++size_group) {
        double size_ap_sum = 0.0;
        size_t size_classes = 0U;
        for (int class_id = 0; class_id < model->spec.num_classes; ++class_id) {
            size_t class_ground_truths = 0U;
            const size_t *indices = class_indices == NULL ? NULL :
                class_indices + class_offsets[class_id];
            float class_ap = compute_ranked_ap(
                predictions, prediction_count, indices, class_counts[class_id],
                ground_truths, ground_truth_count,
                ranges, range_count, class_id, size_group, 0.50f, matched, precisions,
                recalls, &class_ground_truths);
            if (class_ground_truths > 0U) {
                size_ap_sum += class_ap;
                ++size_classes;
            }
        }
        report->size_ap50[size_group] = size_classes > 0U ?
            (float)(size_ap_sum / (double)size_classes) : 0.0f;
    }
    free(matched);
    free(precisions);
    free(recalls);
    free(class_indices);
    return DET_OK;
}

det_status det_predict(const det_model *model, const det_image *image,
                       float score_threshold, det_detection *detections,
                       int capacity, int *count) {
    if (model == NULL || image == NULL || image->data == NULL || detections == NULL ||
        count == NULL || capacity < 0 || !isfinite(score_threshold) ||
        image->channels != model->spec.channels ||
        image->width != model->spec.width || image->height != model->spec.height) {
        return DET_ERR_ARGUMENT;
    }
    size_t image_elements = 0U;
    if (!checked_size3(image->channels, image->height, image->width, &image_elements) ||
        !finite_values(image->data, image_elements)) return DET_ERR_ARGUMENT;
    *count = 0;
    backbone_forward((det_model *)model, image);
    float output[4 + DET_MAX_CLASSES];
    for (int s = 0; s < DET_MAX_SCALES; ++s) {
        const det_head *head = &model->heads[s];
        for (int y = 0; y < head->height; ++y) {
            for (int x = 0; x < head->width; ++x) {
                float features[4];
                extract_feature(model, image, s, y, x, features, NULL);
                if (model->precision == DET_PRECISION_F32) {
                    head_forward(head, features, output);
                } else {
                    head_forward_quant(head, features, model->precision, output);
                }
                int class_id = 0;
                float best = output[4];
                for (int c = 1; c < model->spec.num_classes; ++c) {
                    if (output[c + 4] > best) {
                        best = output[c + 4];
                        class_id = c;
                    }
                }
                float score = det_sigmoid(best);
                if (score < score_threshold) continue;
                float cx = ((float)x + 0.5f) * (float)k_strides[s];
                float cy = ((float)y + 0.5f) * (float)k_strides[s];
                float left = fmaxf(0.0f, output[0]) * (float)k_strides[s];
                float top = fmaxf(0.0f, output[1]) * (float)k_strides[s];
                float right = fmaxf(0.0f, output[2]) * (float)k_strides[s];
                float bottom = fmaxf(0.0f, output[3]) * (float)k_strides[s];
                det_detection candidate;
                candidate.score = score;
                candidate.box.x1 = fmaxf(0.0f, cx - left);
                candidate.box.y1 = fmaxf(0.0f, cy - top);
                candidate.box.x2 = fminf((float)image->width, cx + right);
                candidate.box.y2 = fminf((float)image->height, cy + bottom);
                candidate.box.class_id = class_id;
                insert_detection(detections, count, capacity, &candidate);
            }
        }
    }
    sort_detections_descending(detections, *count);
    return DET_OK;
}

det_status det_evaluate(const det_model *model, const det_dataset *dataset,
                        float score_threshold, det_eval_report *report) {
    if (model == NULL || dataset == NULL || report == NULL || dataset->next == NULL ||
        !isfinite(score_threshold) || score_threshold < 0.0f || score_threshold > 1.0f) {
        return DET_ERR_ARGUMENT;
    }
    if ((size_t)model->spec.max_detections > SIZE_MAX / sizeof(det_detection)) {
        return DET_ERR_MEMORY;
    }
    det_detection *detections = (det_detection *)malloc(
        (size_t)model->spec.max_detections * sizeof(*detections));
    if (detections == NULL) return DET_ERR_MEMORY;
    det_eval_prediction *predictions = NULL;
    det_eval_ground_truth *ground_truths = NULL;
    det_eval_sample_range *ranges = NULL;
    size_t prediction_count = 0U, prediction_capacity = 0U;
    size_t ground_truth_count = 0U, ground_truth_capacity = 0U;
    size_t range_count = 0U, range_capacity = 0U;
    memset(report, 0, sizeof(*report));
    if (dataset->reset != NULL) dataset->reset(dataset->user);
    double iou_sum = 0.0;
    det_status status = DET_OK;
    unsigned char *matched = NULL;
    size_t matched_capacity = 0U;
    size_t sample_index = 0U;
    for (;;) {
        det_sample sample;
        int next_status = dataset->next(dataset->user, &sample);
        if (next_status < 0) {
            status = DET_ERR_IO;
            break;
        }
        if (next_status == 0) break;
        if (!validate_sample(model, &sample)) {
            status = DET_ERR_ARGUMENT;
            break;
        }
        size_t gt_start = ground_truth_count;
        for (int b = 0; b < sample.box_count; ++b) {
            det_eval_ground_truth item = {sample.boxes[b], sample_index};
            if (!append_eval_ground_truth(&ground_truths, &ground_truth_count,
                                          &ground_truth_capacity, &item)) {
                status = DET_ERR_MEMORY;
                break;
            }
            ++report->size_ground_truths[eval_box_size_group(&sample.boxes[b])];
        }
        if (status != DET_OK) break;
        if (!append_eval_range(&ranges, &range_count, &range_capacity,
                               (det_eval_sample_range){gt_start,
                                                       (size_t)sample.box_count})) {
            status = DET_ERR_MEMORY;
            break;
        }
        if ((size_t)sample.box_count > matched_capacity) {
            unsigned char *grown = (unsigned char *)realloc(
                matched, (size_t)sample.box_count);
            if (grown == NULL) {
                status = DET_ERR_MEMORY;
                break;
            }
            matched = grown;
            matched_capacity = (size_t)sample.box_count;
        }
        if (sample.box_count > 0) memset(matched, 0, (size_t)sample.box_count);
        int detection_count = 0;
        status = det_predict(model, &sample.image, 0.0f, detections,
                             model->spec.max_detections, &detection_count);
        if (status != DET_OK) {
            break;
        }
        report->samples_seen += 1U;
        report->ground_truths += (size_t)sample.box_count;
        size_t matched_count = 0U;
        for (int d = 0; d < detection_count; ++d) {
            det_eval_prediction item = {detections[d], sample_index, prediction_count};
            if (!append_eval_prediction(&predictions, &prediction_count,
                                        &prediction_capacity, &item)) {
                status = DET_ERR_MEMORY;
                break;
            }
            if (detections[d].score < score_threshold) continue;
            ++report->predictions;
            int best_index = -1;
            float best_iou = 0.0f;
            for (int b = 0; b < sample.box_count; ++b) {
                if (matched[b] != 0U || sample.boxes[b].class_id != detections[d].box.class_id) {
                    continue;
                }
                float iou = det_iou(&detections[d].box, &sample.boxes[b]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_index = b;
                }
            }
            if (best_index >= 0 && best_iou >= 0.5f) {
                matched[best_index] = 1U;
                ++matched_count;
                ++report->true_positives;
                iou_sum += (double)best_iou;
            } else {
                ++report->false_positives;
            }
        }
        if (status != DET_OK) break;
        report->false_negatives += (size_t)sample.box_count - matched_count;
        ++sample_index;
    }
    free(matched);
    free(detections);
    if (status != DET_OK) {
        free(predictions);
        free(ground_truths);
        free(ranges);
        return status;
    }
    if (report->samples_seen == 0U) {
        free(predictions);
        free(ground_truths);
        free(ranges);
        return DET_ERR_ARGUMENT;
    }
    size_t predicted_positive = report->true_positives + report->false_positives;
    report->precision = predicted_positive > 0U ?
        (float)report->true_positives / (float)predicted_positive : 0.0f;
    report->recall = report->ground_truths > 0U ?
        (float)report->true_positives / (float)report->ground_truths : 0.0f;
    report->mean_iou = report->true_positives > 0U ?
        (float)(iou_sum / (double)report->true_positives) : 0.0f;
    status = finalize_eval_metrics(model, predictions, prediction_count,
                                   ground_truths, ground_truth_count, ranges,
                                   range_count, report);
    free(predictions);
    free(ground_truths);
    free(ranges);
    return status;
}


