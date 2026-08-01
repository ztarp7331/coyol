/* Purpose: allocation-free IoU and class-hit metrics for small calibration sets.
 * Ownership: metrics are caller-owned accumulators; detections are read-only.
 * Failure: malformed boxes/counts or metric overflow return explicit status. */
#include "kshira/eval.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static float maximum(float first, float second) { return first > second ? first : second; }
static float minimum(float first, float second) { return first < second ? first : second; }

static int valid_box(const kshira_rad_box *box) {
    return box != NULL && isfinite(box->x1) && isfinite(box->y1) &&
           isfinite(box->x2) && isfinite(box->y2) && box->class_id >= 0 &&
           box->x2 > box->x1 && box->y2 > box->y1;
}

float kshira_box_iou(const kshira_rad_box *first, const kshira_rad_box *second) {
    float x1;
    float y1;
    float x2;
    float y2;
    float intersection;
    float first_area;
    float second_area;
    float union_area;
    if (!valid_box(first) || !valid_box(second)) return 0.0f;
    x1 = maximum(first->x1, second->x1);
    y1 = maximum(first->y1, second->y1);
    x2 = minimum(first->x2, second->x2);
    y2 = minimum(first->y2, second->y2);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    intersection = (x2 - x1) * (y2 - y1);
    first_area = (first->x2 - first->x1) * (first->y2 - first->y1);
    second_area = (second->x2 - second->x1) * (second->y2 - second->y1);
    union_area = first_area + second_area - intersection;
    if (!isfinite(intersection) || !isfinite(union_area) || union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}

void kshira_proxy_metrics_init(kshira_proxy_metrics *metrics) {
    if (metrics == NULL) return;
    metrics->samples = 0U;
    metrics->class_hits = 0U;
    metrics->iou_sum = 0.0f;
}

kshira_status kshira_proxy_metrics_add(kshira_proxy_metrics *metrics,
                                        const kshira_rad_box *target,
                                        const kshira_rad_detection *detections,
                                        int count) {
    float best_iou = 0.0f;
    int best_class = -1;
    if (metrics == NULL || !valid_box(target) || count < 0 ||
        (count > 0 && detections == NULL)) return KSHIRA_ERR_ARGUMENT;
    if (metrics->samples == SIZE_MAX) return KSHIRA_ERR_RANGE;
    for (int i = 0; i < count; ++i) {
        if (!valid_box(&detections[i].box) || !isfinite(detections[i].score) ||
            !isfinite(detections[i].quality)) return KSHIRA_ERR_ARGUMENT;
    }
    if (count > 0) {
        best_iou = kshira_box_iou(target, &detections[0].box);
        best_class = detections[0].box.class_id;
    }
    if (!isfinite(best_iou) || !isfinite(metrics->iou_sum + best_iou)) {
        return KSHIRA_ERR_RANGE;
    }
    ++metrics->samples;
    metrics->iou_sum += best_iou;
    if (best_iou > 0.0f && best_class == target->class_id) ++metrics->class_hits;
    return KSHIRA_OK;
}

float kshira_proxy_mean_iou(const kshira_proxy_metrics *metrics) {
    if (metrics == NULL || metrics->samples == 0U) return 0.0f;
    return metrics->iou_sum / (float)metrics->samples;
}

float kshira_proxy_class_accuracy(const kshira_proxy_metrics *metrics) {
    if (metrics == NULL || metrics->samples == 0U) return 0.0f;
    return (float)metrics->class_hits / (float)metrics->samples;
}
