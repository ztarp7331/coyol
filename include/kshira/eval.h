#ifndef KSHIRA_EVAL_H
#define KSHIRA_EVAL_H

#include <stddef.h>

#include "kshira/rad.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t samples;
    size_t class_hits;
    float iou_sum;
} kshira_proxy_metrics;

void kshira_proxy_metrics_init(kshira_proxy_metrics *metrics);
kshira_status kshira_proxy_metrics_add(kshira_proxy_metrics *metrics,
                                        const kshira_rad_box *target,
                                        const kshira_rad_detection *detections,
                                        int count);
float kshira_box_iou(const kshira_rad_box *first, const kshira_rad_box *second);
float kshira_proxy_mean_iou(const kshira_proxy_metrics *metrics);
float kshira_proxy_class_accuracy(const kshira_proxy_metrics *metrics);

#ifdef __cplusplus
}
#endif

#endif
