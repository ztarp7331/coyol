#ifndef NATIVE_MULTISCALE_H
#define NATIVE_MULTISCALE_H

#include "native_graph_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
} NGMDetection;

/* Runs the fused multi-scale graph and returns final top-K boxes. */
int ngm_detect_s8(const NGLoadedWeights *weights, const int8_t *input,
                  int32_t input_channels, int32_t height, int32_t width,
                  float score_threshold, int32_t max_detections,
                  NGArena *arena, NGMDetection *detections, int32_t *count);

/* Exact FP32 reference of the same exported graph, used to isolate quantization loss. */
int ngm_detect_f32(const NGLoadedWeights *weights, const float *input,
                   int32_t input_channels, int32_t height, int32_t width,
                   float score_threshold, int32_t max_detections,
                   NGArena *arena, NGMDetection *detections, int32_t *count);

#ifdef __cplusplus
}
#endif

#endif
