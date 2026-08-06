/*
 * Host-side loader for graph-package blobs into NGModelWeights.
 * Uses heap only for loading (not in the flight forward path). Flight builds
 * can instead embed const arrays; this loader is for bring-up and e2e tests.
 */
#ifndef NATIVE_GRAPH_LOADER_H
#define NATIVE_GRAPH_LOADER_H

#include "native_graph_ops.h"
#include "native_graph_model.h"

#include <stddef.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NG_LOADER_MAX_CONVS 128
#define NG_GRAPH_BASE 0
#define NG_GRAPH_MULTISCALE 26

typedef struct {
    NGConv2D convs[NG_LOADER_MAX_CONVS];
    int32_t conv_count;
    NGQuantMode weight_quant;
    int8_t silu_lut[256];
    uint16_t dfl_exp_lut[256];
    NGDflConfig dfl;
    /* Owned blobs (freed by ng_weights_free). */
    uint8_t *weight_blob;
    size_t weight_blob_size;
    float *weight_f32_blob;
    size_t weight_f32_blob_size;
    int32_t *bias_blob;
    size_t bias_count;
    float *bias_f32_blob;
    size_t bias_f32_count;
    int32_t *mult_blob;
    size_t mult_count;
    uint8_t *shift_blob;
    size_t shift_count;
    uint8_t *silu_blob;
    size_t silu_blob_size;
    uint8_t *silu_index_blob;
    size_t silu_index_count;
    float *activation_scale_blob;
    size_t activation_scale_count;
    int32_t class_count;
    NGModelScale scale;
    int32_t input_channels;
    float act_scale;
    int32_t architecture;
} NGLoadedWeights;

/*
 * Load a graph package directory containing:
 *   weights_meta.json, weights.bin, bias_i32.bin, multiplier_i32.bin,
 *   shift_u8.bin, silu_lut_s8.bin, dfl_exp_lut_q15.bin
 * Returns 0 on success.
 */
int ng_weights_load_dir(const char *export_dir, NGLoadedWeights *out);

/* Optional FP32 sidecars are loaded only when explicitly requested. */
int ng_weights_load_dir_mode(const char *export_dir, NGLoadedWeights *out,
                             int load_f32);

/* Fill NGModelWeights view (points into loaded storage). */
void ng_weights_as_model(const NGLoadedWeights *loaded, NGModelWeights *view);

void ng_weights_free(NGLoadedWeights *loaded);

#ifdef __cplusplus
}
#endif

#endif
