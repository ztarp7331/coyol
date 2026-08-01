#ifndef KSHIRA_QUANT_H
#define KSHIRA_QUANT_H

#include "kshira/core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KSHIRA_BITS_INT4 = 4,
    KSHIRA_BITS_INT8 = 8
} kshira_bit_mode;

typedef struct {
    kshira_bit_mode mode;
    float scale;
    size_t count;
    const uint8_t *packed_int4;
    size_t packed_bytes;
    const int8_t *expanded_int8;
} kshira_weight_store;

int kshira_bit_mode_valid(kshira_bit_mode mode);
float kshira_symmetric_scale(float max_abs, kshira_bit_mode mode);
int8_t kshira_quantize_symmetric(float value, float scale, kshira_bit_mode mode);
float kshira_dequantize_symmetric(int8_t value, float scale);
kshira_status kshira_pack_int4(const float *values, size_t count, float scale,
                               uint8_t *packed, size_t packed_bytes);
kshira_status kshira_unpack_int4(const uint8_t *packed, size_t packed_bytes, size_t count,
                                 int8_t *values, size_t value_bytes);
kshira_status kshira_quantize_int8(const float *values, size_t count, float scale,
                                   int8_t *quantized, size_t quantized_bytes);

/* QAS applies the scale correction to real-quantized training gradients. */
kshira_status kshira_apply_qas(float *gradient_weights, size_t weight_count,
                               float *gradient_bias, size_t bias_count,
                               float weight_scale, float input_scale);

kshira_status kshira_weight_store_init(kshira_weight_store *store,
                                       const float *weights, size_t count,
                                       float scale, uint8_t *packed_int4,
                                       size_t packed_bytes);
kshira_status kshira_weight_store_set_mode(kshira_weight_store *store,
                                            kshira_bit_mode mode,
                                            int8_t *expanded_int8,
                                            size_t expanded_bytes);
int8_t kshira_weight_store_at(const kshira_weight_store *store, size_t index);

#ifdef __cplusplus
}
#endif

#endif
