/* Purpose: symmetric INT4/INT8 arithmetic, packed toggle, and QAS scaling.
 * Ownership: callers own all source and destination buffers.
 * Failure: invalid scales, buffers, or capacities return a status; no hidden
 * allocation or persistent state is used. */
#include "kshira/quant.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int8_t clamp_int8(int value, int low, int high) {
    if (value < low) return (int8_t)low;
    if (value > high) return (int8_t)high;
    return (int8_t)value;
}

int kshira_bit_mode_valid(kshira_bit_mode mode) {
    return mode == KSHIRA_BITS_INT4 || mode == KSHIRA_BITS_INT8;
}

float kshira_symmetric_scale(float max_abs, kshira_bit_mode mode) {
    float limit;
    if (!kshira_bit_mode_valid(mode) || !isfinite(max_abs) || max_abs < 0.0f) return 0.0f;
    limit = mode == KSHIRA_BITS_INT4 ? 7.0f : 127.0f;
    return max_abs > 0.0f ? max_abs / limit : 1.0f;
}

int8_t kshira_quantize_symmetric(float value, float scale, kshira_bit_mode mode) {
    int low;
    int high;
    float rounded;
    if (!kshira_bit_mode_valid(mode) || !isfinite(value) || !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }
    low = mode == KSHIRA_BITS_INT4 ? -7 : -127;
    high = mode == KSHIRA_BITS_INT4 ? 7 : 127;
    rounded = nearbyintf(value / scale);
    if (!isfinite(rounded)) return 0;
    if (rounded <= (float)low) return (int8_t)low;
    if (rounded >= (float)high) return (int8_t)high;
    return clamp_int8((int)rounded, low, high);
}

float kshira_dequantize_symmetric(int8_t value, float scale) {
    return isfinite(scale) && scale > 0.0f ? (float)value * scale : 0.0f;
}

kshira_status kshira_pack_int4(const float *values, size_t count, float scale,
                               uint8_t *packed, size_t packed_bytes) {
    size_t required;
    if (values == NULL || packed == NULL || !isfinite(scale) || scale <= 0.0f) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (count > SIZE_MAX - 1U) return KSHIRA_ERR_RANGE;
    required = (count + 1U) / 2U;
    if (packed_bytes < required) return KSHIRA_ERR_RANGE;
    for (size_t i = 0U; i < count; i += 2U) {
        int8_t first = kshira_quantize_symmetric(values[i], scale, KSHIRA_BITS_INT4);
        int8_t second = i + 1U < count ?
            kshira_quantize_symmetric(values[i + 1U], scale, KSHIRA_BITS_INT4) : 0;
        packed[i / 2U] = (uint8_t)((uint8_t)first & 0x0fU) |
                         (uint8_t)(((uint8_t)second & 0x0fU) << 4U);
    }
    return KSHIRA_OK;
}

kshira_status kshira_unpack_int4(const uint8_t *packed, size_t packed_bytes, size_t count,
                                 int8_t *values, size_t value_bytes) {
    size_t required;
    if (packed == NULL || values == NULL || value_bytes < count || count > SIZE_MAX - 1U) {
        return KSHIRA_ERR_ARGUMENT;
    }
    required = (count + 1U) / 2U;
    if (packed_bytes < required) return KSHIRA_ERR_RANGE;
    for (size_t i = 0U; i < count; ++i) {
        unsigned int shift = (unsigned int)((i & 1U) * 4U);
        unsigned int packed_byte = (unsigned int)packed[i / 2U];
        uint8_t nibble = (uint8_t)((packed_byte >> shift) & 0x0fU);
        values[i] = (int8_t)(nibble >= 8U ? (int)nibble - 16 : (int)nibble);
    }
    return KSHIRA_OK;
}

kshira_status kshira_quantize_int8(const float *values, size_t count, float scale,
                                   int8_t *quantized, size_t quantized_bytes) {
    if (values == NULL || quantized == NULL || quantized_bytes < count ||
        !isfinite(scale) || scale <= 0.0f) return KSHIRA_ERR_ARGUMENT;
    for (size_t i = 0U; i < count; ++i) {
        quantized[i] = kshira_quantize_symmetric(values[i], scale, KSHIRA_BITS_INT8);
    }
    return KSHIRA_OK;
}

kshira_status kshira_apply_qas(float *gradient_weights, size_t weight_count,
                               float *gradient_bias, size_t bias_count,
                               float weight_scale, float input_scale) {
    float weight_factor;
    float bias_factor;
    if ((gradient_weights == NULL && weight_count != 0U) ||
        (gradient_bias == NULL && bias_count != 0U) ||
        !isfinite(weight_scale) || !isfinite(input_scale) ||
        weight_scale <= 0.0f || input_scale <= 0.0f) return KSHIRA_ERR_ARGUMENT;
    weight_factor = 1.0f / (weight_scale * weight_scale);
    bias_factor = weight_factor / (input_scale * input_scale);
    for (size_t i = 0U; i < weight_count; ++i) gradient_weights[i] *= weight_factor;
    for (size_t i = 0U; i < bias_count; ++i) gradient_bias[i] *= bias_factor;
    return KSHIRA_OK;
}

kshira_status kshira_weight_store_init(kshira_weight_store *store,
                                       const float *weights, size_t count,
                                       float scale, uint8_t *packed_int4,
                                       size_t packed_bytes) {
    if (store == NULL || weights == NULL || packed_int4 == NULL || count == 0U ||
        !isfinite(scale) || scale <= 0.0f) return KSHIRA_ERR_ARGUMENT;
    if (kshira_pack_int4(weights, count, scale, packed_int4, packed_bytes) != KSHIRA_OK) {
        return KSHIRA_ERR_RANGE;
    }
    store->mode = KSHIRA_BITS_INT4;
    store->scale = scale;
    store->count = count;
    store->packed_int4 = packed_int4;
    store->packed_bytes = (count + 1U) / 2U;
    store->expanded_int8 = NULL;
    return KSHIRA_OK;
}

kshira_status kshira_weight_store_set_mode(kshira_weight_store *store,
                                            kshira_bit_mode mode,
                                            int8_t *expanded_int8,
                                            size_t expanded_bytes) {
    if (store == NULL || !kshira_bit_mode_valid(mode)) return KSHIRA_ERR_ARGUMENT;
    if (mode == KSHIRA_BITS_INT4) {
        store->mode = mode;
        store->expanded_int8 = NULL;
        return KSHIRA_OK;
    }
    if (expanded_int8 == NULL || expanded_bytes < store->count) return KSHIRA_ERR_MEMORY;
    if (kshira_unpack_int4(store->packed_int4, store->packed_bytes, store->count,
                           expanded_int8, expanded_bytes) != KSHIRA_OK) return KSHIRA_ERR_RANGE;
    store->mode = mode;
    store->expanded_int8 = expanded_int8;
    return KSHIRA_OK;
}

int8_t kshira_weight_store_at(const kshira_weight_store *store, size_t index) {
    if (store == NULL || store->packed_int4 == NULL || index >= store->count) return 0;
    if (store->mode == KSHIRA_BITS_INT8 && store->expanded_int8 != NULL) {
        return store->expanded_int8[index];
    }
    {
        unsigned int shift = (unsigned int)((index & 1U) * 4U);
        unsigned int packed_byte = (unsigned int)store->packed_int4[index / 2U];
        uint8_t nibble = (uint8_t)((packed_byte >> shift) & 0x0fU);
        return (int8_t)(nibble >= 8U ? (int)nibble - 16 : (int)nibble);
    }
}
