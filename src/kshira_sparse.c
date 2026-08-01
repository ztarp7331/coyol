/* Purpose: explicit channel masks and conservative sparse-update SRAM bounds.
 * Ownership: callers provide mask storage and decide which tensors are live.
 * Failure: invalid channels or overflow returns an error/SIZE_MAX estimate. */
#include "kshira/sparse.h"

#include <string.h>

static size_t bit_bytes(size_t count) {
    if (count > SIZE_MAX - 7U) return SIZE_MAX;
    return (count + 7U) / 8U;
}

kshira_status kshira_sparse_mask_init(kshira_sparse_mask *mask, uint8_t *storage,
                                       size_t storage_bytes, size_t channel_count) {
    size_t required;
    if (mask == NULL || storage == NULL || channel_count == 0U) return KSHIRA_ERR_ARGUMENT;
    required = bit_bytes(channel_count);
    if (required == SIZE_MAX || storage_bytes < required) return KSHIRA_ERR_RANGE;
    mask->channel_bits = storage;
    mask->channel_count = channel_count;
    memset(storage, 0, required);
    return KSHIRA_OK;
}

void kshira_sparse_mask_clear(kshira_sparse_mask *mask) {
    if (mask == NULL || mask->channel_bits == NULL) return;
    memset(mask->channel_bits, 0, bit_bytes(mask->channel_count));
}

kshira_status kshira_sparse_mask_set(kshira_sparse_mask *mask, size_t channel, int enabled) {
    uint8_t bit;
    if (mask == NULL || mask->channel_bits == NULL || channel >= mask->channel_count ||
        (enabled != 0 && enabled != 1)) return KSHIRA_ERR_ARGUMENT;
    bit = (uint8_t)(1U << (channel & 7U));
    if (enabled != 0) mask->channel_bits[channel / 8U] |= bit;
    else mask->channel_bits[channel / 8U] &= (uint8_t)~bit;
    return KSHIRA_OK;
}

int kshira_sparse_mask_get(const kshira_sparse_mask *mask, size_t channel) {
    if (mask == NULL || mask->channel_bits == NULL || channel >= mask->channel_count) return 0;
    return (mask->channel_bits[channel / 8U] & (uint8_t)(1U << (channel & 7U))) != 0U;
}

size_t kshira_sparse_mask_active(const kshira_sparse_mask *mask) {
    size_t active = 0U;
    if (mask == NULL) return 0U;
    for (size_t i = 0U; i < mask->channel_count; ++i) {
        active += (size_t)kshira_sparse_mask_get(mask, i);
    }
    return active;
}

size_t kshira_sparse_memory_bytes(size_t activation_bytes, size_t weight_bytes,
                                  size_t bias_bytes, size_t channel_count,
                                  kshira_update_mode mode, size_t active_channels) {
    size_t gradient_bytes = 0U;
    if (mode == KSHIRA_UPDATE_BIAS) gradient_bytes = bias_bytes;
    else if (mode == KSHIRA_UPDATE_CHANNELS) {
        size_t per_channel;
        if (channel_count == 0U || active_channels > channel_count) return SIZE_MAX;
        if (active_channels == 0U) gradient_bytes = bias_bytes;
        else {
            if (weight_bytes > SIZE_MAX - (channel_count - 1U)) return SIZE_MAX;
            per_channel = (weight_bytes + channel_count - 1U) / channel_count;
            if (per_channel > SIZE_MAX / active_channels ||
                per_channel * active_channels > SIZE_MAX - bias_bytes) return SIZE_MAX;
            gradient_bytes = per_channel * active_channels + bias_bytes;
        }
    } else if (mode == KSHIRA_UPDATE_FULL) {
        if (weight_bytes > SIZE_MAX - bias_bytes) return SIZE_MAX;
        gradient_bytes = weight_bytes + bias_bytes;
    } else if (mode != KSHIRA_UPDATE_FREEZE) return SIZE_MAX;
    if (activation_bytes > SIZE_MAX - gradient_bytes) return SIZE_MAX;
    return activation_bytes + gradient_bytes;
}

kshira_status kshira_sparse_plan(kshira_update_mode mode, size_t activation_bytes,
                                 size_t weight_bytes, size_t bias_bytes,
                                 size_t channel_count, size_t active_channels,
                                 size_t arena_cap, size_t *peak_bytes) {
    size_t estimate;
    if (peak_bytes == NULL || arena_cap == 0U) return KSHIRA_ERR_ARGUMENT;
    estimate = kshira_sparse_memory_bytes(activation_bytes, weight_bytes, bias_bytes,
                                          channel_count, mode, active_channels);
    if (estimate == SIZE_MAX) return KSHIRA_ERR_RANGE;
    *peak_bytes = estimate;
    return estimate <= arena_cap ? KSHIRA_OK : KSHIRA_ERR_MEMORY;
}
