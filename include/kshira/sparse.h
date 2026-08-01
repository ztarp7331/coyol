#ifndef KSHIRA_SPARSE_H
#define KSHIRA_SPARSE_H

#include <stddef.h>
#include <stdint.h>

#include "kshira/core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KSHIRA_UPDATE_FREEZE = 0,
    KSHIRA_UPDATE_BIAS = 1,
    KSHIRA_UPDATE_CHANNELS = 2,
    KSHIRA_UPDATE_FULL = 3
} kshira_update_mode;

typedef struct {
    uint8_t *channel_bits;
    size_t channel_count;
} kshira_sparse_mask;

kshira_status kshira_sparse_mask_init(kshira_sparse_mask *mask, uint8_t *storage,
                                       size_t storage_bytes, size_t channel_count);
void kshira_sparse_mask_clear(kshira_sparse_mask *mask);
kshira_status kshira_sparse_mask_set(kshira_sparse_mask *mask, size_t channel, int enabled);
int kshira_sparse_mask_get(const kshira_sparse_mask *mask, size_t channel);
size_t kshira_sparse_mask_active(const kshira_sparse_mask *mask);

/* Conservative peak SRAM estimate for a scheduled sparse update. */
size_t kshira_sparse_memory_bytes(size_t activation_bytes, size_t weight_bytes,
                                  size_t bias_bytes, size_t channel_count,
                                  kshira_update_mode mode, size_t active_channels);
kshira_status kshira_sparse_plan(kshira_update_mode mode, size_t activation_bytes,
                                 size_t weight_bytes, size_t bias_bytes,
                                 size_t channel_count, size_t active_channels,
                                 size_t arena_cap, size_t *peak_bytes);

#ifdef __cplusplus
}
#endif

#endif
