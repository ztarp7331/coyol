/* Purpose: compose the caller-owned arena, RAD model, sparse mask, and phase
 * contract into one deterministic PRE/TRAIN/ODT session API.
 * Ownership: the caller owns session storage and backing memory; no allocation
 * occurs in a transition, train step, or prediction call.
 * Failure: invalid phase/config or arena exhaustion returns an explicit status. */
#include "kshira/session.h"

#include <stddef.h>
#include <stdint.h>

static void session_rollback(kshira_session *session, size_t offset, size_t high_water) {
    session->arena.offset = offset;
    session->arena.high_water = high_water;
    session->rad = NULL;
    session->channel_mask = (kshira_sparse_mask){0};
}

static size_t mask_bytes(size_t channels) {
    if (channels > SIZE_MAX - 7U) return SIZE_MAX;
    return (channels + 7U) / 8U;
}

kshira_status kshira_session_init(kshira_session *session, void *memory,
                                   size_t memory_bytes, const kshira_session_spec *spec) {
    size_t bytes;
    size_t start_offset;
    size_t start_high_water;
    uint8_t *mask_storage;
    kshira_status status;
    if (session == NULL || memory == NULL || spec == NULL || spec->arena_cap == 0U ||
        spec->arena_cap > memory_bytes) return KSHIRA_ERR_ARGUMENT;
    if (kshira_arena_init(&session->arena, memory, spec->arena_cap) != KSHIRA_OK ||
        kshira_phase_driver_init(&session->phase, spec->arena_cap) != KSHIRA_OK) {
        return KSHIRA_ERR_ARGUMENT;
    }
    start_offset = session->arena.offset;
    start_high_water = session->arena.high_water;
    session->rad = NULL;
    status = kshira_rad_build(&session->arena, &spec->rad, &session->rad);
    if (status != KSHIRA_OK) {
        session_rollback(session, start_offset, start_high_water);
        return status;
    }
    bytes = mask_bytes((size_t)spec->rad.feature_channels);
    if (bytes == SIZE_MAX) {
        session_rollback(session, start_offset, start_high_water);
        return KSHIRA_ERR_RANGE;
    }
    mask_storage = (uint8_t *)kshira_arena_alloc(&session->arena, bytes, _Alignof(uint8_t));
    if (mask_storage == NULL) {
        session_rollback(session, start_offset, start_high_water);
        return KSHIRA_ERR_MEMORY;
    }
    status = kshira_sparse_mask_init(&session->channel_mask, mask_storage, bytes,
                                     (size_t)spec->rad.feature_channels);
    if (status != KSHIRA_OK) {
        session_rollback(session, start_offset, start_high_water);
        return status;
    }
    return KSHIRA_OK;
}

kshira_status kshira_session_transition(kshira_session *session, kshira_bit_mode bits,
                                         kshira_update_mode update_mode, int qas_enabled) {
    kshira_phase_contract next;
    if (session == NULL) return KSHIRA_ERR_ARGUMENT;
    next.phase = (kshira_phase)(session->phase.contract.phase + 1);
    next.bits = bits;
    next.update_mode = update_mode;
    next.arena_cap = session->phase.contract.arena_cap;
    next.qas_enabled = qas_enabled;
    if (kshira_phase_validate(&next) != KSHIRA_OK) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (kshira_rad_set_bits(session->rad, bits) != KSHIRA_OK) return KSHIRA_ERR_ARGUMENT;
    return kshira_phase_driver_transition(&session->phase, next);
}

kshira_status kshira_session_set_channel(kshira_session *session, size_t channel, int enabled) {
    if (session == NULL) return KSHIRA_ERR_ARGUMENT;
    return kshira_sparse_mask_set(&session->channel_mask, channel, enabled);
}

kshira_status kshira_session_step(kshira_session *session, const kshira_image_f32 *image,
                                   const kshira_rad_box *target, float learning_rate,
                                   float *loss) {
    kshira_rad_train_config config;
    kshira_phase_contract contract;
    if (session == NULL || loss == NULL) return KSHIRA_ERR_ARGUMENT;
    contract = session->phase.contract;
    if (kshira_phase_validate(&contract) != KSHIRA_OK) return KSHIRA_ERR_ARGUMENT;
    if (session->phase.steps == SIZE_MAX) return KSHIRA_ERR_RANGE;
    config.bits = contract.bits;
    config.update_mode = contract.update_mode;
    config.channel_mask = &session->channel_mask;
    config.learning_rate = learning_rate;
    {
        kshira_status status = kshira_rad_train_step(session->rad, image, target, &config, loss);
        if (status != KSHIRA_OK) return status;
    }
    return kshira_phase_driver_step(&session->phase);
}

kshira_status kshira_session_predict(kshira_session *session, const kshira_image_f32 *image,
                                      float threshold, kshira_rad_detection *detections,
                                      int capacity, int *count) {
    if (session == NULL) return KSHIRA_ERR_ARGUMENT;
    return kshira_rad_predict(session->rad, image, threshold, detections, capacity, count);
}

const kshira_phase_contract *kshira_session_contract(const kshira_session *session) {
    return session == NULL ? NULL : &session->phase.contract;
}

size_t kshira_session_arena_high_water(const kshira_session *session) {
    return session == NULL ? 0U : kshira_arena_high_water(&session->arena);
}
