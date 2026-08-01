#ifndef KSHIRA_SESSION_H
#define KSHIRA_SESSION_H

#include <stddef.h>

#include "kshira/phase.h"
#include "kshira/rad.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    kshira_rad_spec rad;
    size_t arena_cap;
} kshira_session_spec;

typedef struct {
    kshira_arena arena;
    kshira_phase_driver phase;
    kshira_rad_model *rad;
    kshira_sparse_mask channel_mask;
} kshira_session;

kshira_status kshira_session_init(kshira_session *session, void *memory,
                                   size_t memory_bytes, const kshira_session_spec *spec);
kshira_status kshira_session_transition(kshira_session *session, kshira_bit_mode bits,
                                         kshira_update_mode update_mode, int qas_enabled);
kshira_status kshira_session_calibrate(kshira_session *session,
                                        const kshira_image_f32 *image);
kshira_status kshira_session_set_channel(kshira_session *session, size_t channel, int enabled);
kshira_status kshira_session_step(kshira_session *session, const kshira_image_f32 *image,
                                   const kshira_rad_box *target, float learning_rate,
                                   float *loss);
kshira_status kshira_session_multiscale_step(kshira_session *session,
                                               const kshira_image_f32 *image,
                                               const kshira_rad_box *target, int level,
                                               float learning_rate, float *loss);
kshira_status kshira_session_predict(kshira_session *session, const kshira_image_f32 *image,
                                      float threshold, kshira_rad_detection *detections,
                                      int capacity, int *count);
const kshira_phase_contract *kshira_session_contract(const kshira_session *session);
size_t kshira_session_arena_high_water(const kshira_session *session);

#ifdef __cplusplus
}
#endif

#endif
