#ifndef KSHIRA_PHASE_H
#define KSHIRA_PHASE_H

#include <stddef.h>

#include "kshira/quant.h"
#include "kshira/sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KSHIRA_PHASE_PRE = 0,
    KSHIRA_PHASE_TRAIN = 1,
    KSHIRA_PHASE_ODT = 2
} kshira_phase;

typedef struct {
    kshira_phase phase;
    kshira_bit_mode bits;
    kshira_update_mode update_mode;
    size_t arena_cap;
    int qas_enabled;
} kshira_phase_contract;

typedef struct {
    kshira_phase_contract contract;
    size_t steps;
    size_t transitions;
} kshira_phase_driver;

kshira_status kshira_phase_validate(const kshira_phase_contract *contract);
const char *kshira_phase_name(kshira_phase phase);
kshira_status kshira_phase_driver_init(kshira_phase_driver *driver, size_t arena_cap);
kshira_status kshira_phase_driver_transition(kshira_phase_driver *driver,
                                              kshira_phase_contract next);
kshira_status kshira_phase_driver_step(kshira_phase_driver *driver);

#ifdef __cplusplus
}
#endif

#endif
