/* Purpose: validate the three-phase PRE/TRAIN/ODT resource contract.
 * Ownership: contracts are caller-owned values and are not retained.
 * Failure: incompatible bit/update/QAS combinations return KSHIRA_ERR_ARGUMENT. */
#include "kshira/phase.h"

kshira_status kshira_phase_validate(const kshira_phase_contract *contract) {
    if (contract == NULL || contract->arena_cap == 0U ||
        !kshira_bit_mode_valid(contract->bits) ||
        contract->update_mode < KSHIRA_UPDATE_FREEZE ||
        contract->update_mode > KSHIRA_UPDATE_FULL ||
        contract->qas_enabled < 0 || contract->qas_enabled > 1) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (contract->phase == KSHIRA_PHASE_PRE) {
        if (contract->update_mode == KSHIRA_UPDATE_FREEZE || contract->qas_enabled != 0) {
            return KSHIRA_ERR_ARGUMENT;
        }
    } else if (contract->phase == KSHIRA_PHASE_TRAIN) {
        if (contract->update_mode == KSHIRA_UPDATE_FREEZE || contract->qas_enabled == 0) {
            return KSHIRA_ERR_ARGUMENT;
        }
    } else if (contract->phase == KSHIRA_PHASE_ODT) {
        if (contract->update_mode == KSHIRA_UPDATE_FULL || contract->qas_enabled == 0) {
            return KSHIRA_ERR_ARGUMENT;
        }
    } else return KSHIRA_ERR_ARGUMENT;
    return KSHIRA_OK;
}

const char *kshira_phase_name(kshira_phase phase) {
    if (phase == KSHIRA_PHASE_PRE) return "PRE";
    if (phase == KSHIRA_PHASE_TRAIN) return "TRAIN";
    if (phase == KSHIRA_PHASE_ODT) return "ODT";
    return "UNKNOWN";
}

kshira_status kshira_phase_driver_init(kshira_phase_driver *driver, size_t arena_cap) {
    if (driver == NULL || arena_cap == 0U) return KSHIRA_ERR_ARGUMENT;
    driver->contract.phase = KSHIRA_PHASE_PRE;
    driver->contract.bits = KSHIRA_BITS_INT8;
    driver->contract.update_mode = KSHIRA_UPDATE_FULL;
    driver->contract.arena_cap = arena_cap;
    driver->contract.qas_enabled = 0;
    driver->steps = 0U;
    driver->transitions = 0U;
    return KSHIRA_OK;
}

kshira_status kshira_phase_driver_transition(kshira_phase_driver *driver,
                                              kshira_phase_contract next) {
    if (driver == NULL || next.arena_cap != driver->contract.arena_cap ||
        next.phase != driver->contract.phase + 1) return KSHIRA_ERR_ARGUMENT;
    if (kshira_phase_validate(&next) != KSHIRA_OK) return KSHIRA_ERR_ARGUMENT;
    driver->contract = next;
    ++driver->transitions;
    return KSHIRA_OK;
}

kshira_status kshira_phase_driver_step(kshira_phase_driver *driver) {
    if (driver == NULL || kshira_phase_validate(&driver->contract) != KSHIRA_OK) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (driver->steps == SIZE_MAX) return KSHIRA_ERR_RANGE;
    ++driver->steps;
    return KSHIRA_OK;
}
