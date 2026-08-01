/* Purpose: caller-owned arena and tensor views for KSHIRA schedules.
 * Ownership: callers own the backing bytes; this module never allocates.
 * Failure: invalid dimensions/alignment or exhausted capacity returns a status
 * or NULL and leaves the arena offset unchanged. */
#include "kshira/core.h"

#include <math.h>
#include <stdint.h>

static int valid_dimension(int value) {
    return value > 0;
}

static size_t aligned_offset(const kshira_arena *arena, size_t alignment) {
    uintptr_t address;
    uintptr_t mask;
    uintptr_t aligned;
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) return SIZE_MAX;
    if ((uintptr_t)arena->offset > UINTPTR_MAX - arena->base_address) return SIZE_MAX;
    address = arena->base_address + (uintptr_t)arena->offset;
    mask = (uintptr_t)alignment - 1U;
    if (address > UINTPTR_MAX - mask) return SIZE_MAX;
    aligned = (address + mask) & ~mask;
    if (aligned < arena->base_address || aligned - arena->base_address > SIZE_MAX) {
        return SIZE_MAX;
    }
    return (size_t)(aligned - arena->base_address);
}

kshira_status kshira_arena_init(kshira_arena *arena, void *memory, size_t capacity) {
    if (arena == NULL || memory == NULL || capacity == 0U) return KSHIRA_ERR_ARGUMENT;
    arena->memory = (uint8_t *)memory;
    arena->capacity = capacity;
    arena->offset = 0U;
    arena->high_water = 0U;
    arena->base_address = (uintptr_t)memory;
    return KSHIRA_OK;
}

void kshira_arena_reset(kshira_arena *arena) {
    if (arena == NULL) return;
    arena->offset = 0U;
}

void *kshira_arena_alloc(kshira_arena *arena, size_t bytes, size_t alignment) {
    size_t start;
    if (arena == NULL || arena->memory == NULL || bytes == 0U) return NULL;
    start = aligned_offset(arena, alignment);
    if (start == SIZE_MAX || start > arena->capacity || bytes > arena->capacity - start) {
        return NULL;
    }
    arena->offset = start + bytes;
    if (arena->offset > arena->high_water) arena->high_water = arena->offset;
    return arena->memory + start;
}

size_t kshira_arena_used(const kshira_arena *arena) {
    return arena == NULL ? 0U : arena->offset;
}

size_t kshira_arena_high_water(const kshira_arena *arena) {
    return arena == NULL ? 0U : arena->high_water;
}

kshira_status kshira_tensor_view(kshira_arena *arena, int channels, int height, int width,
                                 kshira_tensor_f32 *out) {
    size_t elements;
    if (arena == NULL || out == NULL || !valid_dimension(channels) ||
        !valid_dimension(height) || !valid_dimension(width)) return KSHIRA_ERR_ARGUMENT;
    if ((size_t)channels > SIZE_MAX / (size_t)height) return KSHIRA_ERR_RANGE;
    elements = (size_t)channels * (size_t)height;
    if (elements > SIZE_MAX / (size_t)width ||
        elements * (size_t)width > SIZE_MAX / sizeof(float)) return KSHIRA_ERR_RANGE;
    out->data = (float *)kshira_arena_alloc(arena, elements * (size_t)width * sizeof(float),
                                            _Alignof(float));
    if (out->data == NULL) return KSHIRA_ERR_MEMORY;
    out->channels = channels;
    out->height = height;
    out->width = width;
    return KSHIRA_OK;
}
