#ifndef KSHIRA_CORE_H
#define KSHIRA_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KSHIRA_OK = 0,
    KSHIRA_ERR_ARGUMENT = 1,
    KSHIRA_ERR_MEMORY = 2,
    KSHIRA_ERR_RANGE = 3,
    KSHIRA_ERR_UNSUPPORTED = 4
} kshira_status;

/* Caller-owned bump arena. No allocation occurs after initialization. */
typedef struct {
    uint8_t *memory;
    size_t capacity;
    size_t offset;
    size_t high_water;
    uintptr_t base_address;
} kshira_arena;

typedef struct {
    float *data;
    int channels;
    int height;
    int width;
} kshira_tensor_f32;

kshira_status kshira_arena_init(kshira_arena *arena, void *memory, size_t capacity);
void kshira_arena_reset(kshira_arena *arena);
void *kshira_arena_alloc(kshira_arena *arena, size_t bytes, size_t alignment);
size_t kshira_arena_used(const kshira_arena *arena);
size_t kshira_arena_high_water(const kshira_arena *arena);
kshira_status kshira_tensor_view(kshira_arena *arena, int channels, int height, int width,
                                 kshira_tensor_f32 *out);

#ifdef __cplusplus
}
#endif

#endif
