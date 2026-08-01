#include "kshira/core.h"
#include "kshira/rad.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec value;
    (void)timespec_get(&value, TIME_UTC);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

int main(void) {
    enum { WIDTH = 160, HEIGHT = 160, REPEATS = 25, ARENA_BYTES = 256 * 1024 };
    unsigned char arena_memory[ARENA_BYTES];
    float pixels[WIDTH * HEIGHT] = {0.0f};
    kshira_arena arena;
    kshira_rad_model *model = NULL;
    kshira_rad_spec spec = {WIDTH, HEIGHT, 1, 4, 8, 16, 1};
    kshira_image_f32 image = {pixels, 1, HEIGHT, WIDTH};
    kshira_rad_detection detections[16];
    int count = 0;
    double start;
    double elapsed;
    for (int y = 48; y < 96; ++y) {
        for (int x = 48; x < 96; ++x) pixels[y * WIDTH + x] = 1.0f;
    }
    if (kshira_arena_init(&arena, arena_memory, sizeof(arena_memory)) != KSHIRA_OK ||
        kshira_rad_build(&arena, &spec, &model) != KSHIRA_OK ||
        kshira_rad_predict(model, &image, 0.0f, detections, 16, &count) != KSHIRA_OK) {
        fprintf(stderr, "KSHIRA RAD setup failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < 3; ++i) {
        if (kshira_rad_predict(model, &image, 0.25f, detections, 16, &count) != KSHIRA_OK) {
            fprintf(stderr, "KSHIRA RAD warmup failed\n");
            return EXIT_FAILURE;
        }
    }
    start = now_ms();
    for (int i = 0; i < REPEATS; ++i) {
        if (kshira_rad_predict(model, &image, 0.25f, detections, 16, &count) != KSHIRA_OK) {
            fprintf(stderr, "KSHIRA RAD inference failed\n");
            return EXIT_FAILURE;
        }
    }
    elapsed = (now_ms() - start) / (double)REPEATS;
    printf("rad_input=%dx%d map=%dx%d top_k=%d detections=%d latency_ms=%.3f "
           "arena_used=%zu arena_high_water=%zu arena_cap=%u params=%zu activations=%zu\n",
           WIDTH, HEIGHT, kshira_rad_map_width(model), kshira_rad_map_height(model),
           spec.top_k, count, elapsed, kshira_arena_used(&arena),
           kshira_arena_high_water(&arena), (unsigned)sizeof(arena_memory),
           kshira_rad_parameter_bytes(model), kshira_rad_activation_bytes(model));
    if (kshira_arena_high_water(&arena) > sizeof(arena_memory) ||
        !isfinite(elapsed) || elapsed < 0.0) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
