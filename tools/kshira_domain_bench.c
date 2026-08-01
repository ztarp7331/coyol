#include "kshira/domain.h"
#include "kshira/session.h"

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
    enum { WIDTH = 160, HEIGHT = 160, CLASSES = 10, SAMPLES_PER_DOMAIN = 500,
           ARENA_BYTES = 256 * 1024 };
    unsigned char arena_memory[ARENA_BYTES];
    float image_data[WIDTH * HEIGHT];
    int counts[KSHIRA_DOMAIN_COUNT] = {0};
    kshira_domain_stream stream;
    kshira_domain_spec domain_spec = {
        WIDTH, HEIGHT, 1, CLASSES, SAMPLES_PER_DOMAIN, 97U
    };
    kshira_session session;
    kshira_session_spec session_spec = {{WIDTH, HEIGHT, 1, CLASSES, 8, 16, 97},
                                        sizeof(arena_memory)};
    kshira_image_f32 image = {image_data, 1, HEIGHT, WIDTH};
    kshira_rad_box target;
    float loss = 0.0f;
    int domain;
    size_t processed = 0U;
    double start;
    if (kshira_domain_init(&stream, &domain_spec) != KSHIRA_OK ||
        kshira_session_init(&session, arena_memory, sizeof(arena_memory), &session_spec) !=
            KSHIRA_OK) {
        fprintf(stderr, "KSHIRA domain benchmark setup failed\n");
        return EXIT_FAILURE;
    }
    for (int channel = 0; channel < session_spec.rad.feature_channels; ++channel) {
        if (kshira_session_set_channel(&session, (size_t)channel, 1) != KSHIRA_OK) {
            fprintf(stderr, "KSHIRA domain channel setup failed\n");
            return EXIT_FAILURE;
        }
    }
    start = now_ms();
    while (kshira_domain_index(&stream) < kshira_domain_total(&stream)) {
        kshira_status status = kshira_domain_next(
            &stream, image_data, sizeof(image_data) / sizeof(image_data[0]), &target, &domain);
        if (status != KSHIRA_OK ||
            kshira_session_step(&session, &image, &target, 1.0e-4f, &loss) != KSHIRA_OK) {
            fprintf(stderr, "KSHIRA domain training failed at sample %zu\n", processed);
            return EXIT_FAILURE;
        }
        ++counts[domain];
        ++processed;
    }
    {
        double elapsed = now_ms() - start;
        printf("domain_samples=%zu domains=%d train_ms=%.3f per_image_ms=%.6f "
               "arena_high_water=%zu arena_cap=%u image_workspace_bytes=%zu loss=%.6f\n", processed,
               KSHIRA_DOMAIN_COUNT, elapsed, elapsed / (double)processed,
               kshira_session_arena_high_water(&session), (unsigned)sizeof(arena_memory),
               sizeof(image_data), loss);
    }
    for (int i = 0; i < KSHIRA_DOMAIN_COUNT; ++i) {
        if (counts[i] != SAMPLES_PER_DOMAIN) return EXIT_FAILURE;
    }
    return isfinite(loss) ? EXIT_SUCCESS : EXIT_FAILURE;
}
