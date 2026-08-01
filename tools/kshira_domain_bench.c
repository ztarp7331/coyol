#include "kshira/domain.h"
#include "kshira/eval.h"
#include "kshira/session.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    kshira_bit_mode bits;
    double train_ms;
    double eval_ms;
    float mean_loss;
    float mean_iou;
    float class_accuracy;
    size_t arena_high_water;
} mode_result;

static double now_ms(void) {
    struct timespec value;
    (void)timespec_get(&value, TIME_UTC);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static const char *mode_name(kshira_bit_mode bits) {
    if (bits == KSHIRA_BITS_FLOAT) return "F32";
    if (bits == KSHIRA_BITS_INT8) return "INT8";
    return "INT4";
}

static int enable_all_channels(kshira_session *session, int channels) {
    for (int channel = 0; channel < channels; ++channel) {
        if (kshira_session_set_channel(session, (size_t)channel, 1) != KSHIRA_OK) return 0;
    }
    return 1;
}

static int run_mode(kshira_bit_mode bits, mode_result *result) {
    enum { WIDTH = 160, HEIGHT = 160, CLASSES = 10, SAMPLES_PER_DOMAIN = 500,
           EVAL_SAMPLES_PER_DOMAIN = 10, ARENA_BYTES = 256 * 1024 };
    unsigned char arena_memory[ARENA_BYTES];
    float image_data[WIDTH * HEIGHT];
    kshira_domain_stream stream;
    kshira_domain_stream eval_stream;
    kshira_domain_spec domain_spec = {WIDTH, HEIGHT, 1, CLASSES, SAMPLES_PER_DOMAIN, 97U};
    kshira_domain_spec eval_spec = {WIDTH, HEIGHT, 1, CLASSES, EVAL_SAMPLES_PER_DOMAIN, 197U};
    kshira_session session;
    kshira_session_spec session_spec = {{WIDTH, HEIGHT, 1, CLASSES, 8, 16, 97},
                                        sizeof(arena_memory)};
    kshira_image_f32 image = {image_data, 1, HEIGHT, WIDTH};
    kshira_rad_box target;
    kshira_rad_detection detections[16];
    kshira_proxy_metrics metrics;
    float loss_sum = 0.0f;
    float loss = 0.0f;
    size_t processed = 0U;
    int domain;
    int count;
    kshira_status status;
    double start;
    if (result == NULL || kshira_domain_init(&stream, &domain_spec) != KSHIRA_OK ||
        kshira_domain_init(&eval_stream, &eval_spec) != KSHIRA_OK ||
        kshira_session_init(&session, arena_memory, sizeof(arena_memory), &session_spec) !=
            KSHIRA_OK || !enable_all_channels(&session, session_spec.rad.feature_channels)) {
        return 0;
    }
    if (bits != KSHIRA_BITS_FLOAT &&
        kshira_session_transition(&session, bits, KSHIRA_UPDATE_CHANNELS, 1) != KSHIRA_OK) {
        return 0;
    }
    start = now_ms();
    while (kshira_domain_index(&stream) < kshira_domain_total(&stream)) {
        status = kshira_domain_next(&stream, image_data,
                                    sizeof(image_data) / sizeof(image_data[0]), &target,
                                    &domain);
        if (status == KSHIRA_OK) {
            status = kshira_session_step(&session, &image, &target,
                                         bits == KSHIRA_BITS_FLOAT ? 1.0e-4f :
                                         (bits == KSHIRA_BITS_INT8 ? 1.0e-9f : 1.0e-10f),
                                         &loss);
        }
        if (status != KSHIRA_OK || !isfinite(loss)) {
            fprintf(stderr, "curriculum %s failed at sample %zu status=%d loss=%f\n",
                    mode_name(bits), processed, (int)status, loss);
            return 0;
        }
        loss_sum += loss;
        if (!isfinite(loss_sum)) return 0;
        ++processed;
    }
    result->train_ms = now_ms() - start;
    start = now_ms();
    kshira_proxy_metrics_init(&metrics);
    while (kshira_domain_index(&eval_stream) < kshira_domain_total(&eval_stream)) {
        if (kshira_domain_next(&eval_stream, image_data,
                               sizeof(image_data) / sizeof(image_data[0]), &target,
                               &domain) != KSHIRA_OK ||
            kshira_session_predict(&session, &image, 0.25f, detections, 16, &count) !=
                KSHIRA_OK ||
            kshira_proxy_metrics_add(&metrics, &target, detections, count) != KSHIRA_OK) {
            return 0;
        }
    }
    result->eval_ms = now_ms() - start;
    result->bits = bits;
    result->mean_loss = processed == 0U ? 0.0f : loss_sum / (float)processed;
    result->mean_iou = kshira_proxy_mean_iou(&metrics);
    result->class_accuracy = kshira_proxy_class_accuracy(&metrics);
    result->arena_high_water = kshira_session_arena_high_water(&session);
    return isfinite(result->mean_loss) && isfinite(result->mean_iou) &&
           isfinite(result->class_accuracy) && result->arena_high_water <= ARENA_BYTES;
}

int main(void) {
    const kshira_bit_mode modes[] = {KSHIRA_BITS_FLOAT, KSHIRA_BITS_INT8,
                                     KSHIRA_BITS_INT4};
    mode_result results[3];
    for (size_t i = 0U; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        if (!run_mode(modes[i], &results[i])) {
            fprintf(stderr, "KSHIRA %s curriculum run failed\n", mode_name(modes[i]));
            return EXIT_FAILURE;
        }
        printf("rad_bits=%s domain_samples=5000 domains=%d train_ms=%.3f eval_ms=%.3f "
               "mean_loss=%.6f proxy_iou=%.6f proxy_class=%.6f "
               "arena_high_water=%zu arena_cap=%u image_workspace_bytes=%zu\n",
               mode_name(results[i].bits), KSHIRA_DOMAIN_COUNT, results[i].train_ms,
               results[i].eval_ms, results[i].mean_loss, results[i].mean_iou,
               results[i].class_accuracy, results[i].arena_high_water,
               256U * 1024U, (size_t)160 * (size_t)160 * sizeof(float));
    }
    if (results[0].mean_loss > 0.0f) {
        printf("quant_recovery int8_loss_ratio=%.6f int4_loss_ratio=%.6f "
               "int8_iou_ratio=%.6f int4_iou_ratio=%.6f\n",
               results[1].mean_loss / results[0].mean_loss,
               results[2].mean_loss / results[0].mean_loss,
               results[0].mean_iou > 0.0f ? results[1].mean_iou / results[0].mean_iou : 0.0f,
               results[0].mean_iou > 0.0f ? results[2].mean_iou / results[0].mean_iou : 0.0f);
    }
    return EXIT_SUCCESS;
}
