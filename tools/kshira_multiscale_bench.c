#include "kshira/domain.h"
#include "kshira/eval.h"
#include "kshira/session.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    WIDTH = 160,
    HEIGHT = 160,
    CLASSES = 10,
    FEATURES = 8,
    TOP_K = 16,
    SAMPLES_PER_DOMAIN = 500,
    ODT_SAMPLES_PER_DOMAIN = 500,
    EVAL_SAMPLES_PER_DOMAIN = 10,
    CALIBRATION_SAMPLES_PER_DOMAIN = 1,
    ARENA_BYTES = 256 * 1024
};

static double now_ms(void) {
    struct timespec value;
    (void)timespec_get(&value, TIME_UTC);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static const char *mode_name(kshira_bit_mode bits) {
    return bits == KSHIRA_BITS_INT8 ? "INT8" : "INT4";
}

static int target_level(const kshira_rad_box *target) {
    float area = (target->x2 - target->x1) * (target->y2 - target->y1);
    float image_area = (float)WIDTH * (float)HEIGHT;
    return area / image_area < (1.0f / 16.0f) ? 1 : 2;
}

static int enable_all_channels(kshira_session *session) {
    for (int channel = 0; channel < FEATURES; ++channel) {
        if (kshira_session_set_channel(session, (size_t)channel, 1) != KSHIRA_OK) return 0;
    }
    return 1;
}

static int run_mode(kshira_bit_mode bits) {
    unsigned char arena_memory[ARENA_BYTES];
    float image_data[WIDTH * HEIGHT];
    kshira_domain_stream train_stream;
    kshira_domain_stream odt_stream;
    kshira_domain_stream eval_stream;
    kshira_domain_stream calibration_stream;
    kshira_domain_spec train_spec = {WIDTH, HEIGHT, 1, CLASSES, SAMPLES_PER_DOMAIN, 97U};
    kshira_domain_spec odt_spec = {WIDTH, HEIGHT, 1, CLASSES, ODT_SAMPLES_PER_DOMAIN, 297U};
    kshira_domain_spec eval_spec = {WIDTH, HEIGHT, 1, CLASSES, EVAL_SAMPLES_PER_DOMAIN, 197U};
    kshira_domain_spec calibration_spec = {
        WIDTH, HEIGHT, 1, CLASSES, CALIBRATION_SAMPLES_PER_DOMAIN, 53U
    };
    kshira_session session;
    kshira_session_spec session_spec = {
        {WIDTH, HEIGHT, 1, CLASSES, FEATURES, TOP_K, 97, 1}, sizeof(arena_memory)
    };
    kshira_image_f32 image = {image_data, 1, HEIGHT, WIDTH};
    kshira_rad_box target;
    kshira_rad_detection detections[TOP_K];
    kshira_proxy_metrics metrics;
    float loss = 0.0f;
    float loss_sum = 0.0f;
    float odt_loss_sum = 0.0f;
    size_t processed = 0U;
    size_t odt_processed = 0U;
    int domain;
    int count;
    double start;
    double train_ms;
    double odt_ms;
    double eval_ms;
    if (kshira_domain_init(&train_stream, &train_spec) != KSHIRA_OK ||
        kshira_domain_init(&odt_stream, &odt_spec) != KSHIRA_OK ||
        kshira_domain_init(&eval_stream, &eval_spec) != KSHIRA_OK ||
        kshira_domain_init(&calibration_stream, &calibration_spec) != KSHIRA_OK ||
        kshira_session_init(&session, arena_memory, sizeof(arena_memory), &session_spec) !=
            KSHIRA_OK || !enable_all_channels(&session) ||
        !kshira_rad_multiscale_ready(session.rad)) {
        return 0;
    }
    if (kshira_session_transition(&session, bits, KSHIRA_UPDATE_FULL, 1) != KSHIRA_OK) {
        return 0;
    }
    start = now_ms();
    while (kshira_domain_index(&train_stream) < kshira_domain_total(&train_stream)) {
        if (kshira_domain_next(&train_stream, image_data,
                               sizeof(image_data) / sizeof(image_data[0]), &target,
                               &domain) != KSHIRA_OK ||
            kshira_session_step(&session, &image, &target,
                                bits == KSHIRA_BITS_INT8 ? 1.0e-3f : 5.0e-4f,
                                &loss) != KSHIRA_OK || !isfinite(loss)) {
            return 0;
        }
        loss_sum += loss;
        ++processed;
    }
    train_ms = now_ms() - start;
    if (kshira_session_transition(&session, bits, KSHIRA_UPDATE_CHANNELS, 1) != KSHIRA_OK) {
        return 0;
    }
    while (kshira_domain_index(&calibration_stream) <
           kshira_domain_total(&calibration_stream)) {
        if (kshira_domain_next(&calibration_stream, image_data,
                               sizeof(image_data) / sizeof(image_data[0]), &target,
                               &domain) != KSHIRA_OK ||
            kshira_session_calibrate(&session, &image) != KSHIRA_OK) return 0;
    }
    start = now_ms();
    while (kshira_domain_index(&odt_stream) < kshira_domain_total(&odt_stream)) {
        if (kshira_domain_next(&odt_stream, image_data,
                               sizeof(image_data) / sizeof(image_data[0]), &target,
                               &domain) != KSHIRA_OK ||
            kshira_session_multiscale_step(&session, &image, &target,
                                           target_level(&target),
                                           bits == KSHIRA_BITS_INT8 ? 5.0e-4f : 2.5e-4f,
                                           &loss) != KSHIRA_OK || !isfinite(loss)) {
            return 0;
        }
        odt_loss_sum += loss;
        ++odt_processed;
    }
    odt_ms = now_ms() - start;
    start = now_ms();
    kshira_proxy_metrics_init(&metrics);
    while (kshira_domain_index(&eval_stream) < kshira_domain_total(&eval_stream)) {
        if (kshira_domain_next(&eval_stream, image_data,
                               sizeof(image_data) / sizeof(image_data[0]), &target,
                               &domain) != KSHIRA_OK ||
            kshira_session_predict(&session, &image, 0.25f, detections, TOP_K, &count) !=
                KSHIRA_OK ||
            kshira_proxy_metrics_add(&metrics, &target, detections, count) != KSHIRA_OK) {
            return 0;
        }
    }
    eval_ms = now_ms() - start;
    {
        /* Timing gates are reported as diagnostics; ASAN and host jitter can
         * exceed them without making the numerical/arena run invalid. */
        double eval_per_image_ms = eval_ms /
                                    (double)(EVAL_SAMPLES_PER_DOMAIN * KSHIRA_DOMAIN_COUNT);
        int base_gate = train_ms < 1000.0;
        int odt_gate = odt_ms < 1000.0;
        int combined_gate = train_ms + odt_ms < 1000.0;
        int inference_gate = eval_per_image_ms < 33.0;
        printf("multiscale_bits=%s base_samples=%zu odt_samples=%zu base_train_ms=%.3f "
               "odt_ms=%.3f total_train_ms=%.3f eval_ms=%.3f eval_per_image_ms=%.3f "
               "base_mean_loss=%.6f odt_mean_loss=%.6f proxy_iou=%.6f "
               "proxy_class=%.6f base_train_gate=%s odt_train_gate=%s "
               "combined_train_gate=%s inference_gate=%s arena_high_water=%zu "
               "arena_cap=%u\n",
           mode_name(bits), processed, odt_processed, train_ms, odt_ms,
           train_ms + odt_ms, eval_ms, eval_per_image_ms,
           processed == 0U ? 0.0f : loss_sum / (float)processed,
           odt_processed == 0U ? 0.0f : odt_loss_sum / (float)odt_processed,
           kshira_proxy_mean_iou(&metrics), kshira_proxy_class_accuracy(&metrics),
           base_gate ? "PASS" : "FAIL", odt_gate ? "PASS" : "FAIL",
           combined_gate ? "PASS" : "FAIL", inference_gate ? "PASS" : "FAIL",
           kshira_session_arena_high_water(&session), ARENA_BYTES);
    }
    return processed == (size_t)(SAMPLES_PER_DOMAIN * KSHIRA_DOMAIN_COUNT) &&
           odt_processed == (size_t)(ODT_SAMPLES_PER_DOMAIN * KSHIRA_DOMAIN_COUNT) &&
           isfinite(loss_sum) && isfinite(odt_loss_sum) &&
           isfinite(kshira_proxy_mean_iou(&metrics)) &&
           isfinite(kshira_proxy_class_accuracy(&metrics)) &&
           kshira_session_arena_high_water(&session) <= ARENA_BYTES;
}

int main(void) {
    if (!run_mode(KSHIRA_BITS_INT8) || !run_mode(KSHIRA_BITS_INT4)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
