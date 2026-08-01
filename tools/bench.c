#include "det.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

typedef struct {
    float *pixels;
    int width;
    int height;
    int channels;
    size_t total;
    size_t index;
    det_box box;
} bench_dataset;

static int next_sample(void *user, det_sample *sample) {
    bench_dataset *dataset = (bench_dataset *)user;
    if (dataset->index >= dataset->total) return 0;
    memset(dataset->pixels, 0, (size_t)dataset->width * (size_t)dataset->height *
           (size_t)dataset->channels * sizeof(float));
    int x0 = 8 + (int)((dataset->index * 7U) % (size_t)(dataset->width - 32));
    int y0 = 8 + (int)((dataset->index * 11U) % (size_t)(dataset->height - 32));
    int size = 12 + (int)(dataset->index % 20U);
    if (x0 + size >= dataset->width) size = dataset->width - x0 - 1;
    if (y0 + size >= dataset->height) size = dataset->height - y0 - 1;
    for (int y = y0; y < y0 + size; ++y) {
        for (int x = x0; x < x0 + size; ++x) {
            dataset->pixels[(size_t)y * (size_t)dataset->width + (size_t)x] = 1.0f;
        }
    }
    dataset->box.x1 = (float)x0;
    dataset->box.y1 = (float)y0;
    dataset->box.x2 = (float)(x0 + size);
    dataset->box.y2 = (float)(y0 + size);
    dataset->box.class_id = (int)(dataset->index % 4U);
    sample->image.data = dataset->pixels;
    sample->image.channels = dataset->channels;
    sample->image.height = dataset->height;
    sample->image.width = dataset->width;
    sample->boxes = &dataset->box;
    sample->box_count = 1;
    ++dataset->index;
    return 1;
}

static void reset_dataset(void *user) {
    ((bench_dataset *)user)->index = 0;
}

static int parse_positive_int(const char *value, int *out) {
    if (value == NULL || out == NULL) return 0;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 100000000L) return 0;
    *out = (int)parsed;
    return 1;
}

static int parse_threshold(const char *value, float *out) {
    if (value == NULL || out == NULL) return 0;
    char *end = NULL;
    float parsed = strtof(value, &end);
    if (end == value || *end != '\0' || !isfinite(parsed) || parsed < 0.0f || parsed > 1.0f) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int parse_precision(const char *value, det_precision *out) {
    if (value == NULL || out == NULL) return 0;
    if (strcmp(value, "f32") == 0) *out = DET_PRECISION_F32;
    else if (strcmp(value, "int8") == 0) *out = DET_PRECISION_INT8;
    else if (strcmp(value, "w4a8") == 0) *out = DET_PRECISION_W4A8;
    else if (strcmp(value, "int4") == 0) *out = DET_PRECISION_INT4;
    else return 0;
    return 1;
}

static int parse_architecture(const char *value, det_architecture *out) {
    if (value == NULL || out == NULL) return 0;
    if (strcmp(value, "cdet") == 0) *out = DET_ARCH_CDET;
    else if (strcmp(value, "kshira") == 0) *out = DET_ARCH_KSHIRA;
    else return 0;
    return 1;
}

static double wall_now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (QueryPerformanceCounter(&counter) && QueryPerformanceFrequency(&frequency) &&
        frequency.QuadPart > 0) {
        return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
    }
#elif defined(CLOCK_MONOTONIC)
    struct timespec monotonic_ts;
    if (clock_gettime(CLOCK_MONOTONIC, &monotonic_ts) == 0) {
        return (double)monotonic_ts.tv_sec * 1000.0 +
               (double)monotonic_ts.tv_nsec / 1000000.0;
    }
#endif
    struct timespec fallback_ts;
    (void)timespec_get(&fallback_ts, TIME_UTC);
    return (double)fallback_ts.tv_sec * 1000.0 +
           (double)fallback_ts.tv_nsec / 1000000.0;
}

static int reserve_model_path(char *path, size_t capacity) {
    if (path == NULL || capacity == 0U) return 0;
#if defined(_WIN32)
    unsigned long process_id = (unsigned long)_getpid();
#else
    unsigned long process_id = (unsigned long)getpid();
#endif
    for (unsigned int attempt = 0U; attempt < 100U; ++attempt) {
        int written = snprintf(path, capacity, "cdet_bench_%lu_%u.model",
                               process_id, attempt);
        if (written < 0 || (size_t)written >= capacity) return 0;
        FILE *file = fopen(path, "wbx");
        if (file != NULL) {
            return fclose(file) == 0;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int sample_count = 5000;
    int width = 160;
    int height = 160;
    int global = 0;
    const char *manifest_path = NULL;
    const char *eval_manifest_path = NULL;
    det_precision precision = DET_PRECISION_F32;
    det_architecture architecture = DET_ARCH_CDET;
    float threshold = 0.25f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--samples") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &sample_count)) {
                fprintf(stderr, "samples must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--width") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &width)) {
                fprintf(stderr, "width must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--height") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &height)) {
                fprintf(stderr, "height must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--precision") == 0) {
            if (i + 1 >= argc || !parse_precision(argv[++i], &precision)) {
                fprintf(stderr, "precision must be f32, int8, w4a8, or int4\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--architecture") == 0) {
            if (i + 1 >= argc || !parse_architecture(argv[++i], &architecture)) {
                fprintf(stderr, "architecture must be cdet or kshira\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--threshold") == 0) {
            if (i + 1 >= argc || !parse_threshold(argv[++i], &threshold)) {
                fprintf(stderr, "threshold must be a finite value in [0,1]\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--manifest") == 0) {
            if (i + 1 >= argc || argv[++i][0] == '\0') {
                fprintf(stderr, "manifest must be a non-empty path\n");
                return EXIT_FAILURE;
            }
            manifest_path = argv[i];
        }
        else if (strcmp(argv[i], "--eval-manifest") == 0) {
            if (i + 1 >= argc || argv[++i][0] == '\0') {
                fprintf(stderr, "eval manifest must be a non-empty path\n");
                return EXIT_FAILURE;
            }
            eval_manifest_path = argv[i];
        }
        else if (strcmp(argv[i], "--global") == 0) global = 1;
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (width < 33 || height < 33) {
        fprintf(stderr, "width and height must be at least 33\n");
        return EXIT_FAILURE;
    }
    if ((architecture == DET_ARCH_CDET && precision == DET_PRECISION_INT4) ||
        (architecture == DET_ARCH_KSHIRA && precision == DET_PRECISION_W4A8) ||
        (architecture == DET_ARCH_KSHIRA && global)) {
        fprintf(stderr, "requested architecture/precision/training mode combination is unsupported\n");
        return EXIT_FAILURE;
    }
    double e2e_start = wall_now_ms();
    det_context *ctx = NULL;
    det_model *model = NULL;
    int max_detections = architecture == DET_ARCH_KSHIRA ? 64 : 100;
    det_model_spec spec = {width, height, 1, 4, max_detections, 1,
                           architecture, architecture == DET_ARCH_KSHIRA ? 8 : 0};
    size_t arena_bytes = architecture == DET_ARCH_KSHIRA ? 256U << 10 : 8U << 20;
    det_status status = det_context_create(arena_bytes, &ctx);
    if (status == DET_OK) status = det_model_build(ctx, &spec, &model);
    if (status != DET_OK) {
        fprintf(stderr, "bench setup failed: %d\n", status);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    float *pixels = NULL;
    bench_dataset storage = {0};
    det_manifest_dataset *manifest_dataset = NULL;
    det_dataset dataset;
    if (manifest_path != NULL) {
        status = det_manifest_open(manifest_path, width, height, 1, 100, &manifest_dataset);
        if (status == DET_OK) status = det_manifest_dataset_view(manifest_dataset, &dataset);
        if (status != DET_OK || dataset.sample_count == 0U) {
            fprintf(stderr, "manifest setup failed: %d\n", status);
            det_manifest_close(manifest_dataset);
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
    } else {
        pixels = (float *)calloc((size_t)width * (size_t)height, sizeof(float));
        if (pixels == NULL) {
            fprintf(stderr, "pixel allocation failed\n");
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
        storage = (bench_dataset){pixels, width, height, 1, (size_t)sample_count, 0U,
                                  {0, 0, 0, 0, 0}};
        dataset = (det_dataset){&storage, next_sample, reset_dataset, (size_t)sample_count};
    }
    det_train_config config = {global ? DET_TRAIN_GLOBAL_BP : DET_TRAIN_LOCAL_FAST,
                               architecture == DET_ARCH_KSHIRA ? precision : DET_PRECISION_F32,
                               1, 0.01f,
                               architecture == DET_ARCH_KSHIRA ? 0.0f : 0.8f, 0.1f,
                               sample_count, 1, 1};
    det_train_report report;
    double core_start = wall_now_ms();
    status = det_train(model, &dataset, &config, &report);
    if (status != DET_OK) {
        fprintf(stderr, "training failed: %d\n", status);
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    status = det_model_set_precision(model, precision);
    if (status != DET_OK) {
        fprintf(stderr, "precision setup failed: %d\n", status);
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    char model_path[128];
    if (!reserve_model_path(model_path, sizeof(model_path))) {
        fprintf(stderr, "could not reserve a unique benchmark model path\n");
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    double io_start = wall_now_ms();
    det_model *loaded = NULL;
    status = det_save(model, model_path);
    double serialized_at = wall_now_ms();
    double train_core_ms = serialized_at - core_start;
    double synthetic_e2e_ms = serialized_at - e2e_start;
    if (status == DET_OK) status = det_load(ctx, model_path, &loaded);
    double io_ms = wall_now_ms() - io_start;
    (void)remove(model_path);
    if (status != DET_OK) {
        fprintf(stderr, "model I/O failed: %d\n", status);
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    det_sample inference_sample;
    if (dataset.reset != NULL) dataset.reset(dataset.user);
    if (dataset.next(dataset.user, &inference_sample) <= 0) {
        fprintf(stderr, "failed to prepare inference sample\n");
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(loaded);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    det_detection detections[100];
    int detection_count = 0;
    for (int warmup = 0; warmup < 3; ++warmup) {
        status = det_predict(loaded, &inference_sample.image, threshold, detections, 100,
                             &detection_count);
        if (status != DET_OK) break;
    }
    double infer_start = wall_now_ms();
    for (int repeat = 0; status == DET_OK && repeat < 10; ++repeat) {
        status = det_predict(loaded, &inference_sample.image, threshold, detections, 100,
                             &detection_count);
    }
    double infer_ms = (wall_now_ms() - infer_start) / 10.0;
    if (status != DET_OK) {
        fprintf(stderr, "inference failed: %d\n", status);
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(loaded);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_FAILURE;
    }
    det_eval_report evaluation = {0};
    det_manifest_dataset *eval_manifest_dataset = NULL;
    det_dataset eval_dataset = dataset;
    if (eval_manifest_path != NULL) {
        status = det_manifest_open(eval_manifest_path, width, height, 1, 100,
                                   &eval_manifest_dataset);
        if (status == DET_OK) {
            status = det_manifest_dataset_view(eval_manifest_dataset, &eval_dataset);
        }
        if (status != DET_OK || eval_dataset.sample_count == 0U) {
            fprintf(stderr, "eval manifest setup failed: %d\n", status);
            det_manifest_close(eval_manifest_dataset);
            free(pixels);
            det_manifest_close(manifest_dataset);
            det_model_destroy(loaded);
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
    }
    if (manifest_path != NULL || eval_manifest_path != NULL) {
        status = det_evaluate(loaded, &eval_dataset, threshold, &evaluation);
        if (status != DET_OK) {
            fprintf(stderr, "evaluation failed: %d\n", status);
            free(pixels);
            det_manifest_close(eval_manifest_dataset);
            det_manifest_close(manifest_dataset);
            det_model_destroy(loaded);
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
    }
    det_model_destroy(loaded);
    const char *precision_name = precision == DET_PRECISION_INT8 ? "INT8" :
                                 (precision == DET_PRECISION_W4A8 ? "W4A8" :
                                  (precision == DET_PRECISION_INT4 ? "INT4" : "F32"));
    const char *architecture_name = architecture == DET_ARCH_KSHIRA ? "KSHIRA" : "CDET";
    printf("samples=%zu input=%dx%d architecture=%s source=%s mode=%s precision=%s threshold=%.3f %s=%.3f %s=%.3f "
           "infer_ms=%.3f io_ms=%.3f "
           "updates=%zu loss=%.6f images_per_sec=%.2f detections=%d\n",
           report.samples_seen, width, height, architecture_name,
           manifest_path == NULL ? "synthetic" : "manifest",
           global ? "GLOBAL_BP" : "LOCAL_FAST", precision_name, threshold,
           manifest_path == NULL ? "train_core_ms" : "train_plus_decode_ms",
           train_core_ms, manifest_path == NULL ? "synthetic_e2e_ms" : "train_e2e_ms",
           synthetic_e2e_ms, infer_ms, io_ms,
           report.updates, report.mean_loss,
           train_core_ms > 0.0 ? (double)report.samples_seen * 1000.0 / train_core_ms : 0.0,
           detection_count);
    if (manifest_path != NULL || eval_manifest_path != NULL) {
        printf("eval_samples=%zu eval_ground_truths=%zu eval_predictions=%zu eval_tp=%zu "
               "eval_fp=%zu eval_fn=%zu precision=%.4f recall=%.4f mean_iou=%.4f "
               "ap50=%.4f map50_95=%.4f size_gt=%zu,%zu,%zu\n",
               evaluation.samples_seen, evaluation.ground_truths, evaluation.predictions,
               evaluation.true_positives, evaluation.false_positives,
               evaluation.false_negatives, evaluation.precision, evaluation.recall,
               evaluation.mean_iou, evaluation.ap50, evaluation.map50_95,
               evaluation.size_ground_truths[0], evaluation.size_ground_truths[1],
               evaluation.size_ground_truths[2]);
    }
    free(pixels);
    det_manifest_close(eval_manifest_dataset);
    det_manifest_close(manifest_dataset);
    det_model_destroy(model);
    det_context_destroy(ctx);
    return EXIT_SUCCESS;
}
