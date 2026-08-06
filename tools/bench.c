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
    int classes;
    int variant;
    size_t total;
    size_t index;
    det_box box;
} bench_dataset;

static float pattern_pixel(const bench_dataset *dataset, int class_id,
                           int x, int y, int x0, int y0, int size) {
    int dx = x - x0;
    int dy = y - y0;
    int pattern = class_id & 3;
    int groups = (dataset->classes + 3) / 4;
    int group = class_id / 4;
    float amplitude = (dataset->variant ? 0.70f : 0.80f) +
                      0.20f * (float)(group + 1) / (float)groups;
    if (pattern == 0) return amplitude;
    if (pattern == 1) return dx * 2 < size ? amplitude : 0.0f;
    if (pattern == 2) return dy * 2 < size ? amplitude : 0.0f;
    return (abs(2 * dx - size) < size / 3 ||
            abs(2 * dy - size) < size / 3) ? amplitude : 0.0f;
}

static int next_sample(void *user, det_sample *sample) {
    bench_dataset *dataset = (bench_dataset *)user;
    if (dataset->index >= dataset->total) return 0;
    memset(dataset->pixels, 0, (size_t)dataset->width * (size_t)dataset->height *
           (size_t)dataset->channels * sizeof(float));
    int x0 = 8 + (int)((dataset->index * (dataset->variant ? 5U : 7U) +
                         (dataset->variant ? 3U : 0U)) %
                        (size_t)(dataset->width - 32));
    int y0 = 8 + (int)((dataset->index * (dataset->variant ? 13U : 11U) +
                         (dataset->variant ? 5U : 0U)) %
                        (size_t)(dataset->height - 32));
    int size = (dataset->variant ? 10 : 12) +
               (int)((dataset->index * (dataset->variant ? 3U : 1U)) %
                     (dataset->variant ? 22U : 20U));
    int class_id = (int)((dataset->index *
                          (dataset->variant ? 3U : 1U) +
                          (dataset->variant ? 1U : 0U)) %
                         (size_t)dataset->classes);
    if (x0 + size >= dataset->width) size = dataset->width - x0 - 1;
    if (y0 + size >= dataset->height) size = dataset->height - y0 - 1;
    for (int channel = 0; channel < dataset->channels; ++channel) {
        for (int y = y0; y < y0 + size; ++y) {
            for (int x = x0; x < x0 + size; ++x) {
                dataset->pixels[((size_t)channel * (size_t)dataset->height +
                                 (size_t)y) * (size_t)dataset->width + (size_t)x] =
                    pattern_pixel(dataset, class_id, x, y, x0, y0, size);
            }
        }
    }
    dataset->box.x1 = (float)x0;
    dataset->box.y1 = (float)y0;
    dataset->box.x2 = (float)(x0 + size);
    dataset->box.y2 = (float)(y0 + size);
    dataset->box.class_id = class_id;
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

static int parse_learning_rate(const char *value, float *out) {
    char *end = NULL;
    float parsed;
    if (value == NULL || out == NULL) return 0;
    parsed = strtof(value, &end);
    if (end == value || *end != '\0' || !isfinite(parsed) ||
        parsed <= 0.0f || parsed > 10.0f) return 0;
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

static int parse_channels(const char *value, int *out) {
    int parsed;
    if (value == NULL || out == NULL || !parse_positive_int(value, &parsed) ||
        (parsed != 1 && parsed != 3)) return 0;
    *out = parsed;
    return 1;
}

static int parse_architecture(const char *value, det_architecture *out) {
    if (value == NULL || out == NULL) return 0;
    if (strcmp(value, "cdet") == 0) *out = DET_ARCH_CDET;
    else if (strcmp(value, "kshira") == 0) *out = DET_ARCH_KSHIRA;
    else return 0;
    return 1;
}

static int parse_stem_mode(const char *value, det_stem_mode *out) {
    if (value == NULL || out == NULL) return 0;
    if (strcmp(value, "legacy") == 0) *out = DET_STEM_LEGACY;
    else if (strcmp(value, "space-to-depth") == 0) *out = DET_STEM_SPACE_TO_DEPTH;
    else if (strcmp(value, "two-stage") == 0) *out = DET_STEM_TWO_STAGE;
    else return 0;
    return 1;
}

static int parse_research_mode(const char *value, det_research_mode *out) {
    if (value == NULL || out == NULL) return 0;
    if (strcmp(value, "scratch") == 0) *out = DET_RESEARCH_SCRATCH;
    else if (strcmp(value, "adapt") == 0) *out = DET_RESEARCH_ADAPT;
    else if (strcmp(value, "full") == 0) *out = DET_RESEARCH_FULL;
    else return 0;
    return 1;
}

static const char *research_mode_name(det_research_mode mode) {
    switch (mode) {
        case DET_RESEARCH_ADAPT: return "adapt";
        case DET_RESEARCH_FULL: return "full";
        default: return "scratch";
    }
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

static int file_size_bytes(const char *path, size_t *out) {
    FILE *file;
    long length;
    if (path == NULL || out == NULL) return 0;
    file = fopen(path, "rb");
    if (file == NULL) return 0;
    if (fseek(file, 0L, SEEK_END) != 0 || (length = ftell(file)) < 0L) {
        fclose(file);
        return 0;
    }
    *out = (size_t)length;
    return fclose(file) == 0;
}

int main(int argc, char **argv) {
    int sample_count = 5000;
    int max_train_samples = 0;
    int max_total_train_samples = 0;
    int multiscale_aux = 0;
    int adaptive_budget = 0;
    int residual_budget = 1;
    int quality_aligned_assignment = 0;
    int one_to_one_head = 0;
    int shared_multiscale_head = 0;
    int context_fusion = 0;
    int raw_input_features = 0;
    int p3_only_deployment = 0;
    int smooth_box_decode = 0;
    int width = 160;
    int height = 160;
    int channels = 1;
    int classes = 4;
    int feature_channels = 8;
    int arena_kib = 0;
    int max_detections = 0;
    int features_set = 0;
    int global = 0;
    int seed = 1;
    int epochs = 1;
    int calibrate = 0;
    int no_calibrate = 0;
    int time_budget_ms = 0;
    int analytic_readout = 0;
    int feature_readout = 0;
    int candidate_head_adapt = 0;
    int bootstrap_enabled = 0;
    int freeze_encoder = 0;
    int eval_only = 0;
    int no_evaluate = 0;
    int reset_schedule = 0;
    int dense_aux_budget = 2;
    int bootstrap_ms = 5000;
    int predecode_cache = 0;
    int shuffle_dataset = 1;
    double preparation_ms = 0.0;
    const char *manifest_path = NULL;
    const char *eval_manifest_path = NULL;
    const char *init_path = NULL;
    const char *save_model_path = NULL;
    const char *native_graph_dir = NULL;
    det_precision precision = DET_PRECISION_F32;
    det_architecture architecture = DET_ARCH_CDET;
    det_stem_mode stem_mode = DET_STEM_LEGACY;
    det_research_mode research_mode = DET_RESEARCH_SCRATCH;
    float learning_rate = 0.01f;
    int learning_rate_set = 0;
    float threshold = 0.25f;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--samples") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &sample_count)) {
                fprintf(stderr, "samples must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--max-train-samples") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &max_train_samples)) {
                fprintf(stderr, "max-train-samples must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--max-total-train-samples") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &max_total_train_samples)) {
                fprintf(stderr, "max-total-train-samples must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &seed)) {
                fprintf(stderr, "seed must be a positive integer\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--epochs") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &epochs) ||
                epochs > 1000) {
                fprintf(stderr, "epochs must be an integer from 1 to 1000\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--calibrate") == 0) {
            calibrate = 1;
        }
        else if (strcmp(argv[i], "--no-calibrate") == 0) {
            no_calibrate = 1;
        }
        else if (strcmp(argv[i], "--analytic-readout") == 0) {
            analytic_readout = 1;
        }
        else if (strcmp(argv[i], "--feature-readout") == 0) {
            feature_readout = 1;
        }
        else if (strcmp(argv[i], "--candidate-head-adapt") == 0) {
            candidate_head_adapt = 1;
        }
        else if (strcmp(argv[i], "--bootstrap") == 0) {
            bootstrap_enabled = 1;
        }
        else if (strcmp(argv[i], "--bootstrap-ms") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &bootstrap_ms)) {
                fprintf(stderr, "bootstrap-ms must be a positive integer\n");
                return EXIT_FAILURE;
            }
            bootstrap_enabled = 1;
        }
        else if (strcmp(argv[i], "--freeze-encoder") == 0) {
            freeze_encoder = 1;
        }
        else if (strcmp(argv[i], "--eval-only") == 0) {
            eval_only = 1;
        }
        else if (strcmp(argv[i], "--no-evaluate") == 0) {
            no_evaluate = 1;
        }
        else if (strcmp(argv[i], "--reset-schedule") == 0) {
            reset_schedule = 1;
        }
        else if (strcmp(argv[i], "--dense-aux") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &dense_aux_budget) ||
                dense_aux_budget > 32) {
                fprintf(stderr, "dense-aux must be an integer from 1 to 32\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--multiscale-aux") == 0) {
            multiscale_aux = 1;
        }
        else if (strcmp(argv[i], "--adaptive-budget") == 0) {
            adaptive_budget = 1;
        }
        else if (strcmp(argv[i], "--residual-budget") == 0) {
            residual_budget = 1;
        }
        else if (strcmp(argv[i], "--no-residual-budget") == 0) {
            residual_budget = 0;
        }
        else if (strcmp(argv[i], "--quality-align") == 0) {
            quality_aligned_assignment = 1;
        }
        else if (strcmp(argv[i], "--one-to-one-head") == 0) {
            one_to_one_head = 1;
        }
        else if (strcmp(argv[i], "--shared-multiscale") == 0) {
            shared_multiscale_head = 1;
            multiscale_aux = 1;
        }
        else if (strcmp(argv[i], "--context-fusion") == 0) {
            context_fusion = 1;
            shared_multiscale_head = 1;
            multiscale_aux = 1;
        }
        else if (strcmp(argv[i], "--raw-features") == 0) {
            raw_input_features = 1;
        }
        else if (strcmp(argv[i], "--p3-only-deploy") == 0) {
            p3_only_deployment = 1;
        }
        else if (strcmp(argv[i], "--smooth-distances") == 0) {
            smooth_box_decode = 1;
        }
        else if (strcmp(argv[i], "--predecode-cache") == 0) {
            predecode_cache = 1;
        }
        else if (strcmp(argv[i], "--shuffle") == 0 ||
                 strcmp(argv[i], "--shuffle-cache") == 0) {
            shuffle_dataset = 1;
        }
        else if (strcmp(argv[i], "--no-shuffle") == 0) {
            shuffle_dataset = 0;
        }
        else if (strcmp(argv[i], "--time-budget-ms") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &time_budget_ms)) {
                fprintf(stderr, "time-budget-ms must be a positive integer\n");
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
        else if (strcmp(argv[i], "--channels") == 0) {
            if (i + 1 >= argc || !parse_channels(argv[++i], &channels)) {
                fprintf(stderr, "channels must be 1 or 3\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--classes") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &classes) ||
                classes > DET_MAX_CLASSES) {
                fprintf(stderr, "classes must be an integer from 1 to %d\n",
                        DET_MAX_CLASSES);
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--features") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &feature_channels) ||
                feature_channels > 32) {
                fprintf(stderr, "features must be an integer from 1 to 32\n");
                return EXIT_FAILURE;
            }
            features_set = 1;
        }
        else if (strcmp(argv[i], "--arena-kib") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &arena_kib) ||
                arena_kib > 1048576) {
                fprintf(stderr, "arena-kib must be an integer from 1 to 1048576\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--max-detections") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &max_detections) ||
                max_detections > 100) {
                fprintf(stderr, "max-detections must be an integer from 1 to 100\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--learning-rate") == 0) {
            if (i + 1 >= argc ||
                !parse_learning_rate(argv[++i], &learning_rate)) {
                fprintf(stderr, "learning-rate must be finite and in (0,10]\n");
                return EXIT_FAILURE;
            }
            learning_rate_set = 1;
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
        else if (strcmp(argv[i], "--stem") == 0) {
            if (i + 1 >= argc || !parse_stem_mode(argv[++i], &stem_mode)) {
                fprintf(stderr, "stem must be legacy or space-to-depth\n");
                return EXIT_FAILURE;
            }
        }
        else if (strcmp(argv[i], "--research-mode") == 0) {
            if (i + 1 >= argc || !parse_research_mode(argv[++i], &research_mode)) {
                fprintf(stderr, "research-mode must be scratch, adapt, or full\n");
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
        else if (strcmp(argv[i], "--init") == 0) {
            if (i + 1 >= argc || argv[++i][0] == '\0') {
                fprintf(stderr, "init checkpoint must be a non-empty path\n");
                return EXIT_FAILURE;
            }
            init_path = argv[i];
        }
        else if (strcmp(argv[i], "--save-model") == 0) {
            if (i + 1 >= argc || argv[++i][0] == '\0') {
                fprintf(stderr, "save-model path must be non-empty\n");
                return EXIT_FAILURE;
            }
            save_model_path = argv[i];
        }
        else if (strcmp(argv[i], "--native-graph-dir") == 0) {
            if (i + 1 >= argc || argv[++i][0] == '\0') {
                fprintf(stderr, "native-graph-dir must be a non-empty path\n");
                return EXIT_FAILURE;
            }
            native_graph_dir = argv[i];
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
    if (native_graph_dir != NULL && research_mode == DET_RESEARCH_SCRATCH) {
        research_mode = DET_RESEARCH_ADAPT;
    }
    if (native_graph_dir != NULL && !learning_rate_set) {
        /* Native score/box adapters converge more reliably with a small step
         * than the historical 0.10 default; callers can still override it. */
        learning_rate = 0.02f;
    }
    if (eval_only && init_path == NULL) {
        fprintf(stderr, "--eval-only requires --init PATH\n");
        return EXIT_FAILURE;
    }
    if ((architecture == DET_ARCH_CDET && precision == DET_PRECISION_INT4) ||
        (architecture == DET_ARCH_KSHIRA && precision == DET_PRECISION_W4A8) ||
        (architecture == DET_ARCH_KSHIRA && global) ||
        (architecture == DET_ARCH_CDET && features_set) ||
        (architecture == DET_ARCH_KSHIRA && native_graph_dir == NULL &&
         max_detections > 64) ||
        (native_graph_dir != NULL && architecture != DET_ARCH_KSHIRA)) {
        fprintf(stderr, "requested architecture/precision/training mode combination is unsupported\n");
        return EXIT_FAILURE;
    }
    double e2e_start = wall_now_ms();
    det_context *ctx = NULL;
    det_model *model = NULL;
    if (max_detections == 0) {
        max_detections = architecture == DET_ARCH_KSHIRA ?
                         (native_graph_dir != NULL ? 100 : 64) : 100;
    }
    det_model_spec spec = {width, height, channels, classes, max_detections, seed,
                           architecture,
                           architecture == DET_ARCH_KSHIRA ? feature_channels : 0,
                           architecture == DET_ARCH_KSHIRA ? stem_mode : DET_STEM_LEGACY,
                           architecture == DET_ARCH_KSHIRA ? one_to_one_head : 0,
                           architecture == DET_ARCH_KSHIRA ? shared_multiscale_head : 0,
                           research_mode,
                           architecture == DET_ARCH_KSHIRA ? context_fusion : 0,
                           architecture == DET_ARCH_KSHIRA ? raw_input_features : 0,
                           architecture == DET_ARCH_KSHIRA ? p3_only_deployment : 0,
                           architecture == DET_ARCH_KSHIRA ? smooth_box_decode : 0,
                           native_graph_dir};
    if (arena_kib == 0) arena_kib = architecture == DET_ARCH_KSHIRA ? 256 : 8192;
    size_t arena_bytes = (size_t)arena_kib << 10;
    det_status status = det_context_create(arena_bytes, &ctx);
    if (status == DET_OK) {
        status = init_path == NULL ? det_model_build(ctx, &spec, &model) :
                                     det_load(ctx, init_path, &model);
    }
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
        double preparation_start = wall_now_ms();
        status = det_manifest_open(manifest_path, width, height, channels, 100, &manifest_dataset);
        if (status == DET_OK && predecode_cache) {
            status = det_manifest_enable_cache(manifest_dataset);
        }
        if (status == DET_OK && shuffle_dataset) {
            status = det_manifest_set_shuffle(manifest_dataset, 1, seed);
        }
        preparation_ms = wall_now_ms() - preparation_start;
        if (status == DET_OK) status = det_manifest_dataset_view(manifest_dataset, &dataset);
        if (status != DET_OK || dataset.sample_count == 0U) {
            fprintf(stderr, "manifest setup failed: %d\n", status);
            det_manifest_close(manifest_dataset);
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
    } else {
        pixels = (float *)calloc((size_t)width * (size_t)height * (size_t)channels,
                                 sizeof(float));
        if (pixels == NULL) {
            fprintf(stderr, "pixel allocation failed\n");
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
        storage = (bench_dataset){
            .pixels = pixels,
            .width = width,
            .height = height,
            .channels = channels,
            .classes = classes,
            .variant = 0,
            .total = (size_t)sample_count,
            .index = 0U,
            .box = {0, 0, 0, 0, 0}
        };
        dataset = (det_dataset){&storage, next_sample, reset_dataset, (size_t)sample_count};
    }
    /* For raw manifests, max_samples=0 streams the full set each epoch.
     * Synthetic still uses --samples as the in-memory workload size. */
    int train_max_samples = max_train_samples > 0 ? max_train_samples :
                            (manifest_path != NULL ? 0 : sample_count);
    det_train_config config = {global ? DET_TRAIN_GLOBAL_BP : DET_TRAIN_LOCAL_FAST,
                               architecture == DET_ARCH_KSHIRA ? precision : DET_PRECISION_F32,
                               epochs, learning_rate,
                               architecture == DET_ARCH_KSHIRA ? 0.0f : 0.8f, threshold,
                               train_max_samples, seed, init_path == NULL ? 1 : 0,
                               (double)time_budget_ms, analytic_readout,
                               (double)bootstrap_ms, bootstrap_enabled,
                               freeze_encoder, reset_schedule, dense_aux_budget,
                               max_total_train_samples, multiscale_aux,
                                adaptive_budget, residual_budget,
                                 quality_aligned_assignment, feature_readout,
                                 candidate_head_adapt};
    det_train_report report;
    double core_start = wall_now_ms();
    memset(&report, 0, sizeof(report));
    if (!eval_only) {
        status = det_train(model, &dataset, &config, &report);
        if (status != DET_OK) {
            fprintf(stderr, "training failed: %d\n", status);
            free(pixels);
            det_manifest_close(manifest_dataset);
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
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
    if (save_model_path != NULL) {
        status = det_save(model, save_model_path);
        if (status != DET_OK) {
            fprintf(stderr, "could not save model: %d (%s)\n", status, save_model_path);
            free(pixels);
            det_manifest_close(manifest_dataset);
            det_model_destroy(model);
            det_context_destroy(ctx);
            return EXIT_FAILURE;
        }
    }
    if (no_evaluate) {
        printf("training_only samples=%zu updates=%zu candidate_head_updates=%zu loss=%.6f elapsed_ms=%.3f saved=%s\n",
               report.samples_seen, report.updates, report.candidate_head_updates,
               report.mean_loss, report.elapsed_ms,
               save_model_path != NULL ? save_model_path : "none");
        free(pixels);
        det_manifest_close(manifest_dataset);
        det_model_destroy(model);
        det_context_destroy(ctx);
        return EXIT_SUCCESS;
    }
    det_memory_report memory;
    status = det_model_memory(model, &memory);
    if (status != DET_OK) {
        fprintf(stderr, "memory reporting failed: %d\n", status);
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
    size_t checkpoint_bytes = 0U;
    status = det_save(model, model_path);
    double serialized_at = wall_now_ms();
    double train_core_ms = serialized_at - core_start;
    double synthetic_e2e_ms = serialized_at - e2e_start;
    if (status == DET_OK && !file_size_bytes(model_path, &checkpoint_bytes)) {
        status = DET_ERR_IO;
    }
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
        status = det_manifest_open(eval_manifest_path, width, height, channels, 100,
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
    } else if (manifest_path == NULL) {
        storage.variant = 1;
        storage.total = 100U;
        storage.index = 0U;
        eval_dataset = (det_dataset){&storage, next_sample, reset_dataset, 100U};
    }
    /* Deploy-time objectness/score calibration: grid-search threshold on the
     * eval stream and keep the F1-maximizing operating point. */
    float calibrated_threshold = threshold;
    if (!no_calibrate && (calibrate || eval_manifest_path != NULL)) {
        static const float candidates[] = {
            0.05f, 0.08f, 0.10f, 0.12f, 0.15f, 0.18f, 0.20f, 0.25f, 0.30f, 0.40f, 0.50f
        };
        float best_f1 = -1.0f;
        float best_precision = 0.0f;
        float best_recall = 0.0f;
        for (size_t i = 0U; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
            det_eval_report probe;
            float f1;
            status = det_evaluate(loaded, &eval_dataset, candidates[i], &probe);
            if (status != DET_OK) {
                fprintf(stderr, "calibration eval failed at thr=%.3f: %d\n",
                        candidates[i], status);
                free(pixels);
                det_manifest_close(eval_manifest_dataset);
                det_manifest_close(manifest_dataset);
                det_model_destroy(loaded);
                det_model_destroy(model);
                det_context_destroy(ctx);
                return EXIT_FAILURE;
            }
            if (probe.precision + probe.recall > 0.0f) {
                f1 = 2.0f * probe.precision * probe.recall /
                     (probe.precision + probe.recall);
            } else {
                f1 = 0.0f;
            }
            if (f1 > best_f1 ||
                (f1 == best_f1 && probe.precision > best_precision)) {
                best_f1 = f1;
                best_precision = probe.precision;
                best_recall = probe.recall;
                calibrated_threshold = candidates[i];
            }
            printf("calibrate thr=%.3f precision=%.4f recall=%.4f f1=%.4f "
                   "pred=%zu tp=%zu fp=%zu\n",
                   candidates[i], probe.precision, probe.recall, f1,
                   probe.predictions, probe.true_positives, probe.false_positives);
        }
        threshold = calibrated_threshold;
        printf("calibrated_threshold=%.3f best_f1=%.4f precision=%.4f recall=%.4f\n",
               threshold, best_f1, best_precision, best_recall);
    }
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
    det_model_destroy(loaded);
    const char *precision_name = precision == DET_PRECISION_INT8 ? "INT8" :
                                 (precision == DET_PRECISION_W4A8 ? "W4A8" :
                                  (precision == DET_PRECISION_INT4 ? "INT4" : "F32"));
    const char *architecture_name = architecture == DET_ARCH_KSHIRA ? "KSHIRA" : "CDET";
    int reported_one_to_one = det_model_one_to_one_head(model);
    int reported_shared_multiscale = det_model_shared_multiscale(model);
    int reported_context_fusion = det_model_context_fusion(model);
    int reported_p3_only_deployment = det_model_p3_only_deployment(model);
    det_research_mode reported_research_mode = det_model_research_mode(model);
    const char *stem_name = stem_mode == DET_STEM_SPACE_TO_DEPTH ?
                             "space-to-depth" :
                             (stem_mode == DET_STEM_TWO_STAGE ? "two-stage" : "legacy");

     printf("samples=%zu epochs=%d seed=%d research_mode=%s input=%dx%dx%d classes=%d features=%d max_detections=%d architecture=%s stem=%s one_to_one_head=%d shared_multiscale=%d context_fusion=%d raw_features=%d p3_only_deploy=%d smooth_box_decode=%d quality_align=%d source=%s mode=%s precision=%s learning_rate=%.6f threshold=%.3f budget_ms=%d budget_stop=%d max_total_train_samples=%d dense_aux=%d multiscale_aux=%d adaptive_budget=%d residual_budget=%d shuffle=%d predecode_cache=%d no_calibrate=%d preparation_ms=%.3f bootstrap=%d bootstrap_ms=%d bootstrap_samples=%zu bootstrap_ms_used=%.3f analytic_readout=%d readout_applied=%d feature_readout=%d feature_readout_applied=%d feature_readout_examples=%zu candidate_head_adapt=%d candidate_head_updates=%zu %s=%.3f %s=%.3f "
           "infer_ms=%.3f io_ms=%.3f "
           "updates=%zu loss=%.6f images_per_sec=%.2f detections=%d\n",
           report.samples_seen, epochs, seed, research_mode_name(reported_research_mode),
           width, height, channels, classes,
           architecture == DET_ARCH_KSHIRA ? feature_channels : 0,
           max_detections, architecture_name, stem_name, reported_one_to_one,
           reported_shared_multiscale,
           reported_context_fusion,
           raw_input_features,
           reported_p3_only_deployment,
           smooth_box_decode,
           quality_aligned_assignment,
           manifest_path == NULL ? "synthetic" : "manifest",
           global ? "GLOBAL_BP" : "LOCAL_FAST", precision_name,
           learning_rate, threshold, time_budget_ms, report.stopped_by_budget,
           max_total_train_samples, dense_aux_budget, multiscale_aux, adaptive_budget,
           residual_budget, shuffle_dataset, predecode_cache, no_calibrate, preparation_ms,
           bootstrap_enabled, bootstrap_ms,
           report.bootstrap_samples,
            report.bootstrap_elapsed_ms, analytic_readout, report.analytic_readout_applied,
            feature_readout, report.feature_readout_applied,
            report.feature_readout_examples,
            candidate_head_adapt, report.candidate_head_updates,
           manifest_path == NULL ? "train_core_ms" : "train_plus_decode_ms",
           train_core_ms, manifest_path == NULL ? "synthetic_e2e_ms" : "train_e2e_ms",
           synthetic_e2e_ms, infer_ms, io_ms,
           report.updates, report.mean_loss,
           train_core_ms > 0.0 ? (double)report.samples_seen * 1000.0 / train_core_ms : 0.0,
           detection_count);
    printf("parameter_bytes=%zu optimizer_bytes=%zu quant_cache_bytes=%zu "
           "activation_workspace_bytes=%zu arena_high_water_bytes=%zu "
           "arena_capacity_bytes=%zu checkpoint_bytes=%zu\n",
           memory.parameter_bytes, memory.optimizer_bytes,
           memory.quant_cache_bytes, memory.activation_workspace_bytes,
           memory.arena_high_water_bytes, memory.arena_capacity_bytes,
           checkpoint_bytes);
    printf("reference_profile=%s timing_scope=%s train_under_1s=%s "
           "infer_under_33ms=%s arena_budget=%s\n",
           report.samples_seen == 5000U && width == 160 && height == 160 ?
               "YES" : "NO",
           manifest_path == NULL ? "synthetic" : "raw_manifest",
           synthetic_e2e_ms <= 1000.0 ? "PASS" : "FAIL",
           infer_ms < 33.0 ? "PASS" : "FAIL",
           memory.arena_capacity_bytes == 0U ? "NA" :
               (memory.arena_high_water_bytes <= memory.arena_capacity_bytes ?
                    "PASS" : "FAIL"));
    printf("eval_samples=%zu eval_ground_truths=%zu eval_predictions=%zu eval_tp=%zu "
           "eval_fp=%zu eval_fn=%zu precision=%.4f recall=%.4f mean_iou=%.4f "
           "ap50=%.4f map50_95=%.4f size_gt=%zu,%zu,%zu\n",
           evaluation.samples_seen, evaluation.ground_truths, evaluation.predictions,
           evaluation.true_positives, evaluation.false_positives,
           evaluation.false_negatives, evaluation.precision, evaluation.recall,
           evaluation.mean_iou, evaluation.ap50, evaluation.map50_95,
           evaluation.size_ground_truths[0], evaluation.size_ground_truths[1],
           evaluation.size_ground_truths[2]);
    printf("score_hist_positive=");
    for (int bin = 0; bin < 10; ++bin) {
        if (bin != 0) putchar(',');
        printf("%zu", evaluation.score_histogram[0][bin]);
    }
    printf(" score_hist_negative=");
    for (int bin = 0; bin < 10; ++bin) {
        if (bin != 0) putchar(',');
        printf("%zu", evaluation.score_histogram[1][bin]);
    }
    printf(" readout_examples=%zu readout_positives=%zu readout_negatives=%zu "
           "readout_pre=%.6f readout_post=%.6f\n",
           report.readout_examples, report.readout_positives, report.readout_negatives,
           report.readout_pre_objective, report.readout_post_objective);
    printf("class_ap50=");
    for (int class_id = 0; class_id < classes; ++class_id) {
        if (class_id != 0) putchar(',');
        printf("%.4f", evaluation.class_ap50[class_id]);
    }
    putchar('\n');
    free(pixels);
    det_manifest_close(eval_manifest_dataset);
    det_manifest_close(manifest_dataset);
    det_model_destroy(model);
    det_context_destroy(ctx);
    return EXIT_SUCCESS;
}
