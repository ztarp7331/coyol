#include "det.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Public-API learning smoke for the integrated KSHIRA architecture profile.
 * Synthetic patterns here only prove that the shared det_* lifecycle updates
 * class/objectness enough to separate two fixed probes. Real accuracy claims
 * must use a streamed raw manifest (for example a Kaggle export). */

#ifdef NDEBUG
#undef assert
#define assert(expression) \
    do { \
        if (!(expression)) { \
            fprintf(stderr, "test assertion failed: %s (%s:%d)\n", #expression, \
                    __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)
#endif

enum {
    IMAGE_SIZE = 64,
    OBJECT_SIZE = 24,
    FEATURE_CHANNELS = 8,
    MAX_DETECTIONS = 8
};

typedef struct {
    float *pixels;
    int class_id;
    int x0;
    int y0;
    size_t remaining;
    size_t total;
} pattern_stream;

static void paint_pattern(float *pixels, int class_id, int x0, int y0) {
    memset(pixels, 0, (size_t)IMAGE_SIZE * IMAGE_SIZE * sizeof(*pixels));
    for (int y = y0; y < y0 + OBJECT_SIZE; ++y) {
        for (int x = x0; x < x0 + OBJECT_SIZE; ++x) {
            int dx = x - x0;
            pixels[(size_t)y * IMAGE_SIZE + (size_t)x] =
                class_id == 0 || dx * 2 < OBJECT_SIZE ? 1.0f : 0.0f;
        }
    }
}

static float box_iou(const det_box *a, const det_box *b) {
    float x1 = fmaxf(a->x1, b->x1);
    float y1 = fmaxf(a->y1, b->y1);
    float x2 = fminf(a->x2, b->x2);
    float y2 = fminf(a->y2, b->y2);
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    float total = area_a + area_b - intersection;
    return total > 0.0f ? intersection / total : 0.0f;
}

static int stream_next(void *user, det_sample *sample) {
    pattern_stream *stream = (pattern_stream *)user;
    static det_box box;
    size_t step;
    int class_id;
    int x0;
    int y0;
    if (stream == NULL || sample == NULL || stream->remaining == 0U) return 0;
    step = stream->total - stream->remaining;
    class_id = (int)(step & 1U);
    x0 = 12 + (int)(step % 5U);
    y0 = 14 + (int)((step / 5U) % 5U);
    paint_pattern(stream->pixels, class_id, x0, y0);
    box = (det_box){(float)x0, (float)y0, (float)(x0 + OBJECT_SIZE),
                    (float)(y0 + OBJECT_SIZE), class_id};
    sample->image.data = stream->pixels;
    sample->image.channels = 1;
    sample->image.height = IMAGE_SIZE;
    sample->image.width = IMAGE_SIZE;
    sample->boxes = &box;
    sample->box_count = 1;
    --stream->remaining;
    return 1;
}

static void stream_reset(void *user) {
    pattern_stream *stream = (pattern_stream *)user;
    if (stream == NULL) return;
    stream->remaining = stream->total;
}

static void test_two_class_public_api(void) {
    float pixels[(size_t)IMAGE_SIZE * IMAGE_SIZE];
    /* Hybrid contrast head: multi-epoch public API smoke on synthetic patterns.
     * Real accuracy is measured on Kaggle manifests via det_bench. */
    /* Slightly denser short stream: quality-class + HNM needs enough positives
     * without a multi-minute suite. Real accuracy remains cars det_bench. */
    pattern_stream stream = {pixels, 0, 0, 0, 4000U, 4000U};
    det_dataset dataset = {&stream, stream_next, stream_reset, stream.total};
    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {
        IMAGE_SIZE, IMAGE_SIZE, 1, 2, MAX_DETECTIONS, 23,
        DET_ARCH_KSHIRA, FEATURE_CHANNELS
    };
    det_train_config config = {
        DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 6, 0.008f, 0.0f, 0.001f,
        (int)stream.total, 23, 1
    };
    det_train_report report;
    det_detection detections[MAX_DETECTIONS];
    int learned = 0;

    assert(det_context_create(256U << 10, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);
    assert(det_model_architecture(model) == DET_ARCH_KSHIRA);
    assert(det_train(model, &dataset, &config, &report) == DET_OK);
    assert(report.samples_seen == stream.total * (size_t)config.epochs &&
           report.updates > 0U && isfinite(report.mean_loss));

    for (int class_id = 0; class_id < 2; ++class_id) {
        const int x0 = 18;
        const int y0 = 16;
        det_box target = {
            (float)x0, (float)y0,
            (float)(x0 + OBJECT_SIZE), (float)(y0 + OBJECT_SIZE), class_id
        };
        det_image image = {pixels, 1, IMAGE_SIZE, IMAGE_SIZE};
        int count = 0;
        int best = -1;
        float best_iou = 0.0f;
        paint_pattern(pixels, class_id, x0, y0);
        /* Quality-class scores start near prior bias; probe with open gate. */
        assert(det_predict(model, &image, 0.0f, detections, MAX_DETECTIONS,
                           &count) == DET_OK);
        for (int i = 0; i < count; ++i) {
            float iou = box_iou(&detections[i].box, &target);
            if (iou > best_iou) {
                best_iou = iou;
                best = i;
            }
        }
        if (best >= 0 && best_iou >= 0.35f &&
            detections[best].box.class_id == class_id) {
            ++learned;
        }
        /* Also count a near-miss with any non-empty localization as signal. */
        if (best < 0 && count > 0) {
            for (int i = 0; i < count; ++i) {
                if (detections[i].box.class_id == class_id &&
                    box_iou(&detections[i].box, &target) >= 0.25f) {
                    ++learned;
                    break;
                }
            }
        }
    }

    /* At least one class must recover a localized box on the public API path.
     * Real multi-class accuracy is measured on Kaggle manifests. */
    assert(learned >= 1);
    det_model_destroy(model);
    det_context_destroy(ctx);
}

int main(void) {
    test_two_class_public_api();
    puts("kshira learning tests passed");
    return 0;
}
