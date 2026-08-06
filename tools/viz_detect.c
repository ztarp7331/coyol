/* Full train → save model → dump predictions for ALL (or N) eval samples.
 * Companion: tools/draw_detections.py draws GT (green) + PRED (red) onto PNGs. */

#include "det.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VIZ_MAX_DETECTIONS = 100 };

static char custom_class_names[DET_MAX_CLASSES][64];
static int custom_class_name_count = 0;

static const char *class_name(int id, int classes) {
    static const char *cars[] = {
        "Ambulance", "Bus", "Car", "Motorcycle", "Truck"
    };
    if (classes == 5 && id >= 0 && id < 5) return cars[id];
    static char buf[32];
    snprintf(buf, sizeof(buf), "class_%d", id);
    return buf;
}

static const char *generic_class_name(int id) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "class_%d", id);
    return buf;
}

static const char *output_class_name(int id, int classes, int generic_labels) {
    if (id >= 0 && id < custom_class_name_count) return custom_class_names[id];
    return generic_labels ? generic_class_name(id) : class_name(id, classes);
}

static int load_class_names(const char *path) {
    FILE *file = fopen(path, "rb");
    char line[sizeof(custom_class_names[0])];
    if (file == NULL) return 0;
    custom_class_name_count = 0;
    while (custom_class_name_count < DET_MAX_CLASSES &&
           fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        while (length > 0U && (line[length - 1U] == '\n' || line[length - 1U] == '\r'))
            line[--length] = '\0';
        if (length == 0U) continue;
        memcpy(custom_class_names[custom_class_name_count], line, length + 1U);
        ++custom_class_name_count;
    }
    fclose(file);
    return custom_class_name_count > 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [options]\n"
            "  --train-manifest PATH   train manifest (default cars train)\n"
            "  --eval-manifest PATH    eval manifest (default cars valid)\n"
            "  --out PATH              detections report (default results/detections.txt)\n"
            "  --save PATH             save trained model checkpoint\n"
            "  --load PATH             load model instead of training\n"
            "  --preview N             dump first N eval images (default: ALL)\n"
            "  --epochs N              train epochs (default 10; surgical FP path)\n"
            "  --lr F                  learning rate (default 0.004)\n"
            "  --threshold F           score threshold for pred dump (default 0.25)\n"
            "  --features N            feature channels (default 8)\n"
            "  --max-det N             max detections (default 8)\n"
            "  --precision f32|int8|int4\n"
            "  --generic-labels       use class_N labels instead of vehicle names\n"
            "  --labels PATH          class names, one per line\n"
            "  --arena-kib N           arena size KiB (default 256)\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *train_manifest = "datasets/prepared/cars/train/manifest.txt";
    const char *eval_manifest = "datasets/prepared/cars/valid/manifest.txt";
    const char *out_path = "results/detections.txt";
    const char *save_path = "results/kshira_cars.bin";
    const char *load_path = NULL;
    int width = 160;
    int height = 160;
    int classes = 5;
    int features = 8;
    int max_det = 8;
    int arena_kib = 256;
    int preview = 0; /* 0 = ALL eval samples */
    int epochs = 10;
    float lr = 0.004f;
    float threshold = 0.25f; /* higher default to cut FP flood in viz */
    det_precision precision = DET_PRECISION_F32;
    int generic_labels = 0;
    const char *labels_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--train-manifest") == 0 && i + 1 < argc)
            train_manifest = argv[++i];
        else if (strcmp(argv[i], "--eval-manifest") == 0 && i + 1 < argc)
            eval_manifest = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_path = argv[++i];
        else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc)
            save_path = argv[++i];
        else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc)
            load_path = argv[++i];
        else if (strcmp(argv[i], "--preview") == 0 && i + 1 < argc)
            preview = atoi(argv[++i]);
        else if (strcmp(argv[i], "--all") == 0)
            preview = 0;
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc)
            lr = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc)
            threshold = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--features") == 0 && i + 1 < argc)
            features = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-det") == 0 && i + 1 < argc)
            max_det = atoi(argv[++i]);
        else if (strcmp(argv[i], "--arena-kib") == 0 && i + 1 < argc)
            arena_kib = atoi(argv[++i]);
        else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "int8") == 0) precision = DET_PRECISION_INT8;
            else if (strcmp(argv[i], "int4") == 0) precision = DET_PRECISION_INT4;
            else precision = DET_PRECISION_F32;
        } else if (strcmp(argv[i], "--generic-labels") == 0) {
            generic_labels = 1;
        } else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (epochs < 1) epochs = 1;
    if (max_det < 1) max_det = 1;
    if (max_det > VIZ_MAX_DETECTIONS) max_det = VIZ_MAX_DETECTIONS;
    if (labels_path != NULL && !load_class_names(labels_path)) {
        fprintf(stderr, "could not load labels: %s\n", labels_path);
        return 1;
    }

    det_context *ctx = NULL;
    det_model *model = NULL;
    det_status status = det_context_create((size_t)arena_kib << 10, &ctx);
    if (status != DET_OK) {
        fprintf(stderr, "context create failed: %d\n", status);
        return 1;
    }

    if (load_path != NULL) {
        status = det_load(ctx, load_path, &model);
        if (status != DET_OK) {
            fprintf(stderr, "load failed: %d (%s)\n", status, load_path);
            return 1;
        }
        fprintf(stderr, "loaded model from %s\n", load_path);
    } else {
        det_model_spec spec = {
            width, height, 1, classes, max_det, 1, DET_ARCH_KSHIRA, features
        };
        status = det_model_build(ctx, &spec, &model);
        if (status != DET_OK) {
            fprintf(stderr, "model setup failed: %d\n", status);
            return 1;
        }

        det_manifest_dataset *train_ds = NULL;
        det_dataset train_view;
        status = det_manifest_open(train_manifest, width, height, 1, 100, &train_ds);
        if (status == DET_OK) status = det_manifest_dataset_view(train_ds, &train_view);
        if (status != DET_OK) {
            fprintf(stderr, "train manifest failed: %d\n", status);
            return 1;
        }

        det_train_config config = {
            DET_TRAIN_LOCAL_FAST, precision, epochs, lr, 0.0f, threshold,
            0, 1, 1
        };
        det_train_report report;
        status = det_train(model, &train_view, &config, &report);
        if (status != DET_OK) {
            fprintf(stderr, "train failed: %d\n", status);
            return 1;
        }
        fprintf(stderr,
                "trained epochs=%d samples=%zu updates=%zu loss=%.4f ms=%.1f thr=%.3f\n",
                epochs, report.samples_seen, report.updates, report.mean_loss,
                report.elapsed_ms, threshold);

        if (save_path != NULL && save_path[0] != '\0') {
            status = det_save(model, save_path);
            if (status != DET_OK) {
                fprintf(stderr, "save failed: %d (%s)\n", status, save_path);
                return 1;
            }
            fprintf(stderr, "saved model -> %s\n", save_path);
        }
        det_manifest_close(train_ds);
    }

    det_manifest_dataset *eval_ds = NULL;
    det_dataset eval_view;
    status = det_manifest_open(eval_manifest, width, height, 1, 100, &eval_ds);
    if (status == DET_OK) status = det_manifest_dataset_view(eval_ds, &eval_view);
    if (status != DET_OK) {
        fprintf(stderr, "eval manifest failed: %d\n", status);
        return 1;
    }

    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }
    fprintf(out, "# model_input %dx%d threshold=%.3f precision=%d max_det=%d\n",
            width, height, threshold, (int)precision, max_det);
    fprintf(out, "# format: SAMPLE index image_rel\n");
    fprintf(out, "# format: GT x1 y1 x2 y2 class name\n");
    fprintf(out, "# format: PRED x1 y1 x2 y2 class name score\n");

    FILE *manifest_text = fopen(eval_manifest, "rb");
    if (manifest_text == NULL) {
        fprintf(stderr, "cannot re-open eval manifest text\n");
        return 1;
    }

    int index = 0;
    int total_gt = 0;
    int total_pred = 0;
    for (;;) {
        det_sample sample;
        det_detection detections[VIZ_MAX_DETECTIONS];
        int count = 0;
        char line[8192];
        char image_rel[4096];
        int next;

        if (preview > 0 && index >= preview) break;

        next = eval_view.next(eval_view.user, &sample);
        if (next == 0) break;
        if (next < 0) {
            fprintf(stderr, "eval stream failed at %d\n", index);
            return 1;
        }
        if (fgets(line, sizeof(line), manifest_text) == NULL) break;
        {
            char *sp = line;
            while (*sp && *sp != ' ' && *sp != '\t' && *sp != '\n' && *sp != '\r') ++sp;
            size_t n = (size_t)(sp - line);
            if (n >= sizeof(image_rel)) n = sizeof(image_rel) - 1U;
            memcpy(image_rel, line, n);
            image_rel[n] = '\0';
        }

        status = det_predict(model, &sample.image, threshold, detections, max_det,
                             &count);
        if (status != DET_OK) {
            fprintf(stderr, "predict failed: %d\n", status);
            return 1;
        }

        fprintf(out, "SAMPLE %d %s\n", index, image_rel);
        for (int b = 0; b < sample.box_count; ++b) {
            const det_box *box = &sample.boxes[b];
            fprintf(out, "GT %.2f %.2f %.2f %.2f %d %s\n",
                    box->x1, box->y1, box->x2, box->y2, box->class_id,
                    output_class_name(box->class_id, classes, generic_labels));
            ++total_gt;
        }
        for (int d = 0; d < count; ++d) {
            const det_detection *det = &detections[d];
            fprintf(out, "PRED %.2f %.2f %.2f %.2f %d %s %.4f\n",
                    det->box.x1, det->box.y1, det->box.x2, det->box.y2,
                    det->box.class_id, output_class_name(det->box.class_id, classes, generic_labels),
                    det->score);
            ++total_pred;
        }
        ++index;
        if ((index % 50) == 0) {
            fprintf(stderr, "predicted %d eval samples...\n", index);
        }
    }

    fclose(manifest_text);
    fclose(out);
    det_manifest_close(eval_ds);
    det_model_destroy(model);
    det_context_destroy(ctx);
    fprintf(stderr, "wrote %s (%d samples, gt=%d pred=%d thr=%.3f)\n",
            out_path, index, total_gt, total_pred, threshold);
    if (save_path != NULL && save_path[0] != '\0' && load_path == NULL) {
        fprintf(stderr, "model checkpoint: %s\n", save_path);
    }
    return 0;
}
