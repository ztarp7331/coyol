/* Train on a raw manifest, then dump predictions for the first N eval samples
 * as a simple text report. A companion Python script draws boxes onto PNGs. */

#include "det.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *class_name(int id, int classes) {
    static const char *cars[] = {
        "Ambulance", "Bus", "Car", "Motorcycle", "Truck"
    };
    if (classes == 5 && id >= 0 && id < 5) return cars[id];
    static char buf[32];
    snprintf(buf, sizeof(buf), "class_%d", id);
    return buf;
}

int main(int argc, char **argv) {
    const char *train_manifest = "datasets/prepared/cars/train/manifest.txt";
    const char *eval_manifest = "datasets/prepared/cars/valid/manifest.txt";
    const char *out_path = "results/detections.txt";
    int width = 160;
    int height = 160;
    int classes = 5;
    int features = 8;
    int max_det = 8;
    int arena_kib = 256;
    int preview = 12;
    int epochs = 3;
    float lr = 0.01f;
    float threshold = 0.15f;
    det_precision precision = DET_PRECISION_F32;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--train-manifest") == 0 && i + 1 < argc)
            train_manifest = argv[++i];
        else if (strcmp(argv[i], "--eval-manifest") == 0 && i + 1 < argc)
            eval_manifest = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            out_path = argv[++i];
        else if (strcmp(argv[i], "--preview") == 0 && i + 1 < argc)
            preview = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc)
            threshold = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "int8") == 0) precision = DET_PRECISION_INT8;
            else if (strcmp(argv[i], "int4") == 0) precision = DET_PRECISION_INT4;
            else precision = DET_PRECISION_F32;
        }
    }
    if (epochs < 1) epochs = 1;

    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {
        width, height, 1, classes, max_det, 1, DET_ARCH_KSHIRA, features
    };
    det_status status = det_context_create((size_t)arena_kib << 10, &ctx);
    if (status == DET_OK) status = det_model_build(ctx, &spec, &model);
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

    /* Re-open eval manifest as text to recover relative image paths that match
     * the prepared PGM files (boxes are in original 416 coords there). */
    FILE *manifest_text = fopen(eval_manifest, "rb");
    if (manifest_text == NULL) {
        fprintf(stderr, "cannot re-open eval manifest text\n");
        return 1;
    }

    int index = 0;
    while (index < preview) {
        det_sample sample;
        det_detection detections[64];
        int count = 0;
        char line[8192];
        char image_rel[4096];
        int next = eval_view.next(eval_view.user, &sample);
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
                    class_name(box->class_id, classes));
        }
        for (int d = 0; d < count; ++d) {
            const det_detection *det = &detections[d];
            fprintf(out, "PRED %.2f %.2f %.2f %.2f %d %s %.4f\n",
                    det->box.x1, det->box.y1, det->box.x2, det->box.y2,
                    det->box.class_id, class_name(det->box.class_id, classes),
                    det->score);
        }
        ++index;
    }

    fclose(manifest_text);
    fclose(out);
    det_manifest_close(eval_ds);
    det_manifest_close(train_ds);
    det_model_destroy(model);
    det_context_destroy(ctx);
    fprintf(stderr, "wrote %s (%d samples)\n", out_path, index);
    return 0;
}
