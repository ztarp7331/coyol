#include "det.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <direct.h>
#else
#include <unistd.h>
#endif

static float box_iou(const det_box *left, const det_box *right) {
    float x1 = fmaxf(left->x1, right->x1);
    float y1 = fmaxf(left->y1, right->y1);
    float x2 = fminf(left->x2, right->x2);
    float y2 = fminf(left->y2, right->y2);
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float left_area = fmaxf(0.0f, left->x2 - left->x1) *
                      fmaxf(0.0f, left->y2 - left->y1);
    float right_area = fmaxf(0.0f, right->x2 - right->x1) *
                       fmaxf(0.0f, right->y2 - right->y1);
    float union_area = left_area + right_area - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static int next_data_line(FILE *input, char *line, size_t capacity) {
    while (fgets(line, (int)capacity, input) != NULL) {
        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
               *cursor == '\n') ++cursor;
        if (*cursor != '\0' && *cursor != '#') return 1;
    }
    return 0;
}

static int first_token_and_tail(char *line, char **path, char **tail) {
    char *cursor = line;
    char *end;
    while (*cursor == ' ' || *cursor == '\t') ++cursor;
    if (*cursor == '\0' || *cursor == '#' || *cursor == '\r' || *cursor == '\n') {
        return 0;
    }
    end = cursor;
    while (*end != '\0' && *end != ' ' && *end != '\t' &&
           *end != '\r' && *end != '\n') ++end;
    if (*end != '\0') {
        *end = '\0';
        ++end;
        while (*end == ' ' || *end == '\t') ++end;
    }
    *path = cursor;
    *tail = end;
    end = strpbrk(*tail, "\r\n");
    if (end != NULL) *end = '\0';
    return 1;
}

static int parse_threshold(const char *text, float *out) {
    char *end = NULL;
    float value;
    if (text == NULL || out == NULL) return 0;
    errno = 0;
    value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) ||
        value < 0.0f || value > 1.0f) return 0;
    *out = value;
    return 1;
}

static int path_is_absolute(const char *path) {
    if (path == NULL) return 0;
#if defined(_WIN32)
    return path[0] == '/' || path[0] == '\\' ||
           (path[0] != '\0' && path[1] == ':' &&
            (path[2] == '/' || path[2] == '\\'));
#else
    return path[0] == '/';
#endif
}

static int resolve_image_path(const char *manifest_path, const char *image_path,
                              char *resolved, size_t capacity) {
    char current[2048];
    char combined[4096];
    const char *last_slash;
    size_t prefix_length;
    if (manifest_path == NULL || image_path == NULL || resolved == NULL ||
        capacity == 0U) return 0;
    if (path_is_absolute(image_path)) {
        if (strlen(image_path) >= capacity) return 0;
        memcpy(resolved, image_path, strlen(image_path) + 1U);
        return 1;
    }
#if defined(_WIN32)
    if (_getcwd(current, sizeof(current)) == NULL) return 0;
#else
    if (getcwd(current, sizeof(current)) == NULL) return 0;
#endif
    last_slash = strrchr(manifest_path, '/');
    {
        const char *backslash = strrchr(manifest_path, '\\');
        if (backslash != NULL && (last_slash == NULL || backslash > last_slash)) {
            last_slash = backslash;
        }
    }
    if (last_slash == NULL) {
        prefix_length = strlen(current);
        if (snprintf(combined, sizeof(combined), "%s/%s", current, image_path) < 0) {
            return 0;
        }
    } else if (manifest_path[0] == '/' ||
               (manifest_path[0] != '\0' && manifest_path[1] == ':')) {
        prefix_length = (size_t)(last_slash - manifest_path);
        if (prefix_length >= sizeof(combined)) return 0;
        memcpy(combined, manifest_path, prefix_length);
        combined[prefix_length] = '\0';
        if (snprintf(combined + prefix_length, sizeof(combined) - prefix_length,
                     "/%s", image_path) < 0) return 0;
    } else {
        prefix_length = strlen(current);
        if (snprintf(combined, sizeof(combined), "%s/%.*s/%s", current,
                     (int)(last_slash - manifest_path), manifest_path,
                     image_path) < 0) return 0;
    }
#if defined(_WIN32)
    return _fullpath(resolved, combined, capacity) != NULL;
#else
    if (strlen(combined) >= capacity) return 0;
    memcpy(resolved, combined, strlen(combined) + 1U);
    return 1;
#endif
}

int main(int argc, char **argv) {
    const char *graph_dir;
    const char *input_path;
    const char *output_path;
    float threshold = 0.50f;
    det_context *context = NULL;
    det_model *model = NULL;
    det_manifest_dataset *manifest = NULL;
    det_dataset dataset;
    det_model_spec spec;
    det_status status;
    FILE *input = NULL;
    FILE *output = NULL;
    char line[4096];
    size_t samples = 0U;
    size_t teacher_added = 0U;

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s GRAPH_DIR INPUT_MANIFEST OUTPUT_MANIFEST [threshold]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    graph_dir = argv[1];
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5 && !parse_threshold(argv[4], &threshold)) {
        fprintf(stderr, "threshold must be in [0,1]\n");
        return EXIT_FAILURE;
    }

    input = fopen(input_path, "rb");
    output = fopen(output_path, "wb");
    if (input == NULL || output == NULL) {
        fprintf(stderr, "cannot open manifest input/output\n");
        if (input != NULL) fclose(input);
        if (output != NULL) fclose(output);
        return EXIT_FAILURE;
    }
    status = det_context_create(256U * 1024U, &context);
    if (status == DET_OK) {
        status = det_manifest_open(input_path, 160, 160, 1, 100, &manifest);
    }
    if (status == DET_OK) status = det_manifest_dataset_view(manifest, &dataset);
    memset(&spec, 0, sizeof(spec));
    spec.width = 160;
    spec.height = 160;
    spec.channels = 1;
    spec.num_classes = 5;
    spec.max_detections = 100;
    spec.seed = 7;
    spec.architecture = DET_ARCH_KSHIRA;
    spec.feature_channels = 8;
    spec.research_mode = DET_RESEARCH_ADAPT;
    spec.native_graph_dir = graph_dir;
    if (status == DET_OK) status = det_model_build(context, &spec, &model);
    if (status != DET_OK) goto fail;

    while (next_data_line(input, line, sizeof(line))) {
        char *path = NULL;
        char *tail = NULL;
        char resolved_path[4096];
        det_sample sample;
        det_detection detections[100];
        int count = 0;
        int next_status;
        if (!first_token_and_tail(line, &path, &tail)) continue;
        if (!resolve_image_path(input_path, path, resolved_path,
                                sizeof(resolved_path))) {
            status = DET_ERR_IO;
            goto fail;
        }
        next_status = dataset.next(dataset.user, &sample);
        if (next_status <= 0) {
            status = next_status < 0 ? DET_ERR_IO : DET_ERR_FORMAT;
            goto fail;
        }
        status = det_predict(model, &sample.image, threshold, detections,
                             100, &count);
        if (status != DET_OK) goto fail;
        if (fprintf(output, "%s", resolved_path) < 0) {
            status = DET_ERR_IO;
            goto fail;
        }
        if (*tail != '\0' && *tail != '\r' && *tail != '\n' &&
            fprintf(output, " %s", tail) < 0) {
            status = DET_ERR_IO;
            goto fail;
        }
        for (int i = 0; i < count; ++i) {
            int duplicate = 0;
            for (int b = 0; b < sample.box_count; ++b) {
                if (detections[i].box.class_id == sample.boxes[b].class_id &&
                    box_iou(&detections[i].box, &sample.boxes[b]) >= 0.50f) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate || detections[i].score < threshold) continue;
            if (fprintf(output, " %.4f,%.4f,%.4f,%.4f,%d,%.6f",
                        detections[i].box.x1, detections[i].box.y1,
                        detections[i].box.x2, detections[i].box.y2,
                        detections[i].box.class_id, detections[i].score) < 0) {
                status = DET_ERR_IO;
                goto fail;
            }
            ++teacher_added;
        }
        if (fputc('\n', output) == EOF) {
            status = DET_ERR_IO;
            goto fail;
        }
        ++samples;
    }
    if (ferror(input) || fflush(output) != 0) {
        status = DET_ERR_IO;
        goto fail;
    }
    printf("samples=%zu teacher_added=%zu threshold=%.3f output=%s\n",
           samples, teacher_added, threshold, output_path);
    det_model_destroy(model);
    det_manifest_close(manifest);
    det_context_destroy(context);
    fclose(input);
    fclose(output);
    return EXIT_SUCCESS;

fail:
    fprintf(stderr, "native teacher manifest failed: %d\n", status);
    det_model_destroy(model);
    det_manifest_close(manifest);
    det_context_destroy(context);
    if (input != NULL) fclose(input);
    if (output != NULL) fclose(output);
    return EXIT_FAILURE;
}
