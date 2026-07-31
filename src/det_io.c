#include "det.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DET_MANIFEST_LINE 8192
#define DET_MANIFEST_PATH 4096

struct det_manifest_dataset {
    FILE *manifest;
    char base_dir[DET_MANIFEST_PATH];
    int width;
    int height;
    int channels;
    int max_boxes;
    size_t sample_count;
    size_t line_number;
    float *pixels;
    det_box *boxes;
    unsigned char *raw;
    size_t raw_capacity;
    det_status status;
};

static char *skip_space(char *text) {
    while (text != NULL && *text != '\0' && isspace((unsigned char)*text)) ++text;
    return text;
}

static char *trim_line(char *text) {
    text = skip_space(text);
    if (text == NULL) return NULL;
    char *end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return text;
}

static int read_pnm_token(FILE *file, char *token, size_t capacity) {
    if (file == NULL || token == NULL || capacity < 2U) return 0;
    int ch;
    for (;;) {
        ch = fgetc(file);
        if (ch == EOF) return 0;
        if (isspace((unsigned char)ch)) continue;
        if (ch == '#') {
            do ch = fgetc(file); while (ch != EOF && ch != '\n');
            continue;
        }
        break;
    }
    size_t length = 0U;
    do {
        if (isspace((unsigned char)ch) || ch == '#') {
            if (ch == '\r') {
                int next = fgetc(file);
                if (next != '\n' && next != EOF) ungetc(next, file);
            } else if (ch == '#') {
                ungetc(ch, file);
            }
            break;
        }
        if (length + 1U >= capacity) return 0;
        token[length++] = (char)ch;
        ch = fgetc(file);
    } while (ch != EOF);
    token[length] = '\0';
    return length > 0U;
}

static int parse_positive_token(const char *token, int *out) {
    char *end = NULL;
    long value;
    if (token == NULL || out == NULL) return 0;
    errno = 0;
    value = strtol(token, &end, 10);
    if (errno != 0 || end == token || *end != '\0' || value <= 0L || value > 16384L) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_nonnegative_token(const char *token, int *out) {
    char *end = NULL;
    long value;
    if (token == NULL || out == NULL) return 0;
    errno = 0;
    value = strtol(token, &end, 10);
    if (errno != 0 || end == token || *end != '\0' || value < 0L || value > 255L) return 0;
    *out = (int)value;
    return 1;
}

static int parse_box_token(const char *token, det_box *box) {
    float values[4];
    char *cursor = (char *)token;
    char *end = NULL;
    long class_value;
    if (token == NULL || box == NULL) return 0;
    for (int i = 0; i < 4; ++i) {
        errno = 0;
        values[i] = strtof(cursor, &end);
        if (errno != 0 || end == cursor || *end != ',') return 0;
        cursor = end + 1;
    }
    errno = 0;
    class_value = strtol(cursor, &end, 10);
    if (errno != 0 || end == cursor || *end != '\0' || class_value < 0L ||
        class_value > (long)INT_MAX || !isfinite(values[0]) || !isfinite(values[1]) ||
        !isfinite(values[2]) || !isfinite(values[3]) ||
        values[0] < 0.0f || values[1] < 0.0f || values[2] <= values[0] ||
        values[3] <= values[1]) {
        return 0;
    }
    *box = (det_box){values[0], values[1], values[2], values[3], (int)class_value};
    return 1;
}

static int is_absolute_path(const char *path) {
    return path != NULL && (path[0] == '/' || path[0] == '\\' ||
                            (isalpha((unsigned char)path[0]) && path[1] == ':'));
}

static int make_image_path(const det_manifest_dataset *dataset, const char *path,
                           char output[DET_MANIFEST_PATH]) {
    int written;
    if (dataset == NULL || path == NULL || output == NULL) return 0;
    if (is_absolute_path(path)) {
        written = snprintf(output, DET_MANIFEST_PATH, "%s", path);
    } else {
        written = snprintf(output, DET_MANIFEST_PATH, "%s/%s", dataset->base_dir, path);
    }
    return written >= 0 && written < DET_MANIFEST_PATH;
}

static int ensure_raw_capacity(det_manifest_dataset *dataset, size_t bytes) {
    if (bytes <= dataset->raw_capacity) return 1;
    unsigned char *raw = (unsigned char *)realloc(dataset->raw, bytes);
    if (raw == NULL) return 0;
    dataset->raw = raw;
    dataset->raw_capacity = bytes;
    return 1;
}

static int decode_pnm(det_manifest_dataset *dataset, const char *path,
                      int *source_width, int *source_height) {
    FILE *file = NULL;
    char token[64];
    int source_channels;
    int ascii_samples;
    int width, height, max_value;
    size_t pixels, bytes;
    if (dataset == NULL || path == NULL || source_width == NULL || source_height == NULL) return 0;
    file = fopen(path, "rb");
    if (file == NULL) return 0;
    if (!read_pnm_token(file, token, sizeof(token)) ||
        (strcmp(token, "P2") != 0 && strcmp(token, "P3") != 0 &&
         strcmp(token, "P5") != 0 && strcmp(token, "P6") != 0)) goto fail;
    ascii_samples = token[1] == '2' || token[1] == '3';
    source_channels = token[1] == '3' || token[1] == '6' ? 3 : 1;
    if (!read_pnm_token(file, token, sizeof(token)) || !parse_positive_token(token, &width) ||
        !read_pnm_token(file, token, sizeof(token)) || !parse_positive_token(token, &height) ||
        !read_pnm_token(file, token, sizeof(token)) || !parse_positive_token(token, &max_value) ||
        max_value > 255) goto fail;
    if ((size_t)width > SIZE_MAX / (size_t)height) goto fail;
    pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / (size_t)source_channels) goto fail;
    bytes = pixels * (size_t)source_channels;
    if (!ensure_raw_capacity(dataset, bytes)) goto fail;
    if (ascii_samples) {
        for (size_t i = 0U; i < bytes; ++i) {
            int value;
            if (!read_pnm_token(file, token, sizeof(token)) ||
                !parse_nonnegative_token(token, &value) || value > max_value) goto fail;
            dataset->raw[i] = (unsigned char)value;
        }
    } else if (fread(dataset->raw, 1U, bytes, file) != bytes) {
        goto fail;
    } else {
        for (size_t i = 0U; i < bytes; ++i) {
            if ((int)dataset->raw[i] > max_value) goto fail;
        }
    }
    if (fclose(file) != 0) return 0;
    for (int y = 0; y < dataset->height; ++y) {
        int source_y = y * height / dataset->height;
        for (int x = 0; x < dataset->width; ++x) {
            int source_x = x * width / dataset->width;
            size_t source_index = ((size_t)source_y * (size_t)width + (size_t)source_x) *
                                   (size_t)source_channels;
            for (int c = 0; c < dataset->channels; ++c) {
                float value;
                if (dataset->channels == 1 && source_channels == 3) {
                    value = (0.299f * (float)dataset->raw[source_index] +
                             0.587f * (float)dataset->raw[source_index + 1U] +
                             0.114f * (float)dataset->raw[source_index + 2U]) /
                            (float)max_value;
                } else {
                    int source_channel = source_channels == 1 ? 0 : c;
                    value = (float)dataset->raw[source_index + (size_t)source_channel] /
                            (float)max_value;
                }
                dataset->pixels[((size_t)c * (size_t)dataset->height + (size_t)y) *
                                (size_t)dataset->width + (size_t)x] = value;
            }
        }
    }
    *source_width = width;
    *source_height = height;
    return 1;
fail:
    if (file != NULL) fclose(file);
    return 0;
}

static int parse_manifest_line(det_manifest_dataset *dataset, char *line,
                               char *image_path, size_t path_capacity) {
    char *cursor = trim_line(line);
    char *boxes;
    if (cursor == NULL || *cursor == '\0' || *cursor == '#') return 0;
    boxes = cursor;
    while (*boxes != '\0' && !isspace((unsigned char)*boxes)) ++boxes;
    if (*boxes != '\0') {
        *boxes++ = '\0';
        boxes = skip_space(boxes);
    }
    if (snprintf(image_path, path_capacity, "%s", cursor) >= (int)path_capacity) return -1;
    int count = 0;
    while (boxes != NULL && *boxes != '\0') {
        char *token = boxes;
        while (*boxes != '\0' && !isspace((unsigned char)*boxes)) ++boxes;
        if (*boxes != '\0') *boxes++ = '\0';
        boxes = skip_space(boxes);
        if (count >= dataset->max_boxes || !parse_box_token(token, &dataset->boxes[count])) return -1;
        ++count;
    }
    return count + 1;
}

static int manifest_next(void *user, det_sample *sample) {
    det_manifest_dataset *dataset = (det_manifest_dataset *)user;
    char line[DET_MANIFEST_LINE];
    char relative_path[DET_MANIFEST_PATH];
    char image_path[DET_MANIFEST_PATH];
    if (dataset == NULL || sample == NULL || dataset->manifest == NULL) return -1;
    while (fgets(line, sizeof(line), dataset->manifest) != NULL) {
        ++dataset->line_number;
        if (strchr(line, '\n') == NULL && !feof(dataset->manifest)) {
            int ch;
            do ch = fgetc(dataset->manifest); while (ch != EOF && ch != '\n');
            dataset->status = DET_ERR_FORMAT;
            return -1;
        }
        int parsed = parse_manifest_line(dataset, line, relative_path, sizeof(relative_path));
        if (parsed == 0) continue;
        if (parsed < 0 || !make_image_path(dataset, relative_path, image_path)) {
            dataset->status = DET_ERR_FORMAT;
            return -1;
        }
        int source_width = 0;
        int source_height = 0;
        if (!decode_pnm(dataset, image_path, &source_width, &source_height)) {
            dataset->status = DET_ERR_IO;
            return -1;
        }
        float x_scale = (float)dataset->width / (float)source_width;
        float y_scale = (float)dataset->height / (float)source_height;
        for (int i = 0; i < parsed - 1; ++i) {
            dataset->boxes[i].x1 *= x_scale;
            dataset->boxes[i].x2 *= x_scale;
            dataset->boxes[i].y1 *= y_scale;
            dataset->boxes[i].y2 *= y_scale;
        }
        sample->image = (det_image){dataset->pixels, dataset->channels,
                                    dataset->height, dataset->width};
        sample->boxes = dataset->boxes;
        sample->box_count = parsed - 1;
        return 1;
    }
    if (ferror(dataset->manifest)) {
        dataset->status = DET_ERR_IO;
        return -1;
    }
    return 0;
}

static void manifest_reset(void *user) {
    det_manifest_dataset *dataset = (det_manifest_dataset *)user;
    if (dataset == NULL || dataset->manifest == NULL) return;
    rewind(dataset->manifest);
    clearerr(dataset->manifest);
    dataset->line_number = 0U;
    dataset->status = DET_OK;
}

det_status det_manifest_open(const char *manifest_path, int width, int height,
                             int channels, int max_boxes,
                             det_manifest_dataset **out) {
    det_manifest_dataset *dataset;
    char line[DET_MANIFEST_LINE];
    if (manifest_path == NULL || out == NULL || width <= 0 || height <= 0 ||
        (channels != 1 && channels != 3) || max_boxes <= 0) return DET_ERR_ARGUMENT;
    if ((size_t)width > SIZE_MAX / (size_t)height ||
        (size_t)width * (size_t)height > SIZE_MAX / (size_t)channels ||
        (size_t)width * (size_t)height * (size_t)channels >
            SIZE_MAX / sizeof(float) ||
        (size_t)max_boxes > SIZE_MAX / sizeof(det_box)) {
        return DET_ERR_ARGUMENT;
    }
    *out = NULL;
    dataset = (det_manifest_dataset *)calloc(1U, sizeof(*dataset));
    if (dataset == NULL) return DET_ERR_MEMORY;
    dataset->manifest = fopen(manifest_path, "rb");
    if (dataset->manifest == NULL) {
        free(dataset);
        return DET_ERR_IO;
    }
    const char *last_slash = strrchr(manifest_path, '/');
    const char *last_backslash = strrchr(manifest_path, '\\');
    const char *separator = last_slash == NULL ? last_backslash :
                            (last_backslash == NULL ? last_slash :
                             (last_slash > last_backslash ? last_slash : last_backslash));
    int base_length = separator == NULL ? 1 : (int)(separator - manifest_path);
    if (separator == NULL) {
        strcpy(dataset->base_dir, ".");
    } else if (separator == manifest_path) {
        dataset->base_dir[0] = *separator;
        dataset->base_dir[1] = '\0';
    } else if (base_length >= DET_MANIFEST_PATH) {
        det_manifest_close(dataset);
        return DET_ERR_ARGUMENT;
    } else {
        memcpy(dataset->base_dir, manifest_path, (size_t)base_length);
        dataset->base_dir[base_length] = '\0';
    }
    dataset->width = width;
    dataset->height = height;
    dataset->channels = channels;
    dataset->max_boxes = max_boxes;
    dataset->pixels = (float *)malloc((size_t)width * (size_t)height *
                                      (size_t)channels * sizeof(*dataset->pixels));
    dataset->boxes = (det_box *)malloc((size_t)max_boxes * sizeof(*dataset->boxes));
    if (dataset->pixels == NULL || dataset->boxes == NULL) {
        det_manifest_close(dataset);
        return DET_ERR_MEMORY;
    }
    while (fgets(line, sizeof(line), dataset->manifest) != NULL) {
        int has_newline = strchr(line, '\n') != NULL;
        char *text = trim_line(line);
        if (!has_newline && !feof(dataset->manifest)) {
            det_manifest_close(dataset);
            return DET_ERR_FORMAT;
        }
        if (text != NULL && *text != '\0' && *text != '#') ++dataset->sample_count;
    }
    rewind(dataset->manifest);
    clearerr(dataset->manifest);
    dataset->status = DET_OK;
    *out = dataset;
    return DET_OK;
}

det_status det_manifest_dataset_view(det_manifest_dataset *dataset, det_dataset *out) {
    if (dataset == NULL || out == NULL || dataset->manifest == NULL) return DET_ERR_ARGUMENT;
    manifest_reset(dataset);
    *out = (det_dataset){dataset, manifest_next, manifest_reset, dataset->sample_count};
    return DET_OK;
}

det_status det_manifest_status(const det_manifest_dataset *dataset) {
    return dataset == NULL ? DET_ERR_ARGUMENT : dataset->status;
}

void det_manifest_close(det_manifest_dataset *dataset) {
    if (dataset == NULL) return;
    if (dataset->manifest != NULL) fclose(dataset->manifest);
    free(dataset->pixels);
    free(dataset->boxes);
    free(dataset->raw);
    free(dataset);
}
