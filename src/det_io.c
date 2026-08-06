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
#define DET_MANIFEST_MAX_RAW_BYTES (64U << 20)

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
    float *cache_pixels;
    det_box *cache_boxes;
    int *cache_box_counts;
    int cache_enabled;
    long *line_offsets;
    size_t *sample_order;
    size_t sample_cursor;
    size_t sample_epoch;
    uint32_t shuffle_seed;
    int shuffle_enabled;
};

static uint32_t cache_random_next(uint32_t *state) {
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

static void manifest_shuffle_order(det_manifest_dataset *dataset) {
    uint32_t state;
    if (dataset == NULL || dataset->sample_order == NULL || dataset->sample_count == 0U) return;
    state = dataset->shuffle_seed ^ ((uint32_t)dataset->sample_epoch * 0x9e3779b9U);
    for (size_t i = 0U; i < dataset->sample_count; ++i) dataset->sample_order[i] = i;
    for (size_t i = dataset->sample_count - 1U; i > 0U; --i) {
        size_t swap_index = (size_t)(cache_random_next(&state) % (uint32_t)(i + 1U));
        size_t temporary = dataset->sample_order[i];
        dataset->sample_order[i] = dataset->sample_order[swap_index];
        dataset->sample_order[swap_index] = temporary;
    }
    if (dataset->sample_epoch < SIZE_MAX) ++dataset->sample_epoch;
}

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
    float target_weight = 0.0f;
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
    if (errno != 0 || end == cursor || (end[0] != '\0' && end[0] != ',') ||
        class_value < 0L ||
        class_value > (long)INT_MAX || !isfinite(values[0]) || !isfinite(values[1]) ||
        !isfinite(values[2]) || !isfinite(values[3]) ||
        values[0] < 0.0f || values[1] < 0.0f || values[2] <= values[0] ||
        values[3] <= values[1]) {
        return 0;
    }
    if (*end == ',') {
        char *weight_end = NULL;
        errno = 0;
        target_weight = strtof(end + 1, &weight_end);
        if (errno != 0 || weight_end == end + 1 || *weight_end != '\0' ||
            !isfinite(target_weight) || target_weight <= 0.0f || target_weight > 1.0f) {
            return 0;
        }
    }
    *box = (det_box){values[0], values[1], values[2], values[3],
                     (int)class_value, target_weight};
    return 1;
}

static int is_absolute_path(const char *path) {
    if (path == NULL) return 0;
#if defined(_WIN32)
    return path[0] == '/' || path[0] == '\\' ||
           (isalpha((unsigned char)path[0]) && path[1] == ':' &&
            (path[2] == '/' || path[2] == '\\'));
#else
    return path[0] == '/';
#endif
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

static det_status decode_pnm(det_manifest_dataset *dataset, const char *path,
                             int *source_width, int *source_height) {
    FILE *file = NULL;
    char token[64];
    int source_channels;
    int ascii_samples;
    int width, height, max_value;
    size_t pixels, bytes;
    if (dataset == NULL || path == NULL || source_width == NULL || source_height == NULL) {
        return DET_ERR_ARGUMENT;
    }
    file = fopen(path, "rb");
    if (file == NULL) return DET_ERR_IO;
    det_status failure = DET_ERR_FORMAT;
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
    if (bytes > DET_MANIFEST_MAX_RAW_BYTES) goto fail;
    if (!ensure_raw_capacity(dataset, bytes)) {
        failure = DET_ERR_MEMORY;
        goto fail;
    }
    if (ascii_samples) {
        for (size_t i = 0U; i < bytes; ++i) {
            int value;
            if (!read_pnm_token(file, token, sizeof(token)) ||
                !parse_nonnegative_token(token, &value) || value > max_value) goto fail;
            dataset->raw[i] = (unsigned char)value;
        }
    } else if (fread(dataset->raw, 1U, bytes, file) != bytes) {
        failure = DET_ERR_IO;
        goto fail;
    } else {
        for (size_t i = 0U; i < bytes; ++i) {
            if ((int)dataset->raw[i] > max_value) goto fail;
        }
    }
    if (fclose(file) != 0) return DET_ERR_IO;
    for (int y = 0; y < dataset->height; ++y) {
        float source_y = ((float)y + 0.5f) * (float)height /
                         (float)dataset->height - 0.5f;
        int y0 = (int)floorf(source_y);
        float y_weight;
        if (y0 < 0) y0 = 0;
        if (y0 >= height) y0 = height - 1;
        y_weight = source_y - (float)y0;
        if (y_weight < 0.0f) y_weight = 0.0f;
        if (y_weight > 1.0f) y_weight = 1.0f;
        int y1 = y0 + 1 < height ? y0 + 1 : y0;
        for (int x = 0; x < dataset->width; ++x) {
            float source_x = ((float)x + 0.5f) * (float)width /
                             (float)dataset->width - 0.5f;
            int x0 = (int)floorf(source_x);
            float x_weight;
            if (x0 < 0) x0 = 0;
            if (x0 >= width) x0 = width - 1;
            x_weight = source_x - (float)x0;
            if (x_weight < 0.0f) x_weight = 0.0f;
            if (x_weight > 1.0f) x_weight = 1.0f;
            int x1 = x0 + 1 < width ? x0 + 1 : x0;
            for (int c = 0; c < dataset->channels; ++c) {
                int source_channel = source_channels == 1 ? 0 : c;
                size_t index00 = ((size_t)y0 * (size_t)width + (size_t)x0) *
                                 (size_t)source_channels + (size_t)source_channel;
                size_t index01 = ((size_t)y0 * (size_t)width + (size_t)x1) *
                                 (size_t)source_channels + (size_t)source_channel;
                size_t index10 = ((size_t)y1 * (size_t)width + (size_t)x0) *
                                 (size_t)source_channels + (size_t)source_channel;
                size_t index11 = ((size_t)y1 * (size_t)width + (size_t)x1) *
                                 (size_t)source_channels + (size_t)source_channel;
                float top = (float)dataset->raw[index00] * (1.0f - x_weight) +
                            (float)dataset->raw[index01] * x_weight;
                float bottom = (float)dataset->raw[index10] * (1.0f - x_weight) +
                               (float)dataset->raw[index11] * x_weight;
                float value = (top * (1.0f - y_weight) + bottom * y_weight) /
                              (float)max_value;
                if (dataset->channels == 1 && source_channels == 3) {
                    float red_top = (float)dataset->raw[index00 - (size_t)source_channel] *
                                    (1.0f - x_weight) +
                                    (float)dataset->raw[index01 - (size_t)source_channel] * x_weight;
                    float red_bottom = (float)dataset->raw[index10 - (size_t)source_channel] *
                                       (1.0f - x_weight) +
                                       (float)dataset->raw[index11 - (size_t)source_channel] * x_weight;
                    float green_top = (float)dataset->raw[index00 + 1U] *
                                      (1.0f - x_weight) +
                                      (float)dataset->raw[index01 + 1U] * x_weight;
                    float green_bottom = (float)dataset->raw[index10 + 1U] *
                                         (1.0f - x_weight) +
                                         (float)dataset->raw[index11 + 1U] * x_weight;
                    float blue_top = (float)dataset->raw[index00 + 2U] *
                                     (1.0f - x_weight) +
                                     (float)dataset->raw[index01 + 2U] * x_weight;
                    float blue_bottom = (float)dataset->raw[index10 + 2U] *
                                        (1.0f - x_weight) +
                                        (float)dataset->raw[index11 + 2U] * x_weight;
                    value = (0.299f * (red_top * (1.0f - y_weight) + red_bottom * y_weight) +
                             0.587f * (green_top * (1.0f - y_weight) + green_bottom * y_weight) +
                             0.114f * (blue_top * (1.0f - y_weight) + blue_bottom * y_weight)) /
                            (float)max_value;
                }
                dataset->pixels[((size_t)c * (size_t)dataset->height + (size_t)y) *
                                (size_t)dataset->width + (size_t)x] = value;
            }
        }
    }
    *source_width = width;
    *source_height = height;
    return DET_OK;
fail:
    if (file != NULL) fclose(file);
    return failure;
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
    size_t sample_index;
    if (dataset == NULL || sample == NULL || dataset->manifest == NULL ||
        dataset->sample_order == NULL || dataset->line_offsets == NULL) return -1;
    if (dataset->sample_cursor >= dataset->sample_count) return 0;
    sample_index = dataset->sample_order[dataset->sample_cursor];
    if (dataset->cache_enabled) {
        size_t plane = (size_t)dataset->channels * (size_t)dataset->height *
                       (size_t)dataset->width;
        memcpy(dataset->pixels, dataset->cache_pixels + sample_index * plane,
               plane * sizeof(*dataset->pixels));
        if (dataset->cache_box_counts[sample_index] > 0) {
            memcpy(dataset->boxes,
                   dataset->cache_boxes + sample_index * (size_t)dataset->max_boxes,
                   (size_t)dataset->cache_box_counts[sample_index] *
                       sizeof(*dataset->boxes));
        }
        sample->image = (det_image){dataset->pixels, dataset->channels,
                                    dataset->height, dataset->width};
        sample->boxes = dataset->boxes;
        sample->box_count = dataset->cache_box_counts[sample_index];
        ++dataset->sample_cursor;
        return 1;
    }
    if (fseek(dataset->manifest, dataset->line_offsets[sample_index], SEEK_SET) != 0 ||
        fgets(line, sizeof(line), dataset->manifest) == NULL) {
        dataset->status = DET_ERR_IO;
        return -1;
    }
    dataset->line_number = sample_index + 1U;
    if (strchr(line, '\n') == NULL && !feof(dataset->manifest)) {
        dataset->status = DET_ERR_FORMAT;
        return -1;
    }
    {
        int parsed = parse_manifest_line(dataset, line, relative_path, sizeof(relative_path));
        if (parsed == 0) {
            dataset->status = DET_ERR_FORMAT;
            return -1;
        }
        if (parsed < 0 || !make_image_path(dataset, relative_path, image_path)) {
            dataset->status = DET_ERR_FORMAT;
            return -1;
        }
        int source_width = 0;
        int source_height = 0;
        det_status decode_status = decode_pnm(dataset, image_path, &source_width, &source_height);
        if (decode_status != DET_OK) {
            dataset->status = decode_status;
            return -1;
        }
        for (int i = 0; i < parsed - 1; ++i) {
            if (dataset->boxes[i].x2 > (float)source_width ||
                dataset->boxes[i].y2 > (float)source_height) {
                dataset->status = DET_ERR_FORMAT;
                return -1;
            }
        }
        float x_scale = (float)dataset->width / (float)source_width;
        float y_scale = (float)dataset->height / (float)source_height;
        for (int i = 0; i < parsed - 1; ++i) {
            dataset->boxes[i].x1 *= x_scale;
            dataset->boxes[i].x2 *= x_scale;
            dataset->boxes[i].y1 *= y_scale;
            dataset->boxes[i].y2 *= y_scale;
            dataset->boxes[i].x1 = fmaxf(0.0f, fminf((float)dataset->width,
                                                     dataset->boxes[i].x1));
            dataset->boxes[i].x2 = fmaxf(0.0f, fminf((float)dataset->width,
                                                     dataset->boxes[i].x2));
            dataset->boxes[i].y1 = fmaxf(0.0f, fminf((float)dataset->height,
                                                     dataset->boxes[i].y1));
            dataset->boxes[i].y2 = fmaxf(0.0f, fminf((float)dataset->height,
                                                     dataset->boxes[i].y2));
        }
        sample->image = (det_image){dataset->pixels, dataset->channels,
                                    dataset->height, dataset->width};
        sample->boxes = dataset->boxes;
        sample->box_count = parsed - 1;
        ++dataset->sample_cursor;
        return 1;
    }
}

static void manifest_reset(void *user) {
    det_manifest_dataset *dataset = (det_manifest_dataset *)user;
    if (dataset == NULL || dataset->manifest == NULL) return;
    dataset->sample_cursor = 0U;
    if (dataset->shuffle_enabled) manifest_shuffle_order(dataset);
    else if (dataset->sample_order != NULL) {
        for (size_t i = 0U; i < dataset->sample_count; ++i) dataset->sample_order[i] = i;
    }
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
        width > 4096 || height > 4096 ||
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
#if defined(_WIN32)
    const char *last_backslash = strrchr(manifest_path, '\\');
#else
    const char *last_backslash = NULL;
#endif
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
    if (ferror(dataset->manifest)) {
        det_manifest_close(dataset);
        return DET_ERR_IO;
    }
    if (dataset->sample_count == 0U ||
        dataset->sample_count > SIZE_MAX / sizeof(*dataset->line_offsets) ||
        dataset->sample_count > SIZE_MAX / sizeof(*dataset->sample_order)) {
        det_status failure = dataset->sample_count == 0U ? DET_ERR_ARGUMENT : DET_ERR_MEMORY;
        det_manifest_close(dataset);
        return failure;
    }
    dataset->line_offsets = (long *)malloc(dataset->sample_count *
                                           sizeof(*dataset->line_offsets));
    dataset->sample_order = (size_t *)malloc(dataset->sample_count *
                                             sizeof(*dataset->sample_order));
    if (dataset->line_offsets == NULL || dataset->sample_order == NULL) {
        det_manifest_close(dataset);
        return DET_ERR_MEMORY;
    }
    rewind(dataset->manifest);
    clearerr(dataset->manifest);
    {
        size_t index = 0U;
        for (;;) {
            long offset = ftell(dataset->manifest);
            int has_newline;
            char *text;
            if (offset < 0L) {
                det_manifest_close(dataset);
                return DET_ERR_IO;
            }
            if (fgets(line, sizeof(line), dataset->manifest) == NULL) break;
            has_newline = strchr(line, '\n') != NULL;
            text = trim_line(line);
            if (!has_newline && !feof(dataset->manifest)) {
                det_manifest_close(dataset);
                return DET_ERR_FORMAT;
            }
            if (text != NULL && *text != '\0' && *text != '#') {
                if (index >= dataset->sample_count) {
                    det_manifest_close(dataset);
                    return DET_ERR_FORMAT;
                }
                dataset->line_offsets[index] = offset;
                dataset->sample_order[index] = index;
                ++index;
            }
        }
        if (ferror(dataset->manifest) || index != dataset->sample_count) {
            det_status failure = ferror(dataset->manifest) ? DET_ERR_IO : DET_ERR_FORMAT;
            det_manifest_close(dataset);
            return failure;
        }
    }
    dataset->sample_cursor = 0U;
    dataset->sample_epoch = 0U;
    dataset->shuffle_seed = 1U;
    dataset->shuffle_enabled = 0;
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

det_status det_manifest_enable_cache(det_manifest_dataset *dataset) {
    size_t plane;
    size_t pixel_count;
    float *cache_pixels;
    det_box *cache_boxes;
    int *cache_box_counts;
    int shuffle_enabled;
    uint32_t shuffle_seed;
    size_t sample_epoch;
    if (dataset == NULL || dataset->manifest == NULL) return DET_ERR_ARGUMENT;
    if (dataset->cache_enabled) return DET_OK;
    if ((size_t)dataset->channels > SIZE_MAX / (size_t)dataset->height ||
        (size_t)dataset->channels * (size_t)dataset->height >
            SIZE_MAX / (size_t)dataset->width) return DET_ERR_SHAPE;
    plane = (size_t)dataset->channels * (size_t)dataset->height *
            (size_t)dataset->width;
    if (dataset->sample_count > SIZE_MAX / plane ||
        dataset->sample_count * plane > SIZE_MAX / sizeof(*cache_pixels) ||
        dataset->sample_count > SIZE_MAX / (size_t)dataset->max_boxes ||
        dataset->sample_count * (size_t)dataset->max_boxes >
            SIZE_MAX / sizeof(*cache_boxes)) return DET_ERR_MEMORY;
    pixel_count = dataset->sample_count * plane;
    cache_pixels = (float *)malloc(pixel_count * sizeof(*cache_pixels));
    cache_boxes = (det_box *)malloc(dataset->sample_count *
                                    (size_t)dataset->max_boxes * sizeof(*cache_boxes));
    cache_box_counts = (int *)malloc(dataset->sample_count * sizeof(*cache_box_counts));
    if (cache_pixels == NULL || cache_boxes == NULL || cache_box_counts == NULL) {
        free(cache_pixels);
        free(cache_boxes);
        free(cache_box_counts);
        return DET_ERR_MEMORY;
    }
    dataset->cache_enabled = 0;
    shuffle_enabled = dataset->shuffle_enabled;
    shuffle_seed = dataset->shuffle_seed;
    sample_epoch = dataset->sample_epoch;
    dataset->shuffle_enabled = 0;
    manifest_reset(dataset);
    for (size_t index = 0U; index < dataset->sample_count; ++index) {
        det_sample sample;
        int next_status = manifest_next(dataset, &sample);
        if (next_status != 1 || sample.box_count < 0 ||
            sample.box_count > dataset->max_boxes) {
            free(cache_pixels);
            free(cache_boxes);
            free(cache_box_counts);
            dataset->status = next_status < 0 ? dataset->status : DET_ERR_FORMAT;
            dataset->shuffle_enabled = shuffle_enabled;
            dataset->shuffle_seed = shuffle_seed;
            dataset->sample_epoch = sample_epoch;
            manifest_reset(dataset);
            return dataset->status;
        }
        memcpy(cache_pixels + index * plane, sample.image.data,
               plane * sizeof(*cache_pixels));
        if (sample.box_count > 0) {
            memcpy(cache_boxes + index * (size_t)dataset->max_boxes, sample.boxes,
                   (size_t)sample.box_count * sizeof(*cache_boxes));
        }
        cache_box_counts[index] = sample.box_count;
    }
    dataset->cache_pixels = cache_pixels;
    dataset->cache_boxes = cache_boxes;
    dataset->cache_box_counts = cache_box_counts;
    dataset->cache_enabled = 1;
    dataset->shuffle_enabled = shuffle_enabled;
    dataset->shuffle_seed = shuffle_seed;
    dataset->sample_epoch = sample_epoch;
    manifest_reset(dataset);
    return DET_OK;
}

det_status det_manifest_set_shuffle(det_manifest_dataset *dataset, int enabled,
                                    int seed) {
    if (dataset == NULL || dataset->sample_order == NULL ||
        (enabled != 0 && enabled != 1)) {
        return DET_ERR_ARGUMENT;
    }
    if (enabled && seed <= 0) return DET_ERR_ARGUMENT;
    dataset->shuffle_enabled = enabled;
    dataset->shuffle_seed = (uint32_t)(enabled ? seed : 1);
    dataset->sample_epoch = 0U;
    manifest_reset(dataset);
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
    free(dataset->cache_pixels);
    free(dataset->cache_boxes);
    free(dataset->cache_box_counts);
    free(dataset->line_offsets);
    free(dataset->sample_order);
    free(dataset);
}
