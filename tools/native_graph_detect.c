#include "native_graph_loader.h"
#include "native_graph_post.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_token(FILE *file, char *token, size_t capacity) {
    int ch;
    size_t used = 0U;
    if (file == NULL || token == NULL || capacity < 2U) return 0;
    do {
        ch = fgetc(file);
    } while (ch != EOF && ch <= ' ');
    if (ch == '#') {
        do {
            ch = fgetc(file);
        } while (ch != EOF && ch != '\n');
        return read_token(file, token, capacity);
    }
    while (ch != EOF && ch > ' ') {
        if (used + 1U >= capacity) return 0;
        token[used++] = (char)ch;
        ch = fgetc(file);
    }
    token[used] = '\0';
    return used > 0U;
}

static int read_pgm(const char *path, uint8_t **pixels, int32_t *width, int32_t *height) {
    FILE *file;
    char token[32];
    long pixel_count;
    uint8_t *data;
    if (path == NULL || pixels == NULL || width == NULL || height == NULL) return 0;
    file = fopen(path, "rb");
    if (file == NULL || !read_token(file, token, sizeof(token)) || strcmp(token, "P5") != 0 ||
        !read_token(file, token, sizeof(token))) {
        if (file != NULL) fclose(file);
        return 0;
    }
    *width = (int32_t)strtol(token, NULL, 10);
    if (!read_token(file, token, sizeof(token))) {
        fclose(file);
        return 0;
    }
    *height = (int32_t)strtol(token, NULL, 10);
    if (!read_token(file, token, sizeof(token)) || strtol(token, NULL, 10) != 255L ||
        *width <= 0 || *height <= 0) {
        fclose(file);
        return 0;
    }
    pixel_count = (long)*width * (long)*height;
    if (pixel_count <= 0L) {
        fclose(file);
        return 0;
    }
    data = (uint8_t *)malloc((size_t)pixel_count);
    if (data == NULL || fread(data, 1U, (size_t)pixel_count, file) != (size_t)pixel_count) {
        free(data);
        fclose(file);
        return 0;
    }
    fclose(file);
    *pixels = data;
    return 1;
}

int main(int argc, char **argv) {
    NGLoadedWeights weights;
    NGModelWeights view;
    NGModelConfig config;
    NGModelOutput raw;
    Y8DetectPostConfig post;
    Y8DetectResult result;
    Y8LetterboxInfo letterbox;
    NGTensor input;
    uint8_t *source = NULL;
    int8_t *input_data = NULL;
    int8_t *arena_data = NULL;
    int32_t width;
    int32_t height;
    int32_t imgsz = 160;
    int status = EXIT_FAILURE;
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s EXPORT_DIR IMAGE.pgm [imgsz]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4) imgsz = (int32_t)strtol(argv[3], NULL, 10);
    if (imgsz <= 0 || imgsz % 32 != 0 || !read_pgm(argv[2], &source, &width, &height) ||
        ng_weights_load_dir(argv[1], &weights) != 0) {
        fprintf(stderr, "input or weight loading failed\n");
        free(source);
        return EXIT_FAILURE;
    }
    config.scale = weights.scale;
    config.input_channels = weights.input_channels;
    config.class_count = weights.class_count;
    view.convs = weights.convs;
    view.conv_count = weights.conv_count;
    input_data = (int8_t *)calloc((size_t)config.input_channels * (size_t)imgsz *
                                  (size_t)imgsz, sizeof(*input_data));
    arena_data = (int8_t *)malloc(64U * 1024U * 1024U);
    if (input_data == NULL || arena_data == NULL ||
        ng_letterbox_u8_to_s8_scaled(source, width, height, input_data, imgsz,
                                     config.input_channels, 114U, weights.act_scale,
                                     &letterbox) != 0) {
        fprintf(stderr, "input preparation failed\n");
        goto done;
    }
    input = (NGTensor){input_data, 1, config.input_channels, imgsz, imgsz};
    {
        NGArena arena;
        ng_arena_init(&arena, arena_data, 2 * 1024 * 1024);
        if (ng_native_graph_forward_s8(&config, &view, &input, &arena, &raw) != 0) {
            fprintf(stderr, "forward failed\n");
            goto done;
        }
    }
    post = (Y8DetectPostConfig){0.001f, 0.45f, weights.act_scale,
                                NG_MAX_CANDIDATES, NG_MAX_KEEP,
                                &weights.dfl};
    if (ng_detect_post_s8(&raw, config.class_count, &post, &result) != 0 ||
        ng_map_boxes_to_original(result.dets, result.count, &letterbox) != 0) {
        fprintf(stderr, "postprocess failed\n");
        goto done;
    }
    printf("count=%d width=%d height=%d\n", result.count, width, height);
    for (int32_t i = 0; i < result.count; ++i) {
        printf("det=%.6f %.6f %.6f %.6f %.6f %d\n",
               (double)result.dets[i].x1_q12 / 4096.0,
               (double)result.dets[i].y1_q12 / 4096.0,
               (double)result.dets[i].x2_q12 / 4096.0,
               (double)result.dets[i].y2_q12 / 4096.0,
               (double)result.dets[i].score_q15 / 32767.0,
               (int)result.dets[i].class_id);
    }
    status = EXIT_SUCCESS;
done:
    free(arena_data);
    free(input_data);
    free(source);
    ng_weights_free(&weights);
    return status;
}
