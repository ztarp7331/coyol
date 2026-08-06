#include "native_multiscale.h"
#include "native_graph_post.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int token(FILE *file, char *buffer, size_t capacity) {
    int ch;
    size_t used = 0U;
    do { ch = fgetc(file); } while (ch != EOF && ch <= ' ');
    if (ch == '#') {
        do { ch = fgetc(file); } while (ch != EOF && ch != '\n');
        return token(file, buffer, capacity);
    }
    while (ch != EOF && ch > ' ') {
        if (used + 1U >= capacity) return 0;
        buffer[used++] = (char)ch;
        ch = fgetc(file);
    }
    buffer[used] = '\0';
    return used > 0U;
}

static int read_pgm(const char *path, uint8_t **pixels, int32_t *width, int32_t *height) {
    FILE *file = fopen(path, "rb");
    char word[32];
    long bytes;
    uint8_t *data;
    if (!file || !pixels || !width || !height || !token(file, word, sizeof(word)) ||
        strcmp(word, "P5") != 0 || !token(file, word, sizeof(word))) goto fail;
    *width = (int32_t)strtol(word, NULL, 10);
    if (!token(file, word, sizeof(word))) goto fail;
    *height = (int32_t)strtol(word, NULL, 10);
    if (!token(file, word, sizeof(word)) || strtol(word, NULL, 10) != 255L ||
        *width <= 0 || *height <= 0) goto fail;
    bytes = (long)*width * (long)*height;
    if (bytes <= 0L) goto fail;
    data = (uint8_t *)malloc((size_t)bytes);
    if (!data || fread(data, 1U, (size_t)bytes, file) != (size_t)bytes) {
        free(data);
        goto fail;
    }
    fclose(file);
    *pixels = data;
    return 1;
fail:
    if (file) fclose(file);
    return 0;
}

int main(int argc, char **argv) {
    NGLoadedWeights weights;
    Y8LetterboxInfo letterbox;
    NGMDetection detections[300];
    uint8_t *source = NULL;
    int8_t *input = NULL;
    int8_t *arena_data = NULL;
    float *input_f32 = NULL;
    int32_t width;
    int32_t height;
    int32_t imgsz = 160;
    int32_t count = 0;
    int status = EXIT_FAILURE;
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s EXPORT_DIR IMAGE.pgm [imgsz] [f32]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4) imgsz = (int32_t)strtol(argv[3], NULL, 10);
    if (argc == 5 && strcmp(argv[4], "f32") != 0) return EXIT_FAILURE;
    if (imgsz <= 0 || imgsz % 32 != 0 || !read_pgm(argv[2], &source, &width, &height) ||
        ng_weights_load_dir_mode(argv[1], &weights, argc == 5) != 0) {
        fprintf(stderr, "input or weight loading failed\n");
        free(source);
        return EXIT_FAILURE;
    }
    input = (int8_t *)calloc((size_t)3 * (size_t)imgsz * (size_t)imgsz, sizeof(*input));
    arena_data = (int8_t *)malloc(64U * 1024U * 1024U);
    if (!input || !arena_data || ng_letterbox_u8_to_s8_scaled(
            source, width, height, input, imgsz, 3, 114U, weights.act_scale, &letterbox) != 0) {
        fprintf(stderr, "input preparation failed\n");
        goto done;
    }
    if (argc == 5) {
        input_f32 = (float *)malloc((size_t)3 * (size_t)imgsz * (size_t)imgsz * sizeof(float));
        if (!input_f32) goto done;
        if (ng_letterbox_u8_to_f32(source, width, height, input_f32, imgsz, 3,
                                   114U, &letterbox) != 0) goto done;
    }
    {
        NGArena arena;
        ng_arena_init(&arena, arena_data, 4 * 1024 * 1024);
        int result = argc == 5 ?
            ngm_detect_f32(&weights, input_f32, 3, imgsz, imgsz, 0.001f, 300,
                           &arena, detections, &count) :
            ngm_detect_s8(&weights, input, 3, imgsz, imgsz, 0.001f, 300,
                          &arena, detections, &count);
        if (result != 0) {
            fprintf(stderr, "forward failed\n");
            goto done;
        }
    }
    printf("count=%d width=%d height=%d\n", count, width, height);
    for (int32_t i = 0; i < count; ++i) {
        float x1 = (detections[i].x1 - (float)letterbox.pad_x) / letterbox.scale;
        float y1 = (detections[i].y1 - (float)letterbox.pad_y) / letterbox.scale;
        float x2 = (detections[i].x2 - (float)letterbox.pad_x) / letterbox.scale;
        float y2 = (detections[i].y2 - (float)letterbox.pad_y) / letterbox.scale;
        printf("det=%.6f %.6f %.6f %.6f %.6f %d\n",
               fmaxf(0.0f, fminf((float)width, x1)), fmaxf(0.0f, fminf((float)height, y1)),
               fmaxf(0.0f, fminf((float)width, x2)), fmaxf(0.0f, fminf((float)height, y2)),
               detections[i].score, detections[i].class_id);
    }
    status = EXIT_SUCCESS;
done:
    free(arena_data);
    free(input_f32);
    free(input);
    free(source);
    ng_weights_free(&weights);
    return status;
}
