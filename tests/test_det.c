#include "det.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Assertions in this executable are checks, not optional documentation: keep
 * their expressions live in Release so setup calls cannot disappear. */
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

typedef struct {
    det_sample sample;
    int emitted;
} one_sample_dataset;

typedef struct {
    det_sample sample;
    size_t remaining;
    size_t total;
} repeat_sample_dataset;

typedef struct {
    det_sample sample;
    int emitted;
} multi_box_dataset;

typedef struct {
    const det_sample *samples;
    int count;
    int index;
} stream_dataset;

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t classes;
    uint32_t max_detections;
    uint32_t graph_channels;
    uint32_t stage_count;
    uint32_t precision;
    uint64_t train_updates;
} test_file_header;

static uint32_t test_crc32(const unsigned char *data, size_t length) {
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0U; i < length; ++i) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1U);
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

static int rewrite_crc32(const char *path) {
    FILE *file = fopen(path, "r+b");
    if (file == NULL || fseek(file, 0L, SEEK_END) != 0) {
        if (file != NULL) fclose(file);
        return 0;
    }
    long length = ftell(file);
    if (length <= (long)sizeof(uint32_t)) {
        fclose(file);
        return 0;
    }
    size_t payload_length = (size_t)length - sizeof(uint32_t);
    unsigned char *payload = (unsigned char *)malloc(payload_length);
    if (payload == NULL || fseek(file, 0L, SEEK_SET) != 0 ||
        fread(payload, 1U, payload_length, file) != payload_length) {
        free(payload);
        fclose(file);
        return 0;
    }
    uint32_t checksum = test_crc32(payload, payload_length);
    free(payload);
    int ok = fseek(file, (long)payload_length, SEEK_SET) == 0 &&
             fwrite(&checksum, sizeof(checksum), 1U, file) == 1U;
    int close_result = fclose(file);
    return ok && close_result == 0;
}

static int files_equal(const char *left_path, const char *right_path) {
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    if (left == NULL || right == NULL) {
        if (left != NULL) fclose(left);
        if (right != NULL) fclose(right);
        return 0;
    }
    int equal = 1;
    for (;;) {
        unsigned char left_buffer[4096];
        unsigned char right_buffer[4096];
        size_t left_size = fread(left_buffer, 1U, sizeof(left_buffer), left);
        size_t right_size = fread(right_buffer, 1U, sizeof(right_buffer), right);
        if (left_size != right_size || memcmp(left_buffer, right_buffer, left_size) != 0) {
            equal = 0;
            break;
        }
        if (left_size == 0U) break;
    }
    if (fclose(left) != 0 || fclose(right) != 0) equal = 0;
    return equal;
}

static int file_regions_differ(const char *left_path, const char *right_path,
                               size_t offset, size_t length) {
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    unsigned char *left_data = NULL;
    unsigned char *right_data = NULL;
    int different = 0;
    if (left == NULL || right == NULL || length == 0U ||
        fseek(left, (long)offset, SEEK_SET) != 0 ||
        fseek(right, (long)offset, SEEK_SET) != 0) {
        if (left != NULL) fclose(left);
        if (right != NULL) fclose(right);
        return 0;
    }
    left_data = (unsigned char *)malloc(length);
    right_data = (unsigned char *)malloc(length);
    if (left_data != NULL && right_data != NULL &&
        fread(left_data, 1U, length, left) == length &&
        fread(right_data, 1U, length, right) == length) {
        different = memcmp(left_data, right_data, length) != 0;
    }
    free(left_data);
    free(right_data);
    fclose(left);
    fclose(right);
    return different;
}

static int write_truncated_copy(const char *source, const char *destination) {
    FILE *input = fopen(source, "rb");
    if (input == NULL || fseek(input, 0L, SEEK_END) != 0) {
        if (input != NULL) fclose(input);
        return 0;
    }
    long length = ftell(input);
    if (length <= (long)(2U * sizeof(uint32_t)) || fseek(input, 0L, SEEK_SET) != 0) {
        fclose(input);
        return 0;
    }
    size_t payload_length = (size_t)length - sizeof(uint32_t);
    unsigned char *payload = (unsigned char *)malloc(payload_length);
    if (payload == NULL || fread(payload, 1U, payload_length, input) != payload_length) {
        free(payload);
        fclose(input);
        return 0;
    }
    fclose(input);
    size_t shortened = payload_length - 1U;
    uint32_t checksum = test_crc32(payload, shortened);
    FILE *output = fopen(destination, "wb");
    int ok = output != NULL && fwrite(payload, 1U, shortened, output) == shortened &&
             fwrite(&checksum, sizeof(checksum), 1U, output) == 1U;
    if (output != NULL && fclose(output) != 0) ok = 0;
    free(payload);
    return ok;
}

static size_t test_stage_bytes_kernel(int input_channels, int output_channels, int depthwise,
                                      int kernel) {
    size_t weights = (size_t)(depthwise ? output_channels : output_channels * input_channels) *
                     (size_t)kernel * (size_t)kernel;
    size_t packed = (size_t)output_channels *
                    ((((size_t)(depthwise ? 1 : input_channels) * (size_t)kernel *
                       (size_t)kernel) + 1U) / 2U);
    return 10U * sizeof(int) +
           (2U * weights + 2U * (size_t)output_channels) * sizeof(float) +
           weights * sizeof(int8_t) + packed * sizeof(uint8_t) +
           (size_t)output_channels * sizeof(float);
}

static size_t test_stage_bytes(int input_channels, int output_channels, int depthwise) {
    return test_stage_bytes_kernel(input_channels, output_channels, depthwise, 3);
}

static size_t test_head_bytes(int classes) {
    size_t outputs = (size_t)4 + (size_t)classes;
    size_t weights = 4U * outputs;
    size_t packed = outputs * ((4U + 1U) / 2U);
    return 2U * sizeof(int) +
           (2U * weights + 2U * outputs) * sizeof(float) +
           weights * sizeof(int8_t) + packed * sizeof(uint8_t) +
           outputs * sizeof(float);
}

static size_t test_bottomup_weight_offset(int channels, int classes) {
    size_t offset = sizeof(test_file_header);
    int input_channels = channels;
    for (int stage = 0; stage < 5; ++stage) {
        int output_channels = stage == 0 ? 1 : 4;
        offset += test_stage_bytes(input_channels, output_channels, stage > 1);
        input_channels = output_channels;
    }
    offset += 6U * test_head_bytes(classes);
    offset += 3U * test_stage_bytes_kernel(4, 4, 0, 1);
    return offset + 10U * sizeof(int);
}

static int mutate_auxiliary_weight(const char *path, int width, int height,
                                   int channels, int classes) {
    size_t offset = sizeof(test_file_header);
    int input_channels = channels;
    int stage_height = height;
    int stage_width = width;
    for (int s = 0; s < 5; ++s) {
        int output_channels = s == 0 ? 1 : 4;
        int depthwise = s > 1;
        offset += test_stage_bytes(input_channels, output_channels, depthwise);
        input_channels = output_channels;
        stage_height = (stage_height + 1) / 2;
        stage_width = (stage_width + 1) / 2;
    }
    (void)stage_height;
    (void)stage_width;
    offset += 3U * test_head_bytes(classes);
    offset += 2U * sizeof(int);
    FILE *file = fopen(path, "r+b");
    if (file == NULL || fseek(file, (long)offset, SEEK_SET) != 0) {
        if (file != NULL) fclose(file);
        return 0;
    }
    float replacement = 0.1234567f;
    if (fseek(file, (long)offset, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    int ok = fwrite(&replacement, sizeof(replacement), 1U, file) == 1U &&
             fclose(file) == 0;
    return ok && rewrite_crc32(path);
}

static int next_one(void *user, det_sample *sample) {
    one_sample_dataset *dataset = (one_sample_dataset *)user;
    if (dataset->emitted != 0) return 0;
    *sample = dataset->sample;
    dataset->emitted = 1;
    return 1;
}

static void reset_one(void *user) {
    ((one_sample_dataset *)user)->emitted = 0;
}

static int next_repeat(void *user, det_sample *sample) {
    repeat_sample_dataset *dataset = (repeat_sample_dataset *)user;
    if (dataset->remaining == 0U) return 0;
    *sample = dataset->sample;
    --dataset->remaining;
    return 1;
}

static void reset_repeat(void *user) {
    repeat_sample_dataset *dataset = (repeat_sample_dataset *)user;
    dataset->remaining = dataset->total;
}

static int next_multi_box(void *user, det_sample *sample) {
    multi_box_dataset *dataset = (multi_box_dataset *)user;
    if (dataset->emitted != 0) return 0;
    *sample = dataset->sample;
    dataset->emitted = 1;
    return 1;
}

static void reset_multi_box(void *user) {
    ((multi_box_dataset *)user)->emitted = 0;
}

static int next_stream(void *user, det_sample *sample) {
    stream_dataset *dataset = (stream_dataset *)user;
    if (dataset->index >= dataset->count) return 0;
    *sample = dataset->samples[dataset->index++];
    return 1;
}

static void reset_stream(void *user) {
    ((stream_dataset *)user)->index = 0;
}

static int next_io_error(void *user, det_sample *sample) {
    (void)user;
    (void)sample;
    return -1;
}

static void test_manifest_adapter(void) {
    const char *image_path = "det_test_image.pgm";
    const char *manifest_path = "det_test_manifest.txt";
    const unsigned char pixels[] = {0U, 64U, 128U, 255U};
    FILE *image = fopen(image_path, "wb");
    assert(image != NULL);
    assert(fputs("P5\r\n2 2\r\n255\r\n", image) >= 0);
    assert(fwrite(pixels, 1U, sizeof(pixels), image) == sizeof(pixels));
    assert(fclose(image) == 0);
    FILE *manifest = fopen(manifest_path, "wb");
    assert(manifest != NULL);
    assert(fputs("det_test_image.pgm 0,0,2,2,2\n", manifest) >= 0);
    assert(fclose(manifest) == 0);

    det_manifest_dataset *raw = NULL;
    assert(det_manifest_open(manifest_path, 4, 4, 1, 4, &raw) == DET_OK);
    det_dataset dataset;
    assert(det_manifest_dataset_view(raw, &dataset) == DET_OK);
    assert(dataset.sample_count == 1U);
    det_sample sample;
    assert(dataset.next(dataset.user, &sample) == 1);
    assert(sample.image.width == 4 && sample.image.height == 4 && sample.image.channels == 1);
    assert(fabsf(sample.image.data[0] - 0.0f) < 1e-6f);
    assert(fabsf(sample.image.data[2] - 64.0f / 255.0f) < 1e-6f);
    assert(fabsf(sample.image.data[2 * 4 + 2] - 255.0f / 255.0f) < 1e-6f);
    assert(sample.box_count == 1);
    assert(fabsf(sample.boxes[0].x2 - 4.0f) < 1e-6f);
    assert(fabsf(sample.boxes[0].y2 - 4.0f) < 1e-6f);
    assert(sample.boxes[0].class_id == 2);
    assert(dataset.next(dataset.user, &sample) == 0);
    assert(det_manifest_status(raw) == DET_OK);
    det_manifest_close(raw);
    manifest = fopen(manifest_path, "wb");
    assert(manifest != NULL);
    assert(fputs("det_test_image.pgm 0,0,3,2,2\n", manifest) >= 0);
    assert(fclose(manifest) == 0);
    raw = NULL;
    assert(det_manifest_open(manifest_path, 4, 4, 1, 4, &raw) == DET_OK);
    assert(det_manifest_dataset_view(raw, &dataset) == DET_OK);
    assert(dataset.next(dataset.user, &sample) < 0);
    assert(det_manifest_status(raw) == DET_ERR_FORMAT);
    det_manifest_close(raw);
    (void)remove(image_path);
    (void)remove(manifest_path);

    const char *bad_image_path = "det_test_bad_image.pgm";
    const char *bad_manifest_path = "det_test_bad_manifest.txt";
    const unsigned char bad_pixels[] = {0U, 2U, 0U, 1U};
    image = fopen(bad_image_path, "wb");
    assert(image != NULL);
    assert(fputs("P5\n2 2\n1\n", image) >= 0);
    assert(fwrite(bad_pixels, 1U, sizeof(bad_pixels), image) == sizeof(bad_pixels));
    assert(fclose(image) == 0);
    manifest = fopen(bad_manifest_path, "wb");
    assert(manifest != NULL);
    assert(fputs("det_test_bad_image.pgm\n", manifest) >= 0);
    assert(fclose(manifest) == 0);
    raw = NULL;
    assert(det_manifest_open(bad_manifest_path, 2, 2, 1, 1, &raw) == DET_OK);
    assert(det_manifest_dataset_view(raw, &dataset) == DET_OK);
    assert(dataset.next(dataset.user, &sample) < 0);
    assert(det_manifest_status(raw) == DET_ERR_FORMAT);
    det_manifest_close(raw);
    (void)remove(bad_image_path);
    (void)remove(bad_manifest_path);

    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {8, 8, 1, 1, 4, 3};
    assert(det_context_create(1U << 16, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);
    det_dataset failing = {NULL, next_io_error, NULL, 1U};
    det_train_config config = {DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 1, 0.01f,
                               0.8f, 0.1f, 1, 3, 1};
    det_train_report report;
    assert(det_train(model, &failing, &config, &report) == DET_ERR_IO);
    det_model_destroy(model);
    det_context_destroy(ctx);
}

static void test_invalid_dataset_sample(void) {
    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {8, 8, 1, 1, 4, 3};
    assert(det_context_create(1U << 16, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);
    float pixels[64] = {0.0f};
    det_box box = {1, 1, 4, 4, 0};
    one_sample_dataset storage = {{{pixels, 2, 8, 8}, &box, 1}, 0};
    det_dataset dataset = {&storage, next_one, reset_one, 1};
    det_train_config config = {DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 1, 0.01f,
                               0.8f, 0.1f, 1, 3, 1};
    det_train_report report;
    assert(det_train(model, &dataset, &config, &report) == DET_ERR_ARGUMENT);
    det_model_destroy(model);
    det_context_destroy(ctx);
}

static void test_arena_and_conv(void) {
    unsigned char memory[4096];
    det_arena arena;
    assert(det_arena_init(&arena, memory, sizeof(memory)) == DET_OK);
    det_tensor_f32 input;
    det_tensor_f32 output;
    assert(det_tensor_alloc(&arena, 1, 3, 3, &input) == DET_OK);
    assert(det_tensor_alloc(&arena, 1, 1, 1, &output) == DET_OK);
    for (int i = 0; i < 9; ++i) input.data[i] = (float)(i + 1);
    const float weights[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    const float bias[1] = {0};
    assert(det_conv2d_f32(&input, weights, bias, 1, 3, 1, 0, &output) == DET_OK);
    assert(fabsf(output.data[0] + 6.0f) < 1e-6f);

    det_tensor_f32 point_input;
    det_tensor_f32 point_output;
    assert(det_tensor_alloc(&arena, 2, 2, 2, &point_input) == DET_OK);
    assert(det_tensor_alloc(&arena, 2, 2, 2, &point_output) == DET_OK);
    for (int i = 0; i < 4; ++i) {
        point_input.data[i] = (float)(i + 1);
        point_input.data[4 + i] = (float)(i + 5);
    }
    const float point_weights[4] = {1.0f, 2.0f, -1.0f, 0.5f};
    const float point_bias[2] = {0.25f, -0.5f};
    assert(det_conv2d_f32(&point_input, point_weights, point_bias, 2, 1, 1, 0,
                          &point_output) == DET_OK);
    assert(fabsf(point_output.data[0] - 11.25f) < 1e-6f);
    assert(fabsf(point_output.data[4] - 1.0f) < 1e-6f);

    det_tensor_f32 grad_output;
    det_tensor_f32 grad_input;
    assert(det_tensor_alloc(&arena, 1, 1, 1, &grad_output) == DET_OK);
    assert(det_tensor_alloc(&arena, 1, 3, 3, &grad_input) == DET_OK);
    grad_output.data[0] = 1.0f;
    float grad_weights[9];
    float grad_bias[1];
    assert(det_conv2d_backward_f32(&input, weights, &grad_output, 1, 3, 1, 0,
                                   &grad_input, grad_weights, grad_bias) == DET_OK);
    for (int i = 0; i < 9; ++i) assert(fabsf(grad_weights[i] - input.data[i]) < 1e-6f);
    assert(fabsf(grad_bias[0] - 1.0f) < 1e-6f);
}

static void test_stride2_padding_parity(void) {
    const int dimensions[] = {1, 2, 3, 4, 31, 32, 33, 159, 160, 161};
    const int channel_counts[] = {1, 4};
    for (size_t dimension_index = 0U;
         dimension_index < sizeof(dimensions) / sizeof(dimensions[0]); ++dimension_index) {
        int height = dimensions[dimension_index];
        int width = dimensions[(dimension_index * 3U + 2U) %
                                (sizeof(dimensions) / sizeof(dimensions[0]))];
        for (size_t channel_index = 0U;
             channel_index < sizeof(channel_counts) / sizeof(channel_counts[0]); ++channel_index) {
            int channels = channel_counts[channel_index];
            int output_channels = 2;
            int output_height = (height + 1) / 2;
            int output_width = (width + 1) / 2;
            size_t input_count = (size_t)channels * (size_t)height * (size_t)width;
            size_t weight_count = (size_t)output_channels * (size_t)channels * 9U;
            size_t output_count = (size_t)output_channels * (size_t)output_height *
                                  (size_t)output_width;
            float *input_data = (float *)malloc(input_count * sizeof(float));
            float *weights = (float *)malloc(weight_count * sizeof(float));
            float *output_data = (float *)malloc(output_count * sizeof(float));
            float *reference = (float *)malloc(output_count * sizeof(float));
            assert(input_data != NULL && weights != NULL && output_data != NULL &&
                   reference != NULL);
            for (size_t i = 0U; i < input_count; ++i) {
                input_data[i] = (float)((int)(i % 17U) - 8) * 0.07f;
            }
            for (size_t i = 0U; i < weight_count; ++i) {
                weights[i] = (float)((int)(i % 11U) - 5) * 0.03f;
            }
            const float bias[2] = {-0.13f, 0.21f};
            det_tensor_f32 input = {input_data, channels, height, width};
            det_tensor_f32 output = {output_data, output_channels, output_height, output_width};
            assert(det_conv2d_f32(&input, weights, bias, output_channels, 3, 2, 1,
                                  &output) == DET_OK);
            for (int oc = 0; oc < output_channels; ++oc) {
                for (int oy = 0; oy < output_height; ++oy) {
                    for (int ox = 0; ox < output_width; ++ox) {
                        float sum = bias[oc];
                        for (int ic = 0; ic < channels; ++ic) {
                            for (int ky = 0; ky < 3; ++ky) {
                                int iy = oy * 2 + ky - 1;
                                if (iy < 0 || iy >= height) continue;
                                for (int kx = 0; kx < 3; ++kx) {
                                    int ix = ox * 2 + kx - 1;
                                    if (ix < 0 || ix >= width) continue;
                                    size_t input_index = ((size_t)ic * (size_t)height +
                                                          (size_t)iy) * (size_t)width +
                                                         (size_t)ix;
                                    size_t weight_index = (((size_t)oc * (size_t)channels +
                                                            (size_t)ic) * 3U +
                                                           (size_t)ky) * 3U + (size_t)kx;
                                    sum += weights[weight_index] * input_data[input_index];
                                }
                            }
                        }
                        size_t output_index = ((size_t)oc * (size_t)output_height +
                                               (size_t)oy) * (size_t)output_width +
                                              (size_t)ox;
                        reference[output_index] = sum;
                    }
                }
            }
            for (size_t i = 0U; i < output_count; ++i) {
                assert(fabsf(output_data[i] - reference[i]) < 1.0e-6f);
            }
            free(reference);
            free(output_data);
            free(weights);
            free(input_data);
        }
    }
}

static void test_math(void) {
    assert(fabsf(det_sigmoid(0.0f) - 0.5f) < 1e-6f);
    det_box a = {0, 0, 10, 10, 0};
    det_box b = {5, 5, 15, 15, 0};
    assert(fabsf(det_iou(&a, &b) - (25.0f / 175.0f)) < 1e-6f);
    assert(det_quantize_s8(1.0f, 0.01f) == 100);
    assert(det_quantize_s8(3.0f, 0.01f) == 127);
    assert(det_requantize_i32(100, 1, 1, 0) == 50);
    assert(det_requantize_i32(INT64_MAX, INT32_MAX, 0, 0) == 127);
    assert(det_requantize_i32(INT64_MIN, INT32_MAX, 0, 0) == -128);
    assert(det_requantize_i32(1, 1, 1, 0) == 1);
    assert(det_requantize_i32(-1, 1, 1, 0) == -1);
    float weights[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 0.5f};
    int8_t q8[6];
    float scales8[2];
    assert(det_quantize_per_channel_s8(weights, 2, 3, q8, scales8) == DET_OK);
    assert(q8[0] == -127 && q8[2] == 0);
    weights[0] = INFINITY;
    assert(det_quantize_per_channel_s8(weights, 2, 3, q8, scales8) == DET_ERR_ARGUMENT);
    weights[0] = -2.0f;
    uint8_t q4[4];
    float scales4[2];
    assert(det_quantize_pack_w4(weights, 2, 3, q4, scales4) == DET_OK);
    assert((q4[0] & 0x0fU) == 9U);
}

static void test_train_predict_roundtrip(void) {
    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {32, 32, 1, 1, 16, 7};
    assert(det_context_create(1U << 20, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);

    float image_data[32 * 32];
    for (size_t i = 0; i < sizeof(image_data) / sizeof(image_data[0]); ++i) image_data[i] = 0.0f;
    for (int y = 10; y < 22; ++y) {
        for (int x = 10; x < 22; ++x) image_data[y * 32 + x] = 1.0f;
    }
    det_box box = {10, 10, 22, 22, 0};
    one_sample_dataset storage = {{ {image_data, 1, 32, 32}, &box, 1 }, 0};
    det_dataset dataset = {&storage, next_one, reset_one, 1};
    det_train_config config = {DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 40, 0.02f,
                               0.8f, 0.1f, 1, 7, 1};
    det_train_report report;
    assert(det_train(model, &dataset, &config, &report) == DET_OK);
    assert(report.samples_seen == 40U);
    assert(report.updates > 0U);

    det_detection detections[16];
    int count = 0;
    det_image image = {image_data, 1, 32, 32};
    assert(det_predict(model, &image, 0.1f, detections, 16, &count) == DET_OK);
    assert(count > 0);
    assert(detections[0].box.class_id == 0);
    det_detection top_one[1];
    int top_count = 0;
    assert(det_predict(model, &image, 0.0f, top_one, 1, &top_count) == DET_OK);
    assert(top_count == 1);
    assert(top_one[0].score >= detections[0].score - 1e-6f);
    float saved_pixel = image_data[0];
    image_data[0] = NAN;
    assert(det_predict(model, &image, 0.1f, detections, 16, &count) == DET_ERR_ARGUMENT);
    image_data[0] = saved_pixel;

    config.mode = DET_TRAIN_GLOBAL_BP;
    config.epochs = 2;
    config.reset_weights = 0;
    assert(det_train(model, &dataset, &config, &report) == DET_OK);
    assert(report.used_global_backward == 1);
    det_eval_report evaluation;
    assert(det_evaluate(model, &dataset, 0.1f, &evaluation) == DET_OK);
    assert(evaluation.samples_seen == 1U && evaluation.ground_truths == 1U);
    assert(evaluation.predictions > 0U);
    assert(evaluation.true_positives == 1U && evaluation.false_positives == 15U &&
           evaluation.false_negatives == 0U);
    assert(fabsf(evaluation.ap50 - 1.0f) < 1e-6f);
    assert(fabsf(evaluation.ap75) < 1e-6f);
    assert(fabsf(evaluation.map50_95 - 0.4f) < 1e-6f);
    assert(fabsf(evaluation.class_ap50[0] - 1.0f) < 1e-6f);
    assert(fabsf(evaluation.size_ap50[0] - 1.0f) < 1e-6f);
    assert(evaluation.true_positives + evaluation.false_positives == evaluation.predictions);
    assert(evaluation.true_positives + evaluation.false_negatives == evaluation.ground_truths);
    assert(evaluation.precision >= 0.0f && evaluation.precision <= 1.0f);
    assert(evaluation.recall >= 0.0f && evaluation.recall <= 1.0f);
    assert(evaluation.ap50 >= 0.0f && evaluation.ap50 <= 1.0f);
    assert(evaluation.ap75 >= 0.0f && evaluation.ap75 <= 1.0f);
    assert(evaluation.map50_95 >= 0.0f && evaluation.map50_95 <= 1.0f);
    assert(evaluation.class_ap50[0] >= 0.0f && evaluation.class_ap50[0] <= 1.0f);
    for (int size_group = 0; size_group < 3; ++size_group) {
        assert(evaluation.size_ap50[size_group] >= 0.0f &&
               evaluation.size_ap50[size_group] <= 1.0f);
    }
    assert(isfinite(evaluation.precision) && isfinite(evaluation.recall) &&
           isfinite(evaluation.mean_iou) && isfinite(evaluation.ap50));

    const char *path = "det_test_model.cdet";
    assert(det_save(model, path) == DET_OK);
    det_model *loaded = NULL;
    det_status load_status = det_load(ctx, path, &loaded);
    assert(load_status == DET_OK);
    det_detection model_detections[16];
    det_detection loaded_detections[16];
    int current_count = 0;
    assert(det_predict(model, &image, 0.1f, model_detections, 16, &current_count) == DET_OK);
    int loaded_count = 0;
    assert(det_predict(loaded, &image, 0.1f, loaded_detections, 16, &loaded_count) == DET_OK);
    assert(loaded_count == current_count);
    for (int i = 0; i < current_count; ++i) {
        assert(fabsf(model_detections[i].score - loaded_detections[i].score) < 1e-6f);
        assert(fabsf(model_detections[i].box.x1 - loaded_detections[i].box.x1) < 1e-6f);
        assert(fabsf(model_detections[i].box.y1 - loaded_detections[i].box.y1) < 1e-6f);
        assert(fabsf(model_detections[i].box.x2 - loaded_detections[i].box.x2) < 1e-6f);
        assert(fabsf(model_detections[i].box.y2 - loaded_detections[i].box.y2) < 1e-6f);
        assert(model_detections[i].box.class_id == loaded_detections[i].box.class_id);
    }

    /* Resume both copies for one deterministic local step.  Matching output
       after this step covers serialized auxiliary weights and optimizer state,
       not only the deployed inference bank. */
    det_train_config continuation = config;
    continuation.mode = DET_TRAIN_LOCAL_FAST;
    continuation.epochs = 1;
    continuation.reset_weights = 0;
    det_train_report continuation_report;
    assert(det_train(model, &dataset, &continuation, &continuation_report) == DET_OK);
    assert(det_train(loaded, &dataset, &continuation, &continuation_report) == DET_OK);
    assert(det_save(model, path) == DET_OK);
    const char *continuation_path = "det_test_continuation.cdet";
    assert(det_save(loaded, continuation_path) == DET_OK);
    assert(files_equal(path, continuation_path));
    (void)remove(continuation_path);
    assert(det_predict(model, &image, 0.1f, model_detections, 16, &current_count) == DET_OK);
    assert(det_predict(loaded, &image, 0.1f, loaded_detections, 16, &loaded_count) == DET_OK);
    assert(loaded_count == current_count);
    for (int i = 0; i < current_count; ++i) {
        assert(fabsf(model_detections[i].score - loaded_detections[i].score) < 1e-6f);
        assert(fabsf(model_detections[i].box.x1 - loaded_detections[i].box.x1) < 1e-6f);
        assert(fabsf(model_detections[i].box.y1 - loaded_detections[i].box.y1) < 1e-6f);
        assert(fabsf(model_detections[i].box.x2 - loaded_detections[i].box.x2) < 1e-6f);
        assert(fabsf(model_detections[i].box.y2 - loaded_detections[i].box.y2) < 1e-6f);
        assert(model_detections[i].box.class_id == loaded_detections[i].box.class_id);
    }
    det_model_destroy(loaded);
    /* The auxiliary bank is training-only: changing its authoritative FP32
       weight must not alter deployed one-to-one inference. */
    assert(mutate_auxiliary_weight(path, 32, 32, 1, 1));
    det_model *aux_mutated = NULL;
    assert(det_load(ctx, path, &aux_mutated) == DET_OK);
    det_detection aux_detections[16];
    int aux_count = 0;
    assert(det_predict(aux_mutated, &image, 0.1f, aux_detections, 16,
                       &aux_count) == DET_OK);
    assert(aux_count == current_count);
    for (int i = 0; i < current_count; ++i) {
        assert(fabsf(model_detections[i].score - aux_detections[i].score) < 1e-6f);
        assert(fabsf(model_detections[i].box.x1 - aux_detections[i].box.x1) < 1e-6f);
        assert(fabsf(model_detections[i].box.y1 - aux_detections[i].box.y1) < 1e-6f);
        assert(fabsf(model_detections[i].box.x2 - aux_detections[i].box.x2) < 1e-6f);
        assert(fabsf(model_detections[i].box.y2 - aux_detections[i].box.y2) < 1e-6f);
        assert(model_detections[i].box.class_id == aux_detections[i].box.class_id);
    }
    det_model_destroy(aux_mutated);
    const char *truncated_path = "det_test_truncated.cdet";
    assert(write_truncated_copy(path, truncated_path));
    det_model *truncated = NULL;
    assert(det_load(ctx, truncated_path, &truncated) == DET_ERR_FORMAT);
    (void)remove(truncated_path);

    /* A CRC-valid mutation of an inactive FP32 cache must not affect the
       canonical model reconstructed from the learned FP32 tensors. */
    FILE *cache_mutation = fopen(path, "r+b");
    assert(cache_mutation != NULL);
    assert(fseek(cache_mutation, -(long)sizeof(uint32_t) - 1L, SEEK_END) == 0);
    int cache_byte = fgetc(cache_mutation);
    assert(cache_byte != EOF);
    assert(fseek(cache_mutation, -(long)sizeof(uint32_t) - 1L, SEEK_END) == 0);
    unsigned char mutated_cache = (unsigned char)cache_byte ^ 0x7fU;
    assert(fwrite(&mutated_cache, 1U, 1U, cache_mutation) == 1U);
    assert(fclose(cache_mutation) == 0);
    assert(rewrite_crc32(path));
    det_model *canonical = NULL;
    assert(det_load(ctx, path, &canonical) == DET_OK);
    det_detection canonical_detections[16];
    int canonical_count = 0;
    assert(det_predict(canonical, &image, 0.1f, canonical_detections, 16,
                       &canonical_count) == DET_OK);
    assert(canonical_count == current_count);
    for (int i = 0; i < current_count; ++i) {
        assert(fabsf(model_detections[i].score - canonical_detections[i].score) < 1e-6f);
    }
    det_model_destroy(canonical);
    FILE *corrupt = fopen(path, "r+b");
    assert(corrupt != NULL);
    assert(fseek(corrupt, 32L, SEEK_SET) == 0);
    int byte = fgetc(corrupt);
    assert(byte != EOF);
    assert(fseek(corrupt, 32L, SEEK_SET) == 0);
    unsigned char changed = (unsigned char)byte ^ 0x01U;
    assert(fwrite(&changed, 1U, 1U, corrupt) == 1U);
    assert(fclose(corrupt) == 0);
    det_model *corrupt_model = NULL;
    assert(det_load(ctx, path, &corrupt_model) == DET_ERR_FORMAT);
    det_detection f32_detections[16];
    int f32_count = 0;
    assert(det_predict(model, &image, 0.1f, f32_detections, 16, &f32_count) == DET_OK);
    assert(f32_count > 0);
    assert(det_model_set_precision(model, DET_PRECISION_INT8) == DET_OK);
    assert(det_model_precision(model) == DET_PRECISION_INT8);
    int int8_count = 0;
    assert(det_predict(model, &image, 0.1f, detections, 16, &int8_count) == DET_OK);
    assert(int8_count > 0);
    assert(int8_count >= f32_count / 2);
    for (int i = 0; i < int8_count; ++i) assert(isfinite(detections[i].score));
    int int8_compare = f32_count < int8_count ? f32_count : int8_count;
    for (int i = 0; i < int8_compare; ++i) {
        assert(fabsf(f32_detections[i].score - detections[i].score) < 0.35f);
        assert(f32_detections[i].box.class_id == detections[i].box.class_id);
    }
    assert(det_model_set_precision(model, DET_PRECISION_W4A8) == DET_OK);
    int w4_count = 0;
    assert(det_predict(model, &image, 0.1f, detections, 16, &w4_count) == DET_OK);
    assert(w4_count > 0);
    assert(w4_count >= f32_count / 2);
    for (int i = 0; i < w4_count; ++i) assert(isfinite(detections[i].score));
    int w4_compare = f32_count < w4_count ? f32_count : w4_count;
    for (int i = 0; i < w4_compare; ++i) {
        assert(fabsf(f32_detections[i].score - detections[i].score) < 0.5f);
        assert(f32_detections[i].box.class_id == detections[i].box.class_id);
    }
    det_detection w4_detections[16];
    memcpy(w4_detections, detections, (size_t)w4_count * sizeof(*detections));
    const char *quant_path = "det_test_quantized.cdet";
    assert(det_save(model, quant_path) == DET_OK);
    det_model *loaded_quant = NULL;
    assert(det_load(ctx, quant_path, &loaded_quant) == DET_OK);
    assert(det_model_precision(loaded_quant) == DET_PRECISION_W4A8);
    int loaded_quant_count = 0;
    det_detection loaded_quant_detections[16];
    assert(det_predict(loaded_quant, &image, 0.1f, loaded_quant_detections, 16,
                       &loaded_quant_count) == DET_OK);
    assert(loaded_quant_count == w4_count);
    for (int i = 0; i < w4_count; ++i) {
        assert(fabsf(w4_detections[i].score - loaded_quant_detections[i].score) < 1e-6f);
        assert(fabsf(w4_detections[i].box.x1 - loaded_quant_detections[i].box.x1) < 1e-6f);
        assert(w4_detections[i].box.class_id == loaded_quant_detections[i].box.class_id);
    }
    det_model_destroy(loaded_quant);
    (void)remove(quant_path);
    (void)remove(path);
    det_model_destroy(model);
    det_context_destroy(ctx);
}

static void test_odd_shape_neck_roundtrip(void) {
    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model *loaded = NULL;
    det_model_spec spec = {33, 31, 1, 1, 8, 3};
    float pixels[33 * 31] = {0.0f};
    for (int y = 4; y < 12; ++y) {
        for (int x = 4; x < 12; ++x) pixels[y * 33 + x] = 1.0f;
    }
    det_image image = {pixels, 1, 31, 33};
    det_detection before[8];
    det_detection trained[8];
    det_detection after[8];
    int before_count = 0;
    int trained_count = 0;
    int after_count = 0;
    const char *before_path = "det_test_odd_neck_before.cdet";
    const char *path = "det_test_odd_neck.cdet";
    assert(det_context_create(1U << 20, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);
    assert(det_model_reset(model, 11) == DET_OK);
    assert(det_predict(model, &image, 0.1f, before, 8, &before_count) == DET_OK);
    assert(det_save(model, before_path) == DET_OK);
    det_box box = {4.0f, 4.0f, 12.0f, 12.0f, 0};
    one_sample_dataset storage = {{{pixels, 1, 31, 33}, &box, 1}, 0};
    det_dataset dataset = {&storage, next_one, reset_one, 1};
    det_train_config config = {DET_TRAIN_GLOBAL_BP, DET_PRECISION_F32, 1, 0.01f,
                               0.8f, 0.1f, 0, 11, 1};
    det_train_report report;
    assert(det_train(model, &dataset, &config, &report) == DET_OK);
    assert(report.used_global_backward == 1);
    assert(det_predict(model, &image, 0.1f, trained, 8, &trained_count) == DET_OK);
    assert(det_save(model, path) == DET_OK);
    assert(!files_equal(before_path, path));
    size_t bottomup_offset = test_bottomup_weight_offset(1, 1);
    assert(file_regions_differ(before_path, path, bottomup_offset,
                               4U * 9U * sizeof(float)));
    assert(det_load(ctx, path, &loaded) == DET_OK);
    assert(det_predict(loaded, &image, 0.1f, after, 8, &after_count) == DET_OK);
    assert(after_count == trained_count);
    for (int i = 0; i < trained_count; ++i) {
        assert(fabsf(trained[i].score - after[i].score) < 1e-6f);
        assert(fabsf(trained[i].box.x1 - after[i].box.x1) < 1e-6f);
        assert(fabsf(trained[i].box.y1 - after[i].box.y1) < 1e-6f);
        assert(fabsf(trained[i].box.x2 - after[i].box.x2) < 1e-6f);
        assert(fabsf(trained[i].box.y2 - after[i].box.y2) < 1e-6f);
    }
    assert(det_model_set_precision(loaded, DET_PRECISION_INT8) == DET_OK);
    assert(det_predict(loaded, &image, 0.1f, after, 8, &after_count) == DET_OK);
    for (int i = 0; i < after_count; ++i) assert(isfinite(after[i].score));
    assert(det_model_set_precision(loaded, DET_PRECISION_W4A8) == DET_OK);
    assert(det_predict(loaded, &image, 0.1f, after, 8, &after_count) == DET_OK);
    for (int i = 0; i < after_count; ++i) assert(isfinite(after[i].score));
    det_model_destroy(loaded);

    det_model *local_model = NULL;
    const char *local_before_path = "det_test_odd_local_before.cdet";
    const char *local_after_path = "det_test_odd_local_after.cdet";
    assert(det_model_build(ctx, &spec, &local_model) == DET_OK);
    assert(det_model_reset(local_model, 23) == DET_OK);
    assert(det_save(local_model, local_before_path) == DET_OK);
    repeat_sample_dataset local_storage = {storage.sample, 600U, 600U};
    det_dataset local_dataset = {&local_storage, next_repeat, reset_repeat, 600U};
    det_train_config local_config = {DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 2,
                                     0.01f, 0.8f, 0.1f, 0, 23, 1};
    assert(det_train(local_model, &local_dataset, &local_config, &report) == DET_OK);
    assert(det_save(local_model, local_after_path) == DET_OK);
    assert(file_regions_differ(local_before_path, local_after_path,
                               test_bottomup_weight_offset(1, 1) - 10U * sizeof(int),
                               test_stage_bytes_kernel(4, 4, 1, 3)));
    det_model_destroy(local_model);
    det_model_destroy(model);
    det_context_destroy(ctx);
    (void)remove(before_path);
    (void)remove(path);
    (void)remove(local_before_path);
    (void)remove(local_after_path);
}

static void test_multi_box_quantization_gate(void) {
    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {64, 64, 1, 2, 32, 31};
    float pixels[64 * 64] = {0.0f};
    for (int y = 8; y < 24; ++y) {
        for (int x = 8; x < 24; ++x) pixels[y * 64 + x] = 1.0f;
    }
    for (int y = 40; y < 56; ++y) {
        for (int x = 40; x < 56; ++x) pixels[y * 64 + x] = 0.75f;
    }
    det_box boxes[2] = {{8.0f, 8.0f, 24.0f, 24.0f, 0},
                        {40.0f, 40.0f, 56.0f, 56.0f, 1}};
    multi_box_dataset storage = {{{pixels, 1, 64, 64}, boxes, 2}, 0};
    det_dataset dataset = {&storage, next_multi_box, reset_multi_box, 1};
    det_train_config config = {DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 80,
                               0.02f, 0.8f, 0.1f, 1, 31, 1};
    det_train_report train_report;
    det_eval_report f32_report;
    det_eval_report int8_report;
    det_eval_report w4_report;
    assert(det_context_create(2U << 20, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);
    assert(det_train(model, &dataset, &config, &train_report) == DET_OK);
    assert(train_report.samples_seen == 80U && train_report.used_global_backward == 0);
    assert(det_evaluate(model, &dataset, 0.1f, &f32_report) == DET_OK);
    assert(f32_report.samples_seen == 1U && f32_report.ground_truths == 2U);
    assert(f32_report.true_positives >= 2U);
    assert(f32_report.ap50 >= 0.2f && f32_report.map50_95 >= 0.2f);

    assert(det_model_set_precision(model, DET_PRECISION_INT8) == DET_OK);
    assert(det_evaluate(model, &dataset, 0.1f, &int8_report) == DET_OK);
    assert(int8_report.ground_truths == f32_report.ground_truths);
    assert(int8_report.true_positives == f32_report.true_positives);
    assert(int8_report.false_negatives == f32_report.false_negatives);
    assert(int8_report.false_positives <= f32_report.false_positives + 2U);
    assert(int8_report.recall + 0.01f >= f32_report.recall);
    assert(int8_report.precision + 0.02f >= f32_report.precision);
    assert(int8_report.ap50 + 0.02f >= f32_report.ap50);
    assert(int8_report.map50_95 + 0.02f >= f32_report.map50_95);
    assert(int8_report.mean_iou + 0.05f >= f32_report.mean_iou);

    assert(det_model_set_precision(model, DET_PRECISION_W4A8) == DET_OK);
    assert(det_evaluate(model, &dataset, 0.1f, &w4_report) == DET_OK);
    assert(w4_report.ground_truths == f32_report.ground_truths);
    assert(w4_report.true_positives == f32_report.true_positives);
    assert(w4_report.false_negatives == f32_report.false_negatives);
    assert(w4_report.false_positives <= f32_report.false_positives + 2U);
    assert(w4_report.recall + 0.01f >= f32_report.recall);
    assert(w4_report.precision + 0.02f >= f32_report.precision);
    assert(w4_report.ap50 + 0.05f >= f32_report.ap50);
    assert(w4_report.map50_95 + 0.05f >= f32_report.map50_95);
    assert(w4_report.mean_iou + 0.10f >= f32_report.mean_iou);
    det_model_destroy(model);

    det_model *global_model = NULL;
    det_train_config global_config = {DET_TRAIN_GLOBAL_BP, DET_PRECISION_F32, 2,
                                      0.01f, 0.8f, 0.1f, 1, 31, 1};
    det_train_report global_report;
    det_eval_report global_eval;
    assert(det_model_build(ctx, &spec, &global_model) == DET_OK);
    assert(det_train(global_model, &dataset, &global_config, &global_report) == DET_OK);
    assert(global_report.samples_seen == 2U && global_report.used_global_backward == 1);
    assert(det_evaluate(global_model, &dataset, 0.1f, &global_eval) == DET_OK);
    assert(global_eval.samples_seen == 1U && global_eval.ground_truths == 2U);
    assert(isfinite(global_eval.mean_iou) && isfinite(global_eval.map50_95));
    det_model_destroy(global_model);
    det_context_destroy(ctx);
}

static void test_streamed_size_class_stress(void) {
    det_context *ctx = NULL;
    det_model *model = NULL;
    det_model_spec spec = {128, 128, 1, 3, 32, 37};
    float pixels[3][128 * 128] = {{0.0f}};
    det_box boxes[3] = {{4.0f, 4.0f, 12.0f, 12.0f, 0},
                        {16.0f, 16.0f, 56.0f, 56.0f, 1},
                        {20.0f, 20.0f, 120.0f, 120.0f, 2}};
    det_sample samples[3];
    for (int sample_index = 0; sample_index < 3; ++sample_index) {
        const det_box *box = &boxes[sample_index];
        for (int y = (int)box->y1; y < (int)box->y2; ++y) {
            for (int x = (int)box->x1; x < (int)box->x2; ++x) {
                pixels[sample_index][y * 128 + x] = 0.5f + 0.2f * (float)sample_index;
            }
        }
        samples[sample_index] = (det_sample){{pixels[sample_index], 1, 128, 128},
                                              box, 1};
    }
    stream_dataset storage = {samples, 3, 0};
    det_dataset dataset = {&storage, next_stream, reset_stream, 3};
    det_train_config config = {DET_TRAIN_LOCAL_FAST, DET_PRECISION_F32, 20,
                               0.01f, 0.8f, 0.1f, 3, 37, 1};
    det_train_report train_report;
    det_eval_report evaluation;
    assert(det_context_create(8U << 20, &ctx) == DET_OK);
    assert(det_model_build(ctx, &spec, &model) == DET_OK);
    assert(det_train(model, &dataset, &config, &train_report) == DET_OK);
    assert(train_report.samples_seen == 60U && train_report.updates > 0U);
    assert(det_evaluate(model, &dataset, 0.1f, &evaluation) == DET_OK);
    assert(evaluation.samples_seen == 3U && evaluation.ground_truths == 3U);
    assert(evaluation.size_ground_truths[0] == 1U &&
           evaluation.size_ground_truths[1] == 1U &&
           evaluation.size_ground_truths[2] == 1U);
    assert(evaluation.precision >= 0.0f && evaluation.precision <= 1.0f);
    assert(evaluation.recall >= 0.0f && evaluation.recall <= 1.0f);
    assert(evaluation.mean_iou >= 0.0f && evaluation.mean_iou <= 1.0f);
    assert(evaluation.ap50 >= 0.0f && evaluation.ap50 <= 1.0f);
    assert(evaluation.map50_95 >= 0.0f && evaluation.map50_95 <= 1.0f);
    for (int class_id = 0; class_id < 3; ++class_id) {
        assert(evaluation.class_ap50[class_id] >= 0.0f &&
               evaluation.class_ap50[class_id] <= 1.0f);
    }
    for (int size_group = 0; size_group < 3; ++size_group) {
        assert(evaluation.size_ap50[size_group] >= 0.0f &&
               evaluation.size_ap50[size_group] <= 1.0f);
    }
    assert(det_model_set_precision(model, DET_PRECISION_INT8) == DET_OK);
    assert(det_evaluate(model, &dataset, 0.1f, &evaluation) == DET_OK);
    assert(evaluation.samples_seen == 3U && evaluation.ground_truths == 3U);
    assert(isfinite(evaluation.mean_iou) && isfinite(evaluation.map50_95));
    assert(det_model_set_precision(model, DET_PRECISION_W4A8) == DET_OK);
    assert(det_evaluate(model, &dataset, 0.1f, &evaluation) == DET_OK);
    assert(evaluation.samples_seen == 3U && evaluation.ground_truths == 3U);
    assert(isfinite(evaluation.mean_iou) && isfinite(evaluation.map50_95));
    det_model_destroy(model);
    det_context_destroy(ctx);
}

int main(void) {
    test_arena_and_conv();
    test_stride2_padding_parity();
    test_math();
    test_manifest_adapter();
    test_invalid_dataset_sample();
    test_train_predict_roundtrip();
    test_odd_shape_neck_roundtrip();
    test_multi_box_quantization_gate();
    test_streamed_size_class_stress();
    puts("all det tests passed");
    return 0;
}
