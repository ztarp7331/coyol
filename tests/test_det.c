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
    det_model_destroy(loaded);
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

int main(void) {
    test_arena_and_conv();
    test_math();
    test_invalid_dataset_sample();
    test_train_predict_roundtrip();
    puts("all det tests passed");
    return 0;
}
