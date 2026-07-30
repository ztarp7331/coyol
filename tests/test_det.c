#include "det.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    det_sample sample;
    int emitted;
} one_sample_dataset;

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
    float weights[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 0.5f};
    int8_t q8[6];
    float scales8[2];
    assert(det_quantize_per_channel_s8(weights, 2, 3, q8, scales8) == DET_OK);
    assert(q8[0] == -127 && q8[2] == 0);
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

    config.mode = DET_TRAIN_GLOBAL_BP;
    config.epochs = 2;
    config.reset_weights = 0;
    assert(det_train(model, &dataset, &config, &report) == DET_OK);
    assert(report.used_global_backward == 1);

    const char *path = "det_test_model.cdet";
    assert(det_save(model, path) == DET_OK);
    det_model *loaded = NULL;
    assert(det_load(ctx, path, &loaded) == DET_OK);
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
