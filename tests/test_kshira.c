#include "kshira/core.h"
#include "kshira/domain.h"
#include "kshira/eval.h"
#include "kshira/phase.h"
#include "kshira/quant.h"
#include "kshira/rad.h"
#include "kshira/session.h"
#include "kshira/sparse.h"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NDEBUG
#undef assert
#define assert(expression) \
    do { \
        if (!(expression)) { \
            fprintf(stderr, "kshira test assertion failed: %s (%s:%d)\n", \
                    #expression, __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)
#endif

static void test_arena(void) {
    unsigned char memory[256];
    unsigned char *first;
    unsigned char *second;
    kshira_arena arena;
    kshira_tensor_f32 tensor;
    assert(kshira_arena_init(&arena, memory, sizeof(memory)) == KSHIRA_OK);
    first = (unsigned char *)kshira_arena_alloc(&arena, 7U, 1U);
    second = (unsigned char *)kshira_arena_alloc(&arena, 16U, 16U);
    assert(first != NULL && second != NULL);
    assert(((uintptr_t)second % 16U) == 0U);
    assert(kshira_arena_used(&arena) == 32U);
    assert(kshira_arena_high_water(&arena) == 32U);
    assert(kshira_tensor_view(&arena, 2, 2, 2, &tensor) == KSHIRA_OK);
    assert(tensor.data != NULL && tensor.channels == 2 && tensor.height == 2 && tensor.width == 2);
    assert(kshira_arena_alloc(&arena, 1024U, 8U) == NULL);
    kshira_arena_reset(&arena);
    assert(kshira_arena_used(&arena) == 0U && kshira_arena_high_water(&arena) == 64U);
    {
        kshira_arena unaligned;
        void *aligned = NULL;
        assert(kshira_arena_init(&unaligned, memory + 1, 128U) == KSHIRA_OK);
        aligned = kshira_arena_alloc(&unaligned, 4U, 16U);
        assert(aligned != NULL && ((uintptr_t)aligned % 16U) == 0U);
    }
}

static void test_quant(void) {
    const float values[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    uint8_t packed[3] = {0U, 0U, 0U};
    int8_t unpacked[5] = {0};
    int8_t expanded[5] = {0};
    uint8_t store_packed[3] = {0U, 0U, 0U};
    float scale = kshira_symmetric_scale(1.0f, KSHIRA_BITS_INT4);
    kshira_weight_store store;
    assert(fabsf(scale - (1.0f / 7.0f)) < 1e-7f);
    assert(kshira_quantize_symmetric(-1.0f, scale, KSHIRA_BITS_INT4) == -7);
    assert(kshira_quantize_symmetric(1.0f, scale, KSHIRA_BITS_INT4) == 7);
    assert(kshira_quantize_symmetric(1.0e20f, 1.0f, KSHIRA_BITS_INT4) == 7);
    assert(kshira_pack_int4(values, 5U, scale, packed, sizeof(packed)) == KSHIRA_OK);
    assert(kshira_unpack_int4(packed, sizeof(packed), 5U, unpacked,
                              sizeof(unpacked)) == KSHIRA_OK);
    assert(kshira_unpack_int4(packed, 1U, 5U, unpacked, sizeof(unpacked)) == KSHIRA_ERR_RANGE);
    assert(unpacked[0] == -7 && unpacked[2] == 0 && unpacked[4] == 7);
    assert(kshira_weight_store_init(&store, values, 5U, scale, store_packed,
                                    sizeof(store_packed)) == KSHIRA_OK);
    assert(kshira_weight_store_set_mode(&store, KSHIRA_BITS_INT8, expanded,
                                        sizeof(expanded)) == KSHIRA_OK);
    for (size_t i = 0U; i < 5U; ++i) assert(kshira_weight_store_at(&store, i) == unpacked[i]);
    assert(kshira_weight_store_set_mode(&store, KSHIRA_BITS_INT4, NULL, 0U) == KSHIRA_OK);
    assert(kshira_weight_store_at(&store, 4U) == 7);
    assert(kshira_weight_store_set_mode(&store, KSHIRA_BITS_INT8, NULL, 0U) == KSHIRA_ERR_MEMORY);
}

static void test_qas_and_sparse(void) {
    float gradients[2] = {1.0f, -2.0f};
    float bias[1] = {0.5f};
    uint8_t bits[2];
    kshira_sparse_mask mask;
    assert(kshira_apply_qas(gradients, 2U, bias, 1U, 0.5f, 0.25f) == KSHIRA_OK);
    assert(fabsf(gradients[0] - 4.0f) < 1e-6f);
    assert(fabsf(gradients[1] + 8.0f) < 1e-6f);
    assert(fabsf(bias[0] - 32.0f) < 1e-6f);
    assert(kshira_apply_qas(gradients, 2U, bias, 1U, FLT_MIN, 1.0f) == KSHIRA_ERR_RANGE);
    assert(kshira_sparse_mask_init(&mask, bits, sizeof(bits), 10U) == KSHIRA_OK);
    assert(kshira_sparse_mask_set(&mask, 0U, 1) == KSHIRA_OK);
    assert(kshira_sparse_mask_set(&mask, 9U, 1) == KSHIRA_OK);
    assert(kshira_sparse_mask_active(&mask) == 2U);
    assert(kshira_sparse_mask_get(&mask, 9U) == 1);
    assert(kshira_sparse_mask_set(&mask, 10U, 1) == KSHIRA_ERR_ARGUMENT);
    assert(kshira_sparse_memory_bytes(100U, 400U, 20U, 10U,
                                      KSHIRA_UPDATE_CHANNELS, 2U) == 200U);
    assert(kshira_sparse_memory_bytes(100U, 400U, 20U, 10U,
                                      KSHIRA_UPDATE_FULL, 10U) == 520U);
    assert(kshira_sparse_memory_bytes(100U, 401U, 20U, 10U,
                                      KSHIRA_UPDATE_CHANNELS, 1U) == 161U);
    assert(kshira_sparse_memory_bytes(0U, SIZE_MAX, 1U, 1U,
                                      KSHIRA_UPDATE_FULL, 1U) == SIZE_MAX);
    {
        size_t peak = 0U;
        assert(kshira_sparse_plan(KSHIRA_UPDATE_CHANNELS, 100U, 401U, 20U, 10U, 1U,
                                  200U, &peak) == KSHIRA_OK);
        assert(peak == 161U);
        assert(kshira_sparse_plan(KSHIRA_UPDATE_FULL, 100U, 401U, 20U, 10U, 10U,
                                  500U, &peak) == KSHIRA_ERR_MEMORY);
    }
}

static void test_phases(void) {
    kshira_phase_contract contract = {KSHIRA_PHASE_PRE, KSHIRA_BITS_FLOAT,
                                      KSHIRA_UPDATE_FULL, 256U << 10, 0};
    assert(kshira_phase_validate(&contract) == KSHIRA_OK);
    assert(strcmp(kshira_phase_name(KSHIRA_PHASE_PRE), "PRE") == 0);
    contract.phase = KSHIRA_PHASE_TRAIN;
    contract.bits = KSHIRA_BITS_INT4;
    contract.qas_enabled = 1;
    assert(kshira_phase_validate(&contract) == KSHIRA_OK);
    contract.phase = KSHIRA_PHASE_ODT;
    contract.update_mode = KSHIRA_UPDATE_CHANNELS;
    assert(kshira_phase_validate(&contract) == KSHIRA_OK);
    contract.update_mode = KSHIRA_UPDATE_FULL;
    assert(kshira_phase_validate(&contract) == KSHIRA_ERR_ARGUMENT);
    contract.update_mode = (kshira_update_mode)99;
    assert(kshira_phase_validate(&contract) == KSHIRA_ERR_ARGUMENT);
    {
        kshira_phase_driver driver;
        kshira_phase_contract train = {KSHIRA_PHASE_TRAIN, KSHIRA_BITS_INT8,
                                       KSHIRA_UPDATE_CHANNELS, 256U << 10, 1};
        kshira_phase_contract odt = {KSHIRA_PHASE_ODT, KSHIRA_BITS_INT4,
                                     KSHIRA_UPDATE_CHANNELS, 256U << 10, 1};
        assert(kshira_phase_driver_init(&driver, 256U << 10) == KSHIRA_OK);
        assert(kshira_phase_driver_step(&driver) == KSHIRA_OK);
        assert(kshira_phase_driver_transition(&driver, odt) == KSHIRA_ERR_ARGUMENT);
        assert(kshira_phase_driver_transition(&driver, train) == KSHIRA_OK);
        assert(kshira_phase_driver_step(&driver) == KSHIRA_OK);
        assert(kshira_phase_driver_transition(&driver, odt) == KSHIRA_OK);
        assert(driver.transitions == 2U && driver.steps == 2U);
    }
}

static void test_rad_top_k(void) {
    unsigned char memory[256U << 10];
    float pixels[32 * 32] = {0.0f};
    kshira_arena arena;
    kshira_rad_model *model = NULL;
    kshira_rad_spec spec = {32, 32, 1, 3, 4, 3, 17};
    kshira_image_f32 image = {pixels, 1, 32, 32};
    kshira_rad_detection detections[4];
    int count = 0;
    assert(kshira_arena_init(&arena, memory, sizeof(memory)) == KSHIRA_OK);
    for (int y = 8; y < 16; ++y) {
        for (int x = 8; x < 16; ++x) pixels[y * 32 + x] = 1.0f;
    }
    assert(kshira_rad_build(&arena, &spec, &model) == KSHIRA_OK);
    assert(kshira_rad_map_height(model) == 8 && kshira_rad_map_width(model) == 8);
    assert(kshira_rad_parameter_bytes(model) > 0U);
    assert(kshira_rad_activation_bytes(model) > 0U);
    assert(kshira_arena_high_water(&arena) < sizeof(memory));
    assert(kshira_rad_predict(model, &image, 0.0f, detections, 4, &count) == KSHIRA_OK);
    assert(count > 0 && count <= 3);
    for (int i = 0; i < count; ++i) {
        assert(isfinite(detections[i].score) && isfinite(detections[i].quality));
        assert(detections[i].score >= 0.0f && detections[i].score <= 1.0f);
        assert(detections[i].box.class_id >= 0 && detections[i].box.class_id < 3);
        if (i > 0) assert(detections[i - 1].score >= detections[i].score);
    }
    assert(kshira_rad_predict(model, &image, 0.0f, detections, 2, &count) == KSHIRA_OK);
    assert(count == 2);
    pixels[0] = FLT_MAX;
    assert(kshira_rad_predict(model, &image, 0.0f, detections, 4, &count) == KSHIRA_OK);
    for (int i = 0; i < count; ++i) {
        assert(isfinite(detections[i].score));
        assert(isfinite(detections[i].box.x1) && isfinite(detections[i].box.y1));
    }
    pixels[0] = NAN;
    assert(kshira_rad_predict(model, &image, 0.0f, detections, 4, &count) == KSHIRA_ERR_ARGUMENT);
    pixels[0] = 0.0f;
    assert(kshira_rad_set_bits(model, KSHIRA_BITS_INT8) == KSHIRA_OK);
    assert(kshira_rad_bits(model) == KSHIRA_BITS_INT8);
    assert(kshira_rad_predict(model, &image, 0.0f, detections, 4, &count) == KSHIRA_OK);
    for (int i = 0; i < count; ++i) assert(isfinite(detections[i].score));
    assert(kshira_rad_set_bits(model, KSHIRA_BITS_INT4) == KSHIRA_OK);
    assert(kshira_rad_predict(model, &image, 0.0f, detections, 4, &count) == KSHIRA_OK);
    for (int i = 0; i < count; ++i) assert(isfinite(detections[i].score));
    assert(kshira_rad_calibrate(model, &image) == KSHIRA_OK);
    assert(kshira_rad_calibration_ready(model));
    {
        uint8_t channel_bits[1] = {0U};
        kshira_sparse_mask channel_mask;
        kshira_rad_train_config train_config = {
            KSHIRA_BITS_INT4, KSHIRA_UPDATE_CHANNELS, &channel_mask, 1.0e-5f
        };
        kshira_rad_box target = {8.0f, 8.0f, 16.0f, 16.0f, 0};
        float loss = 0.0f;
        assert(kshira_sparse_mask_init(&channel_mask, channel_bits, sizeof(channel_bits), 4U) ==
               KSHIRA_OK);
        assert(kshira_sparse_mask_set(&channel_mask, 0U, 1) == KSHIRA_OK);
        train_config.learning_rate = NAN;
        assert(kshira_rad_train_step(model, &image, &target, &train_config, &loss) ==
               KSHIRA_ERR_ARGUMENT);
        assert(kshira_rad_calibration_ready(model));
        train_config.learning_rate = 1.0e-5f;
        assert(kshira_rad_train_step(model, &image, &target, &train_config, &loss) == KSHIRA_OK);
        assert(isfinite(loss) && loss >= 0.0f);
        assert(!kshira_rad_calibration_ready(model));
        assert(kshira_rad_calibrate(model, &image) == KSHIRA_OK);
        assert(kshira_rad_calibration_ready(model));
        train_config.bits = KSHIRA_BITS_INT8;
        assert(kshira_rad_train_step(model, &image, &target, &train_config, &loss) == KSHIRA_OK);
        assert(kshira_rad_bits(model) == KSHIRA_BITS_INT8);
        assert(!kshira_rad_calibration_ready(model));
        train_config.update_mode = KSHIRA_UPDATE_FULL;
        assert(kshira_rad_train_step(model, &image, &target, &train_config, &loss) == KSHIRA_OK);
        assert(isfinite(loss) && loss >= 0.0f);
    }

    {
        unsigned char first_memory[16U << 10];
        unsigned char second_memory[16U << 10];
        kshira_arena first_arena;
        kshira_arena second_arena;
        kshira_rad_model *first_model = NULL;
        kshira_rad_model *second_model = NULL;
        kshira_rad_train_config train_config = {
            KSHIRA_BITS_INT4, KSHIRA_UPDATE_FULL, NULL, 1.0e-5f
        };
        kshira_rad_box target = {8.0f, 8.0f, 16.0f, 16.0f, 0};
        float first_loss = 0.0f;
        float second_loss = 0.0f;
        assert(kshira_arena_init(&first_arena, first_memory, sizeof(first_memory)) == KSHIRA_OK);
        assert(kshira_arena_init(&second_arena, second_memory, sizeof(second_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&first_arena, &spec, &first_model) == KSHIRA_OK);
        assert(kshira_rad_build(&second_arena, &spec, &second_model) == KSHIRA_OK);
        assert(kshira_rad_set_bits(first_model, KSHIRA_BITS_INT4) == KSHIRA_OK);
        assert(kshira_rad_predict(first_model, &image, 0.0f, detections, 4, &count) ==
               KSHIRA_OK);
        assert(kshira_rad_train_step(first_model, &image, &target, &train_config, &first_loss) ==
               KSHIRA_OK);
        assert(kshira_rad_train_step(second_model, &image, &target, &train_config, &second_loss) ==
               KSHIRA_OK);
        assert(fabsf(first_loss - second_loss) <= 1.0e-6f);
    }

    {
        unsigned char odd_memory[16U << 10];
        float odd_pixels[9 * 10] = {0.0f};
        kshira_arena odd_arena;
        kshira_rad_model *odd_model = NULL;
        kshira_rad_spec odd_spec = {9, 10, 1, 1, 2, 8, 19};
        kshira_image_f32 odd_image = {odd_pixels, 1, 10, 9};
        kshira_rad_detection odd_detections[8];
        int odd_count = 0;
        assert(kshira_arena_init(&odd_arena, odd_memory, sizeof(odd_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&odd_arena, &odd_spec, &odd_model) == KSHIRA_OK);
        assert(kshira_rad_map_height(odd_model) == 3 && kshira_rad_map_width(odd_model) == 3);
        assert(kshira_rad_predict(odd_model, &odd_image, 0.0f, odd_detections, 8,
                                  &odd_count) == KSHIRA_OK);
        for (int i = 0; i < odd_count; ++i) {
            assert(odd_detections[i].box.x1 >= 0.0f && odd_detections[i].box.y1 >= 0.0f);
            assert(odd_detections[i].box.x2 <= 9.0f && odd_detections[i].box.y2 <= 10.0f);
            assert(odd_detections[i].box.x2 > odd_detections[i].box.x1);
            assert(odd_detections[i].box.y2 > odd_detections[i].box.y1);
        }
        assert(kshira_rad_set_bits(odd_model, KSHIRA_BITS_INT8) == KSHIRA_OK);
        assert(kshira_rad_predict(odd_model, &odd_image, 0.0f, odd_detections, 8,
                                  &odd_count) == KSHIRA_OK);
        assert(odd_count >= 0 && odd_count <= 8);
        for (int i = 0; i < odd_count; ++i) {
            assert(odd_detections[i].box.x1 >= 0.0f && odd_detections[i].box.y1 >= 0.0f);
            assert(odd_detections[i].box.x2 <= 9.0f && odd_detections[i].box.y2 <= 10.0f);
            assert(odd_detections[i].box.x2 > odd_detections[i].box.x1);
            assert(odd_detections[i].box.y2 > odd_detections[i].box.y1);
        }
    }
    {
        unsigned char bad_memory[4096];
        kshira_arena bad_arena;
        kshira_rad_model *bad_model = NULL;
        kshira_rad_spec bad_spec = {INT_MAX, 32, 1, 1, 2, 4, 1};
        assert(kshira_arena_init(&bad_arena, bad_memory, sizeof(bad_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&bad_arena, &bad_spec, &bad_model) == KSHIRA_ERR_ARGUMENT);
    }
    {
        unsigned char tiny_memory[4096];
        kshira_arena tiny_arena;
        kshira_rad_model *tiny_model = NULL;
        kshira_rad_spec tiny_spec = {32, 32, 1, 3, 4, 3, 23};
        size_t start_used;
        size_t start_high_water;
        assert(kshira_arena_init(&tiny_arena, tiny_memory, sizeof(tiny_memory)) == KSHIRA_OK);
        assert(kshira_arena_alloc(&tiny_arena, 16U, 8U) != NULL);
        start_used = kshira_arena_used(&tiny_arena);
        start_high_water = kshira_arena_high_water(&tiny_arena);
        assert(kshira_rad_build(&tiny_arena, &tiny_spec, &tiny_model) == KSHIRA_ERR_MEMORY);
        assert(tiny_model == NULL && kshira_arena_used(&tiny_arena) == start_used &&
               kshira_arena_high_water(&tiny_arena) == start_high_water);
    }
    {
        unsigned char multi_memory[64U << 10];
        float multi_pixels[32 * 32] = {0.0f};
        kshira_arena multi_arena;
        kshira_rad_model *multi_model = NULL;
        kshira_rad_spec multi_spec = {32, 32, 1, 3, 4, 3, 31, 1};
        kshira_image_f32 multi_image = {multi_pixels, 1, 32, 32};
        kshira_rad_box multi_target = {8.0f, 8.0f, 16.0f, 16.0f, 0};
        kshira_rad_train_config multi_config = {
            KSHIRA_BITS_FLOAT, KSHIRA_UPDATE_CHANNELS, NULL, 1.0e-4f
        };
        kshira_rad_detection multi_detections[3];
        float multi_loss = 0.0f;
        int multi_count = 0;
        assert(kshira_arena_init(&multi_arena, multi_memory, sizeof(multi_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&multi_arena, &multi_spec, &multi_model) == KSHIRA_OK);
        assert(kshira_rad_multiscale_ready(multi_model));
        assert(kshira_rad_train_multiscale_step(multi_model, &multi_image, &multi_target, 1,
                                                &multi_config, &multi_loss) == KSHIRA_OK);
        assert(isfinite(multi_loss) && multi_loss >= 0.0f);
        multi_config.update_mode = KSHIRA_UPDATE_BIAS;
        assert(kshira_rad_train_multiscale_step(multi_model, &multi_image, &multi_target, 1,
                                                &multi_config, &multi_loss) == KSHIRA_OK);
        assert(isfinite(multi_loss) && multi_loss >= 0.0f);
        multi_config.bits = KSHIRA_BITS_INT8;
        assert(kshira_rad_train_multiscale_step(multi_model, &multi_image, &multi_target, 2,
                                                &multi_config, &multi_loss) == KSHIRA_OK);
        assert(kshira_rad_predict(multi_model, &multi_image, 0.0f, multi_detections, 3,
                                  &multi_count) == KSHIRA_OK);
        assert(multi_count >= 0 && multi_count <= 3);
        assert(kshira_arena_high_water(&multi_arena) < sizeof(multi_memory));
        assert(kshira_rad_train_multiscale_step(multi_model, &multi_image, &multi_target, 0,
                                                &multi_config, &multi_loss) == KSHIRA_ERR_ARGUMENT);
        assert(kshira_rad_train_multiscale_step(multi_model, &multi_image, &multi_target, 3,
                                                &multi_config, &multi_loss) == KSHIRA_ERR_ARGUMENT);
    }
    {
        unsigned char multi_memory[64U << 10] = {0};
        unsigned char base_memory[64U << 10] = {0};
        float pixels[32 * 32] = {0.0f};
        kshira_arena multi_arena;
        kshira_arena base_arena;
        kshira_rad_model *multi_model = NULL;
        kshira_rad_model *base_model = NULL;
        kshira_rad_spec multi_spec = {32, 32, 1, 3, 4, 3, 37, 1};
        kshira_rad_spec base_spec = {32, 32, 1, 3, 4, 3, 37, 0};
        kshira_image_f32 image = {pixels, 1, 32, 32};
        kshira_rad_detection multi_detections[8];
        kshira_rad_detection base_detections[8];
        int multi_count = 0;
        int base_count = 0;
        assert(kshira_arena_init(&multi_arena, multi_memory, sizeof(multi_memory)) == KSHIRA_OK);
        assert(kshira_arena_init(&base_arena, base_memory, sizeof(base_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&multi_arena, &multi_spec, &multi_model) == KSHIRA_OK);
        assert(kshira_rad_build(&base_arena, &base_spec, &base_model) == KSHIRA_OK);
        assert(kshira_rad_activation_bytes(multi_model) ==
               kshira_rad_activation_bytes(base_model));
        assert(kshira_rad_predict(multi_model, &image, 0.0f, multi_detections, 8,
                                  &multi_count) == KSHIRA_OK);
        assert(kshira_rad_predict(base_model, &image, 0.0f, base_detections, 8,
                                  &base_count) == KSHIRA_OK);
        assert(multi_count == base_count);
        for (int i = 0; i < multi_count; ++i) {
            assert(memcmp(&multi_detections[i], &base_detections[i],
                          sizeof(multi_detections[i])) == 0);
        }
    }
    {
        unsigned char zero_memory[256U << 10] = {0};
        unsigned char far_memory[256U << 10] = {0};
        float zero_pixels[160 * 160] = {0.0f};
        float far_pixels[160 * 160] = {0.0f};
        kshira_arena zero_arena;
        kshira_arena far_arena;
        kshira_rad_model *zero_model = NULL;
        kshira_rad_model *far_model = NULL;
        kshira_rad_spec spec = {160, 160, 1, 1, 4, 3, 71, 1};
        kshira_image_f32 zero_image = {zero_pixels, 1, 160, 160};
        kshira_image_f32 far_image = {far_pixels, 1, 160, 160};
        kshira_rad_box target = {76.0f, 76.0f, 84.0f, 84.0f, 0};
        kshira_rad_train_config config = {
            KSHIRA_BITS_FLOAT, KSHIRA_UPDATE_FREEZE, NULL, 1.0e-4f
        };
        float zero_loss = 0.0f;
        float far_loss = 0.0f;
        for (int y = 104; y < 112; ++y) {
            for (int x = 104; x < 112; ++x) far_pixels[y * 160 + x] = 100.0f;
        }
        assert(kshira_arena_init(&zero_arena, zero_memory, sizeof(zero_memory)) == KSHIRA_OK);
        assert(kshira_arena_init(&far_arena, far_memory, sizeof(far_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&zero_arena, &spec, &zero_model) == KSHIRA_OK);
        assert(kshira_rad_build(&far_arena, &spec, &far_model) == KSHIRA_OK);
        assert(kshira_rad_train_multiscale_step(zero_model, &zero_image, &target, 2,
                                                &config, &zero_loss) == KSHIRA_OK);
        assert(kshira_rad_train_multiscale_step(far_model, &far_image, &target, 2,
                                                &config, &far_loss) == KSHIRA_OK);
        assert(isfinite(zero_loss) && isfinite(far_loss) &&
               fabsf(zero_loss - far_loss) > 1.0e-7f);
    }
}

static void test_session_contract(void) {
    unsigned char memory[16U << 10];
    float pixels[32 * 32] = {0.0f};
    kshira_session session;
    kshira_session_spec spec = {{32, 32, 1, 1, 4, 3, 29}, sizeof(memory)};
    kshira_image_f32 image = {pixels, 1, 32, 32};
    kshira_rad_box target = {8.0f, 8.0f, 16.0f, 16.0f, 0};
    kshira_rad_detection detections[3];
    const kshira_phase_contract *contract;
    float loss = 0.0f;
    int count = 0;
    for (size_t i = 0U; i < sizeof(pixels) / sizeof(pixels[0]); ++i) {
        pixels[i] = 0.25f;
    }
    assert(kshira_session_init(&session, memory, sizeof(memory), &spec) == KSHIRA_OK);
    contract = kshira_session_contract(&session);
    assert(contract != NULL && contract->phase == KSHIRA_PHASE_PRE &&
           contract->bits == KSHIRA_BITS_FLOAT);
    assert(kshira_session_step(&session, &image, &target, FLT_MAX, &loss) ==
           KSHIRA_ERR_RANGE);
    assert(session.phase.steps == 0U);
    assert(kshira_session_set_channel(&session, 0U, 1) == KSHIRA_OK);
    assert(kshira_session_step(&session, &image, &target, 1.0e-4f, &loss) == KSHIRA_OK);
    assert(isfinite(loss) && kshira_session_contract(&session)->phase == KSHIRA_PHASE_PRE);
    assert(kshira_session_transition(&session, KSHIRA_BITS_INT8,
                                     KSHIRA_UPDATE_FULL, 1) == KSHIRA_OK);
    assert(kshira_session_calibrate(&session, &image) == KSHIRA_ERR_ARGUMENT);
    assert(kshira_session_step(&session, &image, &target, 1.0e-5f, &loss) == KSHIRA_OK);
    assert(!kshira_rad_calibration_ready(session.rad));
    assert(kshira_session_transition(&session, KSHIRA_BITS_INT8,
                                     KSHIRA_UPDATE_CHANNELS, 1) == KSHIRA_OK);
    assert(kshira_session_calibrate(&session, &image) == KSHIRA_OK);
    assert(kshira_rad_calibration_ready(session.rad));
    assert(kshira_session_step(&session, &image, &target, 1.0e-5f, &loss) == KSHIRA_OK);
    assert(!kshira_rad_calibration_ready(session.rad));
    assert(kshira_session_predict(&session, &image, 0.0f, detections, 3, &count) == KSHIRA_OK);
    assert(count >= 0 && count <= 3 && isfinite(loss));
    assert(kshira_session_arena_high_water(&session) <= sizeof(memory));
    session.phase.steps = SIZE_MAX;
    assert(kshira_session_step(&session, &image, &target, 1.0e-5f, &loss) == KSHIRA_ERR_RANGE);

    {
        unsigned char bad_memory[4096];
        kshira_session bad_session;
        kshira_session_spec bad_spec = {{INT_MAX, 32, 1, 1, 2, 4, 1}, sizeof(bad_memory)};
        assert(kshira_session_init(&bad_session, bad_memory, sizeof(bad_memory), &bad_spec) ==
               KSHIRA_ERR_ARGUMENT);
        assert(bad_session.rad == NULL && kshira_arena_used(&bad_session.arena) == 0U);
    }
}

static void test_multiscale_session(void) {
    unsigned char memory[64U << 10];
    float pixels[32 * 32] = {0.0f};
    kshira_session session;
    kshira_session_spec spec = {{32, 32, 1, 3, 4, 3, 43, 1}, sizeof(memory)};
    kshira_image_f32 image = {pixels, 1, 32, 32};
    kshira_rad_box target = {8.0f, 8.0f, 16.0f, 16.0f, 0};
    float loss = 0.0f;
    for (size_t i = 0U; i < sizeof(pixels) / sizeof(pixels[0]); ++i) pixels[i] = 0.5f;
    assert(kshira_session_init(&session, memory, sizeof(memory), &spec) == KSHIRA_OK);
    assert(kshira_rad_multiscale_ready(session.rad));
    assert(kshira_session_set_channel(&session, 0U, 1) == KSHIRA_OK);
    assert(kshira_session_transition(&session, KSHIRA_BITS_INT8,
                                     KSHIRA_UPDATE_FULL, 1) == KSHIRA_OK);
    assert(kshira_session_step(&session, &image, &target, 1.0e-4f, &loss) == KSHIRA_OK);
    assert(kshira_session_transition(&session, KSHIRA_BITS_INT8,
                                     KSHIRA_UPDATE_CHANNELS, 1) == KSHIRA_OK);
    assert(kshira_session_calibrate(&session, &image) == KSHIRA_OK);
    assert(kshira_session_multiscale_step(&session, &image, &target, 1,
                                          1.0e-4f, &loss) == KSHIRA_OK);
    assert(isfinite(loss) && loss >= 0.0f);
    assert(kshira_session_multiscale_step(&session, &image, &target, 2,
                                          1.0e-4f, &loss) == KSHIRA_OK);
    assert(isfinite(loss) && loss >= 0.0f);
    session.phase.steps = SIZE_MAX;
    loss = -1.0f;
    assert(kshira_session_multiscale_step(&session, &image, &target, 1,
                                          1.0e-4f, &loss) == KSHIRA_ERR_RANGE);
    assert(loss == -1.0f);
    assert(kshira_session_arena_high_water(&session) <= sizeof(memory));
}

static void test_domain_curriculum(void) {
    enum { WIDTH = 32, HEIGHT = 32, SAMPLES = 7 };
    float image[WIDTH * HEIGHT];
    int counts[KSHIRA_DOMAIN_COUNT] = {0};
    kshira_domain_stream stream;
    kshira_domain_spec spec = {WIDTH, HEIGHT, 1, KSHIRA_DOMAIN_COUNT, SAMPLES, 41U};
    kshira_rad_box target;
    int domain;
    assert(kshira_domain_init(&stream, &spec) == KSHIRA_OK);
    assert(kshira_domain_total(&stream) == (size_t)KSHIRA_DOMAIN_COUNT * SAMPLES);
    while (kshira_domain_index(&stream) < kshira_domain_total(&stream)) {
        assert(kshira_domain_next(&stream, image, sizeof(image) / sizeof(image[0]),
                                  &target, &domain) == KSHIRA_OK);
        assert(domain >= 0 && domain < KSHIRA_DOMAIN_COUNT);
        assert(target.x1 >= 0.0f && target.y1 >= 0.0f && target.x2 <= WIDTH &&
               target.y2 <= HEIGHT && target.x2 > target.x1 && target.y2 > target.y1);
        ++counts[domain];
    }
    assert(kshira_domain_next(&stream, image, sizeof(image) / sizeof(image[0]),
                              &target, &domain) == KSHIRA_ERR_RANGE);
    for (int i = 0; i < KSHIRA_DOMAIN_COUNT; ++i) assert(counts[i] == SAMPLES);
}

static void test_proxy_metrics(void) {
    kshira_proxy_metrics metrics;
    kshira_rad_box target = {0.0f, 0.0f, 10.0f, 10.0f, 2};
    kshira_rad_detection detections[2] = {
        {{2.0f, 2.0f, 8.0f, 8.0f, 2}, 0.9f, 0.8f},
        {{20.0f, 20.0f, 24.0f, 24.0f, 1}, 0.8f, 0.7f}
    };
    kshira_proxy_metrics_init(&metrics);
    assert(fabsf(kshira_box_iou(&target, &detections[0].box) - 0.36f) < 1.0e-6f);
    assert(kshira_proxy_metrics_add(&metrics, &target, detections, 2) == KSHIRA_OK);
    assert(kshira_proxy_metrics_add(&metrics, &target, NULL, 0) == KSHIRA_OK);
    assert(fabsf(kshira_proxy_mean_iou(&metrics) - 0.18f) < 1.0e-6f);
    assert(fabsf(kshira_proxy_class_accuracy(&metrics) - 0.5f) < 1.0e-6f);
    {
        kshira_rad_box invalid = {1.0f, 1.0f, 1.0f, 2.0f, 2};
        assert(kshira_proxy_metrics_add(&metrics, &invalid, detections, 1) ==
               KSHIRA_ERR_ARGUMENT);
        detections[1].box.x2 = detections[1].box.x1;
        assert(kshira_proxy_metrics_add(&metrics, &target, detections, 2) ==
               KSHIRA_ERR_ARGUMENT);
    }
}

static void test_rad_state_roundtrip(void) {
    unsigned char first_memory[64U << 10];
    unsigned char second_memory[64U << 10];
    unsigned char state[4096];
    float pixels[32 * 32] = {0.0f};
    kshira_arena first_arena;
    kshira_arena second_arena;
    kshira_rad_model *first = NULL;
    kshira_rad_model *second = NULL;
    kshira_rad_spec spec = {32, 32, 1, 2, 4, 4, 55, 1};
    kshira_image_f32 image = {pixels, 1, 32, 32};
    kshira_rad_box target = {8.0f, 8.0f, 20.0f, 20.0f, 1};
    kshira_rad_train_config config = {
        KSHIRA_BITS_INT8, KSHIRA_UPDATE_FULL, NULL, 1.0e-4f
    };
    kshira_rad_detection first_detections[4];
    kshira_rad_detection second_detections[4];
    float loss = 0.0f;
    size_t written = 0U;
    int first_count = 0;
    int second_count = 0;
    for (int y = 8; y < 20; ++y) {
        for (int x = 8; x < 20; ++x) pixels[y * 32 + x] = 0.75f;
    }
    assert(kshira_arena_init(&first_arena, first_memory, sizeof(first_memory)) == KSHIRA_OK);
    assert(kshira_arena_init(&second_arena, second_memory, sizeof(second_memory)) == KSHIRA_OK);
    assert(kshira_rad_build(&first_arena, &spec, &first) == KSHIRA_OK);
    assert(kshira_rad_build(&second_arena, &spec, &second) == KSHIRA_OK);
    assert(kshira_rad_train_step(first, &image, &target, &config, &loss) == KSHIRA_OK);
    assert(kshira_rad_state_bytes(first) > 0U &&
           kshira_rad_state_bytes(first) <= sizeof(state));
    assert(kshira_rad_export_state(first, state, 1U, &written) == KSHIRA_ERR_MEMORY &&
           written == 0U);
    assert(kshira_rad_export_state(first, state, sizeof(state), &written) == KSHIRA_OK &&
           written == kshira_rad_state_bytes(first));
    assert(kshira_rad_import_state(second, state, written) == KSHIRA_OK);
    assert(kshira_rad_predict(first, &image, 0.0f, first_detections, 4,
                              &first_count) == KSHIRA_OK);
    assert(kshira_rad_predict(second, &image, 0.0f, second_detections, 4,
                              &second_count) == KSHIRA_OK);
    assert(first_count == second_count);
    for (int i = 0; i < first_count; ++i) {
        assert(memcmp(&first_detections[i], &second_detections[i],
                      sizeof(first_detections[i])) == 0);
    }
    state[0] ^= 0xffU;
    assert(kshira_rad_import_state(second, state, written) == KSHIRA_ERR_RANGE);
    assert(kshira_rad_predict(second, &image, 0.0f, second_detections, 4,
                              &second_count) == KSHIRA_OK);
    assert(first_count == second_count);
    for (int i = 0; i < first_count; ++i) {
        assert(memcmp(&first_detections[i], &second_detections[i],
                      sizeof(first_detections[i])) == 0);
    }
}

static void test_background_training(void) {
    unsigned char memory[64U << 10];
    float pixels[32 * 32] = {0.0f};
    kshira_arena arena;
    kshira_rad_model *model = NULL;
    kshira_rad_spec spec = {32, 32, 1, 1, 4, 4, 71, 1};
    kshira_image_f32 image = {pixels, 1, 32, 32};
    kshira_rad_train_config config = {
        KSHIRA_BITS_FLOAT, KSHIRA_UPDATE_FULL, NULL, 0.1f
    };
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    assert(kshira_arena_init(&arena, memory, sizeof(memory)) == KSHIRA_OK);
    assert(kshira_rad_build(&arena, &spec, &model) == KSHIRA_OK);
    assert(kshira_rad_train_background_step(model, &image, -1, 0,
                                             &config, &last_loss) ==
           KSHIRA_ERR_ARGUMENT);
    assert(kshira_rad_train_background_step(model, &image, 0, 0,
                                             &config, &first_loss) == KSHIRA_OK);
    for (int step = 0; step < 20; ++step) {
        assert(kshira_rad_train_background_step(model, &image, step % 8,
                                                 (step * 3) % 8, &config,
                                                 &last_loss) == KSHIRA_OK);
    }
    assert(isfinite(first_loss) && isfinite(last_loss) && last_loss < first_loss);
    config.bits = KSHIRA_BITS_INT8;
    assert(kshira_rad_train_background_step(model, &image, 1, 1,
                                             &config, &last_loss) == KSHIRA_OK);
}

int main(void) {
    test_arena();
    test_quant();
    test_qas_and_sparse();
    test_phases();
    test_rad_top_k();
    test_session_contract();
    test_multiscale_session();
    test_domain_curriculum();
    test_proxy_metrics();
    test_rad_state_roundtrip();
    test_background_training();
    puts("all kshira tests passed");
    return 0;
}
