#include "kshira/core.h"
#include "kshira/phase.h"
#include "kshira/quant.h"
#include "kshira/rad.h"
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
}

static void test_phases(void) {
    kshira_phase_contract contract = {KSHIRA_PHASE_PRE, KSHIRA_BITS_INT8,
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
    }
    {
        unsigned char bad_memory[4096];
        kshira_arena bad_arena;
        kshira_rad_model *bad_model = NULL;
        kshira_rad_spec bad_spec = {INT_MAX, 32, 1, 1, 2, 4, 1};
        assert(kshira_arena_init(&bad_arena, bad_memory, sizeof(bad_memory)) == KSHIRA_OK);
        assert(kshira_rad_build(&bad_arena, &bad_spec, &bad_model) == KSHIRA_ERR_ARGUMENT);
    }
}

int main(void) {
    test_arena();
    test_quant();
    test_qas_and_sparse();
    test_phases();
    test_rad_top_k();
    puts("all kshira tests passed");
    return 0;
}
