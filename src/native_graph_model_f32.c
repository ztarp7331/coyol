#include "native_graph_model.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define NGF32_REG_MAX 16
#define NGF32_LEVELS 3
#define NGF32_MODULES 8
#define NGF32_MAX_REPEATS 6

typedef struct {
    int32_t channels[5];
    int32_t repeats[NGF32_MODULES];
} NGF32Geometry;

typedef struct {
    const NGConv2D *convs;
    int32_t count;
    int32_t next;
} NGF32Cursor;

static int f32_geometry(NGModelScale scale, NGF32Geometry *geometry) {
    if (!geometry) return -1;
    switch (scale) {
        case NG_SCALE_N:
            geometry->channels[0] = 16; geometry->channels[1] = 32; geometry->channels[2] = 64;
            geometry->channels[3] = 128; geometry->channels[4] = 256;
            geometry->repeats[0] = 1; geometry->repeats[1] = 2; geometry->repeats[2] = 2; geometry->repeats[3] = 1;
            geometry->repeats[4] = 1; geometry->repeats[5] = 1; geometry->repeats[6] = 1; geometry->repeats[7] = 1;
            return 0;
        case NG_SCALE_S:
            geometry->channels[0] = 32; geometry->channels[1] = 64; geometry->channels[2] = 128;
            geometry->channels[3] = 256; geometry->channels[4] = 512;
            geometry->repeats[0] = 1; geometry->repeats[1] = 2; geometry->repeats[2] = 2; geometry->repeats[3] = 1;
            geometry->repeats[4] = 1; geometry->repeats[5] = 1; geometry->repeats[6] = 1; geometry->repeats[7] = 1;
            return 0;
        case NG_SCALE_M:
            geometry->channels[0] = 48; geometry->channels[1] = 96; geometry->channels[2] = 192;
            geometry->channels[3] = 384; geometry->channels[4] = 576;
            geometry->repeats[0] = 2; geometry->repeats[1] = 4; geometry->repeats[2] = 4; geometry->repeats[3] = 2;
            geometry->repeats[4] = 2; geometry->repeats[5] = 2; geometry->repeats[6] = 2; geometry->repeats[7] = 2;
            return 0;
        case NG_SCALE_L:
            geometry->channels[0] = 64; geometry->channels[1] = 128; geometry->channels[2] = 256;
            geometry->channels[3] = 512; geometry->channels[4] = 512;
            geometry->repeats[0] = 3; geometry->repeats[1] = 6; geometry->repeats[2] = 6; geometry->repeats[3] = 3;
            geometry->repeats[4] = 3; geometry->repeats[5] = 3; geometry->repeats[6] = 3; geometry->repeats[7] = 3;
            return 0;
        case NG_SCALE_X:
            geometry->channels[0] = 80; geometry->channels[1] = 160; geometry->channels[2] = 320;
            geometry->channels[3] = 512; geometry->channels[4] = 512;
            geometry->repeats[0] = 3; geometry->repeats[1] = 6; geometry->repeats[2] = 6; geometry->repeats[3] = 3;
            geometry->repeats[4] = 3; geometry->repeats[5] = 3; geometry->repeats[6] = 3; geometry->repeats[7] = 3;
            return 0;
        default:
            return -1;
    }
}

static int f32_tensor(NGArena *arena, int32_t c, int32_t h, int32_t w,
                      NGF32Tensor *tensor) {
    int64_t bytes;
    if (!arena || !tensor || !arena->data || arena->capacity < 0 || arena->used < 0 ||
        c <= 0 || h <= 0 || w <= 0) return -1;
    bytes = (int64_t)c * h * w * (int64_t)sizeof(float);
    if (bytes > INT32_MAX || bytes > arena->capacity - arena->used) return -1;
    tensor->data = (float *)(void *)(arena->data + arena->used);
    tensor->n = 1;
    tensor->c = c;
    tensor->h = h;
    tensor->w = w;
    arena->used += (int32_t)bytes;
    return 0;
}

static int32_t f32_index(const NGF32Tensor *tensor, int32_t c, int32_t y, int32_t x) {
    return (c * tensor->h + y) * tensor->w + x;
}

static const NGConv2D *f32_take(NGF32Cursor *cursor) {
    if (!cursor || !cursor->convs || cursor->next < 0 || cursor->next >= cursor->count) return NULL;
    return &cursor->convs[cursor->next++];
}

static float f32_silu(float value) {
    return value / (1.0f + expf(-value));
}

static int f32_conv(NGF32Cursor *cursor, const NGF32Tensor *input,
                    NGF32Tensor *output, NGArena *arena) {
    const NGConv2D *layer = f32_take(cursor);
    int32_t channels_per_group;
    int32_t outputs_per_group;
    int32_t expected_h;
    int32_t expected_w;
    if (!layer || !input || !output || !input->data || !layer->weights_f32 ||
        !layer->bias_f32 || input->n != 1 || input->c != layer->in_channels ||
        layer->groups <= 0 || layer->in_channels % layer->groups ||
        layer->out_channels % layer->groups || layer->kernel <= 0 || layer->stride <= 0)
        return -1;
    expected_h = (input->h + 2 * layer->padding - layer->kernel) / layer->stride + 1;
    expected_w = (input->w + 2 * layer->padding - layer->kernel) / layer->stride + 1;
    if (output->data == NULL) {
        if (f32_tensor(arena, layer->out_channels, expected_h, expected_w, output)) {
            return -1;
        }
    } else if (output->n != 1 || output->c != layer->out_channels ||
               output->h != expected_h || output->w != expected_w) {
        return -1;
    }
    channels_per_group = layer->in_channels / layer->groups;
    outputs_per_group = layer->out_channels / layer->groups;
    for (int32_t oc = 0; oc < output->c; ++oc) {
        int32_t group = oc / outputs_per_group;
        for (int32_t oy = 0; oy < output->h; ++oy) {
            for (int32_t ox = 0; ox < output->w; ++ox) {
                float acc = layer->bias_f32[oc];
                for (int32_t ic = 0; ic < channels_per_group; ++ic) {
                    for (int32_t ky = 0; ky < layer->kernel; ++ky) {
                        int32_t iy = oy * layer->stride + ky - layer->padding;
                        if (iy < 0 || iy >= input->h) continue;
                        for (int32_t kx = 0; kx < layer->kernel; ++kx) {
                            int32_t ix = ox * layer->stride + kx - layer->padding;
                            int32_t wi;
                            if (ix < 0 || ix >= input->w) continue;
                            wi = (((oc * channels_per_group + ic) * layer->kernel + ky) *
                                  layer->kernel + kx);
                            acc += input->data[f32_index(input, group * channels_per_group + ic, iy, ix)] *
                                   layer->weights_f32[wi];
                        }
                    }
                }
                if (layer->activation == NG_ACT_SILU) acc = f32_silu(acc);
                output->data[f32_index(output, oc, oy, ox)] = acc;
            }
        }
    }
    return 0;
}

static int f32_view(const NGF32Tensor *source, int32_t first, int32_t count,
                    NGF32Tensor *view) {
    int64_t offset;
    if (!source || !source->data || !view || first < 0 || count <= 0 ||
        first > source->c - count) return -1;
    offset = (int64_t)first * source->h * source->w;
    if (offset > INT32_MAX) return -1;
    *view = (NGF32Tensor){source->data + offset, 1, count, source->h, source->w};
    return 0;
}

static int f32_add(const NGF32Tensor *left, const NGF32Tensor *right,
                   NGF32Tensor *output) {
    int64_t elements;
    if (!left || !right || !output || !left->data || !right->data || !output->data ||
        left->c != right->c || left->h != right->h || left->w != right->w ||
        output->c != left->c || output->h != left->h || output->w != left->w) return -1;
    elements = (int64_t)left->c * left->h * left->w;
    for (int64_t i = 0; i < elements; ++i) output->data[i] = left->data[i] + right->data[i];
    return 0;
}

static int f32_concat(const NGF32Tensor *inputs, int32_t count,
                      NGF32Tensor *output) {
    int32_t offset = 0;
    if (!inputs || !output || !output->data || count < 1) return -1;
    for (int32_t i = 0; i < count; ++i) {
        if (!inputs[i].data || inputs[i].h != output->h || inputs[i].w != output->w) return -1;
        for (int32_t c = 0; c < inputs[i].c; ++c)
            for (int32_t y = 0; y < output->h; ++y)
                for (int32_t x = 0; x < output->w; ++x)
                    output->data[f32_index(output, offset + c, y, x)] =
                        inputs[i].data[f32_index(&inputs[i], c, y, x)];
        offset += inputs[i].c;
    }
    return offset == output->c ? 0 : -1;
}

static int f32_upsample(const NGF32Tensor *input, NGArena *arena, NGF32Tensor *output) {
    if (!input || !arena || !output || !input->data ||
        f32_tensor(arena, input->c, input->h * 2, input->w * 2, output)) return -1;
    for (int32_t c = 0; c < output->c; ++c)
        for (int32_t y = 0; y < output->h; ++y)
            for (int32_t x = 0; x < output->w; ++x)
                output->data[f32_index(output, c, y, x)] =
                    input->data[f32_index(input, c, y / 2, x / 2)];
    return 0;
}

static int f32_maxpool(const NGF32Tensor *input, NGF32Tensor *output,
                       int32_t kernel, int32_t stride, int32_t padding,
                       NGArena *arena) {
    if (!input || !output || !arena || !input->data ||
        f32_tensor(arena, input->c, input->h, input->w, output)) return -1;
    for (int32_t c = 0; c < output->c; ++c)
        for (int32_t y = 0; y < output->h; ++y)
            for (int32_t x = 0; x < output->w; ++x) {
                float maximum = -FLT_MAX;
                for (int32_t ky = 0; ky < kernel; ++ky)
                    for (int32_t kx = 0; kx < kernel; ++kx) {
                        int32_t iy = y * stride + ky - padding;
                        int32_t ix = x * stride + kx - padding;
                        if (iy >= 0 && iy < input->h && ix >= 0 && ix < input->w) {
                            float value = input->data[f32_index(input, c, iy, ix)];
                            if (value > maximum) maximum = value;
                        }
                    }
                output->data[f32_index(output, c, y, x)] = maximum;
            }
    return 0;
}

static int f32_c2f(NGF32Cursor *cursor, const NGF32Tensor *input,
                   NGF32Tensor *output, int32_t repeats, int shortcut,
                   NGArena *arena) {
    NGF32Tensor split = {0};
    NGF32Tensor branches[2 + NGF32_MAX_REPEATS] = {{0}};
    NGF32Tensor concat = {0};
    int32_t hidden;
    if (!input || !output || !arena || repeats < 1 || repeats > NGF32_MAX_REPEATS) return -1;
    if (f32_conv(cursor, input, &split, arena) || split.c <= 0 || split.c % 2) return -1;
    hidden = split.c / 2;
    if (f32_view(&split, 0, hidden, &branches[0]) ||
        f32_view(&split, hidden, hidden, &branches[1])) return -1;
    for (int32_t i = 0; i < repeats; ++i) {
        NGF32Tensor first = {0};
        if (f32_conv(cursor, &branches[i + 1], &first, arena) ||
            f32_conv(cursor, &first, &branches[i + 2], arena)) return -1;
        if (shortcut && f32_add(&branches[i + 1], &branches[i + 2], &branches[i + 2])) return -1;
    }
    if (f32_tensor(arena, (2 + repeats) * hidden, input->h, input->w, &concat) ||
        f32_concat(branches, 2 + repeats, &concat)) return -1;
    return f32_conv(cursor, &concat, output, arena);
}

static int f32_sppf(NGF32Cursor *cursor, const NGF32Tensor *input,
                    NGF32Tensor *output, NGArena *arena) {
    NGF32Tensor x = {0}, y1 = {0}, y2 = {0}, y3 = {0}, concat = {0};
    NGF32Tensor inputs[4];
    if (f32_conv(cursor, input, &x, arena) ||
        f32_maxpool(&x, &y1, 5, 1, 2, arena) ||
        f32_maxpool(&y1, &y2, 5, 1, 2, arena) ||
        f32_maxpool(&y2, &y3, 5, 1, 2, arena)) return -1;
    inputs[0] = x; inputs[1] = y1; inputs[2] = y2; inputs[3] = y3;
    if (f32_tensor(arena, 4 * x.c, x.h, x.w, &concat) || f32_concat(inputs, 4, &concat)) return -1;
    return f32_conv(cursor, &concat, output, arena);
}

static int f32_detect(NGF32Cursor *cursor, const NGF32Tensor *features,
                      int32_t class_count, NGArena *arena,
                      NGF32ModelOutput *output) {
    for (int32_t i = 0; i < NGF32_LEVELS; ++i) {
        NGF32Tensor reg1 = {0}, reg2 = {0}, reg3 = {0};
        NGF32Tensor cls1 = {0}, cls2 = {0}, cls3 = {0};
        NGF32Tensor branches[2];
        if (!features[i].data || f32_tensor(arena, 4 * NGF32_REG_MAX + class_count,
                                            features[i].h, features[i].w,
                                            &output->prediction[i])) return -1;
        if (f32_conv(cursor, &features[i], &reg1, arena) ||
            f32_conv(cursor, &reg1, &reg2, arena) ||
            f32_conv(cursor, &reg2, &reg3, arena) ||
            f32_conv(cursor, &features[i], &cls1, arena) ||
            f32_conv(cursor, &cls1, &cls2, arena) ||
            f32_conv(cursor, &cls2, &cls3, arena)) return -1;
        output->class_feature[i] = cls2;
        branches[0] = reg3; branches[1] = cls3;
        if (f32_concat(branches, 2, &output->prediction[i])) return -1;
    }
    output->stride[0] = 8; output->stride[1] = 16; output->stride[2] = 32;
    return 0;
}

int ng_native_graph_forward_f32(const NGModelConfig *config, const NGModelWeights *weights,
                                const float *input, int32_t input_channels,
                                int32_t height, int32_t width, NGArena *arena,
                                NGF32ModelOutput *output) {
    NGF32Geometry geometry;
    NGF32Cursor cursor;
    NGF32Tensor x[22] = {{0}};
    NGF32Tensor concat_inputs[2];
    NGF32Tensor detect_inputs[NGF32_LEVELS];
    int32_t checkpoint;
    int32_t scratch;
    if (!config || !weights || !weights->convs || !input || !arena || !output ||
        !arena->data || input_channels != config->input_channels || height <= 0 || width <= 0 ||
        height % 32 || width % 32 || config->class_count <= 0 ||
        weights->conv_count != ng_native_graph_conv_count(config->scale) ||
        f32_geometry(config->scale, &geometry)) return -1;
    checkpoint = arena->used;
    for (int32_t i = 0; i < NGF32_LEVELS; ++i) {
        output->prediction[i].data = NULL;
        output->prediction[i].n = output->prediction[i].c =
            output->prediction[i].h = output->prediction[i].w = 0;
        output->class_feature[i].data = NULL;
        output->class_feature[i].n = output->class_feature[i].c =
            output->class_feature[i].h = output->class_feature[i].w = 0;
        output->stride[i] = 0;
    }
    cursor = (NGF32Cursor){weights->convs, weights->conv_count, 0};
    if (f32_conv(&cursor, &(NGF32Tensor){(float *)input, 1, input_channels, height, width}, &x[0], arena) ||
        f32_conv(&cursor, &x[0], &x[1], arena)) goto fail;
    if (f32_tensor(arena, geometry.channels[1], x[1].h, x[1].w, &x[2])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[1], &x[2], geometry.repeats[0], 1, arena)) goto fail;
    arena->used = scratch;
    if (f32_conv(&cursor, &x[2], &x[3], arena)) goto fail;
    if (f32_tensor(arena, geometry.channels[2], x[3].h, x[3].w, &x[4])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[3], &x[4], geometry.repeats[1], 1, arena)) goto fail;
    arena->used = scratch;
    if (f32_conv(&cursor, &x[4], &x[5], arena)) goto fail;
    if (f32_tensor(arena, geometry.channels[3], x[5].h, x[5].w, &x[6])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[5], &x[6], geometry.repeats[2], 1, arena)) goto fail;
    arena->used = scratch;
    if (f32_conv(&cursor, &x[6], &x[7], arena)) goto fail;
    if (f32_tensor(arena, geometry.channels[4], x[7].h, x[7].w, &x[8])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[7], &x[8], geometry.repeats[3], 1, arena)) goto fail;
    arena->used = scratch;
    if (f32_tensor(arena, geometry.channels[4], x[8].h, x[8].w, &x[9])) goto fail;
    scratch = arena->used;
    if (f32_sppf(&cursor, &x[8], &x[9], arena)) goto fail;
    arena->used = scratch;
    if (f32_upsample(&x[9], arena, &x[10])) goto fail;
    concat_inputs[0] = x[10]; concat_inputs[1] = x[6];
    if (f32_tensor(arena, x[10].c + x[6].c, x[10].h, x[10].w, &x[11]) ||
        f32_concat(concat_inputs, 2, &x[11])) goto fail;
    if (f32_tensor(arena, geometry.channels[3], x[11].h, x[11].w, &x[12])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[11], &x[12], geometry.repeats[4], 0, arena)) goto fail;
    arena->used = scratch;
    if (f32_upsample(&x[12], arena, &x[13])) goto fail;
    concat_inputs[0] = x[13]; concat_inputs[1] = x[4];
    if (f32_tensor(arena, x[13].c + x[4].c, x[13].h, x[13].w, &x[14]) ||
        f32_concat(concat_inputs, 2, &x[14])) goto fail;
    if (f32_tensor(arena, geometry.channels[2], x[14].h, x[14].w, &x[15])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[14], &x[15], geometry.repeats[5], 0, arena)) goto fail;
    arena->used = scratch;
    if (f32_conv(&cursor, &x[15], &x[16], arena)) goto fail;
    concat_inputs[0] = x[16]; concat_inputs[1] = x[12];
    if (f32_tensor(arena, x[16].c + x[12].c, x[16].h, x[16].w, &x[17]) ||
        f32_concat(concat_inputs, 2, &x[17])) goto fail;
    if (f32_tensor(arena, geometry.channels[3], x[17].h, x[17].w, &x[18])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[17], &x[18], geometry.repeats[6], 0, arena)) goto fail;
    arena->used = scratch;
    if (f32_conv(&cursor, &x[18], &x[19], arena)) goto fail;
    concat_inputs[0] = x[19]; concat_inputs[1] = x[9];
    if (f32_tensor(arena, x[19].c + x[9].c, x[19].h, x[19].w, &x[20]) ||
        f32_concat(concat_inputs, 2, &x[20])) goto fail;
    if (f32_tensor(arena, geometry.channels[4], x[20].h, x[20].w, &x[21])) goto fail;
    scratch = arena->used;
    if (f32_c2f(&cursor, &x[20], &x[21], geometry.repeats[7], 0, arena)) goto fail;
    arena->used = scratch;
    detect_inputs[0] = x[15]; detect_inputs[1] = x[18]; detect_inputs[2] = x[21];
    if (f32_detect(&cursor, detect_inputs, config->class_count, arena, output) ||
        cursor.next != cursor.count) goto fail;
    return 0;
fail:
    arena->used = checkpoint;
    return -1;
}
