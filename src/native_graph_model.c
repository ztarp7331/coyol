#include "native_graph_model.h"

#include <limits.h>
#include <stddef.h>

#define NG_REG_MAX 16
#define NG_DETECT_LEVELS 3
#define NG_C2F_MODULES 8
#define NG_MAX_C2F_REPEATS 6

typedef struct {
    int32_t channels[5];
    int32_t repeats[NG_C2F_MODULES];
} NGGeometry;

typedef struct {
    const NGConv2D *convs;
    int32_t count;
    int32_t next;
} NGConvCursor;

static int geometry_for_scale(NGModelScale scale, NGGeometry *geometry) {
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

void ng_arena_init(NGArena *arena, int8_t *data, int32_t capacity) {
    if (!arena) return;
    arena->data = data;
    arena->capacity = capacity;
    arena->used = 0;
}

void ng_arena_reset(NGArena *arena) {
    if (arena) arena->used = 0;
}

int32_t ng_native_graph_conv_count(NGModelScale scale) {
    switch (scale) {
        case NG_SCALE_N:
        case NG_SCALE_S:
            return 63;
        case NG_SCALE_M:
            return 83;
        case NG_SCALE_L:
        case NG_SCALE_X:
            return 103;
        default:
            return -1;
    }
}

static int arena_tensor(NGArena *arena, int32_t c, int32_t h, int32_t w, NGTensor *tensor) {
    int64_t bytes;
    if (!arena || !tensor || !arena->data || arena->capacity < 0 || arena->used < 0 || c <= 0 || h <= 0 || w <= 0) return -1;
    bytes = (int64_t)c * h * w;
    if (bytes > INT32_MAX || bytes > arena->capacity - arena->used) return -1;
    tensor->data = arena->data + arena->used;
    tensor->n = 1;
    tensor->c = c;
    tensor->h = h;
    tensor->w = w;
    arena->used += (int32_t)bytes;
    return 0;
}

static const NGConv2D *take_conv(NGConvCursor *cursor) {
    if (!cursor || !cursor->convs || cursor->next >= cursor->count) return NULL;
    return &cursor->convs[cursor->next++];
}

static int conv_new(NGConvCursor *cursor, const NGTensor *input, NGArena *arena, NGTensor *output) {
    const NGConv2D *conv = take_conv(cursor);
    int32_t h;
    int32_t w;
    if (!conv || !input || input->n != 1 || conv->kernel <= 0 || conv->stride <= 0) return -1;
    h = (input->h + 2 * conv->padding - conv->kernel) / conv->stride + 1;
    w = (input->w + 2 * conv->padding - conv->kernel) / conv->stride + 1;
    if (arena_tensor(arena, conv->out_channels, h, w, output)) return -1;
    return ng_conv2d_s8(input, output, conv);
}

static int tensor_channel_view_n1(const NGTensor *source, int32_t first_channel, int32_t channels, NGTensor *view) {
    int64_t offset;
    if (!source || !source->data || !view || source->n != 1 || first_channel < 0 || channels <= 0 ||
        first_channel > source->c - channels) return -1;
    offset = (int64_t)first_channel * source->h * source->w;
    if (offset > INT32_MAX) return -1;
    view->data = source->data + offset;
    view->n = 1;
    view->c = channels;
    view->h = source->h;
    view->w = source->w;
    return 0;
}

static int c2f_forward(NGConvCursor *cursor, const NGTensor *input, NGTensor *output, int32_t repeats, int shortcut,
                       NGArena *arena) {
    const NGConv2D *conv;
    NGTensor split;
    NGTensor branches[2 + NG_MAX_C2F_REPEATS];
    NGTensor concat;
    int32_t hidden;

    if (!input || !output || !arena || repeats < 1 || repeats > NG_MAX_C2F_REPEATS || input->n != 1) return -1;
    conv = take_conv(cursor);
    if (!conv || conv->out_channels <= 0 || conv->out_channels % 2) return -1;
    if (arena_tensor(arena, conv->out_channels, input->h, input->w, &split) || ng_conv2d_s8(input, &split, conv)) return -1;
    hidden = conv->out_channels / 2;
    if (tensor_channel_view_n1(&split, 0, hidden, &branches[0]) ||
        tensor_channel_view_n1(&split, hidden, hidden, &branches[1])) return -1;

    for (int32_t i = 0; i < repeats; ++i) {
        NGTensor first;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &first) ||
            ng_conv2d_s8(&branches[i + 1], &first, conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &branches[i + 2]) ||
            ng_conv2d_s8(&first, &branches[i + 2], conv)) return -1;
        if (shortcut && ng_add_s8(&branches[i + 1], &branches[i + 2], &branches[i + 2])) return -1;
    }
    if (arena_tensor(arena, (2 + repeats) * hidden, input->h, input->w, &concat) ||
        ng_concat_channels_s8(branches, 2 + repeats, &concat)) return -1;
    conv = take_conv(cursor);
    return conv ? ng_conv2d_s8(&concat, output, conv) : -1;
}

static int sppf_forward(NGConvCursor *cursor, const NGTensor *input, NGTensor *output, NGArena *arena) {
    const NGConv2D *conv;
    NGTensor x, y1, y2, y3, concat;
    NGTensor inputs[4];
    if (!input || !output || !arena || input->n != 1) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &x) || ng_conv2d_s8(input, &x, conv)) return -1;
    if (arena_tensor(arena, x.c, x.h, x.w, &y1) || ng_maxpool2d_s8(&x, &y1, 5, 1, 2) ||
        arena_tensor(arena, x.c, x.h, x.w, &y2) || ng_maxpool2d_s8(&y1, &y2, 5, 1, 2) ||
        arena_tensor(arena, x.c, x.h, x.w, &y3) || ng_maxpool2d_s8(&y2, &y3, 5, 1, 2)) return -1;
    inputs[0] = x; inputs[1] = y1; inputs[2] = y2; inputs[3] = y3;
    if (arena_tensor(arena, 4 * x.c, x.h, x.w, &concat) || ng_concat_channels_s8(inputs, 4, &concat)) return -1;
    conv = take_conv(cursor);
    return conv ? ng_conv2d_s8(&concat, output, conv) : -1;
}

static int upsample_new(const NGTensor *input, NGArena *arena, NGTensor *output) {
    if (!input || !arena || !output || input->n != 1 || arena_tensor(arena, input->c, input->h * 2, input->w * 2, output)) return -1;
    return ng_upsample_nearest2_s8(input, output);
}

static int concat_new(const NGTensor *inputs, int32_t count, NGArena *arena, NGTensor *output) {
    int32_t channels = 0;
    if (!inputs || !arena || !output || count < 1 || inputs[0].n != 1) return -1;
    for (int32_t i = 0; i < count; ++i) {
        if (inputs[i].n != 1 || inputs[i].h != inputs[0].h || inputs[i].w != inputs[0].w || inputs[i].c <= 0) return -1;
        if (channels > INT32_MAX - inputs[i].c) return -1;
        channels += inputs[i].c;
    }
    if (arena_tensor(arena, channels, inputs[0].h, inputs[0].w, output)) return -1;
    return ng_concat_channels_s8(inputs, count, output);
}

static int detect_forward(NGConvCursor *cursor, const NGTensor *features, int32_t class_count, NGArena *arena,
                          NGModelOutput *output) {
    for (int32_t i = 0; i < NG_DETECT_LEVELS; ++i) {
        NGTensor reg1, reg2, reg3, cls1, cls2, cls3;
        NGTensor branches[2];
        const NGConv2D *conv;
        if (features[i].n != 1 || arena_tensor(arena, 4 * NG_REG_MAX + class_count, features[i].h, features[i].w,
                                                 &output->prediction[i])) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, features[i].h, features[i].w, &reg1) ||
            ng_conv2d_s8(&features[i], &reg1, conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, features[i].h, features[i].w, &reg2) ||
            ng_conv2d_s8(&reg1, &reg2, conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, features[i].h, features[i].w, &reg3) ||
            ng_conv2d_s8(&reg2, &reg3, conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, features[i].h, features[i].w, &cls1) ||
            ng_conv2d_s8(&features[i], &cls1, conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, features[i].h, features[i].w, &cls2) ||
            ng_conv2d_s8(&cls1, &cls2, conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, features[i].h, features[i].w, &cls3) ||
            ng_conv2d_s8(&cls2, &cls3, conv)) return -1;
        output->class_feature[i] = cls2;
        branches[0] = reg3;
        branches[1] = cls3;
        if (ng_concat_channels_s8(branches, 2, &output->prediction[i])) return -1;
    }
    output->stride[0] = 8;
    output->stride[1] = 16;
    output->stride[2] = 32;
    return 0;
}

int ng_native_graph_forward_s8(const NGModelConfig *config, const NGModelWeights *weights, const NGTensor *input,
                          NGArena *arena, NGModelOutput *output) {
    NGGeometry geometry;
    NGConvCursor cursor;
    NGTensor x[22];
    NGTensor concat_inputs[2];
    NGTensor detect_inputs[NG_DETECT_LEVELS];
    int32_t checkpoint;
    int32_t scratch;

    if (!config || !weights || !weights->convs || !input || !input->data || !arena || !output || !arena->data ||
        input->n != 1 || input->c != config->input_channels || input->h <= 0 || input->w <= 0 || input->h % 32 ||
        input->w % 32 || config->class_count <= 0 || weights->conv_count != ng_native_graph_conv_count(config->scale) ||
        geometry_for_scale(config->scale, &geometry)) return -1;
    checkpoint = arena->used;
    for (int32_t i = 0; i < NG_DETECT_LEVELS; ++i) {
        output->prediction[i].data = NULL;
        output->prediction[i].n = output->prediction[i].c = output->prediction[i].h = output->prediction[i].w = 0;
        output->class_feature[i].data = NULL;
        output->class_feature[i].n = output->class_feature[i].c =
            output->class_feature[i].h = output->class_feature[i].w = 0;
        output->stride[i] = 0;
    }
    cursor.convs = weights->convs;
    cursor.count = weights->conv_count;
    cursor.next = 0;

    if (conv_new(&cursor, input, arena, &x[0]) || conv_new(&cursor, &x[0], arena, &x[1])) goto fail;
    if (arena_tensor(arena, geometry.channels[1], x[1].h, x[1].w, &x[2])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[1], &x[2], geometry.repeats[0], 1, arena)) goto fail;
    arena->used = scratch;
    if (conv_new(&cursor, &x[2], arena, &x[3])) goto fail;
    if (arena_tensor(arena, geometry.channels[2], x[3].h, x[3].w, &x[4])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[3], &x[4], geometry.repeats[1], 1, arena)) goto fail;
    arena->used = scratch;
    if (conv_new(&cursor, &x[4], arena, &x[5])) goto fail;
    if (arena_tensor(arena, geometry.channels[3], x[5].h, x[5].w, &x[6])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[5], &x[6], geometry.repeats[2], 1, arena)) goto fail;
    arena->used = scratch;
    if (conv_new(&cursor, &x[6], arena, &x[7])) goto fail;
    if (arena_tensor(arena, geometry.channels[4], x[7].h, x[7].w, &x[8])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[7], &x[8], geometry.repeats[3], 1, arena)) goto fail;
    arena->used = scratch;
    if (arena_tensor(arena, geometry.channels[4], x[8].h, x[8].w, &x[9])) goto fail;
    scratch = arena->used;
    if (sppf_forward(&cursor, &x[8], &x[9], arena)) goto fail;
    arena->used = scratch;
    if (upsample_new(&x[9], arena, &x[10])) goto fail;
    concat_inputs[0] = x[10]; concat_inputs[1] = x[6];
    if (concat_new(concat_inputs, 2, arena, &x[11])) goto fail;
    if (arena_tensor(arena, geometry.channels[3], x[11].h, x[11].w, &x[12])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[11], &x[12], geometry.repeats[4], 0, arena)) goto fail;
    arena->used = scratch;
    if (upsample_new(&x[12], arena, &x[13])) goto fail;
    concat_inputs[0] = x[13]; concat_inputs[1] = x[4];
    if (concat_new(concat_inputs, 2, arena, &x[14])) goto fail;
    if (arena_tensor(arena, geometry.channels[2], x[14].h, x[14].w, &x[15])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[14], &x[15], geometry.repeats[5], 0, arena)) goto fail;
    arena->used = scratch;
    if (conv_new(&cursor, &x[15], arena, &x[16])) goto fail;
    concat_inputs[0] = x[16]; concat_inputs[1] = x[12];
    if (concat_new(concat_inputs, 2, arena, &x[17])) goto fail;
    if (arena_tensor(arena, geometry.channels[3], x[17].h, x[17].w, &x[18])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[17], &x[18], geometry.repeats[6], 0, arena)) goto fail;
    arena->used = scratch;
    if (conv_new(&cursor, &x[18], arena, &x[19])) goto fail;
    concat_inputs[0] = x[19]; concat_inputs[1] = x[9];
    if (concat_new(concat_inputs, 2, arena, &x[20])) goto fail;
    if (arena_tensor(arena, geometry.channels[4], x[20].h, x[20].w, &x[21])) goto fail;
    scratch = arena->used;
    if (c2f_forward(&cursor, &x[20], &x[21], geometry.repeats[7], 0, arena)) goto fail;
    arena->used = scratch;

    detect_inputs[0] = x[15]; detect_inputs[1] = x[18]; detect_inputs[2] = x[21];
    if (detect_forward(&cursor, detect_inputs, config->class_count, arena, output) || cursor.next != cursor.count) goto fail;
    return 0;

fail:
    arena->used = checkpoint;
    return -1;
}
