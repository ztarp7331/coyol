#include "native_multiscale.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    const NGConv2D *convs;
    int32_t count;
    int32_t next;
} NGMCursor;

typedef struct {
    NGMDetection detection;
    int32_t anchor;
} NGMRawCandidate;

enum {
    NGM_BLOCK_BOTTLENECK = 0,
    NGM_BLOCK_C3K = 1,
    NGM_BLOCK_BOTTLENECK_PSA = 2
};

static int arena_tensor(NGArena *arena, int32_t c, int32_t h, int32_t w,
                        NGTensor *tensor) {
    int64_t bytes;
    if (!arena || !tensor || !arena->data || c <= 0 || h <= 0 || w <= 0 ||
        arena->used < 0 || arena->capacity < arena->used) return -1;
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

static const NGConv2D *take_conv(NGMCursor *cursor) {
    if (!cursor || !cursor->convs || cursor->next < 0 || cursor->next >= cursor->count) return NULL;
    return &cursor->convs[cursor->next++];
}

static int conv_new(NGMCursor *cursor, const NGTensor *input, NGArena *arena,
                    NGTensor *output) {
    const NGConv2D *conv = take_conv(cursor);
    int32_t h;
    int32_t w;
    if (!conv || !input || !output || input->n != 1) return -1;
    h = (input->h + 2 * conv->padding - conv->kernel) / conv->stride + 1;
    w = (input->w + 2 * conv->padding - conv->kernel) / conv->stride + 1;
    if (arena_tensor(arena, conv->out_channels, h, w, output)) return -1;
    return ng_conv2d_s8(input, output, conv);
}

static int view_channels(const NGTensor *source, int32_t first, int32_t count,
                         NGTensor *view) {
    int64_t offset;
    if (!source || !source->data || !view || first < 0 || count <= 0 ||
        first > source->c - count) return -1;
    offset = (int64_t)first * source->h * source->w;
    if (offset > INT32_MAX) return -1;
    *view = (NGTensor){source->data + offset, 1, count, source->h, source->w};
    return 0;
}

static int bottleneck_forward(NGMCursor *cursor, const NGTensor *input,
                              NGTensor *output, NGArena *arena, int shortcut) {
    NGTensor first;
    const NGConv2D *conv;
    if (!input || !output || input->n != 1 || output->c <= 0) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &first) ||
        ng_conv2d_s8(input, &first, conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || ng_conv2d_s8(&first, output, conv)) return -1;
    if (shortcut && ng_add_s8(input, output, output)) return -1;
    return 0;
}

static int c3k_forward(NGMCursor *cursor, const NGTensor *input,
                       NGTensor *output, NGArena *arena) {
    int32_t base = arena->used;
    NGTensor left;
    NGTensor right;
    NGTensor hidden[2];
    NGTensor concat;
    const NGConv2D *conv;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &left) ||
        ng_conv2d_s8(input, &left, conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &right) ||
        ng_conv2d_s8(input, &right, conv)) return -1;
    for (int i = 0; i < 2; ++i) {
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, left.h, left.w, &hidden[0]) ||
            ng_conv2d_s8(&left, &hidden[0], conv)) return -1;
        conv = take_conv(cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, left.h, left.w, &hidden[1]) ||
            ng_conv2d_s8(&hidden[0], &hidden[1], conv)) return -1;
        if (ng_add_s8(&left, &hidden[1], &left)) return -1;
    }
    {
        NGTensor parts[2] = {left, right};
        if (arena_tensor(arena, left.c + right.c, left.h, left.w, &concat) ||
            ng_concat_channels_s8(parts, 2, &concat)) return -1;
    }
    conv = take_conv(cursor);
    if (!conv || ng_conv2d_s8(&concat, output, conv)) return -1;
    arena->used = base;
    return 0;
}

static int attention_forward(NGMCursor *cursor, const NGTensor *input,
                             NGTensor *output, NGArena *arena,
                             float act_scale) {
    const NGConv2D *qkv_conv = take_conv(cursor);
    const NGConv2D *pe_conv;
    const NGConv2D *proj_conv;
    NGTensor qkv;
    NGTensor value;
    NGTensor pe;
    NGTensor mixed;
    NGTensor projected;
    int32_t channels;
    int32_t heads;
    int32_t head_dim;
    int32_t key_dim;
    int32_t qkv_chunk;
    int32_t spatial;
    float scale;
    float qkv_scale;
    float pe_scale;
    float mixed_scale;
    if (!qkv_conv || !input || !output || act_scale <= 0.0f) return -1;
    qkv_scale = qkv_conv->output_scale > 0.0f ? qkv_conv->output_scale : act_scale;
    channels = input->c;
    heads = channels / 64;
    if (heads < 1) heads = 1;
    head_dim = channels / heads;
    key_dim = head_dim / 2;
    qkv_chunk = 2 * key_dim + head_dim;
    spatial = input->h * input->w;
    if (qkv_conv->out_channels != heads * qkv_chunk ||
        arena_tensor(arena, qkv_conv->out_channels, input->h, input->w, &qkv) ||
        ng_conv2d_s8(input, &qkv, qkv_conv)) return -1;
    pe_conv = take_conv(cursor);
    if (!pe_conv || arena_tensor(arena, channels, input->h, input->w, &value) ||
        arena_tensor(arena, channels, input->h, input->w, &pe) ||
        arena_tensor(arena, channels, input->h, input->w, &mixed)) return -1;
    for (int32_t h = 0; h < heads; ++h) {
        for (int32_t d = 0; d < head_dim; ++d) {
            int32_t qkv_channel = h * qkv_chunk + 2 * key_dim + d;
            int32_t value_channel = h * head_dim + d;
            for (int32_t position = 0; position < spatial; ++position)
                value.data[value_channel * spatial + position] =
                qkv.data[qkv_channel * spatial + position];
        }
    }
    if (ng_conv2d_s8(&value, &pe, pe_conv)) return -1;
    pe_scale = pe_conv->output_scale > 0.0f ? pe_conv->output_scale : qkv_scale;
    mixed_scale = pe_scale;
    scale = 1.0f / sqrtf((float)key_dim);
    for (int32_t h = 0; h < heads; ++h) {
        for (int32_t query = 0; query < spatial; ++query) {
            float scores[64];
            float maximum = -INFINITY;
            float sum = 0.0f;
            for (int32_t key = 0; key < spatial; ++key) {
                float score = 0.0f;
                for (int32_t d = 0; d < key_dim; ++d) {
                    int32_t q_channel = h * qkv_chunk + d;
                    int32_t k_channel = h * qkv_chunk + key_dim + d;
                    score += (float)qkv.data[q_channel * spatial + query] *
                             (float)qkv.data[k_channel * spatial + key];
                }
                scores[key] = score * qkv_scale * qkv_scale * scale;
                if (scores[key] > maximum) maximum = scores[key];
            }
            for (int32_t key = 0; key < spatial; ++key) {
                scores[key] = expf(scores[key] - maximum);
                sum += scores[key];
            }
            for (int32_t d = 0; d < head_dim; ++d) {
                float value_sum = 0.0f;
                int32_t output_channel = h * head_dim + d;
                int32_t value_channel = output_channel;
                for (int32_t key = 0; key < spatial; ++key)
                    value_sum += (scores[key] / sum) *
                                 ((float)value.data[value_channel * spatial + key] * qkv_scale);
                value_sum += (float)pe.data[output_channel * spatial + query] * pe_scale;
                mixed.data[output_channel * spatial + query] =
                    (int8_t)fmaxf(-128.0f, fminf(127.0f, roundf(value_sum / mixed_scale)));
            }
        }
    }
    proj_conv = take_conv(cursor);
    if (!proj_conv || arena_tensor(arena, channels, input->h, input->w, &projected) ||
        ng_conv2d_s8(&mixed, &projected, proj_conv) ||
        ng_add_s8(input, &projected, output)) return -1;
    return 0;
}

static int psa_block_forward(NGMCursor *cursor, const NGTensor *input,
                             NGTensor *output, NGArena *arena, float act_scale) {
    int32_t base = arena->used;
    NGTensor attn;
    NGTensor ffn0;
    const NGConv2D *conv;
    if (arena_tensor(arena, input->c, input->h, input->w, &attn) ||
        attention_forward(cursor, input, &attn, arena, act_scale)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &ffn0) ||
        ng_conv2d_s8(&attn, &ffn0, conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || ng_conv2d_s8(&ffn0, output, conv) ||
        ng_add_s8(&attn, output, output)) return -1;
    arena->used = base;
    return 0;
}

static int c3k2_forward(NGMCursor *cursor, const NGTensor *input,
                         NGTensor *output, NGArena *arena, int mode,
                         float act_scale) {
    int32_t base = arena->used;
    const NGConv2D *conv;
    NGTensor split;
    NGTensor branches[3];
    NGTensor concat;
    int32_t hidden;
    conv = take_conv(cursor);
    if (!conv || conv->out_channels % 2 != 0 ||
        arena_tensor(arena, conv->out_channels, input->h, input->w, &split) ||
        ng_conv2d_s8(input, &split, conv)) return -1;
    hidden = conv->out_channels / 2;
    if (view_channels(&split, 0, hidden, &branches[0]) ||
        view_channels(&split, hidden, hidden, &branches[1])) return -1;
    if (mode == NGM_BLOCK_C3K) {
        if (arena_tensor(arena, hidden, input->h, input->w, &branches[2]) ||
            c3k_forward(cursor, &branches[1], &branches[2], arena)) return -1;
    } else if (mode == NGM_BLOCK_BOTTLENECK_PSA) {
        if (arena_tensor(arena, hidden, input->h, input->w, &branches[2]) ||
            bottleneck_forward(cursor, &branches[1], &branches[2], arena, 1)) return -1;
        {
            NGTensor psa_output;
            if (arena_tensor(arena, hidden, input->h, input->w, &psa_output) ||
                psa_block_forward(cursor, &branches[2], &psa_output, arena, act_scale)) return -1;
            branches[2] = psa_output;
        }
    } else {
        if (arena_tensor(arena, hidden, input->h, input->w, &branches[2]) ||
            bottleneck_forward(cursor, &branches[1], &branches[2], arena, 1)) return -1;
    }
    {
        NGTensor parts[3] = {branches[0], branches[1], branches[2]};
        if (arena_tensor(arena, hidden * 3, input->h, input->w, &concat) ||
            ng_concat_channels_s8(parts, 3, &concat)) return -1;
    }
    conv = take_conv(cursor);
    if (!conv || ng_conv2d_s8(&concat, output, conv)) return -1;
    arena->used = base;
    return 0;
}

static int sppf_forward(NGMCursor *cursor, const NGTensor *input,
                        NGTensor *output, NGArena *arena) {
    int32_t base = arena->used;
    NGTensor x;
    NGTensor y1;
    NGTensor y2;
    NGTensor y3;
    NGTensor concat;
    NGTensor parts[4];
    const NGConv2D *conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, input->h, input->w, &x) ||
        ng_conv2d_s8(input, &x, conv) ||
        arena_tensor(arena, x.c, x.h, x.w, &y1) || ng_maxpool2d_s8(&x, &y1, 5, 1, 2) ||
        arena_tensor(arena, x.c, x.h, x.w, &y2) || ng_maxpool2d_s8(&y1, &y2, 5, 1, 2) ||
        arena_tensor(arena, x.c, x.h, x.w, &y3) || ng_maxpool2d_s8(&y2, &y3, 5, 1, 2)) return -1;
    parts[0] = x; parts[1] = y1; parts[2] = y2; parts[3] = y3;
    if (arena_tensor(arena, x.c * 4, x.h, x.w, &concat) ||
        ng_concat_channels_s8(parts, 4, &concat)) return -1;
    conv = take_conv(cursor);
    if (!conv || ng_conv2d_s8(&concat, output, conv) ||
        ng_add_s8(input, output, output)) return -1;
    arena->used = base;
    return 0;
}

static int insert_detection(NGMDetection *detections, int32_t *count,
                            int32_t capacity, const NGMDetection *candidate) {
    int32_t position;
    if (!detections || !count || !candidate || capacity <= 0) return -1;
    if (*count < capacity) {
        position = *count;
        ++*count;
    } else if (candidate->score <= detections[*count - 1].score) {
        return 0;
    } else {
        position = *count - 1;
    }
    while (position > 0 && detections[position - 1].score < candidate->score) {
        detections[position] = detections[position - 1];
        --position;
    }
    detections[position] = *candidate;
    return 0;
}

static int detect_head(NGMCursor *cursor, const NGTensor *feature,
                       int32_t stride, int32_t class_count, float act_scale,
                       float threshold, int32_t anchor_offset,
                       NGMRawCandidate *raw, int32_t *raw_count,
                       int32_t raw_capacity, NGArena *arena) {
    (void)act_scale;
    NGTensor reg[3];
    NGTensor cls[4];
    NGTensor reg_out;
    NGTensor cls_out;
    const NGConv2D *conv;
    const NGConv2D *reg_final;
    const NGConv2D *cls_final;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &reg[0]) ||
        ng_conv2d_s8(feature, &reg[0], conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &reg[1]) ||
        ng_conv2d_s8(&reg[0], &reg[1], conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &reg_out) ||
        ng_conv2d_s8(&reg[1], &reg_out, conv)) return -1;
    reg_final = conv;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &cls[0]) ||
        ng_conv2d_s8(feature, &cls[0], conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &cls[1]) ||
        ng_conv2d_s8(&cls[0], &cls[1], conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &cls[2]) ||
        ng_conv2d_s8(&cls[1], &cls[2], conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &cls[3]) ||
        ng_conv2d_s8(&cls[2], &cls[3], conv)) return -1;
    conv = take_conv(cursor);
    if (!conv || arena_tensor(arena, conv->out_channels, feature->h, feature->w, &cls_out) ||
        ng_conv2d_s8(&cls[3], &cls_out, conv)) return -1;
    cls_final = conv;
    {
        const float box_scale = reg_final->output_scale > 0.0f ? reg_final->output_scale : 0.125f;
        const float class_scale = cls_final->output_scale > 0.0f ? cls_final->output_scale : 0.0625f;
        for (int32_t y = 0; y < feature->h; ++y) {
            for (int32_t x = 0; x < feature->w; ++x) {
                int32_t position = y * feature->w + x;
                float left = (float)reg_out.data[position] * box_scale;
                float top = (float)reg_out.data[feature->h * feature->w + position] * box_scale;
                float right = (float)reg_out.data[2 * feature->h * feature->w + position] * box_scale;
                float bottom = (float)reg_out.data[3 * feature->h * feature->w + position] * box_scale;
                float anchor_x = (float)x + 0.5f;
                float anchor_y = (float)y + 0.5f;
                for (int32_t class_id = 0; class_id < class_count; ++class_id) {
                    float score = 1.0f / (1.0f + expf(-(float)cls_out.data[class_id * feature->h * feature->w + position] * class_scale));
                    NGMDetection candidate;
                    if (score < threshold || *raw_count >= raw_capacity) continue;
                    candidate = (NGMDetection){(anchor_x - left) * (float)stride,
                                               (anchor_y - top) * (float)stride,
                                               (anchor_x + right) * (float)stride,
                                               (anchor_y + bottom) * (float)stride,
                                               score, class_id};
                    raw[*raw_count].detection = candidate;
                    raw[*raw_count].anchor = anchor_offset + position;
                    ++*raw_count;
                }
            }
        }
    }
    return 0;
}

int ngm_detect_s8(const NGLoadedWeights *weights, const int8_t *input,
                  int32_t input_channels, int32_t height, int32_t width,
                  float score_threshold, int32_t max_detections,
                  NGArena *arena, NGMDetection *detections, int32_t *count) {
    NGMCursor cursor;
    NGTensor input_tensor;
    NGTensor x[23];
    NGTensor concat_inputs[2];
    NGMRawCandidate raw[525 * 80];
    float anchor_scores[525];
    int32_t selected_anchors[300];
    int32_t raw_count = 0;
    int32_t selected_count = 0;
    if (!weights || !input || !arena || !detections || !count ||
        input_channels != 3 || height <= 0 || width <= 0 || height % 32 || width % 32 ||
        weights->conv_count != 102 || weights->class_count <= 0 || weights->class_count > 80 ||
        max_detections <= 0 || max_detections > 300 ||
        score_threshold < 0.0f || score_threshold > 1.0f) {
        fprintf(stderr, "y26 args fail input=%p arena=%p channels=%d hw=%dx%d convs=%d classes=%d max=%d threshold=%f\\n",
                (const void *)input, (void *)arena, input_channels, height, width,
                weights ? weights->conv_count : -1, weights ? weights->class_count : -1,
                max_detections, score_threshold);
        return -1;
    }
    cursor = (NGMCursor){weights->convs, weights->conv_count, 0};
    input_tensor = (NGTensor){(int8_t *)input, 1, input_channels, height, width};
    *count = 0;
    if (conv_new(&cursor, &input_tensor, arena, &x[0])) { fprintf(stderr, "y26 stem0 fail next=%d\n", cursor.next); return -1; }
    if (conv_new(&cursor, &x[0], arena, &x[1])) { fprintf(stderr, "y26 stem1 fail next=%d\n", cursor.next); return -1; }
    if (arena_tensor(arena, 64, x[1].h, x[1].w, &x[2]) ||
        c3k2_forward(&cursor, &x[1], &x[2], arena, NGM_BLOCK_BOTTLENECK, weights->act_scale)) { fprintf(stderr, "y26 block2 fail next=%d\n", cursor.next); return -1; }
    if (conv_new(&cursor, &x[2], arena, &x[3])) { fprintf(stderr, "y26 stem3 fail next=%d\n", cursor.next); return -1; }
    if (arena_tensor(arena, 128, x[3].h, x[3].w, &x[4]) ||
        c3k2_forward(&cursor, &x[3], &x[4], arena, NGM_BLOCK_BOTTLENECK, weights->act_scale)) { fprintf(stderr, "y26 block4 fail next=%d\n", cursor.next); return -1; }
    if (conv_new(&cursor, &x[4], arena, &x[5])) { fprintf(stderr, "y26 stem5 fail next=%d\n", cursor.next); return -1; }
    if (arena_tensor(arena, 128, x[5].h, x[5].w, &x[6]) ||
        c3k2_forward(&cursor, &x[5], &x[6], arena, NGM_BLOCK_C3K, weights->act_scale)) { fprintf(stderr, "y26 block6 fail next=%d\n", cursor.next); return -1; }
    if (conv_new(&cursor, &x[6], arena, &x[7])) { fprintf(stderr, "y26 stem7 fail next=%d\n", cursor.next); return -1; }
    if (arena_tensor(arena, 256, x[7].h, x[7].w, &x[8]) ||
        c3k2_forward(&cursor, &x[7], &x[8], arena, NGM_BLOCK_C3K, weights->act_scale)) { fprintf(stderr, "y26 block8 fail next=%d\n", cursor.next); return -1; }
    if (arena_tensor(arena, 256, x[8].h, x[8].w, &x[9]) || sppf_forward(&cursor, &x[8], &x[9], arena)) { fprintf(stderr, "y26 sppf fail next=%d\n", cursor.next); return -1; }
    if (arena_tensor(arena, 256, x[9].h, x[9].w, &x[10])) { fprintf(stderr, "y26 output10 alloc fail next=%d\n", cursor.next); return -1; }
    {
        int32_t base = arena->used;
        NGTensor split;
        NGTensor a;
        NGTensor b;
        NGTensor psa;
        NGTensor joined;
        const NGConv2D *conv = take_conv(&cursor);
        if (!conv || arena_tensor(arena, conv->out_channels, x[9].h, x[9].w, &split) ||
            ng_conv2d_s8(&x[9], &split, conv) || view_channels(&split, 0, 128, &a) ||
            view_channels(&split, 128, 128, &b) || arena_tensor(arena, 128, b.h, b.w, &psa) ||
            psa_block_forward(&cursor, &b, &psa, arena, weights->act_scale)) { fprintf(stderr, "y26 c2psa branch fail next=%d\n", cursor.next); return -1; }
        {
            NGTensor parts[2] = {a, psa};
            if (arena_tensor(arena, 256, split.h, split.w, &joined) ||
                ng_concat_channels_s8(parts, 2, &joined)) { fprintf(stderr, "y26 c2psa concat fail next=%d\n", cursor.next); return -1; }
        }
        conv = take_conv(&cursor);
        if (!conv || ng_conv2d_s8(&joined, &x[10], conv)) { fprintf(stderr, "y26 c2psa cv2 fail next=%d\n", cursor.next); return -1; }
        arena->used = base;
    }
    if (arena_tensor(arena, 256, x[10].h * 2, x[10].w * 2, &x[11]) ||
        ng_upsample_nearest2_s8(&x[10], &x[11])) { fprintf(stderr, "y26 neck up1 fail next=%d\\n", cursor.next); return -1; }
    concat_inputs[0] = x[11]; concat_inputs[1] = x[6];
    if (arena_tensor(arena, 384, x[11].h, x[11].w, &x[12]) ||
        ng_concat_channels_s8(concat_inputs, 2, &x[12]) || arena_tensor(arena, 128, x[12].h, x[12].w, &x[13]) ||
        c3k2_forward(&cursor, &x[12], &x[13], arena, NGM_BLOCK_C3K, weights->act_scale) ||
        arena_tensor(arena, 128, x[13].h * 2, x[13].w * 2, &x[14]) || ng_upsample_nearest2_s8(&x[13], &x[14])) { fprintf(stderr, "y26 neck p4 fail next=%d\\n", cursor.next); return -1; }
    concat_inputs[0] = x[14]; concat_inputs[1] = x[4];
    if (arena_tensor(arena, 256, x[14].h, x[14].w, &x[15]) || ng_concat_channels_s8(concat_inputs, 2, &x[15]) ||
        arena_tensor(arena, 64, x[15].h, x[15].w, &x[16]) || c3k2_forward(&cursor, &x[15], &x[16], arena, NGM_BLOCK_C3K, weights->act_scale) ||
        conv_new(&cursor, &x[16], arena, &x[17])) { fprintf(stderr, "y26 neck p3 fail next=%d\\n", cursor.next); return -1; }
    concat_inputs[0] = x[17]; concat_inputs[1] = x[13];
    if (arena_tensor(arena, 192, x[17].h, x[17].w, &x[18]) || ng_concat_channels_s8(concat_inputs, 2, &x[18]) ||
        arena_tensor(arena, 128, x[18].h, x[18].w, &x[19]) || c3k2_forward(&cursor, &x[18], &x[19], arena, NGM_BLOCK_C3K, weights->act_scale) ||
        conv_new(&cursor, &x[19], arena, &x[20])) { fprintf(stderr, "y26 neck p4out fail next=%d\\n", cursor.next); return -1; }
    concat_inputs[0] = x[20]; concat_inputs[1] = x[10];
    if (arena_tensor(arena, 384, x[20].h, x[20].w, &x[21]) || ng_concat_channels_s8(concat_inputs, 2, &x[21]) ||
        arena_tensor(arena, 256, x[21].h, x[21].w, &x[22]) || c3k2_forward(&cursor, &x[21], &x[22], arena, NGM_BLOCK_BOTTLENECK_PSA, weights->act_scale)) { fprintf(stderr, "y26 neck p5out fail next=%d\\n", cursor.next); return -1; }
    if (detect_head(&cursor, &x[16], 8, weights->class_count, weights->act_scale, score_threshold, 0, raw, &raw_count, (int32_t)(sizeof(raw) / sizeof(raw[0])), arena)) { fprintf(stderr, "y26 head p3 fail next=%d\\n", cursor.next); return -1; }
    if (detect_head(&cursor, &x[19], 16, weights->class_count, weights->act_scale, score_threshold, 400, raw, &raw_count, (int32_t)(sizeof(raw) / sizeof(raw[0])), arena)) { fprintf(stderr, "y26 head p4 fail next=%d\\n", cursor.next); return -1; }
    if (detect_head(&cursor, &x[22], 32, weights->class_count, weights->act_scale, score_threshold, 500, raw, &raw_count, (int32_t)(sizeof(raw) / sizeof(raw[0])), arena)) { fprintf(stderr, "y26 head p5 fail next=%d\\n", cursor.next); return -1; }
    for (int32_t i = 0; i < 525; ++i) anchor_scores[i] = -1.0f;
    for (int32_t i = 0; i < raw_count; ++i)
        if (raw[i].detection.score > anchor_scores[raw[i].anchor])
            anchor_scores[raw[i].anchor] = raw[i].detection.score;
    for (int32_t k = 0; k < max_detections; ++k) {
        int32_t best = -1;
        for (int32_t i = 0; i < 525; ++i) {
            int already = 0;
            for (int32_t j = 0; j < selected_count; ++j)
                if (selected_anchors[j] == i) { already = 1; break; }
            if (!already && (best < 0 || anchor_scores[i] > anchor_scores[best])) best = i;
        }
        if (best < 0 || anchor_scores[best] < 0.0f) break;
        selected_anchors[selected_count++] = best;
    }
    for (int32_t i = 0; i < raw_count; ++i) {
        int selected = 0;
        for (int32_t j = 0; j < selected_count; ++j)
            if (selected_anchors[j] == raw[i].anchor) { selected = 1; break; }
        if (selected && insert_detection(detections, count, max_detections, &raw[i].detection)) return -1;
    }
    if (cursor.next != cursor.count) { fprintf(stderr, "y26 cursor mismatch next=%d count=%d\\n", cursor.next, cursor.count); return -1; }
    return 0;
}
