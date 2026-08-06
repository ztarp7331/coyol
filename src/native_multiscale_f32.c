#include "native_multiscale.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const NGConv2D *convs;
    int32_t count;
    int32_t next;
} NGMF32Cursor;

typedef struct {
    float *data;
    int32_t n;
    int32_t c;
    int32_t h;
    int32_t w;
} NGMF32Tensor;

typedef struct {
    NGMDetection detection;
    int32_t anchor;
} NGMF32RawCandidate;

enum {
    NGMF32_BLOCK_BOTTLENECK = 0,
    NGMF32_BLOCK_C3K = 1,
    NGMF32_BLOCK_BOTTLENECK_PSA = 2
};

static int f32_trace_enabled(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        enabled = getenv("NGM_TRACE") != NULL;
        initialized = 1;
    }
    return enabled;
}

static void f32_trace(const char *name, const NGMF32Tensor *tensor) {
    double sum = 0.0;
    float minimum = FLT_MAX;
    float maximum = -FLT_MAX;
    int64_t elements;
    if (!f32_trace_enabled() || !name || !tensor || !tensor->data) return;
    elements = (int64_t)tensor->c * tensor->h * tensor->w;
    for (int64_t i = 0; i < elements; ++i) {
        float value = tensor->data[i];
        sum += value;
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
    }
    printf("trace=%s c=%d h=%d w=%d mean=%.9g min=%.9g max=%.9g\n",
           name, tensor->c, tensor->h, tensor->w,
           elements > 0 ? sum / (double)elements : 0.0, minimum, maximum);
}

static int f32_tensor(NGArena *arena, int32_t c, int32_t h, int32_t w,
                      NGMF32Tensor *tensor) {
    int64_t elements;
    int64_t bytes;
    if (!arena || !tensor || !arena->data || c <= 0 || h <= 0 || w <= 0 ||
        arena->used < 0 || arena->capacity < arena->used) return -1;
    elements = (int64_t)c * h * w;
    bytes = elements * (int64_t)sizeof(float);
    if (elements > INT32_MAX || bytes > INT32_MAX || bytes > arena->capacity - arena->used)
        return -1;
    tensor->data = (float *)(void *)(arena->data + arena->used);
    tensor->n = 1;
    tensor->c = c;
    tensor->h = h;
    tensor->w = w;
    arena->used += (int32_t)bytes;
    return 0;
}

static int32_t f32_index(const NGMF32Tensor *tensor, int32_t n, int32_t c,
                         int32_t y, int32_t x) {
    return ((n * tensor->c + c) * tensor->h + y) * tensor->w + x;
}

static const NGConv2D *f32_take(NGMF32Cursor *cursor) {
    if (!cursor || !cursor->convs || cursor->next < 0 || cursor->next >= cursor->count)
        return NULL;
    return &cursor->convs[cursor->next++];
}

static float f32_silu(float value) {
    return value / (1.0f + expf(-value));
}

static int f32_conv(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                    NGMF32Tensor *output, NGArena *arena) {
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
        if (f32_tensor(arena, layer->out_channels, expected_h, expected_w, output)) return -1;
    } else if (output->n != input->n || output->c != layer->out_channels ||
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
                            acc += input->data[f32_index(input, 0, group * channels_per_group + ic, iy, ix)] *
                                   layer->weights_f32[wi];
                        }
                    }
                }
                if (layer->activation == NG_ACT_SILU) acc = f32_silu(acc);
                output->data[f32_index(output, 0, oc, oy, ox)] = acc;
            }
        }
    }
    return 0;
}

static int f32_view(const NGMF32Tensor *source, int32_t first, int32_t count,
                    NGMF32Tensor *view) {
    int64_t offset;
    if (!source || !source->data || !view || first < 0 || count <= 0 ||
        first > source->c - count) return -1;
    offset = (int64_t)first * source->h * source->w;
    if (offset > INT32_MAX) return -1;
    *view = (NGMF32Tensor){source->data + offset, 1, count, source->h, source->w};
    return 0;
}

static int f32_add(const NGMF32Tensor *a, const NGMF32Tensor *b,
                   NGMF32Tensor *output) {
    int64_t elements;
    if (!a || !b || !output || !a->data || !b->data || !output->data ||
        a->n != b->n || a->c != b->c || a->h != b->h || a->w != b->w ||
        a->n != output->n || a->c != output->c || a->h != output->h || a->w != output->w)
        return -1;
    elements = (int64_t)a->n * a->c * a->h * a->w;
    for (int64_t i = 0; i < elements; ++i) output->data[i] = a->data[i] + b->data[i];
    return 0;
}

static int f32_concat(const NGMF32Tensor *inputs, int32_t input_count,
                      NGMF32Tensor *output) {
    int32_t offset = 0;
    if (!inputs || !output || !output->data || input_count <= 0) return -1;
    for (int32_t i = 0; i < input_count; ++i) {
        if (!inputs[i].data || inputs[i].n != output->n || inputs[i].h != output->h ||
            inputs[i].w != output->w) return -1;
        for (int32_t c = 0; c < inputs[i].c; ++c)
            for (int32_t y = 0; y < output->h; ++y)
                for (int32_t x = 0; x < output->w; ++x)
                    output->data[f32_index(output, 0, offset + c, y, x)] =
                        inputs[i].data[f32_index(&inputs[i], 0, c, y, x)];
        offset += inputs[i].c;
    }
    return offset == output->c ? 0 : -1;
}

static int f32_upsample(const NGMF32Tensor *input, NGMF32Tensor *output) {
    if (!input || !output || !input->data || !output->data || input->n != output->n ||
        input->c != output->c || output->h != input->h * 2 || output->w != input->w * 2)
        return -1;
    for (int32_t c = 0; c < output->c; ++c)
        for (int32_t y = 0; y < output->h; ++y)
            for (int32_t x = 0; x < output->w; ++x)
                output->data[f32_index(output, 0, c, y, x)] =
                    input->data[f32_index(input, 0, c, y / 2, x / 2)];
    return 0;
}

static int f32_maxpool(const NGMF32Tensor *input, NGMF32Tensor *output,
                       int32_t kernel, int32_t stride, int32_t padding) {
    if (!input || !output || !input->data || !output->data || kernel <= 0 || stride <= 0 ||
        input->n != output->n || input->c != output->c ||
        output->h != (input->h + 2 * padding - kernel) / stride + 1 ||
        output->w != (input->w + 2 * padding - kernel) / stride + 1) return -1;
    for (int32_t c = 0; c < output->c; ++c) {
        for (int32_t oy = 0; oy < output->h; ++oy) {
            for (int32_t ox = 0; ox < output->w; ++ox) {
                float maximum = -FLT_MAX;
                for (int32_t ky = 0; ky < kernel; ++ky) {
                    for (int32_t kx = 0; kx < kernel; ++kx) {
                        int32_t iy = oy * stride + ky - padding;
                        int32_t ix = ox * stride + kx - padding;
                        if (iy >= 0 && iy < input->h && ix >= 0 && ix < input->w) {
                            float value = input->data[f32_index(input, 0, c, iy, ix)];
                            if (value > maximum) maximum = value;
                        }
                    }
                }
                output->data[f32_index(output, 0, c, oy, ox)] = maximum;
            }
        }
    }
    return 0;
}

static int f32_bottleneck(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                          NGMF32Tensor *output, NGArena *arena, int shortcut) {
    int32_t base = arena->used;
    NGMF32Tensor first = {0};
    if (f32_conv(cursor, input, &first, arena) || f32_conv(cursor, &first, output, arena)) return -1;
    if (shortcut && f32_add(input, output, output)) return -1;
    arena->used = base;
    return 0;
}

static int f32_c3k(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                   NGMF32Tensor *output, NGArena *arena) {
    int32_t base = arena->used;
    NGMF32Tensor left = {0};
    NGMF32Tensor right = {0};
    NGMF32Tensor hidden0 = {0};
    NGMF32Tensor hidden1 = {0};
    NGMF32Tensor concat = {0};
    if (f32_conv(cursor, input, &left, arena) || f32_conv(cursor, input, &right, arena)) return -1;
    for (int i = 0; i < 2; ++i) {
        if (f32_conv(cursor, &left, &hidden0, arena) || f32_conv(cursor, &hidden0, &hidden1, arena) ||
            f32_add(&left, &hidden1, &left)) return -1;
    }
    {
        NGMF32Tensor parts[2] = {left, right};
        if (f32_tensor(arena, left.c + right.c, left.h, left.w, &concat) ||
            f32_concat(parts, 2, &concat) || f32_conv(cursor, &concat, output, arena)) return -1;
    }
    arena->used = base;
    return 0;
}

static int f32_attention(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                         NGMF32Tensor *output, NGArena *arena) {
    const NGConv2D *qkv_conv;
    const NGConv2D *pe_conv;
    const NGConv2D *proj_conv;
    NGMF32Tensor qkv = {0};
    NGMF32Tensor value = {0};
    NGMF32Tensor pe = {0};
    NGMF32Tensor mixed = {0};
    NGMF32Tensor projected = {0};
    int32_t heads;
    int32_t head_dim;
    int32_t key_dim;
    int32_t qkv_chunk;
    int32_t spatial;
    if (!cursor || cursor->next < 0 || cursor->next >= cursor->count || !input || !output) return -1;
    qkv_conv = &cursor->convs[cursor->next];
    heads = input->c / 64;
    if (heads < 1) heads = 1;
    head_dim = input->c / heads;
    key_dim = head_dim / 2;
    qkv_chunk = 2 * key_dim + head_dim;
    spatial = input->h * input->w;
    if (qkv_conv->out_channels != heads * qkv_chunk ||
        f32_conv(cursor, input, &qkv, arena) ||
        f32_tensor(arena, input->c, input->h, input->w, &value)) return -1;
    for (int32_t h = 0; h < heads; ++h) {
        for (int32_t d = 0; d < head_dim; ++d) {
            int32_t qkv_channel = h * qkv_chunk + 2 * key_dim + d;
            int32_t value_channel = h * head_dim + d;
            for (int32_t position = 0; position < spatial; ++position)
                value.data[value_channel * spatial + position] =
                    qkv.data[qkv_channel * spatial + position];
        }
    }
    f32_trace("attn_qkv", &qkv);
    f32_trace("attn_value", &value);
    if (cursor->next < 0 || cursor->next >= cursor->count) return -1;
    pe_conv = &cursor->convs[cursor->next];
    if (!pe_conv || f32_conv(cursor, &value, &pe, arena) ||
        f32_tensor(arena, input->c, input->h, input->w, &mixed)) return -1;
    f32_trace("attn_pe", &pe);
    for (int32_t h = 0; h < heads; ++h) {
        for (int32_t qpos = 0; qpos < spatial; ++qpos) {
            float scores[64];
            float maximum = -FLT_MAX;
            float sum = 0.0f;
            for (int32_t kpos = 0; kpos < spatial; ++kpos) {
                float dot = 0.0f;
                for (int32_t d = 0; d < key_dim; ++d) {
                    int32_t q_channel = h * qkv_chunk + d;
                    int32_t k_channel = h * qkv_chunk + key_dim + d;
                    dot += qkv.data[q_channel * spatial + qpos] *
                           qkv.data[k_channel * spatial + kpos];
                }
                scores[kpos] = dot / sqrtf((float)key_dim);
                if (scores[kpos] > maximum) maximum = scores[kpos];
            }
            for (int32_t kpos = 0; kpos < spatial; ++kpos) {
                scores[kpos] = expf(scores[kpos] - maximum);
                sum += scores[kpos];
            }
            for (int32_t d = 0; d < head_dim; ++d) {
                float value_sum = 0.0f;
                int32_t output_channel = h * head_dim + d;
                for (int32_t kpos = 0; kpos < spatial; ++kpos)
                    value_sum += (scores[kpos] / sum) *
                                 value.data[(output_channel * spatial) + kpos];
                value_sum += pe.data[(output_channel * spatial) + qpos];
                mixed.data[(output_channel * spatial) + qpos] = value_sum;
            }
        }
    }
    f32_trace("attn_mixed", &mixed);
    if (cursor->next < 0 || cursor->next >= cursor->count) return -1;
    proj_conv = &cursor->convs[cursor->next];
    if (!proj_conv || f32_conv(cursor, &mixed, &projected, arena) ||
        f32_add(input, &projected, output)) return -1;
    f32_trace("attn_projected", &projected);
    return 0;
}

static int f32_psa_block(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                         NGMF32Tensor *output, NGArena *arena) {
    int32_t base = arena->used;
    NGMF32Tensor attn = {0};
    NGMF32Tensor ffn0 = {0};
    if (f32_tensor(arena, input->c, input->h, input->w, &attn) ||
        f32_attention(cursor, input, &attn, arena)) return -1;
    f32_trace("psa_attn", &attn);
    if (f32_conv(cursor, &attn, &ffn0, arena) ||
        f32_conv(cursor, &ffn0, output, arena) ||
        f32_add(&attn, output, output)) return -1;
    f32_trace("psa_output", output);
    arena->used = base;
    return 0;
}

static int f32_c3k2(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                    NGMF32Tensor *output, NGArena *arena, int mode) {
    int32_t base = arena->used;
    NGMF32Tensor split = {0};
    NGMF32Tensor branches[3] = {{0}, {0}, {0}};
    NGMF32Tensor concat = {0};
    int32_t hidden;
    const NGConv2D *split_conv;
    if (!cursor || cursor->next < 0 || cursor->next >= cursor->count) return -1;
    split_conv = &cursor->convs[cursor->next];
    if (f32_conv(cursor, input, &split, arena) || split_conv->out_channels % 2 != 0) return -1;
    hidden = split_conv->out_channels / 2;
    if (f32_view(&split, 0, hidden, &branches[0]) ||
        f32_view(&split, hidden, hidden, &branches[1]) ||
        f32_tensor(arena, hidden, input->h, input->w, &branches[2])) return -1;
    if (mode == NGMF32_BLOCK_C3K) {
        if (f32_c3k(cursor, &branches[1], &branches[2], arena)) return -1;
    } else if (mode == NGMF32_BLOCK_BOTTLENECK_PSA) {
        if (f32_bottleneck(cursor, &branches[1], &branches[2], arena, 1)) return -1;
        {
            NGMF32Tensor psa_output = {0};
            if (f32_tensor(arena, hidden, input->h, input->w, &psa_output) ||
                f32_psa_block(cursor, &branches[2], &psa_output, arena)) return -1;
            branches[2] = psa_output;
        }
    } else if (f32_bottleneck(cursor, &branches[1], &branches[2], arena, 1)) {
        return -1;
    }
    {
        NGMF32Tensor parts[3] = {branches[0], branches[1], branches[2]};
        if (f32_tensor(arena, hidden * 3, input->h, input->w, &concat) ||
            f32_concat(parts, 3, &concat) || f32_conv(cursor, &concat, output, arena)) return -1;
    }
    arena->used = base;
    return 0;
}

static int f32_sppf(NGMF32Cursor *cursor, const NGMF32Tensor *input,
                    NGMF32Tensor *output, NGArena *arena) {
    int32_t base = arena->used;
    NGMF32Tensor x = {0};
    NGMF32Tensor y1 = {0};
    NGMF32Tensor y2 = {0};
    NGMF32Tensor y3 = {0};
    NGMF32Tensor concat = {0};
    if (f32_conv(cursor, input, &x, arena) ||
        f32_tensor(arena, x.c, x.h, x.w, &y1) || f32_maxpool(&x, &y1, 5, 1, 2) ||
        f32_tensor(arena, x.c, x.h, x.w, &y2) || f32_maxpool(&y1, &y2, 5, 1, 2) ||
        f32_tensor(arena, x.c, x.h, x.w, &y3) || f32_maxpool(&y2, &y3, 5, 1, 2)) return -1;
    {
        NGMF32Tensor parts[4] = {x, y1, y2, y3};
        if (f32_tensor(arena, x.c * 4, x.h, x.w, &concat) ||
            f32_concat(parts, 4, &concat) || f32_conv(cursor, &concat, output, arena) ||
            f32_add(input, output, output)) return -1;
    }
    arena->used = base;
    return 0;
}

static int f32_insert(NGMDetection *detections, int32_t *count, int32_t capacity,
                      const NGMDetection *candidate) {
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

static int f32_head(NGMF32Cursor *cursor, const NGMF32Tensor *feature,
                    int32_t stride, int32_t class_count, float threshold,
                    int32_t anchor_offset, NGMF32RawCandidate *raw,
                    int32_t *raw_count, int32_t raw_capacity, NGArena *arena) {
    NGMF32Tensor reg0 = {0};
    NGMF32Tensor reg1 = {0};
    NGMF32Tensor reg_out = {0};
    NGMF32Tensor cls0 = {0};
    NGMF32Tensor cls1 = {0};
    NGMF32Tensor cls2 = {0};
    NGMF32Tensor cls3 = {0};
    NGMF32Tensor cls_out = {0};
    if (f32_conv(cursor, feature, &reg0, arena) || f32_conv(cursor, &reg0, &reg1, arena) ||
        f32_conv(cursor, &reg1, &reg_out, arena) || f32_conv(cursor, feature, &cls0, arena) ||
        f32_conv(cursor, &cls0, &cls1, arena) || f32_conv(cursor, &cls1, &cls2, arena) ||
        f32_conv(cursor, &cls2, &cls3, arena) || f32_conv(cursor, &cls3, &cls_out, arena)) return -1;
    for (int32_t y = 0; y < feature->h; ++y) {
        for (int32_t x = 0; x < feature->w; ++x) {
            int32_t position = y * feature->w + x;
            for (int32_t class_id = 0; class_id < class_count; ++class_id) {
                float score = 1.0f / (1.0f + expf(-cls_out.data[class_id * feature->h * feature->w + position]));
                NGMDetection candidate;
                if (score < threshold || *raw_count >= raw_capacity) continue;
                candidate = (NGMDetection){
                    ((float)x + 0.5f - reg_out.data[position]) * (float)stride,
                    ((float)y + 0.5f - reg_out.data[feature->h * feature->w + position]) * (float)stride,
                    ((float)x + 0.5f + reg_out.data[2 * feature->h * feature->w + position]) * (float)stride,
                    ((float)y + 0.5f + reg_out.data[3 * feature->h * feature->w + position]) * (float)stride,
                    score, class_id};
                raw[*raw_count].detection = candidate;
                raw[*raw_count].anchor = anchor_offset + position;
                ++*raw_count;
            }
        }
    }
    return 0;
}

int ngm_detect_f32(const NGLoadedWeights *weights, const float *input,
                   int32_t input_channels, int32_t height, int32_t width,
                   float score_threshold, int32_t max_detections,
                   NGArena *arena, NGMDetection *detections, int32_t *count) {
    NGMF32Cursor cursor;
    NGMF32Tensor input_tensor;
    NGMF32Tensor x[23] = {{0}};
    NGMF32Tensor concat_inputs[2];
    NGMF32RawCandidate raw[525 * 80];
    float anchor_scores[525];
    int32_t selected_anchors[300];
    int32_t raw_count = 0;
    int32_t selected_count = 0;
    if (!weights || !input || !arena || !detections || !count ||
        !weights->weight_f32_blob || !weights->bias_f32_blob ||
        input_channels != 3 || height <= 0 || width <= 0 || height % 32 || width % 32 ||
        weights->conv_count != 102 || weights->class_count <= 0 || weights->class_count > 80 ||
        max_detections <= 0 || max_detections > 300 || score_threshold < 0.0f ||
        score_threshold > 1.0f)
        return -1;
    cursor = (NGMF32Cursor){weights->convs, weights->conv_count, 0};
    input_tensor = (NGMF32Tensor){(float *)input, 1, input_channels, height, width};
    *count = 0;
    if (f32_conv(&cursor, &input_tensor, &x[0], arena) ||
        f32_conv(&cursor, &x[0], &x[1], arena) ||
        f32_tensor(arena, 64, x[1].h, x[1].w, &x[2]) ||
        f32_c3k2(&cursor, &x[1], &x[2], arena, NGMF32_BLOCK_BOTTLENECK) ||
        f32_conv(&cursor, &x[2], &x[3], arena) ||
        f32_tensor(arena, 128, x[3].h, x[3].w, &x[4]) ||
        f32_c3k2(&cursor, &x[3], &x[4], arena, NGMF32_BLOCK_BOTTLENECK) ||
        f32_conv(&cursor, &x[4], &x[5], arena) ||
        f32_tensor(arena, 128, x[5].h, x[5].w, &x[6]) ||
        f32_c3k2(&cursor, &x[5], &x[6], arena, NGMF32_BLOCK_C3K) ||
        f32_conv(&cursor, &x[6], &x[7], arena) ||
        f32_tensor(arena, 256, x[7].h, x[7].w, &x[8]) ||
        f32_c3k2(&cursor, &x[7], &x[8], arena, NGMF32_BLOCK_C3K) ||
        f32_tensor(arena, 256, x[8].h, x[8].w, &x[9]) ||
        f32_sppf(&cursor, &x[8], &x[9], arena) ||
        f32_tensor(arena, 256, x[9].h, x[9].w, &x[10])) return -1;
    f32_trace("x0", &x[0]);
    f32_trace("x1", &x[1]);
    f32_trace("x2", &x[2]);
    f32_trace("x4", &x[4]);
    f32_trace("x6", &x[6]);
    f32_trace("x8", &x[8]);
    f32_trace("x9", &x[9]);
    {
        int32_t base = arena->used;
        NGMF32Tensor split = {0};
        NGMF32Tensor a = {0};
        NGMF32Tensor b = {0};
        NGMF32Tensor psa = {0};
        NGMF32Tensor joined = {0};
        if (f32_conv(&cursor, &x[9], &split, arena) || f32_view(&split, 0, 128, &a) ||
            f32_view(&split, 128, 128, &b) || f32_tensor(arena, 128, b.h, b.w, &psa) ||
            f32_psa_block(&cursor, &b, &psa, arena)) return -1;
        f32_trace("c2psa_split", &split);
        f32_trace("c2psa_b", &b);
        f32_trace("c2psa_psa", &psa);
        {
            NGMF32Tensor parts[2] = {a, psa};
            if (f32_tensor(arena, 256, split.h, split.w, &joined) ||
                f32_concat(parts, 2, &joined) || f32_conv(&cursor, &joined, &x[10], arena)) return -1;
            f32_trace("c2psa_joined", &joined);
        }
        arena->used = base;
    }
    f32_trace("x10", &x[10]);
    if (f32_tensor(arena, 256, x[10].h * 2, x[10].w * 2, &x[11]) ||
        f32_upsample(&x[10], &x[11])) return -1;
    concat_inputs[0] = x[11]; concat_inputs[1] = x[6];
    if (f32_tensor(arena, 384, x[11].h, x[11].w, &x[12]) ||
        f32_concat(concat_inputs, 2, &x[12]) || f32_tensor(arena, 128, x[12].h, x[12].w, &x[13]) ||
        f32_c3k2(&cursor, &x[12], &x[13], arena, NGMF32_BLOCK_C3K) ||
        f32_tensor(arena, 128, x[13].h * 2, x[13].w * 2, &x[14]) ||
        f32_upsample(&x[13], &x[14])) return -1;
    f32_trace("x13", &x[13]);
    concat_inputs[0] = x[14]; concat_inputs[1] = x[4];
    if (f32_tensor(arena, 256, x[14].h, x[14].w, &x[15]) ||
        f32_concat(concat_inputs, 2, &x[15]) || f32_tensor(arena, 64, x[15].h, x[15].w, &x[16]) ||
        f32_c3k2(&cursor, &x[15], &x[16], arena, NGMF32_BLOCK_C3K) ||
        f32_conv(&cursor, &x[16], &x[17], arena)) return -1;
    f32_trace("x16", &x[16]);
    f32_trace("x17", &x[17]);
    concat_inputs[0] = x[17]; concat_inputs[1] = x[13];
    if (f32_tensor(arena, 192, x[17].h, x[17].w, &x[18]) ||
        f32_concat(concat_inputs, 2, &x[18]) || f32_tensor(arena, 128, x[18].h, x[18].w, &x[19]) ||
        f32_c3k2(&cursor, &x[18], &x[19], arena, NGMF32_BLOCK_C3K) ||
        f32_conv(&cursor, &x[19], &x[20], arena)) return -1;
    f32_trace("x19", &x[19]);
    f32_trace("x20", &x[20]);
    concat_inputs[0] = x[20]; concat_inputs[1] = x[10];
    if (f32_tensor(arena, 384, x[20].h, x[20].w, &x[21]) ||
        f32_concat(concat_inputs, 2, &x[21]) || f32_tensor(arena, 256, x[21].h, x[21].w, &x[22]) ||
        f32_c3k2(&cursor, &x[21], &x[22], arena, NGMF32_BLOCK_BOTTLENECK_PSA)) return -1;
    f32_trace("x22", &x[22]);
    if (f32_head(&cursor, &x[16], 8, weights->class_count, score_threshold, 0,
                 raw, &raw_count, (int32_t)(sizeof(raw) / sizeof(raw[0])), arena) ||
        f32_head(&cursor, &x[19], 16, weights->class_count, score_threshold, 400,
                 raw, &raw_count, (int32_t)(sizeof(raw) / sizeof(raw[0])), arena) ||
        f32_head(&cursor, &x[22], 32, weights->class_count, score_threshold, 500,
                 raw, &raw_count, (int32_t)(sizeof(raw) / sizeof(raw[0])), arena)) return -1;
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
        if (selected && f32_insert(detections, count, max_detections, &raw[i].detection)) return -1;
    }
    return cursor.next == cursor.count ? 0 : -1;
}
