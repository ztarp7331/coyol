#include "native_graph_ops.h"

#include <limits.h>
#include <stddef.h>


static int8_t clamp_s8(int64_t value) {
    if (value > INT8_MAX) return INT8_MAX;
    if (value < INT8_MIN) return INT8_MIN;
    return (int8_t)value;
}

static int8_t weight_at(const NGConv2D *layer, int32_t index) {
    if (layer->weight_quant == NG_QUANT_INT8) return ((const int8_t *)layer->weights)[index];

    uint8_t packed = ((const uint8_t *)layer->weights)[index >> 1];
    int8_t nibble = (int8_t)((index & 1) ? (packed >> 4) : (packed & 0x0f));
    return (nibble & 0x08) ? (int8_t)(nibble | (int8_t)0xf0) : nibble;
}

static int32_t tensor_index(const NGTensor *tensor, int32_t n, int32_t c, int32_t y, int32_t x) {
    return ((n * tensor->c + c) * tensor->h + y) * tensor->w + x;
}

static int conv2d_s8_dense3(const NGTensor *input, NGTensor *output,
                            const NGConv2D *layer) {
    const int8_t *weights = (const int8_t *)layer->weights;
    int32_t channels = layer->in_channels;
    int32_t width = input->w;
    int32_t height = input->h;
    int32_t stride = layer->stride;
    for (int32_t n = 0; n < output->n; ++n) {
        const int8_t *input_batch = input->data + (int64_t)n * input->c * height * width;
        int8_t *output_batch = output->data + (int64_t)n * output->c * output->h * output->w;
        for (int32_t oc = 0; oc < output->c; ++oc) {
            const int8_t *row_weights = weights + (int64_t)oc * channels * 9;
            for (int32_t oy = 0; oy < output->h; ++oy) {
                int32_t y0 = oy * stride - layer->padding;
                int32_t ky_begin = y0 < 0 ? -y0 : 0;
                int32_t ky_end = height - y0;
                if (ky_end > 3) ky_end = 3;
                for (int32_t ox = 0; ox < output->w; ++ox) {
                    int32_t x0 = ox * stride - layer->padding;
                    int32_t kx_begin = x0 < 0 ? -x0 : 0;
                    int32_t kx_end = width - x0;
                    int64_t accumulator = layer->bias[oc];
                    if (kx_end > 3) kx_end = 3;
                    for (int32_t ic = 0; ic < channels; ++ic) {
                        const int8_t *input_channel = input_batch + (int64_t)ic * height * width;
                        const int8_t *weight_channel = row_weights + (int64_t)ic * 9;
                        for (int32_t ky = ky_begin; ky < ky_end; ++ky) {
                            const int8_t *input_row = input_channel +
                                (int64_t)(y0 + ky) * width + x0 + kx_begin;
                            const int8_t *weight_row = weight_channel + ky * 3 + kx_begin;
                            for (int32_t kx = kx_begin; kx < kx_end; ++kx)
                                accumulator += (int32_t)input_row[kx - kx_begin] *
                                               (int32_t)weight_row[kx - kx_begin];
                        }
                    }
                    accumulator *= layer->multiplier[oc];
                    if (layer->shift[oc] > 0)
                        accumulator = (accumulator + ((int64_t)1 << (layer->shift[oc] - 1))) >>
                                     layer->shift[oc];
                    if (accumulator > INT8_MAX) accumulator = INT8_MAX;
                    if (accumulator < INT8_MIN) accumulator = INT8_MIN;
                    {
                        int8_t value = (int8_t)accumulator;
                        if (layer->activation == NG_ACT_SILU && layer->silu_lut)
                            value = layer->silu_lut[(uint8_t)value];
                        output_batch[(oc * output->h + oy) * output->w + ox] = value;
                    }
                }
            }
        }
    }
    return 0;
}

static int conv2d_s8_depthwise3(const NGTensor *input, NGTensor *output,
                                const NGConv2D *layer) {
    const int8_t *weights = (const int8_t *)layer->weights;
    int32_t width = input->w;
    int32_t height = input->h;
    int32_t stride = layer->stride;
    for (int32_t n = 0; n < output->n; ++n) {
        const int8_t *input_batch = input->data + (int64_t)n * input->c * height * width;
        int8_t *output_batch = output->data + (int64_t)n * output->c * output->h * output->w;
        for (int32_t oc = 0; oc < output->c; ++oc) {
            const int8_t *input_channel = input_batch + (int64_t)oc * height * width;
            const int8_t *weight_channel = weights + (int64_t)oc * 9;
            for (int32_t oy = 0; oy < output->h; ++oy) {
                int32_t y0 = oy * stride - layer->padding;
                int32_t ky_begin = y0 < 0 ? -y0 : 0;
                int32_t ky_end = height - y0;
                if (ky_end > 3) ky_end = 3;
                for (int32_t ox = 0; ox < output->w; ++ox) {
                    int32_t x0 = ox * stride - layer->padding;
                    int32_t kx_begin = x0 < 0 ? -x0 : 0;
                    int32_t kx_end = width - x0;
                    int64_t accumulator = layer->bias[oc];
                    if (kx_end > 3) kx_end = 3;
                    for (int32_t ky = ky_begin; ky < ky_end; ++ky) {
                        const int8_t *input_row = input_channel +
                            (int64_t)(y0 + ky) * width + x0 + kx_begin;
                        const int8_t *weight_row = weight_channel + ky * 3 + kx_begin;
                        for (int32_t kx = kx_begin; kx < kx_end; ++kx)
                            accumulator += (int32_t)input_row[kx - kx_begin] *
                                           (int32_t)weight_row[kx - kx_begin];
                    }
                    accumulator *= layer->multiplier[oc];
                    if (layer->shift[oc] > 0)
                        accumulator = (accumulator + ((int64_t)1 << (layer->shift[oc] - 1))) >>
                                     layer->shift[oc];
                    if (accumulator > INT8_MAX) accumulator = INT8_MAX;
                    if (accumulator < INT8_MIN) accumulator = INT8_MIN;
                    {
                        int8_t value = (int8_t)accumulator;
                        if (layer->activation == NG_ACT_SILU && layer->silu_lut)
                            value = layer->silu_lut[(uint8_t)value];
                        output_batch[(oc * output->h + oy) * output->w + ox] = value;
                    }
                }
            }
        }
    }
    return 0;
}

int ng_conv2d_s8(const NGTensor *input, NGTensor *output, const NGConv2D *layer) {
    int32_t channels_per_group;
    int32_t outputs_per_group;
    int32_t expected_h;
    int32_t expected_w;

    if (!input || !output || !layer || !input->data || !output->data || !layer->weights || !layer->bias ||
        !layer->multiplier || !layer->shift || input->n != output->n || input->c != layer->in_channels ||
        output->c != layer->out_channels || layer->groups <= 0 || layer->in_channels % layer->groups ||
        layer->out_channels % layer->groups || layer->kernel <= 0 || layer->stride <= 0) return -1;

    expected_h = (input->h + 2 * layer->padding - layer->kernel) / layer->stride + 1;
    expected_w = (input->w + 2 * layer->padding - layer->kernel) / layer->stride + 1;
    if (output->h != expected_h || output->w != expected_w) return -1;

#ifndef NG_DISABLE_S8_FAST_PATH
    if (layer->weight_quant == NG_QUANT_INT8 && layer->kernel == 3 &&
        layer->groups == 1) {
        return conv2d_s8_dense3(input, output, layer);
    }
    if (layer->weight_quant == NG_QUANT_INT8 && layer->kernel == 3 &&
        layer->groups == layer->in_channels && layer->in_channels == layer->out_channels) {
        return conv2d_s8_depthwise3(input, output, layer);
    }
#endif

    channels_per_group = layer->in_channels / layer->groups;
    outputs_per_group = layer->out_channels / layer->groups;
    for (int32_t n = 0; n < output->n; ++n) {
        for (int32_t oc = 0; oc < output->c; ++oc) {
            int32_t group = oc / outputs_per_group;
            for (int32_t oy = 0; oy < output->h; ++oy) {
                for (int32_t ox = 0; ox < output->w; ++ox) {
                    int64_t acc = layer->bias[oc];
                    for (int32_t ic = 0; ic < channels_per_group; ++ic) {
                        for (int32_t ky = 0; ky < layer->kernel; ++ky) {
                            int32_t iy = oy * layer->stride + ky - layer->padding;
                            if (iy < 0 || iy >= input->h) continue;
                            for (int32_t kx = 0; kx < layer->kernel; ++kx) {
                                int32_t ix = ox * layer->stride + kx - layer->padding;
                                int32_t wi = (((oc * channels_per_group + ic) * layer->kernel + ky) * layer->kernel + kx);
                                if (ix < 0 || ix >= input->w) continue;
                                acc += (int32_t)input->data[tensor_index(input, n, group * channels_per_group + ic, iy, ix)] *
                                       weight_at(layer, wi);
                            }
                        }
                    }
                    acc *= layer->multiplier[oc];
                    if (layer->shift[oc] > 0) acc = (acc + ((int64_t)1 << (layer->shift[oc] - 1))) >> layer->shift[oc];
                    int8_t value = clamp_s8(acc);
                    if (layer->activation == NG_ACT_SILU && layer->silu_lut) value = layer->silu_lut[(uint8_t)value];
                    output->data[tensor_index(output, n, oc, oy, ox)] = value;
                }
            }
        }
    }
    return 0;
}

int ng_maxpool2d_s8(const NGTensor *input, NGTensor *output, int32_t kernel, int32_t stride, int32_t padding) {
    if (!input || !output || !input->data || !output->data || kernel <= 0 || stride <= 0 || input->n != output->n || input->c != output->c ||
        output->h != (input->h + 2 * padding - kernel) / stride + 1 || output->w != (input->w + 2 * padding - kernel) / stride + 1) return -1;
    for (int32_t n = 0; n < output->n; ++n) for (int32_t c = 0; c < output->c; ++c)
        for (int32_t oy = 0; oy < output->h; ++oy) for (int32_t ox = 0; ox < output->w; ++ox) {
            int8_t maximum = INT8_MIN;
            for (int32_t ky = 0; ky < kernel; ++ky) for (int32_t kx = 0; kx < kernel; ++kx) {
                int32_t iy = oy * stride + ky - padding, ix = ox * stride + kx - padding;
                if (iy >= 0 && iy < input->h && ix >= 0 && ix < input->w) {
                    int8_t value = input->data[tensor_index(input, n, c, iy, ix)];
                    if (value > maximum) maximum = value;
                }
            }
            output->data[tensor_index(output, n, c, oy, ox)] = maximum;
        }
    return 0;
}

int ng_upsample_nearest2_s8(const NGTensor *input, NGTensor *output) {
    if (!input || !output || !input->data || !output->data || input->n != output->n || input->c != output->c ||
        output->h != input->h * 2 || output->w != input->w * 2) return -1;
    for (int32_t n = 0; n < output->n; ++n) for (int32_t c = 0; c < output->c; ++c)
        for (int32_t y = 0; y < output->h; ++y) for (int32_t x = 0; x < output->w; ++x)
            output->data[tensor_index(output, n, c, y, x)] = input->data[tensor_index(input, n, c, y / 2, x / 2)];
    return 0;
}

int ng_concat_channels_s8(const NGTensor *inputs, int32_t input_count, NGTensor *output) {
    int32_t offset = 0;
    if (!inputs || !output || !output->data || input_count <= 0) return -1;
    for (int32_t i = 0; i < input_count; ++i) {
        if (!inputs[i].data || inputs[i].n != output->n || inputs[i].h != output->h || inputs[i].w != output->w) return -1;
        for (int32_t n = 0; n < output->n; ++n) for (int32_t c = 0; c < inputs[i].c; ++c)
            for (int32_t y = 0; y < output->h; ++y) for (int32_t x = 0; x < output->w; ++x)
                output->data[tensor_index(output, n, offset + c, y, x)] = inputs[i].data[tensor_index(&inputs[i], n, c, y, x)];
        offset += inputs[i].c;
    }
    return offset == output->c ? 0 : -1;
}

int ng_add_s8(const NGTensor *a, const NGTensor *b, NGTensor *output) {
    int64_t elements;
    if (!a || !b || !output || !a->data || !b->data || !output->data || a->n != b->n || a->c != b->c ||
        a->h != b->h || a->w != b->w || a->n != output->n || a->c != output->c || a->h != output->h ||
        a->w != output->w) return -1;
    elements = (int64_t)a->n * a->c * a->h * a->w;
    for (int64_t i = 0; i < elements; ++i) output->data[i] = clamp_s8((int32_t)a->data[i] + b->data[i]);
    return 0;
}

int ng_decode_dfl_s8(const int8_t *logits, int32_t grid_x, int32_t grid_y, int32_t stride,
                     const NGDflConfig *config, NGDetection *out) {
    int32_t distance_q12[4];
    if (!logits || !config || !config->exp_lut_q15 || !out || stride <= 0) return -1;
    for (int32_t side = 0; side < 4; ++side) {
        int64_t weighted_sum = 0;
        uint32_t sum = 0;
        for (int32_t bin = 0; bin < 16; ++bin) {
            uint16_t probability = config->exp_lut_q15[(uint8_t)logits[side * 16 + bin]];
            sum += probability;
            weighted_sum += (int64_t)bin * (int64_t)probability;
        }
        if (!sum) return -1;
        distance_q12[side] = (int32_t)((weighted_sum << 12) / sum) * stride;
    }
    int32_t cx = ((grid_x << 12) + 2048) * stride;
    int32_t cy = ((grid_y << 12) + 2048) * stride;
    out->x1_q12 = cx - distance_q12[0]; out->y1_q12 = cy - distance_q12[1];
    out->x2_q12 = cx + distance_q12[2]; out->y2_q12 = cy + distance_q12[3];
    return 0;
}

uint16_t ng_iou_q15(const NGDetection *a, const NGDetection *b) {
    int32_t left = a->x1_q12 > b->x1_q12 ? a->x1_q12 : b->x1_q12;
    int32_t top = a->y1_q12 > b->y1_q12 ? a->y1_q12 : b->y1_q12;
    int32_t right = a->x2_q12 < b->x2_q12 ? a->x2_q12 : b->x2_q12;
    int32_t bottom = a->y2_q12 < b->y2_q12 ? a->y2_q12 : b->y2_q12;
    int64_t inter = (right > left && bottom > top) ? (int64_t)(right - left) * (bottom - top) : 0;
    int64_t area_a = (a->x2_q12 > a->x1_q12 && a->y2_q12 > a->y1_q12) ? (int64_t)(a->x2_q12 - a->x1_q12) * (a->y2_q12 - a->y1_q12) : 0;
    int64_t area_b = (b->x2_q12 > b->x1_q12 && b->y2_q12 > b->y1_q12) ? (int64_t)(b->x2_q12 - b->x1_q12) * (b->y2_q12 - b->y1_q12) : 0;
    int64_t union_area = area_a + area_b - inter;
    return union_area > 0 ? (uint16_t)((inter << 15) / union_area) : 0;
}

int32_t ng_nms(NGDetection *detections, int32_t count, uint16_t iou_threshold_q15,
               NGDetection *kept, int32_t kept_capacity) {
    int32_t kept_count = 0;
    if (!detections || !kept || count < 0 || kept_capacity < 0) return -1;
    for (int32_t i = 0; i < count; ++i) for (int32_t j = i + 1; j < count; ++j)
        if (detections[j].score_q15 > detections[i].score_q15) { NGDetection temp = detections[i]; detections[i] = detections[j]; detections[j] = temp; }
    for (int32_t i = 0; i < count && kept_count < kept_capacity; ++i) {
        int suppress = 0;
        for (int32_t j = 0; j < kept_count; ++j)
            if (detections[i].class_id == kept[j].class_id && ng_iou_q15(&detections[i], &kept[j]) > iou_threshold_q15) { suppress = 1; break; }
        if (!suppress) kept[kept_count++] = detections[i];
    }
    return kept_count;
}
