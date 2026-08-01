/* Backbone, neck, forward quantization, and explicit global backward kernels. */
static void stage_forward(det_stage *stage, const det_tensor_f32 *input) {
    det_tensor_f32 output = {stage->activation, stage->output_channels,
                             stage->output_height, stage->output_width};
    if (!stage->depthwise) {
        (void)det_conv2d_f32(input, stage->weights, stage->bias,
                             stage->output_channels, stage->kernel, stage->stride,
                             stage->padding, &output);
        det_relu_inplace(&output);
        return;
    }
    for (int c = 0; c < stage->output_channels; ++c) {
        for (int oy = 0; oy < stage->output_height; ++oy) {
            for (int ox = 0; ox < stage->output_width; ++ox) {
                float sum = stage->bias[c];
                for (int ky = 0; ky < stage->kernel; ++ky) {
                    int iy = oy * stage->stride + ky - stage->padding;
                    if (iy < 0 || iy >= stage->input_height) continue;
                    for (int kx = 0; kx < stage->kernel; ++kx) {
                        int ix = ox * stage->stride + kx - stage->padding;
                        if (ix < 0 || ix >= stage->input_width) continue;
                        size_t weight_index = stage_weight_index(stage, c, c, ky, kx);
                        sum += stage->weights[weight_index] *
                               input->data[tensor_index(input, c, iy, ix)];
                    }
                }
                if (sum < 0.0f) sum = 0.0f;
                output.data[tensor_index(&output, c, oy, ox)] = sum;
            }
        }
    }
}

static int quantized_stage_weight(const det_stage *stage, det_precision precision,
                                  size_t index) {
    if (precision == DET_PRECISION_INT8) return (int)stage->quant_weights[index];
    size_t cols = (size_t)(stage->depthwise ? 1 : stage->input_channels) *
                  (size_t)stage->kernel * (size_t)stage->kernel;
    size_t row = index / cols;
    size_t col = index % cols;
    size_t packed_index = row * ((cols + 1U) / 2U) + col / 2U;
    unsigned int packed = (unsigned int)stage->packed_weights[packed_index];
    int value = (int)((packed >> ((col & 1U) * 4U)) & 0x0fU);
    return value >= 8 ? value - 16 : value;
}

static float tensor_quant_scale(const det_tensor_f32 *tensor) {
    size_t count = (size_t)tensor->channels * (size_t)tensor->height *
                   (size_t)tensor->width;
    float maximum = 0.0f;
    for (size_t i = 0U; i < count; ++i) {
        float value = fabsf(tensor->data[i]);
        if (value > maximum) maximum = value;
    }
    return maximum > 0.0f ? maximum / 127.0f : 1.0f;
}

static int stage_parameters_are_zero(const det_stage *stage) {
    size_t weights = stage_weight_count(stage);
    for (size_t i = 0U; i < weights; ++i) {
        if (stage->weights[i] != 0.0f) return 0;
    }
    for (int i = 0; i < stage->output_channels; ++i) {
        if (stage->bias[i] != 0.0f) return 0;
    }
    return 1;
}

static void stage_forward_quant(const det_stage *stage, const det_tensor_f32 *input,
                                det_precision precision, float *output_data) {
    det_tensor_f32 output = {output_data, stage->output_channels,
                             stage->output_height, stage->output_width};
    float input_scale = tensor_quant_scale(input);
    for (int oc = 0; oc < stage->output_channels; ++oc) {
        for (int oy = 0; oy < stage->output_height; ++oy) {
            for (int ox = 0; ox < stage->output_width; ++ox) {
                int32_t accumulator = 0;
                int channel_start = stage->depthwise ? oc : 0;
                int channel_end = stage->depthwise ? oc + 1 : stage->input_channels;
                for (int ic = channel_start; ic < channel_end; ++ic) {
                    for (int ky = 0; ky < stage->kernel; ++ky) {
                        int iy = oy * stage->stride + ky - stage->padding;
                        if (iy < 0 || iy >= stage->input_height) continue;
                        for (int kx = 0; kx < stage->kernel; ++kx) {
                            int ix = ox * stage->stride + kx - stage->padding;
                            if (ix < 0 || ix >= stage->input_width) continue;
                            size_t weight_index = stage_weight_index(stage, oc, ic, ky, kx);
                            int input_q = (int)det_quantize_s8(
                                input->data[tensor_index(input, ic, iy, ix)], input_scale);
                            accumulator += input_q * quantized_stage_weight(stage, precision,
                                                                             weight_index);
                        }
                    }
                }
                float value = (float)accumulator * input_scale * stage->quant_scales[oc] +
                              stage->bias[oc];
                if (value < 0.0f) value = 0.0f;
                output.data[tensor_index(&output, oc, oy, ox)] = value;
            }
        }
    }
}

static int quantized_head_weight(const det_head *head, det_precision precision,
                                 size_t index) {
    if (precision == DET_PRECISION_INT8) return (int)head->quant_weights[index];
    size_t col = index % (size_t)head->channels;
    size_t row = index / (size_t)head->channels;
    size_t packed_index = row * (((size_t)head->channels + 1U) / 2U) + col / 2U;
    unsigned int packed = (unsigned int)head->packed_weights[packed_index];
    int value = (int)((packed >> ((col & 1U) * 4U)) & 0x0fU);
    return value >= 8 ? value - 16 : value;
}

static void head_forward_quant(const det_head *head, const float features[4],
                               det_precision precision, float *output) {
    float maximum = 0.0f;
    for (int c = 0; c < head->channels; ++c) {
        float value = fabsf(features[c]);
        if (value > maximum) maximum = value;
    }
    float input_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    for (int o = 0; o < head->outputs; ++o) {
        int32_t accumulator = 0;
        for (int c = 0; c < head->channels; ++c) {
            int input_q = (int)det_quantize_s8(features[c], input_scale);
            size_t weight_index = (size_t)o * (size_t)head->channels + (size_t)c;
            accumulator += input_q * quantized_head_weight(head, precision, weight_index);
        }
        output[o] = (float)accumulator * input_scale * head->quant_scales[o] + head->bias[o];
    }
}

static void stage_backward(det_stage *stage, const det_tensor_f32 *input,
                           const det_tensor_f32 *grad_output,
                           det_tensor_f32 *grad_input) {
    if (!stage->depthwise) {
        (void)det_conv2d_backward_f32(input, stage->weights, grad_output,
                                      stage->output_channels, stage->kernel,
                                      stage->stride, stage->padding, grad_input,
                                      stage->gradient_w, stage->gradient_b);
        return;
    }
    size_t input_count = (size_t)stage->input_channels * (size_t)stage->input_height *
                         (size_t)stage->input_width;
    size_t weight_count = stage_weight_count(stage);
    memset(grad_input->data, 0, input_count * sizeof(float));
    memset(stage->gradient_w, 0, weight_count * sizeof(float));
    memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
    for (int c = 0; c < stage->output_channels; ++c) {
        for (int oy = 0; oy < stage->output_height; ++oy) {
            for (int ox = 0; ox < stage->output_width; ++ox) {
                float go = grad_output->data[tensor_index(grad_output, c, oy, ox)];
                stage->gradient_b[c] += go;
                for (int ky = 0; ky < stage->kernel; ++ky) {
                    int iy = oy * stage->stride + ky - stage->padding;
                    if (iy < 0 || iy >= stage->input_height) continue;
                    for (int kx = 0; kx < stage->kernel; ++kx) {
                        int ix = ox * stage->stride + kx - stage->padding;
                        if (ix < 0 || ix >= stage->input_width) continue;
                        size_t weight_index = stage_weight_index(stage, c, c, ky, kx);
                        size_t input_index = tensor_index(input, c, iy, ix);
                        stage->gradient_w[weight_index] += go * input->data[input_index];
                        grad_input->data[input_index] += go * stage->weights[weight_index];
                    }
                }
            }
        }
    }
}

static void fuse_neck_forward(det_model *model);

static void backbone_forward(det_model *model, const det_image *image) {
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
        det_stage *stage = &model->stages[i];
        det_tensor_f32 input;
        if (i == 0) {
            input.data = (float *)image->data;
            input.channels = image->channels;
            input.height = image->height;
            input.width = image->width;
        } else {
            det_stage *previous = &model->stages[i - 1];
            input.data = previous->activation;
            input.channels = previous->output_channels;
            input.height = previous->output_height;
            input.width = previous->output_width;
        }
        if (model->precision == DET_PRECISION_F32) {
            stage_forward(stage, &input);
        } else {
            stage_forward_quant(stage, &input, model->precision, stage->activation);
        }
    }
    fuse_neck_forward(model);
}

static size_t feature_offset(int height, int width, int c, int y, int x) {
    return ((size_t)c * (size_t)height + (size_t)y) * (size_t)width + (size_t)x;
}

static void fuse_neck_forward(det_model *model) {
    for (int scale = DET_NECK_STAGES - 1; scale >= 0; --scale) {
        det_stage *lateral = &model->neck[scale];
        const det_stage *backbone = &model->stages[scale + 2];
        det_tensor_f32 input = {backbone->activation, backbone->output_channels,
                                backbone->output_height, backbone->output_width};
        if (model->precision == DET_PRECISION_F32) {
            stage_forward(lateral, &input);
        } else {
            stage_forward_quant(lateral, &input, model->precision, lateral->activation);
        }
        int height = lateral->output_height;
        int width = lateral->output_width;
        size_t count = (size_t)DET_GRAPH_CHANNELS * (size_t)height * (size_t)width;
        memcpy(model->neck_topdown[scale], lateral->activation, count * sizeof(float));
        if (scale + 1 < DET_NECK_STAGES) {
            const det_stage *parent = &model->neck[scale + 1];
            const float *parent_data = model->neck_topdown[scale + 1];
            int parent_height = parent->output_height;
            int parent_width = parent->output_width;
            for (int c = 0; c < DET_GRAPH_CHANNELS; ++c) {
                for (int y = 0; y < height; ++y) {
                    int py = (y * parent_height) / height;
                    for (int x = 0; x < width; ++x) {
                        int px = (x * parent_width) / width;
                        size_t dst = feature_offset(height, width, c, y, x);
                        size_t src = feature_offset(parent_height, parent_width, c, py, px);
                        model->neck_topdown[scale][dst] += parent_data[src];
                    }
                }
            }
        }
    }
    for (int scale = 0; scale < DET_NECK_STAGES; ++scale) {
        int height = model->neck[scale].output_height;
        int width = model->neck[scale].output_width;
        size_t count = (size_t)DET_GRAPH_CHANNELS * (size_t)height * (size_t)width;
        memcpy(model->neck_fused[scale], model->neck_topdown[scale], count * sizeof(float));
    }
    for (int scale = 0; scale < DET_BOTTOMUP_STAGES; ++scale) {
        det_stage *down = &model->bottomup[scale];
        const float *input_data = scale == 0 ? model->neck_topdown[0] :
                                  model->neck_fused[scale];
        det_tensor_f32 input = { (float *)input_data, DET_GRAPH_CHANNELS,
                                 down->input_height, down->input_width };
        int height = down->output_height;
        int width = down->output_width;
        size_t count = (size_t)DET_GRAPH_CHANNELS * (size_t)height * (size_t)width;
        if (stage_parameters_are_zero(down)) {
            memset(down->activation, 0, count * sizeof(float));
        } else if (model->precision == DET_PRECISION_F32) {
            stage_forward(down, &input);
        } else {
            stage_forward_quant(down, &input, model->precision, down->activation);
        }
        for (size_t i = 0U; i < count; ++i) {
            model->neck_fused[scale + 1][i] += down->activation[i];
        }
    }
}

static void extract_feature(const det_model *model, const det_image *image, int scale,
                            int cell_y, int cell_x, float features[4], float base[4]) {
    (void)image;
    const det_stage *stage = &model->neck[scale];
    det_tensor_f32 tensor = {model->neck_fused[scale], stage->output_channels,
                             stage->output_height, stage->output_width};
    for (int c = 0; c < DET_GRAPH_CHANNELS; ++c) {
        float value = tensor.data[tensor_index(&tensor, c, cell_y, cell_x)];
        features[c] = value;
        if (base != NULL) base[c] = value;
    }
}

static void head_forward(const det_head *head, const float features[4], float *output) {
    for (int o = 0; o < head->outputs; ++o) {
        float value = head->bias[o];
        for (int c = 0; c < head->channels; ++c) {
            value += head->weights[(size_t)o * (size_t)head->channels + (size_t)c] * features[c];
        }
        output[o] = value;
    }
}

static int choose_scale(const det_box *box, int width, int height) {
    float bw = (box->x2 - box->x1) / (float)width;
    float bh = (box->y2 - box->y1) / (float)height;
    float size = fmaxf(bw, bh);
    return size < 0.2f ? 0 : (size < 0.5f ? 1 : 2);
}

static float smooth_l1_loss(float value) {
    float magnitude = fabsf(value);
    return magnitude < 1.0f ? 0.5f * magnitude * magnitude : magnitude - 0.5f;
}

static float smooth_l1_gradient(float value) {
    float magnitude = fabsf(value);
    if (magnitude < 1.0f) return value;
    return value < 0.0f ? -1.0f : 1.0f;
}

static float centered_distance_iou(const float prediction[4], const float target[4]) {
    float pl = fmaxf(0.0f, prediction[0]);
    float pt = fmaxf(0.0f, prediction[1]);
    float pr = fmaxf(0.0f, prediction[2]);
    float pb = fmaxf(0.0f, prediction[3]);
    float tl = fmaxf(0.0f, target[0]);
    float tt = fmaxf(0.0f, target[1]);
    float tr = fmaxf(0.0f, target[2]);
    float tb = fmaxf(0.0f, target[3]);
    float intersection = (fminf(pl, tl) + fminf(pr, tr)) *
                         (fminf(pt, tt) + fminf(pb, tb));
    float prediction_area = (pl + pr) * (pt + pb);
    float target_area = (tl + tr) * (tt + tb);
    float union_area = prediction_area + target_area - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static void centered_iou_gradient(const float prediction[4], const float target[4],
                                  float gradient[4]) {
    const float epsilon = 1.0e-3f;
    for (int i = 0; i < 4; ++i) {
        float plus[4] = {prediction[0], prediction[1], prediction[2], prediction[3]};
        float minus[4] = {prediction[0], prediction[1], prediction[2], prediction[3]};
        plus[i] += epsilon;
        minus[i] -= epsilon;
        gradient[i] = (centered_distance_iou(plus, target) -
                       centered_distance_iou(minus, target)) / (2.0f * epsilon);
    }
}

static float prediction_loss_gradient(const float *prediction, const float *target,
                                      int outputs, int positive, int regression_enabled,
                                      float gradient[4 + DET_MAX_CLASSES]) {
    float loss = 0.0f;
    for (int o = 0; o < outputs; ++o) gradient[o] = 0.0f;
    if (positive >= 0 && regression_enabled) {
        float iou = centered_distance_iou(prediction, target);
        float iou_gradient[4];
        centered_iou_gradient(prediction, target, iou_gradient);
        loss += 0.5f * (1.0f - iou);
        for (int o = 0; o < 4; ++o) {
            float difference = prediction[o] - target[o];
            loss += 0.5f * smooth_l1_loss(difference);
            gradient[o] = 0.5f * smooth_l1_gradient(difference) - 0.5f * iou_gradient[o];
        }
    }
    for (int o = 4; o < outputs; ++o) {
        float logit = prediction[o];
        float target_value = target[o];
        float softplus = logit > 0.0f ? logit + log1pf(expf(-logit)) : log1pf(expf(logit));
        loss += softplus - target_value * logit;
        gradient[o] = det_sigmoid(logit) - target_value;
    }
    return loss;
}

static void update_head(det_head *head, const float features[4], const float *target,
                        int positive, int regression_enabled, float lr, float momentum) {
    float prediction[4 + DET_MAX_CLASSES] = {0.0f};
    float gradient[4 + DET_MAX_CLASSES];
    head_forward(head, features, prediction);
    (void)prediction_loss_gradient(prediction, target, head->outputs, positive,
                                   regression_enabled, gradient);
    if (positive < 0) {
        /* LOCAL_FAST sees many more background cells than positives. Keep a
           sparse negative signal without allowing BCE bias updates to erase
           the one-to-one class evidence. GLOBAL_BP keeps the full loss. */
        for (int o = 4; o < head->outputs; ++o) gradient[o] *= 0.001f;
    }
    for (int o = 0; o < head->outputs; ++o) {
        size_t base = (size_t)o * (size_t)head->channels;
        for (int c = 0; c < head->channels; ++c) {
            float grad = gradient[o] * features[c];
            head->velocity_w[base + (size_t)c] = momentum * head->velocity_w[base + (size_t)c] + grad;
            head->weights[base + (size_t)c] -= lr * head->velocity_w[base + (size_t)c];
        }
        head->velocity_b[o] = momentum * head->velocity_b[o] + gradient[o];
        head->bias[o] -= lr * head->velocity_b[o];
    }
}

static void feature_gradient_from_error(const det_head *head, const float *gradient,
                                        float feature_gradient[4]) {
    for (int c = 0; c < 4; ++c) feature_gradient[c] = 0.0f;
    for (int o = 0; o < head->outputs; ++o) {
        for (int c = 0; c < 4; ++c) {
            feature_gradient[c] += gradient[o] *
                                   head->weights[(size_t)o * (size_t)head->channels + (size_t)c];
        }
    }
}

static void apply_head_gradient(det_head *head, float gradient_w[][4],
                                const float *gradient_b, float lr, float momentum,
                                float scale) {
    for (int o = 0; o < head->outputs; ++o) {
        size_t base = (size_t)o * (size_t)head->channels;
        for (int c = 0; c < head->channels; ++c) {
            float grad = gradient_w[o][c] * scale;
            head->velocity_w[base + (size_t)c] = momentum * head->velocity_w[base + (size_t)c] + grad;
            head->weights[base + (size_t)c] -= lr * head->velocity_w[base + (size_t)c];
        }
        float grad_b = gradient_b[o] * scale;
        head->velocity_b[o] = momentum * head->velocity_b[o] + grad_b;
        head->bias[o] -= lr * head->velocity_b[o];
    }
}

static void clear_backbone_gradients(det_model *model) {
    for (int i = 0; i < DET_GRAPH_STAGES; ++i) {
        det_stage *stage = &model->stages[i];
        size_t output_count = (size_t)stage->output_channels *
                              (size_t)stage->output_height * (size_t)stage->output_width;
        size_t weight_count = stage_weight_count(stage);
        memset(stage->gradient_output, 0, output_count * sizeof(float));
        memset(stage->gradient_w, 0, weight_count * sizeof(float));
        memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
    }
    for (int i = 0; i < DET_NECK_STAGES; ++i) {
        det_stage *stage = &model->neck[i];
        size_t output_count = (size_t)stage->output_channels *
                              (size_t)stage->output_height * (size_t)stage->output_width;
        memset(model->neck_gradient[i], 0, output_count * sizeof(float));
        memset(stage->gradient_output, 0, output_count * sizeof(float));
        memset(stage->gradient_w, 0, stage_weight_count(stage) * sizeof(float));
        memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
    }
    for (int i = 0; i < DET_BOTTOMUP_STAGES; ++i) {
        det_stage *stage = &model->bottomup[i];
        size_t output_count = (size_t)stage->output_channels *
                              (size_t)stage->output_height * (size_t)stage->output_width;
        memset(stage->gradient_output, 0, output_count * sizeof(float));
        memset(stage->gradient_w, 0, stage_weight_count(stage) * sizeof(float));
        memset(stage->gradient_b, 0, (size_t)stage->output_channels * sizeof(float));
    }
}

static void add_feature_gradient_to_backbone(det_model *model, int scale, int y, int x,
                                             const float gradient[4]) {
    const det_stage *stage = &model->neck[scale];
    for (int c = 0; c < DET_GRAPH_CHANNELS; ++c) {
        size_t index = ((size_t)c * (size_t)stage->output_height + (size_t)y) *
                       (size_t)stage->output_width + (size_t)x;
        model->neck_gradient[scale][index] += gradient[c];
    }
}

static void neck_backward(det_model *model) {
    for (int scale = DET_BOTTOMUP_STAGES - 1; scale >= 0; --scale) {
        det_stage *down = &model->bottomup[scale];
        int neutral_stage = stage_parameters_are_zero(down);
        size_t count = (size_t)down->output_channels * (size_t)down->output_height *
                       (size_t)down->output_width;
        for (size_t i = 0U; i < count; ++i) {
            /* A neutral zero stage needs one straight-through step to escape
             * zero initialization; learned stages use ordinary ReLU gating. */
            down->gradient_output[i] = (neutral_stage || down->activation[i] > 0.0f) ?
                model->neck_gradient[scale + 1][i] : 0.0f;
        }
        const float *input_data = scale == 0 ? model->neck_topdown[0] :
                                  model->neck_fused[scale];
        det_tensor_f32 input = {(float *)input_data, DET_GRAPH_CHANNELS,
                                down->input_height, down->input_width};
        det_tensor_f32 grad_output = {down->gradient_output, down->output_channels,
                                      down->output_height, down->output_width};
        det_tensor_f32 grad_input = {down->gradient_input, down->input_channels,
                                     down->input_height, down->input_width};
        stage_backward(down, &input, &grad_output, &grad_input);
        size_t input_count = (size_t)down->input_channels * (size_t)down->input_height *
                             (size_t)down->input_width;
        for (size_t i = 0U; i < input_count; ++i) {
            model->neck_gradient[scale][i] += down->gradient_input[i];
        }
    }
    for (int scale = 0; scale + 1 < DET_NECK_STAGES; ++scale) {
        const det_stage *child = &model->neck[scale];
        const det_stage *parent = &model->neck[scale + 1];
        int child_height = child->output_height;
        int child_width = child->output_width;
        int parent_height = parent->output_height;
        int parent_width = parent->output_width;
        for (int c = 0; c < DET_GRAPH_CHANNELS; ++c) {
            for (int y = 0; y < child_height; ++y) {
                int py = (y * parent_height) / child_height;
                for (int x = 0; x < child_width; ++x) {
                    int px = (x * parent_width) / child_width;
                    size_t child_index = feature_offset(child_height, child_width, c, y, x);
                    size_t parent_index = feature_offset(parent_height, parent_width, c, py, px);
                    model->neck_gradient[scale + 1][parent_index] +=
                        model->neck_gradient[scale][child_index];
                }
            }
        }
    }
    for (int scale = 0; scale < DET_NECK_STAGES; ++scale) {
        det_stage *lateral = &model->neck[scale];
        const det_stage *backbone = &model->stages[scale + 2];
        size_t count = (size_t)lateral->output_channels *
                       (size_t)lateral->output_height * (size_t)lateral->output_width;
        for (size_t i = 0U; i < count; ++i) {
            lateral->gradient_output[i] = lateral->activation[i] > 0.0f ?
                model->neck_gradient[scale][i] : 0.0f;
        }
        det_tensor_f32 input = {backbone->activation, backbone->output_channels,
                                backbone->output_height, backbone->output_width};
        det_tensor_f32 grad_output = {lateral->gradient_output, lateral->output_channels,
                                      lateral->output_height, lateral->output_width};
        det_tensor_f32 grad_input = {lateral->gradient_input, lateral->input_channels,
                                     lateral->input_height, lateral->input_width};
        stage_backward(lateral, &input, &grad_output, &grad_input);
        for (size_t i = 0U; i < count; ++i) {
            ((det_stage *)backbone)->gradient_output[i] += lateral->gradient_input[i];
        }
    }
}

static void accumulate_previous_stage_gradient(det_stage *stage, det_stage *previous) {
    size_t count = (size_t)stage->input_channels * (size_t)stage->input_height *
                   (size_t)stage->input_width;
    for (size_t i = 0U; i < count; ++i) previous->gradient_output[i] += stage->gradient_input[i];
}

static void backbone_backward(det_model *model, const det_image *image) {
    for (int i = DET_GRAPH_STAGES - 1; i >= 0; --i) {
        det_stage *stage = &model->stages[i];
        size_t output_count = (size_t)stage->output_channels *
                              (size_t)stage->output_height * (size_t)stage->output_width;
        for (size_t j = 0U; j < output_count; ++j) {
            if (stage->activation[j] <= 0.0f) stage->gradient_output[j] = 0.0f;
        }
        det_tensor_f32 input;
        if (i == 0) {
            input.data = (float *)image->data;
            input.channels = image->channels;
            input.height = image->height;
            input.width = image->width;
        } else {
            det_stage *previous = &model->stages[i - 1];
            input.data = previous->activation;
            input.channels = previous->output_channels;
            input.height = previous->output_height;
            input.width = previous->output_width;
        }
        det_tensor_f32 grad_output = {stage->gradient_output, stage->output_channels,
                                      stage->output_height, stage->output_width};
        det_tensor_f32 grad_input = {stage->gradient_input, stage->input_channels,
                                     stage->input_height, stage->input_width};
        stage_backward(stage, &input, &grad_output, &grad_input);
        if (i > 0) accumulate_previous_stage_gradient(stage, &model->stages[i - 1]);
    }
}

static void apply_stage_gradient(det_stage *stage, float lr, float momentum, float scale) {
