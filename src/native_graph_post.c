#include "native_graph_post.h"

#include <float.h>
#include <math.h>
#include <string.h>

#define NG_REG_MAX 16

static int8_t u8_to_s8_act(uint8_t p) {
    /* q â‰ˆ round(pixel * 127 / 255); pairs with act_scale = 1/127 on f = p/255 */
    int32_t q = ((int32_t)p * 127 + 127) / 255;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static int8_t u8_to_s8_scaled(uint8_t p, float act_scale) {
    float q;
    if (act_scale <= 0.0f) act_scale = 1.0f / 127.0f;
    if (act_scale == 1.0f / 127.0f) return u8_to_s8_act(p);
    q = ((float)p / 255.0f) / act_scale;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (int8_t)(q >= 0.0f ? q + 0.5f : q - 0.5f);
}

static int8_t f32_to_s8_scaled(float p, float act_scale) {
    float q;
    if (act_scale <= 0.0f) act_scale = 1.0f / 127.0f;
    q = (p / 255.0f) / act_scale;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (int8_t)(q >= 0.0f ? q + 0.5f : q - 0.5f);
}

static void fill_nchw_constant(int8_t *dst, int32_t c, int32_t h, int32_t w, int8_t value) {
    int64_t n = (int64_t)c * h * w;
    for (int64_t i = 0; i < n; ++i) dst[i] = value;
}

static void fill_nchw_constant_f32(float *dst, int32_t c, int32_t h, int32_t w, float value) {
    int64_t n = (int64_t)c * h * w;
    for (int64_t i = 0; i < n; ++i) dst[i] = value;
}

int ng_letterbox_u8_to_s8(const uint8_t *src, int32_t src_w, int32_t src_h,
                          int8_t *dst_nchw, int32_t imgsz, int32_t out_channels,
                          uint8_t pad_value, Y8LetterboxInfo *info) {
    return ng_letterbox_u8_to_s8_scaled(src, src_w, src_h, dst_nchw, imgsz,
                                        out_channels, pad_value, 1.0f / 127.0f, info);
}

int ng_letterbox_u8_to_s8_scaled(const uint8_t *src, int32_t src_w, int32_t src_h,
                                 int8_t *dst_nchw, int32_t imgsz, int32_t out_channels,
                                 uint8_t pad_value, float act_scale, Y8LetterboxInfo *info) {
    float r;
    int32_t new_w, new_h, left, top;
    int8_t pad_q;
    if (!src || !dst_nchw || !info || src_w <= 0 || src_h <= 0 || imgsz <= 0 || (imgsz % 32) ||
        (out_channels != 1 && out_channels != 3))
        return -1;

    r = (float)imgsz / (float)src_w;
    if ((float)imgsz / (float)src_h < r) r = (float)imgsz / (float)src_h;
    new_w = (int32_t)((float)src_w * r + 0.5f);
    new_h = (int32_t)((float)src_h * r + 0.5f);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;
    left = (imgsz - new_w) / 2;
    top = (imgsz - new_h) / 2;
    pad_q = u8_to_s8_scaled(pad_value, act_scale);

    fill_nchw_constant(dst_nchw, out_channels, imgsz, imgsz, pad_q);

    for (int32_t y = 0; y < new_h; ++y) {
        float source_y = ((float)y + 0.5f) / r - 0.5f;
        int32_t sy0 = (int32_t)floorf(source_y);
        int32_t sy1;
        float wy;
        if (sy0 < 0) sy0 = 0;
        if (sy0 >= src_h) sy0 = src_h - 1;
        sy1 = sy0 + 1 < src_h ? sy0 + 1 : sy0;
        wy = source_y - floorf(source_y);
        if (wy < 0.0f) wy = 0.0f;
        if (wy > 1.0f) wy = 1.0f;
        for (int32_t x = 0; x < new_w; ++x) {
            float source_x = ((float)x + 0.5f) / r - 0.5f;
            int32_t sx0 = (int32_t)floorf(source_x);
            int32_t sx1;
            float wx;
            float blend_top;
            float blend_bottom;
            float pixel;
            int8_t q;
            int32_t dx, dy;
            if (sx0 < 0) sx0 = 0;
            if (sx0 >= src_w) sx0 = src_w - 1;
            sx1 = sx0 + 1 < src_w ? sx0 + 1 : sx0;
            wx = source_x - floorf(source_x);
            if (wx < 0.0f) wx = 0.0f;
            if (wx > 1.0f) wx = 1.0f;
            blend_top = (float)src[sy0 * src_w + sx0] * (1.0f - wx) +
                        (float)src[sy0 * src_w + sx1] * wx;
            blend_bottom = (float)src[sy1 * src_w + sx0] * (1.0f - wx) +
                           (float)src[sy1 * src_w + sx1] * wx;
            pixel = blend_top * (1.0f - wy) + blend_bottom * wy;
            q = f32_to_s8_scaled(pixel, act_scale);
            dy = top + y;
            dx = left + x;
            if (dy < 0 || dy >= imgsz || dx < 0 || dx >= imgsz) continue;
            for (int32_t c = 0; c < out_channels; ++c)
                dst_nchw[(c * imgsz + dy) * imgsz + dx] = q;
        }
    }

    info->orig_w = src_w;
    info->orig_h = src_h;
    info->imgsz = imgsz;
    info->scale = r;
    info->pad_x = left;
    info->pad_y = top;
    return 0;
}

int ng_letterbox_u8_to_f32(const uint8_t *src, int32_t src_w, int32_t src_h,
                           float *dst_nchw, int32_t imgsz, int32_t out_channels,
                           uint8_t pad_value, Y8LetterboxInfo *info) {
    float r;
    int32_t new_w, new_h, left, top;
    if (!src || !dst_nchw || !info || src_w <= 0 || src_h <= 0 || imgsz <= 0 ||
        (out_channels != 1 && out_channels != 3)) return -1;
    r = (float)imgsz / (float)src_w;
    if ((float)imgsz / (float)src_h < r) r = (float)imgsz / (float)src_h;
    new_w = (int32_t)((float)src_w * r + 0.5f);
    new_h = (int32_t)((float)src_h * r + 0.5f);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;
    left = (imgsz - new_w) / 2;
    top = (imgsz - new_h) / 2;
    fill_nchw_constant_f32(dst_nchw, out_channels, imgsz, imgsz,
                           (float)pad_value / 255.0f);
    for (int32_t y = 0; y < new_h; ++y) {
        float source_y = ((float)y + 0.5f) / r - 0.5f;
        int32_t sy0 = (int32_t)floorf(source_y);
        int32_t sy1;
        float wy = source_y - floorf(source_y);
        if (sy0 < 0) sy0 = 0;
        if (sy0 >= src_h) sy0 = src_h - 1;
        sy1 = sy0 + 1 < src_h ? sy0 + 1 : sy0;
        if (wy < 0.0f) wy = 0.0f;
        if (wy > 1.0f) wy = 1.0f;
        for (int32_t x = 0; x < new_w; ++x) {
            float source_x = ((float)x + 0.5f) / r - 0.5f;
            int32_t sx0 = (int32_t)floorf(source_x);
            int32_t sx1;
            float wx = source_x - floorf(source_x);
            float blend_top;
            float blend_bottom;
            float pixel;
            int32_t dx;
            int32_t dy;
            if (sx0 < 0) sx0 = 0;
            if (sx0 >= src_w) sx0 = src_w - 1;
            sx1 = sx0 + 1 < src_w ? sx0 + 1 : sx0;
            if (wx < 0.0f) wx = 0.0f;
            if (wx > 1.0f) wx = 1.0f;
            blend_top = (float)src[sy0 * src_w + sx0] * (1.0f - wx) +
                        (float)src[sy0 * src_w + sx1] * wx;
            blend_bottom = (float)src[sy1 * src_w + sx0] * (1.0f - wx) +
                           (float)src[sy1 * src_w + sx1] * wx;
            pixel = (blend_top * (1.0f - wy) + blend_bottom * wy) / 255.0f;
            dy = top + y;
            dx = left + x;
            if (dy < 0 || dy >= imgsz || dx < 0 || dx >= imgsz) continue;
            for (int32_t c = 0; c < out_channels; ++c)
                dst_nchw[(c * imgsz + dy) * imgsz + dx] = pixel;
        }
    }
    info->orig_w = src_w;
    info->orig_h = src_h;
    info->imgsz = imgsz;
    info->scale = r;
    info->pad_x = left;
    info->pad_y = top;
    return 0;
}

int ng_letterbox_u16_to_s8(const uint16_t *src, int32_t src_w, int32_t src_h, uint16_t max_value,
                           int8_t *dst_nchw, int32_t imgsz, int32_t out_channels,
                           uint8_t pad_value, Y8LetterboxInfo *info) {
    static uint8_t u8buf[640 * 640];
    int64_t n;
    uint32_t mv;
    if (!src || max_value == 0 || src_w <= 0 || src_h <= 0) return -1;
    n = (int64_t)src_w * src_h;
    if (n > (int64_t)sizeof(u8buf)) return -1;
    mv = max_value;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t v = src[i] > mv ? mv : src[i];
        u8buf[i] = (uint8_t)((v * 255u) / mv);
    }
    return ng_letterbox_u8_to_s8(u8buf, src_w, src_h, dst_nchw, imgsz, out_channels, pad_value, info);
}

uint16_t ng_conf_to_q15(float conf) {
    if (conf <= 0.f) return 0;
    if (conf >= 1.f) return 32767;
    return (uint16_t)(conf * 32767.f + 0.5f);
}

uint16_t ng_iou_to_q15(float iou) {
    if (iou <= 0.f) return 0;
    if (iou >= 1.f) return 32767;
    return (uint16_t)(iou * 32767.f + 0.5f);
}

/* Host-friendly sigmoid; replace with LUT for HLS. logit scaled by 1/32. */
static int16_t sigmoid_s8_to_q15(int8_t logit, float logit_scale) {
    float z = (float)logit * (logit_scale > 0.0f ? logit_scale : 1.0f / 32.0f);
    float ex = 1.f;
    float p = 1.f;
    float az = z < 0.f ? -z : z;
    float s;
    for (int k = 1; k <= 10; ++k) {
        p *= az / (float)k;
        ex += p;
    }
    if (z >= 0.f)
        s = ex / (1.f + ex);
    else
        s = 1.f / (1.f + ex);
    return (int16_t)(s * 32767.f + 0.5f);
}

int ng_detect_post_s8(const NGModelOutput *raw, int32_t class_count,
                      const Y8DetectPostConfig *cfg, Y8DetectResult *out) {
    NGDetection candidates[NG_MAX_CANDIDATES];
    int32_t cand = 0;
    int32_t max_cand;
    int16_t conf_q15;
    uint16_t iou_q15;
    int32_t kept;
    int32_t keep_cap;

    if (!raw || !cfg || !cfg->dfl || !out || class_count <= 0) return -1;
    max_cand = cfg->max_candidates > 0 && cfg->max_candidates < NG_MAX_CANDIDATES ? cfg->max_candidates
                                                                                  : NG_MAX_CANDIDATES;
    keep_cap = cfg->max_keep > 0 && cfg->max_keep < NG_MAX_KEEP ? cfg->max_keep : NG_MAX_KEEP;
    conf_q15 = (int16_t)ng_conf_to_q15(cfg->conf_threshold);
    iou_q15 = ng_iou_to_q15(cfg->iou_threshold);

    for (int32_t lvl = 0; lvl < 3; ++lvl) {
        const NGTensor *t = &raw->prediction[lvl];
        int32_t stride = raw->stride[lvl];
        int32_t reg_ch = 4 * NG_REG_MAX;
        if (!t->data || t->n != 1 || t->c != reg_ch + class_count || stride <= 0) return -1;
        for (int32_t gy = 0; gy < t->h; ++gy) {
            for (int32_t gx = 0; gx < t->w; ++gx) {
                int32_t best_cls = 0;
                int16_t best_score = -1;
                int8_t logits[4 * NG_REG_MAX];
                NGDetection det;
                for (int32_t c = 0; c < class_count; ++c) {
                    int64_t idx = ((int64_t)(reg_ch + c) * t->h + gy) * t->w + gx;
                    int16_t sc = sigmoid_s8_to_q15(t->data[idx], cfg->logit_scale);
                    if (sc > best_score) {
                        best_score = sc;
                        best_cls = c;
                    }
                }
                if (best_score < conf_q15) continue;
                if (cand >= max_cand) goto do_nms;
                for (int32_t k = 0; k < 4 * NG_REG_MAX; ++k) {
                    int64_t idx = ((int64_t)k * t->h + gy) * t->w + gx;
                    logits[k] = t->data[idx];
                }
                if (ng_decode_dfl_s8(logits, gx, gy, stride, cfg->dfl, &det)) return -1;
                det.score_q15 = best_score;
                det.class_id = (int16_t)best_cls;
                candidates[cand++] = det;
            }
        }
    }
do_nms:
    kept = ng_nms(candidates, cand, iou_q15, out->dets, keep_cap);
    if (kept < 0) return -1;
    out->count = kept;
    return 0;
}

static float f32_sigmoid(float value) {
    if (value >= 0.0f) {
        float e = expf(-value);
        return 1.0f / (1.0f + e);
    }
    {
        float e = expf(value);
        return e / (1.0f + e);
    }
}

static float f32_iou(const NGF32Detection *left, const NGF32Detection *right) {
    float x1 = fmaxf(left->x1, right->x1);
    float y1 = fmaxf(left->y1, right->y1);
    float x2 = fminf(left->x2, right->x2);
    float y2 = fminf(left->y2, right->y2);
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_left = fmaxf(0.0f, left->x2 - left->x1) *
                      fmaxf(0.0f, left->y2 - left->y1);
    float area_right = fmaxf(0.0f, right->x2 - right->x1) *
                       fmaxf(0.0f, right->y2 - right->y1);
    float union_area = area_left + area_right - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static int f32_decode_dfl(const float *logits, int32_t grid_x, int32_t grid_y,
                          int32_t stride, NGF32Detection *out) {
    float distance[4];
    if (!logits || !out || stride <= 0) return -1;
    for (int32_t side = 0; side < 4; ++side) {
        float maximum = -FLT_MAX;
        float denominator = 0.0f;
        float numerator = 0.0f;
        for (int32_t bin = 0; bin < 16; ++bin) {
            if (logits[side * 16 + bin] > maximum) maximum = logits[side * 16 + bin];
        }
        for (int32_t bin = 0; bin < 16; ++bin) {
            float probability = expf(logits[side * 16 + bin] - maximum);
            denominator += probability;
            numerator += (float)bin * probability;
        }
        if (!(denominator > 0.0f)) return -1;
        distance[side] = numerator / denominator * (float)stride;
    }
    {
        float center_x = ((float)grid_x + 0.5f) * (float)stride;
        float center_y = ((float)grid_y + 0.5f) * (float)stride;
        out->x1 = center_x - distance[0];
        out->y1 = center_y - distance[1];
        out->x2 = center_x + distance[2];
        out->y2 = center_y + distance[3];
    }
    return 0;
}

int ng_detect_post_f32(const NGF32ModelOutput *raw, int32_t class_count,
                       const Y8F32DetectPostConfig *cfg, Y8F32DetectResult *out) {
    NGF32Detection candidates[NG_MAX_CANDIDATES];
    NGF32Detection kept[NG_MAX_KEEP];
    int32_t candidate_count = 0;
    int32_t keep_count = 0;
    int32_t max_candidates;
    int32_t max_keep;
    if (!raw || !cfg || !out || class_count <= 0 || class_count > 80 ||
        cfg->conf_threshold < 0.0f || cfg->conf_threshold > 1.0f ||
        cfg->iou_threshold < 0.0f || cfg->iou_threshold > 1.0f) return -1;
    max_candidates = cfg->max_candidates > 0 && cfg->max_candidates < NG_MAX_CANDIDATES ?
                     cfg->max_candidates : NG_MAX_CANDIDATES;
    max_keep = cfg->max_keep > 0 && cfg->max_keep < NG_MAX_KEEP ? cfg->max_keep : NG_MAX_KEEP;
    for (int32_t level = 0; level < 3; ++level) {
        const NGF32Tensor *tensor = &raw->prediction[level];
        int32_t stride = raw->stride[level];
        int32_t regression_channels = 4 * 16;
        if (!tensor->data || tensor->n != 1 || tensor->c != regression_channels + class_count || stride <= 0)
            return -1;
        for (int32_t y = 0; y < tensor->h; ++y) {
            for (int32_t x = 0; x < tensor->w; ++x) {
                int32_t best_class = 0;
                float best_score = -1.0f;
                float logits[4 * 16];
                NGF32Detection candidate;
                for (int32_t class_id = 0; class_id < class_count; ++class_id) {
                    int64_t index = ((int64_t)(regression_channels + class_id) * tensor->h + y) * tensor->w + x;
                    float score = f32_sigmoid(tensor->data[index]);
                    if (score > best_score) {
                        best_score = score;
                        best_class = class_id;
                    }
                }
                if (best_score < cfg->conf_threshold) continue;
                if (candidate_count >= max_candidates) goto do_f32_nms;
                for (int32_t k = 0; k < regression_channels; ++k) {
                    int64_t index = ((int64_t)k * tensor->h + y) * tensor->w + x;
                    logits[k] = tensor->data[index];
                }
                if (f32_decode_dfl(logits, x, y, stride, &candidate)) return -1;
                candidate.score = best_score;
                candidate.class_id = best_class;
                candidates[candidate_count++] = candidate;
            }
        }
    }
do_f32_nms:
    for (int32_t i = 0; i < candidate_count; ++i) {
        for (int32_t j = i + 1; j < candidate_count; ++j) {
            if (candidates[j].score > candidates[i].score) {
                NGF32Detection temp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = temp;
            }
        }
    }
    for (int32_t i = 0; i < candidate_count && keep_count < max_keep; ++i) {
        int suppress = 0;
        for (int32_t j = 0; j < keep_count; ++j) {
            if (candidates[i].class_id == kept[j].class_id &&
                f32_iou(&candidates[i], &kept[j]) > cfg->iou_threshold) {
                suppress = 1;
                break;
            }
        }
        if (!suppress) kept[keep_count++] = candidates[i];
    }
    memcpy(out->dets, kept, (size_t)keep_count * sizeof(kept[0]));
    out->count = keep_count;
    return 0;
}

int ng_map_boxes_to_original(NGDetection *dets, int32_t count, const Y8LetterboxInfo *lb) {
    if (!dets || !lb || count < 0 || lb->scale <= 0.f) return -1;
    for (int32_t i = 0; i < count; ++i) {
        float x1 = (float)dets[i].x1_q12 / 4096.f;
        float y1 = (float)dets[i].y1_q12 / 4096.f;
        float x2 = (float)dets[i].x2_q12 / 4096.f;
        float y2 = (float)dets[i].y2_q12 / 4096.f;
        x1 = (x1 - (float)lb->pad_x) / lb->scale;
        y1 = (y1 - (float)lb->pad_y) / lb->scale;
        x2 = (x2 - (float)lb->pad_x) / lb->scale;
        y2 = (y2 - (float)lb->pad_y) / lb->scale;
        dets[i].x1_q12 = (int32_t)(x1 * 4096.f);
        dets[i].y1_q12 = (int32_t)(y1 * 4096.f);
        dets[i].x2_q12 = (int32_t)(x2 * 4096.f);
        dets[i].y2_q12 = (int32_t)(y2 * 4096.f);
    }
    return 0;
}

int ng_check_docking_pattern(const NGDetection *dets, int32_t count, int16_t class_id,
                             int32_t expected_count, int32_t spacing_tol_q12,
                             int32_t *pose_out6) {
    int32_t idx[16];
    int32_t n = 0;
    int32_t cx = 0, cy = 0;
    int64_t sum = 0;
    int32_t pairs = 0;
    int32_t min_d = 0x7fffffff, max_d = 0;

    if (!dets || count < 0 || expected_count < 1 || expected_count > 16) return -1;
    for (int32_t i = 0; i < count && n < 16; ++i) {
        if (dets[i].class_id == class_id) idx[n++] = i;
    }
    if (n != expected_count) return 0;

    for (int32_t i = 0; i < n; ++i) {
        const NGDetection *d = &dets[idx[i]];
        cx += (d->x1_q12 + d->x2_q12) / 2;
        cy += (d->y1_q12 + d->y2_q12) / 2;
    }
    cx /= n;
    cy /= n;

    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = i + 1; j < n; ++j) {
            const NGDetection *a = &dets[idx[i]];
            const NGDetection *b = &dets[idx[j]];
            int32_t ax = (a->x1_q12 + a->x2_q12) / 2;
            int32_t ay = (a->y1_q12 + a->y2_q12) / 2;
            int32_t bx = (b->x1_q12 + b->x2_q12) / 2;
            int32_t by = (b->y1_q12 + b->y2_q12) / 2;
            int32_t dx = ax - bx, dy = ay - by;
            int32_t dist = (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
            sum += dist;
            if (dist < min_d) min_d = dist;
            if (dist > max_d) max_d = dist;
            pairs++;
        }
    }
    if (pairs < 1) return 0;
    if (spacing_tol_q12 > 0 && (max_d - min_d) > spacing_tol_q12) return 0;
    if (pose_out6) {
        pose_out6[0] = cx;
        pose_out6[1] = cy;
        pose_out6[2] = (int32_t)(sum / pairs);
        pose_out6[3] = 0;
        pose_out6[4] = dets[idx[0]].score_q15;
        pose_out6[5] = n;
    }
    return 1;
}
