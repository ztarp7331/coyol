#include "native_graph_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int32_t index;
    int32_t in_channels;
    int32_t out_channels;
    int32_t kernel;
    int32_t stride;
    int32_t padding;
    int32_t groups;
    int32_t act; /* 1 = silu */
    int32_t bits;
    int32_t weight_offset;
    uint32_t weight_bytes;
    uint32_t bias_offset;
    uint32_t mult_len;
} Y8ConvTableRec;

static int8_t loaded_weight_at(const NGConv2D *conv, size_t index) {
    if (conv->weight_quant == NG_QUANT_INT8)
        return ((const int8_t *)conv->weights)[index];
    {
        uint8_t packed = ((const uint8_t *)conv->weights)[index >> 1U];
        int8_t nibble = (int8_t)((index & 1U) != 0U ? (packed >> 4U) : (packed & 0x0fU));
        return (nibble & 0x08) != 0 ? (int8_t)(nibble | (int8_t)0xf0) : nibble;
    }
}

static int build_f32_sidecar(NGLoadedWeights *loaded) {
    size_t total_weights = 0U;
    size_t weight_index = 0U;
    if (loaded == NULL || loaded->conv_count <= 0 || loaded->bias_count == 0U) return -1;
    for (int32_t i = 0; i < loaded->conv_count; ++i) {
        const NGConv2D *conv = &loaded->convs[i];
        size_t elements;
        if (conv->in_channels <= 0 || conv->out_channels <= 0 || conv->groups <= 0 ||
            conv->in_channels % conv->groups != 0) return -1;
        elements = (size_t)conv->out_channels *
                   (size_t)(conv->in_channels / conv->groups) *
                   (size_t)conv->kernel * (size_t)conv->kernel;
        if (elements > SIZE_MAX / sizeof(float) || total_weights > SIZE_MAX - elements)
            return -1;
        total_weights += elements;
    }
    loaded->weight_f32_blob = (float *)malloc(total_weights * sizeof(float));
    loaded->bias_f32_blob = (float *)malloc(loaded->bias_count * sizeof(float));
    if (loaded->weight_f32_blob == NULL || loaded->bias_f32_blob == NULL) {
        free(loaded->weight_f32_blob);
        free(loaded->bias_f32_blob);
        loaded->weight_f32_blob = NULL;
        loaded->bias_f32_blob = NULL;
        return -1;
    }
    loaded->weight_f32_blob_size = total_weights * sizeof(float);
    loaded->bias_f32_count = loaded->bias_count;
    for (int32_t i = 0; i < loaded->conv_count; ++i) {
        NGConv2D *conv = &loaded->convs[i];
        size_t elements = (size_t)conv->out_channels *
                          (size_t)(conv->in_channels / conv->groups) *
                          (size_t)conv->kernel * (size_t)conv->kernel;
        float input_scale = conv->input_scale > 0.0f ? conv->input_scale : loaded->act_scale;
        float output_scale = conv->output_scale > 0.0f ? conv->output_scale : loaded->act_scale;
        float *weights = loaded->weight_f32_blob + weight_index;
        if (!(input_scale > 0.0f) || !(output_scale > 0.0f)) return -1;
        for (int32_t oc = 0; oc < conv->out_channels; ++oc) {
            float weight_scale = ldexpf((float)conv->multiplier[oc],
                                        -(int)conv->shift[oc]) * output_scale / input_scale;
            size_t row = (size_t)oc * (size_t)(conv->in_channels / conv->groups) *
                         (size_t)conv->kernel * (size_t)conv->kernel;
            size_t bias_index = (size_t)(conv->bias - loaded->bias_blob) + (size_t)oc;
            if (!isfinite(weight_scale) || weight_scale == 0.0f ||
                bias_index >= loaded->bias_f32_count) return -1;
            for (size_t j = 0U; j < elements / (size_t)conv->out_channels; ++j)
                weights[row + j] = (float)loaded_weight_at(conv, row + j) * weight_scale;
            loaded->bias_f32_blob[bias_index] = (float)conv->bias[oc] * input_scale * weight_scale;
        }
        conv->weights_f32 = weights;
        conv->bias_f32 = loaded->bias_f32_blob + (size_t)(conv->bias - loaded->bias_blob);
        weight_index += elements;
    }
    return 0;
}

static int read_file(const char *path, void **data_out, size_t *size_out) {
    FILE *f;
    long sz;
    void *buf;
    if (!path || !data_out || !size_out) return -1;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return -1;
    }
    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return -1;
    }
    buf = malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *data_out = buf;
    *size_out = (size_t)sz;
    return 0;
}

static void join_path(char *out, size_t out_sz, const char *dir, const char *name) {
    size_t n = strlen(dir);
    int need_sep = n > 0 && dir[n - 1] != '/' && dir[n - 1] != '\\';
    if (need_sep)
        snprintf(out, out_sz, "%s/%s", dir, name);
    else
        snprintf(out, out_sz, "%s%s", dir, name);
}

void ng_weights_free(NGLoadedWeights *loaded) {
    if (!loaded) return;
    free(loaded->weight_blob);
    free(loaded->weight_f32_blob);
    free(loaded->bias_blob);
    free(loaded->bias_f32_blob);
    free(loaded->mult_blob);
    free(loaded->shift_blob);
    free(loaded->silu_blob);
    free(loaded->silu_index_blob);
    free(loaded->activation_scale_blob);
    memset(loaded, 0, sizeof(*loaded));
}

void ng_weights_as_model(const NGLoadedWeights *loaded, NGModelWeights *view) {
    if (!loaded || !view) return;
    view->convs = loaded->convs;
    view->conv_count = loaded->conv_count;
}

int ng_weights_load_dir_mode(const char *export_dir, NGLoadedWeights *out,
                             int load_f32) {
    char path[1024];
    void *raw = NULL;
    size_t raw_sz = 0;
    Y8ConvTableRec *table = NULL;
    size_t table_bytes = 0;
    int32_t i;
    uint8_t info[24];
    FILE *fi;

    if (!export_dir || !out) return -1;
    memset(out, 0, sizeof(*out));
    out->dfl.exp_lut_q15 = out->dfl_exp_lut;
    out->input_channels = 3;
    out->class_count = 80;
    out->scale = NG_SCALE_N;
    out->act_scale = 1.f / 127.f;
    out->weight_quant = NG_QUANT_INT8;

    join_path(path, sizeof(path), export_dir, "model_info.bin");
    fi = fopen(path, "rb");
    if (fi) {
        if (fread(info, 1, sizeof(info), fi) == sizeof(info) && memcmp(info, "Y8W1", 4) == 0) {
            uint8_t scale_code = info[4];
            uint8_t bits = info[5];
            uint16_t architecture_code;
            int32_t conv_count, class_count, input_channels;
            float act_scale;
            memcpy(&conv_count, info + 8, 4);
            memcpy(&class_count, info + 12, 4);
            memcpy(&input_channels, info + 16, 4);
            memcpy(&act_scale, info + 20, 4);
            memcpy(&architecture_code, info + 6, 2);
            if (scale_code <= 4) out->scale = (NGModelScale)scale_code;
            out->weight_quant = (bits == 4) ? NG_QUANT_INT4 : NG_QUANT_INT8;
            out->conv_count = conv_count;
            out->class_count = class_count;
            out->input_channels = input_channels;
            out->act_scale = act_scale;
            out->architecture = (int32_t)architecture_code;
        }
        fclose(fi);
    }

    join_path(path, sizeof(path), export_dir, "weights.bin");
    if (read_file(path, (void **)&out->weight_blob, &out->weight_blob_size)) goto fail;

    if (load_f32) {
        join_path(path, sizeof(path), export_dir, "weights_f32.bin");
        if (read_file(path, (void **)&out->weight_f32_blob, &out->weight_f32_blob_size)) {
            out->weight_f32_blob = NULL;
            out->weight_f32_blob_size = 0;
        }
    }

    join_path(path, sizeof(path), export_dir, "bias_i32.bin");
    if (read_file(path, (void **)&out->bias_blob, &raw_sz)) goto fail;
    out->bias_count = raw_sz / sizeof(int32_t);

    if (load_f32) {
        join_path(path, sizeof(path), export_dir, "bias_f32.bin");
        if (read_file(path, (void **)&out->bias_f32_blob, &raw_sz) == 0) {
            out->bias_f32_count = raw_sz / sizeof(float);
        }
    }

    join_path(path, sizeof(path), export_dir, "multiplier_i32.bin");
    if (read_file(path, (void **)&out->mult_blob, &raw_sz)) goto fail;
    out->mult_count = raw_sz / sizeof(int32_t);

    join_path(path, sizeof(path), export_dir, "shift_u8.bin");
    if (read_file(path, (void **)&out->shift_blob, &out->shift_count)) goto fail;

    join_path(path, sizeof(path), export_dir, "silu_lut_s8.bin");
    if (read_file(path, &raw, &raw_sz) || raw_sz < 256) goto fail;
    memcpy(out->silu_lut, raw, 256);
    out->silu_blob = (uint8_t *)raw;
    out->silu_blob_size = raw_sz;
    raw = NULL;

    join_path(path, sizeof(path), export_dir, "dfl_exp_lut_q15.bin");
    if (read_file(path, &raw, &raw_sz) || raw_sz < 512) goto fail;
    memcpy(out->dfl_exp_lut, raw, 512);
    free(raw);
    raw = NULL;

    join_path(path, sizeof(path), export_dir, "conv_table.bin");
    if (read_file(path, (void **)&table, &table_bytes)) goto fail;
    if (table_bytes % sizeof(Y8ConvTableRec)) goto fail;
    {
        int32_t n = (int32_t)(table_bytes / sizeof(Y8ConvTableRec));
        if (out->conv_count <= 0) out->conv_count = n;
        if (n != out->conv_count || n <= 0 || n > NG_LOADER_MAX_CONVS) goto fail;
    }

    join_path(path, sizeof(path), export_dir, "silu_lut_index.bin");
    if (read_file(path, (void **)&out->silu_index_blob, &out->silu_index_count)) {
        out->silu_index_blob = NULL;
        out->silu_index_count = 0;
    }
    join_path(path, sizeof(path), export_dir, "activation_scales.bin");
    if (read_file(path, (void **)&out->activation_scale_blob, &raw_sz) == 0) {
        if (raw_sz != (size_t)out->conv_count * 2U * sizeof(float)) goto fail;
        out->activation_scale_count = raw_sz / sizeof(float);
    }

    for (i = 0; i < out->conv_count; ++i) {
        const Y8ConvTableRec *r = &table[i];
        NGConv2D *c = &out->convs[i];
        size_t bias_idx = (size_t)r->bias_offset / sizeof(int32_t);
        if (r->weight_offset < 0 || (size_t)r->weight_offset + r->weight_bytes > out->weight_blob_size) goto fail;
        if (bias_idx + (size_t)r->out_channels > out->bias_count) goto fail;
        if (bias_idx + (size_t)r->out_channels > out->mult_count) goto fail;
        if (bias_idx + (size_t)r->out_channels > out->shift_count) goto fail;

        c->weights = out->weight_blob + r->weight_offset;
        c->weights_f32 = NULL;
        c->bias = out->bias_blob + bias_idx;
        c->bias_f32 = NULL;
        c->multiplier = out->mult_blob + bias_idx;
        c->shift = out->shift_blob + bias_idx;
        c->silu_lut = out->silu_lut;
        if (out->weight_f32_blob != NULL && out->bias_f32_blob != NULL) {
            size_t weight_index = 0U;
            for (int32_t j = 0; j < i; ++j) {
                const Y8ConvTableRec *previous = &table[j];
                int64_t elements = (int64_t)previous->out_channels *
                                   (previous->in_channels / previous->groups) *
                                   previous->kernel * previous->kernel;
                if (elements <= 0) goto fail;
                weight_index += (size_t)elements;
            }
            {
                int64_t elements = (int64_t)r->out_channels *
                                   (r->in_channels / r->groups) * r->kernel * r->kernel;
                if (elements <= 0 ||
                    (weight_index + (size_t)elements) * sizeof(float) > out->weight_f32_blob_size ||
                    bias_idx + (size_t)r->out_channels > out->bias_f32_count) goto fail;
                c->weights_f32 = out->weight_f32_blob + weight_index;
                c->bias_f32 = out->bias_f32_blob + bias_idx;
            }
        }
        c->input_scale = 0.0f;
        c->output_scale = 0.0f;
        if (out->activation_scale_blob && out->activation_scale_count >= (size_t)(2 * (i + 1))) {
            c->input_scale = out->activation_scale_blob[2 * i];
            c->output_scale = out->activation_scale_blob[2 * i + 1];
        }
        if (out->silu_index_blob && (size_t)i < out->silu_index_count) {
            size_t lut_offset = (size_t)out->silu_index_blob[i] * 256U;
            if (lut_offset + 256U <= out->silu_blob_size)
                c->silu_lut = (const int8_t *)(out->silu_blob + lut_offset);
        }
        c->in_channels = r->in_channels;
        c->out_channels = r->out_channels;
        c->kernel = r->kernel;
        c->stride = r->stride;
        c->padding = r->padding;
        c->groups = r->groups > 0 ? r->groups : 1;
        c->weight_quant = (r->bits == 4) ? NG_QUANT_INT4 : NG_QUANT_INT8;
        c->activation = r->act ? NG_ACT_SILU : NG_ACT_IDENTITY;
        out->weight_quant = c->weight_quant;
    }

    if (load_f32 && (out->weight_f32_blob == NULL || out->bias_f32_blob == NULL)) {
        free(out->weight_f32_blob);
        free(out->bias_f32_blob);
        out->weight_f32_blob = NULL;
        out->weight_f32_blob_size = 0U;
        out->bias_f32_blob = NULL;
        out->bias_f32_count = 0U;
        if (build_f32_sidecar(out) != 0) goto fail;
    }

    free(table);
    return 0;

fail:
    free(raw);
    free(table);
    ng_weights_free(out);
    return -1;
}

int ng_weights_load_dir(const char *export_dir, NGLoadedWeights *out) {
    return ng_weights_load_dir_mode(export_dir, out, 0);
}
