det_status det_quantize_per_channel_s8(const float *weights, int rows, int cols,
                                        int8_t *quantized, float *scales) {
    if (weights == NULL || quantized == NULL || scales == NULL || rows <= 0 || cols <= 0) {
        return DET_ERR_ARGUMENT;
    }
    for (int row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = fabsf(weights[(size_t)row * (size_t)cols + (size_t)col]);
            if (!isfinite(value)) return DET_ERR_ARGUMENT;
            if (value > max_abs) max_abs = value;
        }
        scales[row] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        if (!isfinite(scales[row]) || !(scales[row] > 0.0f)) return DET_ERR_ARGUMENT;
        for (int col = 0; col < cols; ++col) {
            quantized[(size_t)row * (size_t)cols + (size_t)col] =
                det_quantize_s8(weights[(size_t)row * (size_t)cols + (size_t)col], scales[row]);
        }
    }
    return DET_OK;
}

det_status det_quantize_pack_w4(const float *weights, int rows, int cols,
                                uint8_t *packed, float *scales) {
    if (weights == NULL || packed == NULL || scales == NULL || rows <= 0 || cols <= 0) {
        return DET_ERR_ARGUMENT;
    }
    int packed_cols = (cols + 1) / 2;
    for (int row = 0; row < rows; ++row) {
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = fabsf(weights[(size_t)row * (size_t)cols + (size_t)col]);
            if (!isfinite(value)) return DET_ERR_ARGUMENT;
            if (value > max_abs) max_abs = value;
        }
        scales[row] = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
        if (!isfinite(scales[row]) || !(scales[row] > 0.0f)) return DET_ERR_ARGUMENT;
        for (int byte_index = 0; byte_index < packed_cols; ++byte_index) {
            int col0 = byte_index * 2;
            int col1 = col0 + 1;
            int8_t q0 = col0 < cols ? det_quantize_s8(
                weights[(size_t)row * (size_t)cols + (size_t)col0], scales[row]) : 0;
            int8_t q1 = col1 < cols ? det_quantize_s8(
                weights[(size_t)row * (size_t)cols + (size_t)col1], scales[row]) : 0;
            if (q0 > 7) q0 = 7;
            if (q0 < -8) q0 = -8;
            if (q1 > 7) q1 = 7;
            if (q1 < -8) q1 = -8;
            packed[(size_t)row * (size_t)packed_cols + (size_t)byte_index] =
                (uint8_t)(((uint8_t)q0 & 0x0fU) | (((uint8_t)q1 & 0x0fU) << 4U));
        }
    }
    return DET_OK;
}


