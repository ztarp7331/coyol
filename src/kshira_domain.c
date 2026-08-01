/* Purpose: deterministic, balanced ten-domain curriculum samples for PRE.
 * Ownership: callers provide the stream object, image buffer, and target box.
 * Failure: invalid shapes/capacity or exhausted streams return explicit status. */
#include "kshira/domain.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static uint32_t next_u32(uint32_t *state) {
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

static uint32_t pixel_hash(uint32_t seed, int x, int y, int channel) {
    uint32_t value = seed ^ ((uint32_t)x * 0x9e3779b9U) ^
                     ((uint32_t)y * 0x85ebca6bU) ^ ((uint32_t)channel * 0xc2b2ae35U);
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    return value;
}

static int inside_box(int x, int y, const kshira_rad_box *box) {
    return x >= (int)box->x1 && x < (int)box->x2 &&
           y >= (int)box->y1 && y < (int)box->y2;
}

static float domain_value(int domain, int x, int y, int channel,
                          const kshira_rad_box *box, uint32_t seed, size_t sample) {
    int left = (int)box->x1;
    int top = (int)box->y1;
    int right = (int)box->x2;
    int bottom = (int)box->y2;
    int local_x = x - left;
    int local_y = y - top;
    int width = right - left;
    int height = bottom - top;
    int dx = 2 * x - (left + right - 1);
    int dy = 2 * y - (top + bottom - 1);
    uint32_t hash = pixel_hash(seed ^ (uint32_t)sample, x, y, channel);
    float value = 0.0f;

    if (!inside_box(x, y, box)) {
        return domain == 9 ? (float)(hash % 25U) / 1000.0f : 0.0f;
    }
    switch (domain) {
        case 0: value = 1.0f; break; /* solid blob */
        case 1: value = ((local_y + (int)(sample & 7U)) % 8 < 4) ? 1.0f : 0.2f; break;
        case 2: value = ((local_x + (int)(sample & 7U)) % 8 < 4) ? 1.0f : 0.2f; break;
        case 3: value = (((local_x / 4) + (local_y / 4)) & 1) ? 0.9f : 0.15f; break;
        case 4: value = ((x + y + (int)(sample & 15U)) % 10 < 5) ? 1.0f : 0.1f; break;
        case 5: {
            int distance = dx * dx + dy * dy;
            int radius = (width < height ? width : height) / 2;
            int outer = radius * radius;
            int inner = (radius / 2) * (radius / 2);
            value = distance <= outer && distance >= inner ? 1.0f : 0.05f;
            break;
        }
        case 6: {
            int distance = dx * dx + dy * dy;
            int radius = (width < height ? width : height) / 2;
            int outer = radius * radius;
            value = outer > 0 && distance < outer ?
                    1.0f - (float)distance / (float)outer : 0.05f;
            break;
        }
        case 7: {
            int edge_x = local_x < 4 || local_x >= width - 4;
            int edge_y = local_y < 4 || local_y >= height - 4;
            value = edge_x && edge_y ? 1.0f : 0.08f;
            break;
        }
        case 8: value = (hash & 7U) == 0U ? 1.0f : 0.02f; break;
        default: value = (float)(hash % 1000U) / 999.0f; break; /* noise texture */
    }
    return value * (0.70f + 0.15f * (float)((domain + channel) % 3));
}

kshira_status kshira_domain_init(kshira_domain_stream *stream,
                                  const kshira_domain_spec *spec) {
    if (stream == NULL || spec == NULL || spec->width < 8 || spec->width > 4096 ||
        spec->height < 8 || spec->height > 4096 || spec->channels < 1 ||
        spec->channels > 4 || spec->classes < 1 || spec->classes > 80 ||
        spec->samples_per_domain == 0U ||
        spec->samples_per_domain > SIZE_MAX / (size_t)KSHIRA_DOMAIN_COUNT) {
        return KSHIRA_ERR_ARGUMENT;
    }
    stream->spec = *spec;
    stream->index = 0U;
    stream->total_samples = spec->samples_per_domain * (size_t)KSHIRA_DOMAIN_COUNT;
    return KSHIRA_OK;
}

kshira_status kshira_domain_next(kshira_domain_stream *stream, float *image,
                                  size_t image_capacity, kshira_rad_box *target,
                                  int *domain_id) {
    size_t image_elements;
    size_t sample;
    int domain;
    uint32_t state;
    int min_width;
    int min_height;
    int box_width;
    int box_height;
    if (stream == NULL || image == NULL || target == NULL || domain_id == NULL) {
        return KSHIRA_ERR_ARGUMENT;
    }
    if (stream->index >= stream->total_samples) return KSHIRA_ERR_RANGE;
    if ((size_t)stream->spec.channels > SIZE_MAX / (size_t)stream->spec.height ||
        (size_t)stream->spec.channels * (size_t)stream->spec.height >
            SIZE_MAX / (size_t)stream->spec.width) {
        return KSHIRA_ERR_RANGE;
    }
    image_elements = (size_t)stream->spec.channels * (size_t)stream->spec.height *
                     (size_t)stream->spec.width;
    if (image_capacity < image_elements) return KSHIRA_ERR_MEMORY;
    domain = (int)(stream->index % (size_t)KSHIRA_DOMAIN_COUNT);
    sample = stream->index / (size_t)KSHIRA_DOMAIN_COUNT;
    state = stream->spec.seed ^ ((uint32_t)domain * 0x9e3779b9U) ^
            (uint32_t)sample * 0x85ebca6bU;
    min_width = stream->spec.width / 8;
    min_height = stream->spec.height / 8;
    if (min_width < 2) min_width = 2;
    if (min_height < 2) min_height = 2;
    box_width = min_width + (int)(next_u32(&state) %
                                  (uint32_t)(stream->spec.width / 2 - min_width + 1));
    box_height = min_height + (int)(next_u32(&state) %
                                    (uint32_t)(stream->spec.height / 2 - min_height + 1));
    target->x1 = (float)(next_u32(&state) % (uint32_t)(stream->spec.width - box_width + 1));
    target->y1 = (float)(next_u32(&state) % (uint32_t)(stream->spec.height - box_height + 1));
    target->x2 = target->x1 + (float)box_width;
    target->y2 = target->y1 + (float)box_height;
    target->class_id = domain % stream->spec.classes;
    if (domain != 9) {
        memset(image, 0, image_elements * sizeof(*image));
    }
    for (int channel = 0; channel < stream->spec.channels; ++channel) {
        if (domain == 9) {
            for (int y = 0; y < stream->spec.height; ++y) {
                for (int x = 0; x < stream->spec.width; ++x) {
                    size_t index = ((size_t)channel * (size_t)stream->spec.height +
                                    (size_t)y) * (size_t)stream->spec.width + (size_t)x;
                    image[index] = (float)(pixel_hash(stream->spec.seed ^ (uint32_t)sample,
                                                       x, y, channel) % 25U) / 1000.0f;
                }
            }
        }
        for (int y = (int)target->y1; y < (int)target->y2; ++y) {
            for (int x = (int)target->x1; x < (int)target->x2; ++x) {
                image[((size_t)channel * (size_t)stream->spec.height + (size_t)y) *
                      (size_t)stream->spec.width + (size_t)x] = domain_value(
                          domain, x, y, channel, target, stream->spec.seed, sample);
            }
        }
    }
    *domain_id = domain;
    ++stream->index;
    return KSHIRA_OK;
}

size_t kshira_domain_total(const kshira_domain_stream *stream) {
    return stream == NULL ? 0U : stream->total_samples;
}

size_t kshira_domain_index(const kshira_domain_stream *stream) {
    return stream == NULL ? 0U : stream->index;
}
