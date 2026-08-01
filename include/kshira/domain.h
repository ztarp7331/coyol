#ifndef KSHIRA_DOMAIN_H
#define KSHIRA_DOMAIN_H

#include <stddef.h>
#include <stdint.h>

#include "kshira/rad.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { KSHIRA_DOMAIN_COUNT = 10 };

typedef struct {
    int width;
    int height;
    int channels;
    int classes;
    size_t samples_per_domain;
    uint32_t seed;
} kshira_domain_spec;

typedef struct {
    kshira_domain_spec spec;
    size_t index;
    size_t total_samples;
} kshira_domain_stream;

kshira_status kshira_domain_init(kshira_domain_stream *stream,
                                  const kshira_domain_spec *spec);
kshira_status kshira_domain_next(kshira_domain_stream *stream, float *image,
                                  size_t image_capacity, kshira_rad_box *target,
                                  int *domain_id);
size_t kshira_domain_total(const kshira_domain_stream *stream);
size_t kshira_domain_index(const kshira_domain_stream *stream);

#ifdef __cplusplus
}
#endif

#endif
