#include "det.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    det_context *context = NULL;
    det_model *model = NULL;
    det_model *loaded = NULL;
    det_manifest_dataset *manifest = NULL;
    det_dataset dataset;
    det_sample sample;
    det_detection detections_before[64];
    det_detection detections_after[64];
    det_model_spec spec;
    int count_before = 0;
    int count_after = 0;
    const char *checkpoint = "runs/native_lifecycle_smoke.cdet";
    det_status status;

    if (argc != 3) {
        fprintf(stderr, "usage: %s NATIVE_EXPORT_DIR MANIFEST\n", argv[0]);
        return EXIT_FAILURE;
    }
    status = det_context_create(256U * 1024U, &context);
    if (status != DET_OK) return EXIT_FAILURE;
    status = det_manifest_open(argv[2], 160, 160, 1, 100, &manifest);
    if (status == DET_OK) status = det_manifest_dataset_view(manifest, &dataset);
    if (status != DET_OK || dataset.next(dataset.user, &sample) <= 0) goto fail;
    spec = (det_model_spec){160, 160, 1, 5, 64, 7, DET_ARCH_KSHIRA,
                            8, DET_STEM_LEGACY, 0, 0, DET_RESEARCH_ADAPT,
                            0, 0, 0, 0, argv[1]};
    status = det_model_build(context, &spec, &model);
    if (status != DET_OK) goto fail;
    status = det_predict(model, &sample.image, 0.001f, detections_before, 64,
                         &count_before);
    if (status != DET_OK) goto fail;
    status = det_save(model, checkpoint);
    if (status != DET_OK) goto fail;
    status = det_load(context, checkpoint, &loaded);
    if (status != DET_OK) goto fail;
    status = det_predict(loaded, &sample.image, 0.001f, detections_after, 64,
                         &count_after);
    if (status != DET_OK) goto fail;
    if (count_before != count_after) {
        status = DET_ERR_FORMAT;
        goto fail;
    }
    for (int i = 0; i < count_before; ++i) {
        const det_detection *before = &detections_before[i];
        const det_detection *after = &detections_after[i];
        if (before->box.class_id != after->box.class_id ||
            before->score != after->score || before->box.x1 != after->box.x1 ||
            before->box.y1 != after->box.y1 || before->box.x2 != after->box.x2 ||
            before->box.y2 != after->box.y2) {
            status = DET_ERR_FORMAT;
            goto fail;
        }
    }
    printf("before=%d after=%d checkpoint_roundtrip=1\n", count_before, count_after);
    remove(checkpoint);
    det_model_destroy(loaded);
    det_model_destroy(model);
    det_manifest_close(manifest);
    det_context_destroy(context);
    return EXIT_SUCCESS;

fail:
    fprintf(stderr, "native lifecycle smoke failed: %d\n", status);
    remove(checkpoint);
    det_model_destroy(loaded);
    det_model_destroy(model);
    det_manifest_close(manifest);
    det_context_destroy(context);
    return EXIT_FAILURE;
}
