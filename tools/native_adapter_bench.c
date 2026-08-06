#include "det.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    det_context *context = NULL;
    det_model *model = NULL;
    det_manifest_dataset *manifest = NULL;
    det_dataset dataset;
    det_model_spec spec;
    det_train_config config;
    det_train_report report;
    det_memory_report memory;
    float gain = 0.0f;
    float bias = 0.0f;
    det_status status;
    int budget_ms = 30000;

    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s NATIVE_EXPORT_DIR TRAIN_MANIFEST [budget_ms] [save_path]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4) budget_ms = atoi(argv[3]);
    if (budget_ms <= 0) return EXIT_FAILURE;
    status = det_context_create(256U * 1024U, &context);
    if (status == DET_OK) {
        status = det_manifest_open(argv[2], 160, 160, 1, 100, &manifest);
    }
    if (status == DET_OK) status = det_manifest_dataset_view(manifest, &dataset);
    if (status != DET_OK) goto fail;
    spec = (det_model_spec){160, 160, 1, 5, 100, 7, DET_ARCH_KSHIRA,
                            8, DET_STEM_LEGACY, 0, 0, DET_RESEARCH_ADAPT,
                            0, 0, 0, 0, argv[1]};
    status = det_model_build(context, &spec, &model);
    if (status != DET_OK) goto fail;
    memset(&config, 0, sizeof(config));
    config.mode = DET_TRAIN_LOCAL_FAST;
    config.precision = DET_PRECISION_F32;
    config.epochs = 1000;
    config.learning_rate = 0.10f;
    config.max_samples = 0;
    config.max_total_samples = 0;
    config.time_budget_ms = (double)budget_ms;
    config.seed = 7;
    config.reset_weights = 0;
    status = det_train(model, &dataset, &config, &report);
    if (status != DET_OK) goto fail;
    status = det_model_native_graph_calibration(model, &gain, &bias);
    if (status != DET_OK) goto fail;
    status = det_model_memory(model, &memory);
    if (status != DET_OK) goto fail;
    if (argc == 5) {
        status = det_save(model, argv[4]);
        if (status != DET_OK) goto fail;
    }
    printf("samples=%zu updates=%zu elapsed_ms=%.3f loss=%.6f budget_stop=%d\n",
           report.samples_seen, report.updates, report.elapsed_ms,
           report.mean_loss, report.stopped_by_budget);
    printf("score_gain=%.6f score_bias=%.6f\n", gain, bias);
    printf("activation_workspace_bytes=%zu arena_high_water_bytes=%zu "
           "arena_capacity_bytes=%zu\n",
           memory.activation_workspace_bytes, memory.arena_high_water_bytes,
           memory.arena_capacity_bytes);
    det_model_destroy(model);
    det_manifest_close(manifest);
    det_context_destroy(context);
    return EXIT_SUCCESS;

fail:
    fprintf(stderr, "native adapter bench failed: %d\n", status);
    det_model_destroy(model);
    det_manifest_close(manifest);
    det_context_destroy(context);
    return EXIT_FAILURE;
}
