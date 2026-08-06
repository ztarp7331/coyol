#include "kshira/budget.h"

#include <math.h>

void kshira_budget_init(kshira_budget *budget, double total_ms,
                        double reserve_ms, double start_ms) {
    if (budget == NULL) return;
    *budget = (kshira_budget){0};
    if (!isfinite(total_ms) || total_ms <= 0.0 ||
        !isfinite(reserve_ms) || reserve_ms < 0.0 ||
        !isfinite(start_ms)) return;
    if (reserve_ms >= total_ms) reserve_ms = total_ms * 0.10;
    budget->total_ms = total_ms;
    budget->reserve_ms = reserve_ms;
    budget->start_ms = start_ms;
    budget->work_ms = total_ms - reserve_ms;
    budget->last_ms = start_ms;
    budget->active = budget->work_ms > 0.0;
}

void kshira_budget_observe(kshira_budget *budget, double now_ms) {
    double delta;
    if (budget == NULL || !budget->active || !isfinite(now_ms)) return;
    if (now_ms < budget->last_ms) return;
    delta = now_ms - budget->last_ms;
    budget->last_ms = now_ms;
    if (budget->samples == 0U) budget->sample_ms_ema = delta;
    else budget->sample_ms_ema = 0.8 * budget->sample_ms_ema + 0.2 * delta;
    if (budget->samples < (size_t)-1) ++budget->samples;
}

double kshira_budget_remaining(const kshira_budget *budget, double now_ms) {
    double elapsed;
    if (budget == NULL || !budget->active || !isfinite(now_ms)) return 0.0;
    elapsed = now_ms - budget->start_ms;
    if (elapsed < 0.0) elapsed = 0.0;
    if (elapsed >= budget->work_ms) return 0.0;
    return budget->work_ms - elapsed;
}

int kshira_budget_should_stop(const kshira_budget *budget, double now_ms) {
    return budget != NULL && budget->active &&
           kshira_budget_remaining(budget, now_ms) <= 0.0;
}

int kshira_budget_stage(const kshira_budget *budget, double now_ms) {
    double elapsed;
    if (budget == NULL || !budget->active || !isfinite(now_ms)) return 2;
    elapsed = now_ms - budget->start_ms;
    if (elapsed < budget->work_ms * 0.125) return 0;
    if (elapsed < budget->work_ms * 0.375) return 1;
    return 2;
}
