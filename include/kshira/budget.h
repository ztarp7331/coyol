#ifndef KSHIRA_BUDGET_H
#define KSHIRA_BUDGET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double total_ms;
    double reserve_ms;
    double start_ms;
    double work_ms;
    double last_ms;
    double sample_ms_ema;
    size_t samples;
    int active;
} kshira_budget;

void kshira_budget_init(kshira_budget *budget, double total_ms,
                        double reserve_ms, double start_ms);
void kshira_budget_observe(kshira_budget *budget, double now_ms);
double kshira_budget_remaining(const kshira_budget *budget, double now_ms);
int kshira_budget_should_stop(const kshira_budget *budget, double now_ms);
/* 0=dense warm-up, 1=mixed, 2=residual/refinement. */
int kshira_budget_stage(const kshira_budget *budget, double now_ms);

#ifdef __cplusplus
}
#endif

#endif
