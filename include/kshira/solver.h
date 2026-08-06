#ifndef KSHIRA_SOLVER_H
#define KSHIRA_SOLVER_H

#include <stddef.h>

#include "kshira/core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Solve A X = B for a positive-definite A.  The caller owns all buffers;
 * workspace is an n*n row-major matrix used for the Cholesky factor. */
kshira_status kshira_solver_cholesky(const double *matrix, const double *rhs,
                                      size_t n, size_t columns, double *solution,
                                      double *workspace);

#ifdef __cplusplus
}
#endif

#endif
