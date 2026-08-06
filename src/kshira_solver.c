#include "kshira/solver.h"

#include <math.h>
#include <string.h>

kshira_status kshira_solver_cholesky(const double *matrix, const double *rhs,
                                      size_t n, size_t columns, double *solution,
                                      double *workspace) {
    if (matrix == NULL || rhs == NULL || solution == NULL || workspace == NULL ||
        n == 0U || columns == 0U || n > SIZE_MAX / n ||
        n * n > SIZE_MAX / sizeof(*workspace) ||
        n > SIZE_MAX / columns || n * columns > SIZE_MAX / sizeof(*solution)) {
        return KSHIRA_ERR_ARGUMENT;
    }
    memcpy(workspace, matrix, n * n * sizeof(*workspace));
    for (size_t row = 0U; row < n; ++row) {
        for (size_t col = 0U; col < row; ++col) {
            double value = workspace[row * n + col];
            for (size_t k = 0U; k < col; ++k) {
                value -= workspace[row * n + k] * workspace[col * n + k];
            }
            if (!isfinite(value) || workspace[col * n + col] <= 0.0) {
                return KSHIRA_ERR_RANGE;
            }
            workspace[row * n + col] = value / workspace[col * n + col];
        }
        {
            double diagonal = workspace[row * n + row];
            for (size_t k = 0U; k < row; ++k) {
                diagonal -= workspace[row * n + k] * workspace[row * n + k];
            }
            if (!isfinite(diagonal) || diagonal <= 1.0e-12) return KSHIRA_ERR_RANGE;
            workspace[row * n + row] = sqrt(diagonal);
        }
        for (size_t col = row + 1U; col < n; ++col) workspace[row * n + col] = 0.0;
    }
    for (size_t col = 0U; col < columns; ++col) {
        for (size_t row = 0U; row < n; ++row) {
            double value = rhs[row * columns + col];
            for (size_t k = 0U; k < row; ++k) {
                value -= workspace[row * n + k] * solution[k * columns + col];
            }
            if (!isfinite(value)) return KSHIRA_ERR_RANGE;
            solution[row * columns + col] = value / workspace[row * n + row];
        }
        for (size_t row = n; row > 0U; --row) {
            size_t index = row - 1U;
            double value = solution[index * columns + col];
            for (size_t k = index + 1U; k < n; ++k) {
                value -= workspace[k * n + index] * solution[k * columns + col];
            }
            if (!isfinite(value)) return KSHIRA_ERR_RANGE;
            solution[index * columns + col] = value / workspace[index * n + index];
        }
    }
    return KSHIRA_OK;
}
