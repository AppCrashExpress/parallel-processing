#include <stdio.h>
#include <memory>
#include <utility>
#include <cmath>

const double EPS = 1e-7;

using matrix_p = std::unique_ptr<double[]>;

void read_matrix(matrix_p& matrix_m,
                 long& n, long& m, long& k) {
    scanf("%ld %ld %ld", &n, &m, &k);
    matrix_m = matrix_p(new double[(m + k) * n]);

    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < m; ++j) {
            scanf("%lf", &matrix_m[(j * n) + i]);
        }
    }

    for (long i = 0; i < n; ++i) {
        for (long j = m; j < (m + k); ++j) {
            scanf("%lf", &matrix_m[(j * n) + i]);
        }
    }
}

void write_matrix(matrix_p& matrix_x,
                  long m, long k) {

    for (long j = 0; j < m; ++j) {
        for (long i = 0; i < k; ++i) {
            printf("%.10lf ", matrix_x[i * m + j]);
        }
        printf("\n");
    }
}
bool is_close(double a, double b) {
    return std::fabs(a - b) < EPS;
}

long argmax_col_range(const matrix_p& matrix_m, long n,
                      long col, long start, long end) {
    double max = std::fabs(matrix_m[n * col + start]);
    long argmax = start;

    for (long i = start+1; i < end; ++i) {
        double val = std::fabs(matrix_m[n * col + i]);

        if (val > max) {
            max = val;
            argmax = i;
        }
    }

    return argmax;
}

void swap_rows(matrix_p& matrix_m, 
              long m, long n, long k, long row_a, long row_b) {
    for (long i = 0; i < (m + k); ++i) {
        long col = i * n;
        double t = matrix_m[col + row_a];
        matrix_m[col + row_a] = matrix_m[col + row_b];
        matrix_m[col + row_b] = t;
    }
}

void normalize_row(matrix_p& matrix_a, matrix_p& matrix_b, long n, long m, long k, long norm_col, long row) {
    double norm = matrix_a[norm_col * n + row];

    for (long i = norm_col; i < m; ++i) {
        matrix_a[i * n + row] /= norm;
    }

    for (long i = 0; i < k; ++i) {
        matrix_b[i * n + row] /= norm;
    }
}

void diff_rows(matrix_p& matrix_m,
               long n, long m, long k, long start_col, long max_row) {
    for (long i = start_col + 1; i < (m + k); ++i) {
        long col = i * n;
        double val = matrix_m[col + max_row] / matrix_m[start_col * n + max_row];

        for (long j = 0; j < n; ++j) {
            if (j == max_row) {
                // Don't let it destroy itself
                continue;
            }

            matrix_m[col + j] -= val * matrix_m[start_col * n + j];
        }
    }

}

void reduce_gauss(matrix_p& matrix_m,
                  long n, long m, long k) {

    long col = 0;

    for (long row = 0; row < n; ++row) {
        long max_row = 0;
        
        for (; col < m; ++col) {
            max_row = argmax_col_range(matrix_m, n, col, row, n);
            if (!is_close(matrix_m[n * col + max_row], 0.0)) {
                break;
            }
        }

        if (col == m) {
            // Cannot reduce further?
            break;
        }

        if (row != max_row) {
            swap_rows(matrix_m, m, n, k, row, max_row);
        }

        // normalize_row(matrix_a, matrix_b, n, m, k, max_col, row);
        diff_rows(matrix_m, n, m, k, col, row);

        ++col;
    }

}

matrix_p solve(matrix_p&& matrix_m,
               long n, long m, long k) {
    reduce_gauss(matrix_m, n, m, k);

    matrix_p matrix_x = matrix_p(new double[m * k]);

    // Scan rows for first non zero values
    long scanned_rows = 0;
    long col = 0;
    for (; col < m && scanned_rows < n; ++col) {
        if (is_close(matrix_m[col * n + scanned_rows], 0.0)) {
            for (long i = 0; i < k; ++i) {
                matrix_x[i * m + col] = 0;
            }

        } else {
            for (long i = 0; i < k; ++i) {
                matrix_x[i * m + col] = matrix_m[(m + i) * n + scanned_rows] / matrix_m[col * n + scanned_rows];
            }

            ++scanned_rows;
        }
    }

    for (; col < m; ++col) {
        for (long i = 0; i < k; ++i) {
            matrix_x[i * m + col] = 0;
        }
    }

    return matrix_x;
}

int main() {
    long n, m, k;
    // matrix_m for matrix_merged
    matrix_p matrix_m, matrix_x;
    
    read_matrix(matrix_m, n, m, k);
    matrix_x = solve(std::move(matrix_m), n, m, k);

    write_matrix(matrix_x, m, k);
}
