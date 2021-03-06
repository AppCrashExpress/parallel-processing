#include <stdio.h>
#include <memory>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cmath>

#define CSC(call)                                             \
do {                                                          \
    cudaError_t res = call;                                   \
    if (res != cudaSuccess) {                                 \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

const double EPS = 1e-7;

using matrix_p = std::unique_ptr<double[]>;

struct Comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return std::fabs(a) < std::fabs(b);
    }
};

bool is_close(double a, double b) {
    return std::fabs(a - b) < EPS;
}

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

__global__
void gpu_swap_rows(double* matrix_m, 
                   long n, long m, long k, long row_a, long row_b) {
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = idx; i < (m + k); i += stride) {
        long col = i * n;
        double t = matrix_m[col + row_a];
        matrix_m[col + row_a] = matrix_m[col + row_b];
        matrix_m[col + row_b] = t;
    }
}

__global__
void gpu_diff_rows(double* matrix_m,
               long n, long m, long k, long start_col, long max_row) {
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long idy = blockDim.y * blockIdx.y + threadIdx.y;
    long stridex = blockDim.x * gridDim.x;
    long stridey = blockDim.y * gridDim.y;

    for (long i = idy + start_col + 1; i < (m + k); i += stridey) {
        long col = i * n;
        double val = matrix_m[col + max_row] / matrix_m[start_col * n + max_row];

        for (long j = idx + max_row + 1; j < n; j += stridex) {
            matrix_m[col + j] -= val * matrix_m[start_col * n + j];
        }
    }

}

__global__
void gpu_normalize_rows(double* matrix_m,
                    long n, long m, long k, long col, long row) {
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long idy = blockDim.y * blockIdx.y + threadIdx.y;
    long stridex = blockDim.x * gridDim.x;
    long stridey = blockDim.y * gridDim.y;

    double div = matrix_m[col * n + row];
    
    for (long i = idy + col + 1; i < (m + k); i += stridey) {
        double temp = matrix_m[i * n + row];
        
        for (long j = idx; j < row; j += stridex) {
            matrix_m[i * n + j] -= temp * matrix_m[col * n + j] / div;
        }
    }
}

void reduce_gauss(double* dev_matrix_m,
                  std::vector<long>& nonzero_cols,
                  long n, long m, long k) {

    Comparator comp;
    long col = 0;

    for (long row = 0; row < n; ++row) {
        thrust::device_ptr<double> col_ptr, max_ptr;
        
        for (; col < m; ++col) {
            col_ptr = thrust::device_pointer_cast(dev_matrix_m + col * n);
            max_ptr = thrust::max_element(col_ptr + row, col_ptr + n, comp);

            if (!is_close(*max_ptr, 0.0)) {
                break;
            }
        }

        if (col == m) {
            // Cannot reduce further?
            break;
        }
        nonzero_cols.push_back(col);

        long max_row = max_ptr - col_ptr;

        if (row != max_row) {
            gpu_swap_rows<<<64, 64>>>(dev_matrix_m, n, m, k, row, max_row);
        }

        gpu_diff_rows<<<dim3(8, 64), dim3(8, 64)>>>(dev_matrix_m, n, m, k, col, row);

        ++col;
    }

}

matrix_p solve(matrix_p&& matrix_m,
               long n, long m, long k) {
    std::vector<long> nonzero_cols;
    nonzero_cols.reserve(m);

    double* dev_matrix_m;
    size_t size = sizeof(double) * (m + k) * n;
    CSC(cudaMalloc(&dev_matrix_m, size));
    CSC(cudaMemcpy(dev_matrix_m, matrix_m.get(), size, cudaMemcpyHostToDevice));

    reduce_gauss(dev_matrix_m, nonzero_cols, n, m, k);

    for (long start_row = 1; start_row < nonzero_cols.size(); ++start_row) {
        long start_col = nonzero_cols[start_row];
        gpu_normalize_rows<<<dim3(8, 64), dim3(8, 64)>>>(dev_matrix_m, n, m, k, start_col, start_row);
    }

    CSC(cudaMemcpy(matrix_m.get(), dev_matrix_m, size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_matrix_m));

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
