#include <stdio.h>
#include <stdlib.h>

void read_vec(double* vec, unsigned int num_elements) {
    for (unsigned int i = 0; i < num_elements; ++i) {
        scanf("%lf", &vec[i]);
    }
}

__global__
void vec_diff(double* vec1, double* vec2, unsigned int num_elements) {
    unsigned int thread_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride   = blockDim.x * gridDim.x;
    for (unsigned int i = thread_i; i < num_elements; i += stride) {
        vec1[i] -= vec2[i];
    }
}

int main() {
    unsigned int num_elements;
    scanf("%u", &num_elements);
    size_t mem_size = num_elements * sizeof(double);

    double *buffer = (double*) malloc(mem_size);
    double *d_a;
    double *d_b;

    read_vec(buffer, num_elements);
    cudaMalloc(&d_a, mem_size);
    cudaMemcpy(d_a, buffer, mem_size, cudaMemcpyHostToDevice);

    read_vec(buffer, num_elements);
    cudaMalloc(&d_b, mem_size);
    cudaMemcpy(d_b, buffer, mem_size, cudaMemcpyHostToDevice);

    vec_diff<<<512, 512>>>(d_a, d_b, num_elements);

    cudaMemcpy(buffer, d_a, mem_size, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < num_elements; ++i) {
        printf("%.10lf ", buffer[i]);
    }
    printf("\n");

    free(buffer);
    cudaFree(d_a);
    cudaFree(d_b);
}
