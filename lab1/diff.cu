#include <stdio.h>
#include <stdlib.h>

void read_vec(double* vec, unsigned int num_elements) {
    for (unsigned int i = 0; i < num_elements; ++i) {
        scanf("%lf", &vec[i]);
    }
}

__global__
void vec_diff(double* vec1, double* vec2, unsigned int num_elements) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    vec1[i] -= vec2[i];
}

int main() {
    unsigned int num_elements;
    scanf("%u", &num_elements);
    size_t mem_size = num_elements * sizeof(double);

    double *h_a = (double*) malloc(mem_size);
    double *h_b = (double*) malloc(mem_size);
    read_vec(h_a, num_elements);
    read_vec(h_b, num_elements);

    double *d_a;
    double *d_b;
    cudaMalloc(&d_a, mem_size);
    cudaMalloc(&d_b, mem_size);
    cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);

    unsigned int blockSize = 512;
    unsigned int gridSize  = num_elements / blockSize + 1;
    vec_diff<<<gridSize, blockSize>>>(d_a, d_b, num_elements);

    cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < num_elements; ++i) {
        printf("%.10lf ", h_a[i]);
    }
    printf("\n");

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
}
