#include <iostream>
#include <memory>
#include <limits>
#include <cstring>
#include <utility>

#define CSC(call)                                             \
do {                                                          \
    cudaError_t res = call;                                   \
    if (res != cudaSuccess) {                                 \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

using ElemT = int;
using ArrayPtr = std::unique_ptr<ElemT[]>;

const ElemT MAX_VALUE = std::numeric_limits<ElemT>::max();
const size_t BLOCK_SIZE = 256; // Must be power of 2


void read_array(ArrayPtr& array_ptr, ElemT& size, std::istream& is) {
    is.read(reinterpret_cast<char*>(&size), sizeof(ElemT));
    array_ptr = ArrayPtr(new ElemT[size]);
    is.read(reinterpret_cast<char*>(array_ptr.get()), size * sizeof(ElemT));
}

void write_array(const ArrayPtr& array_ptr, const ElemT& size, std::ostream& os) {
    os.write(reinterpret_cast<char*>(array_ptr.get()), size * sizeof(ElemT));
}

ElemT pad_array(ArrayPtr& array_ptr, ElemT real_size) {
    if (real_size % BLOCK_SIZE == 0) {
        return real_size;
    }

    int padded_size = (real_size / BLOCK_SIZE + 1) * BLOCK_SIZE;
    ArrayPtr new_array_ptr = ArrayPtr(new ElemT[padded_size]);
    mempcpy(new_array_ptr.get(), array_ptr.get(), sizeof(ElemT) * real_size);
    for (ElemT i = real_size; i < padded_size; ++i) {
        new_array_ptr[i] = MAX_VALUE;
    }

    array_ptr = std::move(new_array_ptr);
    return padded_size;
}

__device__
void swap(__volatile__ ElemT *array, ElemT i, ElemT j) {
    ElemT t = array[i];
    array[i] = array[j];
    array[j] = t;
}

__global__
void presort_blocks_bitonic(ElemT *array, ElemT size) {
    int block_size = blockDim.x;
    int block_no   = blockIdx.x;
    int idx = block_size * block_no + threadIdx.x;
    int i   = threadIdx.x;

    __shared__ ElemT sharr[BLOCK_SIZE];

    sharr[i] = array[idx];
    __syncthreads();

    for (ElemT k = 2; k <= block_size; k *= 2) {
        for (ElemT j = k / 2; j > 0; j /= 2) {
            ElemT r = i ^ j;

            if (i < r) {
                if ((i & k) == 0 && sharr[i] > sharr[r]) swap(sharr, i, r);
                if ((i & k) != 0 && sharr[i] < sharr[r]) swap(sharr, i, r);
            }

            __syncthreads();
        }
    }

    array[idx] = sharr[i];
    __syncthreads();

}

__global__
void merge_pairs_bitonic(ElemT *array, ElemT size, ElemT offset) {
    int block_size = blockDim.x;
    int block_no   = blockIdx.x;
    int idx = offset + block_size * block_no + threadIdx.x;
    int i   = threadIdx.x;
    int rev_i = block_size - 1 - i + block_size / 2;

    __shared__ ElemT sharr[BLOCK_SIZE * 2];

    if (i < block_size / 2) {
        sharr[i] = array[idx];
    } else {
        sharr[rev_i] = array[idx];
    }
    __syncthreads();

    for (ElemT j = block_size / 2; j > 0; j /= 2) {
        ElemT r = i ^ j;

        if (i < r) {
            ElemT min_val = min(sharr[i], sharr[r]);
            ElemT max_val = max(sharr[i], sharr[r]);

            sharr[i] = min_val;
            sharr[r] = max_val;
        }

        __syncthreads();
    }

    array[idx] = sharr[i];
    __syncthreads();

}

void print_array(ElemT *array, ElemT size) {
    for (ElemT i = 0; i < size; ++i) {
        std::cout << array[i] << ' ';
    }
    std::cout << std::endl;
}

void sort(ArrayPtr& array_ptr, ElemT size, size_t block_size) {
    ElemT block_count = size / block_size;

    if (block_count == 0) {
        return;
    }

    ElemT *d_array;
    size_t mem_size = size * sizeof(ElemT);
    CSC(cudaMalloc(&d_array, mem_size));
    CSC(cudaMemcpy(d_array, array_ptr.get(), mem_size, cudaMemcpyHostToDevice));

    presort_blocks_bitonic<<<block_count, block_size>>>(d_array, size);
    CSC(cudaPeekAtLastError());

    size_t even_blocks = block_count / 2;
    size_t odd_blocks  = (block_count - 1) / 2;
    for (ElemT pass = 0; pass < block_count; ++pass) {

        // Even 
        if (even_blocks != 0) {
            merge_pairs_bitonic<<<even_blocks, 2 * block_size>>>(d_array, size, 0);
            CSC(cudaPeekAtLastError());
        }

        // Odd
        if (odd_blocks != 0) {
            merge_pairs_bitonic<<<odd_blocks, 2 * block_size>>>(d_array, size, block_size);
            CSC(cudaPeekAtLastError());
        }

    }

    CSC(cudaMemcpy(array_ptr.get(), d_array, mem_size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_array));
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    ArrayPtr array_ptr;
    ElemT real_size;
    read_array(array_ptr, real_size, std::cin);
    // Need to pad array for proper block sorting
    ElemT padded_size = pad_array(array_ptr, real_size);

    sort(array_ptr, padded_size, BLOCK_SIZE);

    write_array(array_ptr, real_size, std::cout);
}
