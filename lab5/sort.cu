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
const size_t BLOCK_SIZE = 8; // Must be power of 2


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
void swap(ElemT *array, ElemT i, ElemT j) {
    ElemT t = array[i];
    array[i] = array[j];
    array[j] = t;
}

__global__
void sort_bitonic(ElemT *array, ElemT size) {
    int block_size = blockDim.x;
    int block_no   = blockIdx.x;
    int idx = block_size * block_no + threadIdx.x;

    for (ElemT k = 2; k <= block_size; k *= 2) {
        for (ElemT j = k / 2; j > 0; j /= 2) {
            ElemT r = idx ^ j;

            if (idx < r) {
                if ((idx & k) == 0 && array[idx] > array[r]) swap(array, idx, r);
                if ((idx & k) != 0 && array[idx] < array[r]) swap(array, idx, r);
            }
        }
    }
}

__global__
void correct_bitonic(ElemT *array, ElemT size) {
    int block_size = blockDim.x;
    int block_no   = blockIdx.x;
    int idx = block_size * block_no + threadIdx.x;

    for (ElemT j = block_size / 2; j > 0; j /= 2) {
        ElemT r = idx ^ j;

        if (idx < r) {
            ElemT min_val = min(array[idx], array[r]);
            ElemT max_val = max(array[idx], array[r]);

            array[idx] = min_val;
            array[r]   = max_val;
        }
    }
}

__global__
void merge_bitonic(ElemT *array, ElemT size, ElemT offset) {
    int block_size = blockDim.x * 2;
    int block_no   = blockIdx.x;

    int block_mid  = block_size / 2;

    ElemT i = threadIdx.x;

    if (i >= block_mid) {
        return;
    }

    ElemT l = offset + block_size * block_no + i;
    ElemT r = offset + block_size * (block_no + 1) - (i + 1);
    
    ElemT min_val = min(array[l], array[r]);
    ElemT max_val = max(array[l], array[r]);

    array[l] = min_val;
    array[r] = max_val;
}

void print_array(ElemT *array, ElemT size) {
    for (ElemT i = 0; i < size; ++i) {
        std::cout << array[i] << ' ';
    }
    std::cout << std::endl;
}

void sort(ArrayPtr& array_ptr, ElemT size, size_t block_size) {
    ElemT block_count = size / block_size;

    ElemT *d_array;
    size_t mem_size = size * sizeof(ElemT);
    CSC(cudaMalloc(&d_array, mem_size));
    CSC(cudaMemcpy(d_array, array_ptr.get(), mem_size, cudaMemcpyHostToDevice));

    sort_bitonic<<<block_count, block_size>>>(d_array, size);
    CSC(cudaPeekAtLastError());

    size_t even_blocks = block_count / 2;
    size_t odd_blocks  = (block_count - 1) / 2;
    for (ElemT pass = 0; pass < block_count; ++pass) {

        // Even 
        if (even_blocks != 0) {
            merge_bitonic<<<even_blocks, block_size>>>(d_array, size, 0);
            CSC(cudaPeekAtLastError());
            correct_bitonic<<<block_count, block_size>>>(d_array, size);
            CSC(cudaPeekAtLastError());
        }

        // Odd
        if (odd_blocks != 0) {
            merge_bitonic<<<odd_blocks, block_size>>>(d_array, size, block_size);
            CSC(cudaPeekAtLastError());
            correct_bitonic<<<block_count, block_size>>>(d_array, size);
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
