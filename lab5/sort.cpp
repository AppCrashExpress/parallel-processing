#include <iostream>
#include <memory>
#include <limits>
#include <cstring>
#include <utility>

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

void sort_bitonic(ElemT *array, ElemT size, ElemT order, ElemT dist) {
    // Make amount of threads half the size of block
    // Find current block by index of block

    for (ElemT i = 0; i < size; ++i) {
        // Why XOR? i is in first half, then this will be i + dist
        // If i in in second half, then thil will be i - dist
        // Natural check for threads
        ElemT r = i ^ dist;

        if (i < r) {
            if ((i & order) == 0 && array[i] > array[r]) std::swap(array[i], array[r]);
            if ((i & order) != 0 && array[i] < array[r]) std::swap(array[i], array[r]);
        }
    }
}

void correct_bitonic(ElemT *array, ElemT size, ElemT order, ElemT dist) {
    for (ElemT i = 0; i < size; ++i) {
        ElemT r = i ^ dist;

        if (i < r) {
            ElemT min = std::min(array[i], array[r]);
            ElemT max = std::max(array[i], array[r]);

            array[i] = min;
            array[r] = max;
        }
    }
}

void merge_bitonic(ElemT *array, ElemT size, size_t block_size, ElemT block_no, ElemT offset) {
    ElemT block_mid = block_size / 2;
    for (ElemT i = 0; i < block_mid; ++i) {
        ElemT l = offset + block_size * block_no + i;
        ElemT r = offset + block_size * (block_no + 1) - (i + 1);
        
        ElemT min = std::min(array[l], array[r]);
        ElemT max = std::max(array[l], array[r]);

        array[l] = min;
        array[r] = max;
    }
}

void sort(ArrayPtr& array_ptr, ElemT& size, size_t block_size) {
    ElemT *array = array_ptr.get();

    for (ElemT k = 2; k < block_size; k *= 2) {
        for (ElemT j = k / 2; j > 0; j /= 2) {
            sort_bitonic(array, size, k, j);
        }
    }
    for (ElemT j = block_size / 2; j > 0; j /= 2) {
        sort_bitonic(array, size, block_size, j);
    }

    ElemT block_count = size / block_size;
    for (ElemT pass = 0; pass < block_count; ++pass) {
        for (ElemT i = 0; i < block_count - 1; i += 2) {
            merge_bitonic(array, size, block_size * 2, i / 2, 0);
        }
        for (ElemT j = block_size / 2; j > 0; j /= 2) {
            correct_bitonic(array, size, block_size, j);
        }

        for (ElemT i = 1; i < block_count - 1; i += 2) {
            merge_bitonic(array, size, block_size * 2, i / 2, block_size);
        }
        for (ElemT j = block_size / 2; j > 0; j /= 2) {
            correct_bitonic(array, size, block_size, j);
        }
    }
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
