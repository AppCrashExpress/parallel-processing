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

void merge_bitonic(int *array, ElemT start, ElemT end) {
    ElemT order = end - start;
    ElemT half_len = (end - start) / 2;
    for (ElemT i = 0; i < half_len; ++i) {
        ElemT l = start + i;
        ElemT r = end - i - 1;

        if (array[r] < array[l]) {
            std::swap(array[r], array[l]);
        }
    }

    for (ElemT step = order / 2; step > 1; step /= 2) {
        for (ElemT subblock = start; subblock < end; subblock += step) {
            for (ElemT i = 0; i < step / 2; ++i) {
                if (array[subblock + step / 2 + i] < array[subblock + i]) {
                    std::swap(array[subblock + step / 2 + i], array[subblock + i]);
                }
            }
        }
    }
}

void sort_block_bitonic(ElemT *array, ElemT start, ElemT end) {
    ElemT len = end - start;
    for (ElemT k = 1; k < len; k *= 2){ 
        for (ElemT i = start; i < end; i += 2 * k) {
            merge_bitonic(array, i, i + 2 * k);
        }
    }
}

void sort(ArrayPtr& array_ptr, ElemT& size, size_t block_size) {
    ElemT *array = array_ptr.get();

    for (ElemT i = 0; i < size; i += block_size) {
        sort_block_bitonic(array, i, i + block_size);
    }

    ElemT block_count = size / block_size;
    for (ElemT pass = 0; pass < block_count; ++pass) {
        // Even 
        for (ElemT i = 0; i < block_count - 1; i += 2) {
            merge_bitonic(array, i * block_size, (i + 2) * block_size);
        }

        // Odd
        for (ElemT i = 1; i < block_count - 1; i += 2) {
            merge_bitonic(array, i * block_size, (i + 2) * block_size);
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
