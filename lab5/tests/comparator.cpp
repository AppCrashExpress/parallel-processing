#include <iostream>
#include <fstream>

using ElemT = int;

bool compare_streams(std::istream& is_a, std::istream& is_b) {
    bool equal = true;

    while(equal) {
        ElemT a;
        ElemT b;

        is_a.read(reinterpret_cast<char*>(&a), sizeof(ElemT));
        is_b.read(reinterpret_cast<char*>(&b), sizeof(ElemT));

        if (is_a.eof() || is_b.eof()) {
            equal &= is_a.eof() & is_b.eof();
            break;
        }

        equal &= (a == b);
    }

    return equal;
}

int main(int argc, char *argv[]) {
    std::ifstream real_file(argv[1], std::ios::in | std::ios::binary);

    bool are_equal = compare_streams(real_file, std::cin);

    real_file.close();

    if (are_equal) {
        return 0;
    } else {
        return 1;
    }
}
