#include <iostream>
#include <fstream>

using ElemT = double;

const double eps = 1e-5;

bool is_equal(ElemT a, ElemT b) {
    return (a - b) < eps;
}

bool compare_streams(std::istream& is_a, std::istream& is_b) {
    bool equal = true;

    while(equal) {
        ElemT a;
        ElemT b;

        is_a >> a;
        is_b >> b;

        if (is_a.eof() || is_b.eof()) {
            equal &= is_a.eof() & is_b.eof();
            break;
        }

        equal &= is_equal(a, b);
    }

    return equal;
}

int main(int argc, char *argv[]) {
    std::ifstream real_file(argv[1], std::ios::in);

    bool are_equal = compare_streams(real_file, std::cin);

    real_file.close();

    if (are_equal) {
        return 0;
    } else {
        return 1;
    }
}
