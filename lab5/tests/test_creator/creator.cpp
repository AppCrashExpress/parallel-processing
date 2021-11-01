#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

using ElemT = int;

const std::string IN_FILE_NAME  = "in.data";
const std::string OUT_FILE_NAME = "out.data";

std::vector<ElemT> read_array(std::istream& is) {
    ElemT size;
    is >> size;

    std::vector<ElemT> arr(size);

    for (ElemT i = 0; i < size; ++i) {
        is >> arr[i];
    }

    return arr;
}

void write_array_to_binary_stream(std::ostream& os, const std::vector<ElemT>& arr, ElemT size) {
    os.write(reinterpret_cast<const char*>(arr.data()), sizeof(ElemT) * size);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "argv[1] must be text file name" << std::endl;
        return 1;
    }
    
    std::ifstream in_file(argv[1], std::ios::in | std::ios::binary);
    std::vector<ElemT> arr = read_array(in_file);
    ElemT size = arr.size();
    in_file.close();

    std::ofstream binary_in;
    binary_in = std::ofstream(IN_FILE_NAME, std::ios::out | std::ios::binary);
    binary_in.write(reinterpret_cast<const char*>(&size), sizeof(ElemT));
    write_array_to_binary_stream(binary_in, arr, size);

    std::sort(arr.begin(), arr.end());

    std::ofstream binary_out;
    binary_out = std::ofstream(OUT_FILE_NAME, std::ios::out | std::ios::binary);
    write_array_to_binary_stream(binary_out, arr, size);
}
