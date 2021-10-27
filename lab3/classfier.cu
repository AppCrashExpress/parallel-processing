#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <float.h>
#include <memory>

#define CSC(call)                                             \
do {                                                          \
    cudaError_t res = call;                                   \
    if (res != cudaSuccess) {                                 \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0);                                              \
    }                                                         \
} while(0)

struct ClassData {
    double mean[3];
    double inv_corr[3][3];
};

__constant__ ClassData classes_global[32];

void read_file(std::unique_ptr<uchar4[]>& image,
               int32_t& w,
               int32_t& h,
               const std::string& in_file) {
    std::ifstream input_file(in_file, std::ios::in | std::ios::binary);

    input_file.read(reinterpret_cast<char*>(&w), sizeof(int32_t));
    input_file.read(reinterpret_cast<char*>(&h), sizeof(int32_t));

    image = std::unique_ptr<uchar4[]>(new uchar4[w * h]);

    input_file.read(reinterpret_cast<char*>(image.get()), sizeof(uchar4) * w * h);

    input_file.close();
}

void write_file(const std::unique_ptr<uchar4[]>& image,
                const int32_t& w,
                const int32_t& h,
                const std::string& out_file) {
    std::ofstream output_file(out_file, std::ios::out | std::ios::binary);

    output_file.write(reinterpret_cast<const char*>(&w), sizeof(int32_t));
    output_file.write(reinterpret_cast<const char*>(&h), sizeof(int32_t));
    output_file.write(reinterpret_cast<const char*>(image.get()), sizeof(uchar4) * w * h);

    output_file.close();
}

__global__
void kernel(uchar4 *image, int32_t size, short class_count) {
    int id     = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; id < size; id += stride) {
        uchar4& pixel = image[id];

        short  max_class_i = 0;
        double max_distance = -DBL_MAX;

        for (short cl = 0; cl < class_count; ++cl) {
            ClassData& curr_class = classes_global[cl];

            double centered_pixel[3];
            centered_pixel[0] = pixel.x - curr_class.mean[0];
            centered_pixel[1] = pixel.y - curr_class.mean[1];
            centered_pixel[2] = pixel.z - curr_class.mean[2];

            double tmp[3] = {};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    tmp[i] += centered_pixel[j] * curr_class.inv_corr[j][i];
                }
            }

            double distance = 0;
            for (int i = 0; i < 3; ++i) {
                distance += tmp[i] * centered_pixel[i];
            }
            distance = -1 * distance;

            if (distance > max_distance) {
                max_class_i = cl;
                max_distance = distance;
            }
        }

        pixel.w = max_class_i;
    }
}

int main() {
    std::string in_file, out_file;
    std::cin >> in_file >> out_file;

    int32_t w, h;
    std::unique_ptr<uchar4[]> image;

    read_file(image, w, h, in_file);

    short class_count;
    ClassData classes[32] = {};
    std::cin >> class_count;

    for (short cl = 0; cl < class_count; ++cl) {
        ClassData& current_class = classes[cl];

        size_t pixel_count;
        std::cin >> pixel_count;
        std::vector< std::array<double, 3> > pixels(pixel_count);

        for (auto& p : pixels) {
            size_t x, y;
            std::cin >> x >> y;
            uchar4& pixel = image[y * w + x];
            p[0] = pixel.x;
            p[1] = pixel.y;
            p[2] = pixel.z;
        }

        for (const auto& p : pixels) {
            // Upper bound of sum should be 255 * 2^19 = 133693440
            // Which should be within max range of double
            for (int i = 0; i < 3; ++i) {
                current_class.mean[i] += p[i];
            }
        }

        for (int i = 0; i < 3; ++i) {
            current_class.mean[i] /= pixel_count;
        }

        for (auto& p : pixels) {
            for (int i = 0; i < 3; ++i) {
                p[i] -= current_class.mean[i];
            }
        }

        double corr[3][3] = {};
        for (auto& p : pixels) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    corr[i][j] += p[i] * p[j];
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                corr[i][j] /= pixel_count - 1;
            }
        }

        // my eyes
        double det = corr[0][0] * (corr[1][1] * corr[2][2] - corr[2][1] * corr[1][2]) -
                     corr[0][1] * (corr[1][0] * corr[2][2] - corr[1][2] * corr[2][0]) +
                     corr[0][2] * (corr[1][0] * corr[2][1] - corr[1][1] * corr[2][0]);
        double inv_det = 1 / det;

        current_class.inv_corr[0][0] = (corr[1][1] * corr[2][2] - corr[2][1] * corr[1][2]) * inv_det;
        current_class.inv_corr[0][1] = (corr[0][2] * corr[2][1] - corr[0][1] * corr[2][2]) * inv_det;
        current_class.inv_corr[0][2] = (corr[0][1] * corr[1][2] - corr[0][2] * corr[1][1]) * inv_det;
        current_class.inv_corr[1][0] = (corr[1][2] * corr[2][0] - corr[1][0] * corr[2][2]) * inv_det;
        current_class.inv_corr[1][1] = (corr[0][0] * corr[2][2] - corr[0][2] * corr[2][0]) * inv_det;
        current_class.inv_corr[1][2] = (corr[1][0] * corr[0][2] - corr[0][0] * corr[1][2]) * inv_det;
        current_class.inv_corr[2][0] = (corr[1][0] * corr[2][1] - corr[2][0] * corr[1][1]) * inv_det;
        current_class.inv_corr[2][1] = (corr[2][0] * corr[0][1] - corr[0][0] * corr[2][1]) * inv_det;
        current_class.inv_corr[2][2] = (corr[0][0] * corr[1][1] - corr[1][0] * corr[0][1]) * inv_det;
    }

    CSC(cudaMemcpyToSymbol(classes_global, classes, sizeof(ClassData) * 32));

    uchar4 *dev_image;
    CSC(cudaMalloc(&dev_image, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_image, image.get(), sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    kernel<<<32, 32>>>(dev_image, w * h, class_count);

    CSC(cudaMemcpy(image.get(), dev_image, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaFree(dev_image));

    write_file(image, w, h, out_file);
}
