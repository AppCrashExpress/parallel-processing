#include <iostream>
#include <fstream>
#include <memory>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

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

__device__
double convert_greyscale(uchar4 pixel) {
    // Alpha ignored
	return 0.299 * pixel.x + 
           0.587 * pixel.y + 
           0.114 * pixel.z;
}

__global__
void kernel(uchar4 *image, int32_t w, int32_t h) {
    int idx     = blockDim.x * blockIdx.x + threadIdx.x;
	int idy     = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

    int x_filter[3][3] = {
        {-1,  0,  1},
        {-1,  0,  1},
        {-1,  0,  1}
    };
    int y_filter[3][3] = {
        {-1, -1, -1},
        { 0,  0,  0},
        { 1,  1,  1}
    };

	for(int y = idy; y < h; y += offsety) {
		for(int x = idx; x < w; x += offsetx) {
            double gx = 0;
            double gy = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double grey_pixel = convert_greyscale( tex2D(tex, x + i - 1, y + j - 1) );
                    gx += x_filter[i][j] * grey_pixel;
                    gy += y_filter[i][j] * grey_pixel;
                }
            }

            // Must be 0 to 255 inclusive, else overflow
            unsigned char g = min( sqrt(gx * gx + gy * gy), 255.0 );

            int offset = y * w + x;
            image[offset].x = g;
            image[offset].y = g;
            image[offset].z = g;
            image[offset].w = 0;
		}
    }

}

int main() {
    std::string in_file, out_file;
    std::cin >> in_file >> out_file;

    int32_t w, h;
    // ama big boy
    std::unique_ptr<uchar4[]> image;

    read_file(image, w, h, in_file);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
	CSC(cudaMemcpyToArray(arr, 0, 0, image.get(), sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4 *dev_image;
	CSC(cudaMalloc(&dev_image, sizeof(uchar4) * w * h));

    kernel<<< dim3(32, 32), dim3(32, 32) >>>(dev_image, w, h);

	CSC(cudaMemcpy(image.get(), dev_image, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_image));

    write_file(image, w, h, out_file);
}
