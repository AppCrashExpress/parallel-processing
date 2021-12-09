#include <iostream>
#include <algorithm>
#include "mpi.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call)                                             \
do {                                                          \
    cudaError_t res = call;                                   \
    if (res != cudaSuccess) {                                 \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

#define _i(x, y, z) (((z) + 1) * (block_y + 2) * (block_x + 2) + \
                     ((y) + 1) * (block_x + 2) + \
                      (x) + 1)

__global__ // For X
void init_yz_layer(double *grid,
                  int block_x, int block_y, int block_z,
                  int x, double init_val) {
    // Do not get confused for 2D kernels
    // Yes, idy uses x-wise indexing but thats
    // simple because x-wise is first to index
    // then y-wise

    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = blockDim.x * gridDim.x;
    int stridez = blockDim.y * gridDim.y;

    for (int z = idz; z < block_z; z += stridez) {
        for (int y = idy; y < block_y; y += stridey) {
            grid[_i(x, y, z)] = init_val;
        }
    }
}

__global__ // For Y
void init_xz_layer(double *grid,
                  int block_x, int block_y, int block_z,
                  int y, double init_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int stridex = blockDim.x * gridDim.x;
    int stridez = blockDim.y * gridDim.y;

    for (int z = idz; z < block_z; z += stridez) {
        for (int x = idx; x < block_x; x += stridex) {
            grid[_i(x, y, z)] = init_val;
        }
    }
}

__global__ // For Z
void init_xy_layer(double *grid,
                  int block_x, int block_y, int block_z,
                  int z, double init_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for (int y = idy; y < block_y; y += stridex) {
        for (int x = idx; x < block_x; x += stridey) {
            grid[_i(x, y, z)] = init_val;
        }
    }
}

__global__ // For X
void copy_yz_layer(double *grid, double *buffer,
                  int block_x, int block_y, int block_z,
                  int x, bool copy_from_buffer) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int stridey = blockDim.x * gridDim.x;
    int stridez = blockDim.y * gridDim.y;

    for (int z = idz; z < block_z; z += stridez) {
        for (int y = idy; y < block_y; y += stridey) {
            if (copy_from_buffer) {
                grid[_i(x, y, z)] = buffer[z * block_y + y];
            } else {
                buffer[z * block_y + y] = grid[_i(x, y, z)];
            }
        }
    }
}

__global__ // For Y
void copy_xz_layer(double *grid, double *buffer,
                  int block_x, int block_y, int block_z,
                  int y, bool copy_from_buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idz = blockIdx.y * blockDim.y + threadIdx.y;
    int stridex = blockDim.x * gridDim.x;
    int stridez = blockDim.y * gridDim.y;

    for (int z = idz; z < block_z; z += stridez) {
        for (int x = idx; x < block_x; x += stridex) {
            if (copy_from_buffer) {
                grid[_i(x, y, z)] = buffer[z * block_x + x];
            } else {
                buffer[z * block_x + x] = grid[_i(x, y, z)];
            }
        }
    }
}

__global__ // For Z
void copy_xy_layer(double *grid, double *buffer,
                  int block_x, int block_y, int block_z,
                  int z, bool copy_from_buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for (int y = idy; y < block_y; y += stridey) {
        for (int x = idx; x < block_x; x += stridex) {
            if (copy_from_buffer) {
                grid[_i(x, y, z)] = buffer[y * block_x + x];
            } else {
                buffer[y * block_x + x] = grid[_i(x, y, z)];
            }
        }
    }
}

__global__
void calc_grid(double *new_grid, double *old_grid,
               int    block_x, int    block_y, int    block_z,
               double hx,      double hy,      double hz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;
    int stridez = blockDim.z * gridDim.z;

    double invhx = 1 / (hx * hx);
    double invhy = 1 / (hy * hy);
    double invhz = 1 / (hz * hz);

    // Block size is not padded
    for (int z = idz; z < block_z; z += stridez) {
        for (int y = idy; y < block_y; y += stridey) {
            for (int x = idx; x < block_x; x += stridex) {
                new_grid[_i(x, y, z)] = 0.5 * ((old_grid[_i(x + 1, y    , z    )] + old_grid[_i(x - 1, y,     z    )]) * invhx +
                                               (old_grid[_i(x,     y + 1, z    )] + old_grid[_i(x,     y - 1, z    )]) * invhy +
                                               (old_grid[_i(x,     y,     z + 1)] + old_grid[_i(x,     y,     z - 1)]) * invhz) /
                                       (invhx + invhy + invhz);
            }
        }
    }
}

__global__
void calc_error(double *new_grid, double *old_grid,
               int    block_x, int    block_y, int    block_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;
    int stridez = blockDim.z * gridDim.z;


    for (int z = idz - 1; z < block_z + 1; z += stridez) {
        for (int y = idy - 1; y < block_y + 1; y += stridey) {
            for (int x = idx - 1; x < block_x + 1; x += stridex) {
                bool is_edge_elem = (x == -1      || y == -1      || z == -1 ||
                                     x == block_x || y == block_y || z == block_z);
                size_t i = _i(x, y, z);
                old_grid[i] = (!is_edge_elem) * fabs(new_grid[i] - old_grid[i]);
            }
        }
    }
}

struct absolute_difference
{
    __host__ __device__
        double operator()(thrust::tuple<double,double> t)
        {
            double a = thrust::get<0>(t);
            double b = thrust::get<1>(t);

            return fabs(a - b);
        }
};

int main(int argc, char **argv) {
    enum Side {
        down_s,
        up_s,
        left_s,
        right_s,
        front_s,
        back_s
    };

    enum Direction {
        x_dir,
        y_dir,
        z_dir
    };

    int proc_size[3];
    int block_size[3];
    char out_file_name[1024];
    double eps;
    double l[3];
    double u[7];
    double u0;

    MPI_Init(&argc, &argv);

    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0) {
        std::cin >> proc_size[x_dir] >> proc_size[y_dir] >> proc_size[z_dir]
                 >> block_size[x_dir] >> block_size[y_dir] >> block_size[z_dir]
                 >> out_file_name
                 >> eps
                 >> l[x_dir] >> l[y_dir] >> l[z_dir]
                 >> u[down_s] >> u[up_s] >> u[left_s] >> u[right_s] >> u[front_s] >> u[back_s]
                 >> u0;
    }

    MPI_Bcast(proc_size,      3,     MPI_INT,     0,  MPI_COMM_WORLD);
    MPI_Bcast(block_size,     3,     MPI_INT,     0,  MPI_COMM_WORLD);
    MPI_Bcast(&eps,           1,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(l,              3,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(u,              6,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(&u0,            1,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(out_file_name,  1024,  MPI_CHAR,    0,  MPI_COMM_WORLD);

    /*
     * z
     *  X → x
     *  ↓
     *  y
     *
     *  z goes into screen
     */

    int proc_x =  id % proc_size[x_dir];
    int proc_y = (id / proc_size[x_dir]) % proc_size[y_dir];
    int proc_z =  id / (proc_size[x_dir] * proc_size[y_dir]);

    int pad_block_size[3];
    
    pad_block_size[x_dir] = block_size[x_dir] + 2;
    pad_block_size[y_dir] = block_size[y_dir] + 2;
    pad_block_size[z_dir] = block_size[z_dir] + 2;

    size_t elem_count = pad_block_size[x_dir] * pad_block_size[y_dir] * pad_block_size[z_dir];
    double *old_grid = new double[elem_count];
    double *new_grid = new double[elem_count];

    double *d_old_grid;
    double *d_new_grid;
    CSC(cudaMalloc(&d_old_grid, sizeof(double) * elem_count));
    CSC(cudaMalloc(&d_new_grid, sizeof(double) * elem_count));
    
    int edge_side_len = std::max({ block_size[x_dir], block_size[y_dir], block_size[z_dir] });
    int edge_buff_size = edge_side_len * edge_side_len;
    double *edge_buff = new double[edge_buff_size];

    double *d_edge_buff;
    CSC(cudaMalloc(&d_edge_buff, sizeof(double) * edge_buff_size));

    auto calc_1d = [pad_block_size](int x, int y, int z) {
        return (z * pad_block_size[y_dir] + y) * pad_block_size[x_dir] + x;
    };

    auto calc_rank = [proc_size](int x, int y, int z) {
        return (z * proc_size[y_dir] + y) * proc_size[x_dir] + x;
    };

    // --------------- //
    // Initialize grid //
    // --------------- //
    
    for (size_t i = 0; i < elem_count; ++i) {
        old_grid[i] = u0;
    }

    // -------------- //
    // Calculate grid //
    // -------------- //
    const dim3 BLOCKS = dim3(32,32);
    const dim3 THREADS = dim3(32,32);

    int proc_count = proc_size[x_dir] * proc_size[y_dir] * proc_size[z_dir];
    double *block_maxes = new double[proc_count];

    int buffer_size;
    MPI_Pack_size(edge_buff_size, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
    double *send_buffer = new double[buffer_size];
    MPI_Buffer_attach(send_buffer, buffer_size);

    MPI_Status status;

    double hx = 1.0f * l[x_dir] / (proc_size[x_dir] * block_size[x_dir]);
    double hy = 1.0f * l[y_dir] / (proc_size[y_dir] * block_size[y_dir]);
    double hz = 1.0f * l[z_dir] / (proc_size[z_dir] * block_size[z_dir]);

    cudaMemcpy(d_old_grid, old_grid, sizeof(double) * elem_count, cudaMemcpyHostToDevice);

    while (true) {

        // Initialize boundary conditions on CUDA

        if (proc_x == 0) {
            init_yz_layer<<<BLOCKS, THREADS>>>(d_old_grid,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    -1, u[left_s]);
        }

        if (proc_x == proc_size[x_dir] - 1) {
            init_yz_layer<<<BLOCKS, THREADS>>>(d_old_grid,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[x_dir], u[right_s]);
        }

        if (proc_y == 0) {
            init_xz_layer<<<BLOCKS, THREADS>>>(d_old_grid,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    -1, u[front_s]);
        }

        if (proc_y == proc_size[y_dir] - 1) {
            init_xz_layer<<<BLOCKS, THREADS>>>(d_old_grid,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[y_dir], u[back_s]);
        }

        if (proc_z == 0) {
            init_xy_layer<<<BLOCKS, THREADS>>>(d_old_grid,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    -1, u[down_s]);
        }

        if (proc_z == proc_size[z_dir] - 1) {
            init_xy_layer<<<BLOCKS, THREADS>>>(d_old_grid,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[z_dir], u[up_s]);
        }

        /*
        cudaMemcpy(old_grid, d_old_grid, sizeof(double) * elem_count, cudaMemcpyDeviceToHost);
        for (int z = 0; z < pad_block_size[z_dir] - 0; ++z) {
            for (int y = 0; y < pad_block_size[y_dir] - 0; ++y) {
                for (int x = 0; x < pad_block_size[x_dir] - 0; ++x) {
                    std::cerr << old_grid[calc_1d(x,y,z)] << ' ';
                }
                std::cerr << std::endl;
            }
            std::cerr << std::endl;
        }
        */

        calc_grid<<<BLOCKS, THREADS>>>(d_new_grid, d_old_grid,
                block_size[x_dir], block_size[y_dir], block_size[z_dir],
                hx, hy, hz);


        calc_error<<<BLOCKS, THREADS>>>(d_new_grid, d_old_grid,
                block_size[x_dir], block_size[y_dir], block_size[z_dir]);

        thrust::device_ptr<double> p_old_grid = thrust::device_pointer_cast(d_old_grid);
        thrust::device_ptr<double> max_diff_ptr = thrust::max_element(p_old_grid, p_old_grid + elem_count);
        double max_diff = *max_diff_ptr;

        MPI_Allgather(&max_diff, 1, MPI_DOUBLE, block_maxes, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < proc_count; ++i) {
            max_diff = std::max(max_diff, block_maxes[i]);
        }

        if (max_diff < eps) {
            break;
        }

        // Send buffers to neighbours
        // Visually copies like:
        //
        // n-1  n   |  -1   0
        //  |   ↑       ↑   |
        //  |   |-------|---|
        //  |-----------|

        if (proc_x < proc_size[x_dir] - 1) {
            copy_yz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[x_dir] - 1, false); 
            CSC(cudaMemcpy(edge_buff, d_edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x + 1, proc_y, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_x > 0) {
            copy_yz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    0, false); 
            CSC(cudaMemcpy(edge_buff, d_edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x - 1, proc_y, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_y < proc_size[y_dir] - 1) {
            copy_xz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[y_dir] - 1, false);
            CSC(cudaMemcpy(edge_buff, d_edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y + 1, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_y > 0) {
            copy_xz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    0, false);
            CSC(cudaMemcpy(edge_buff, d_edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y - 1, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_z < proc_size[z_dir] - 1) {
            copy_xy_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[z_dir] - 1, false);
            CSC(cudaMemcpy(edge_buff, d_edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y, proc_z + 1),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_z > 0) {
            copy_xy_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    0, false);
            CSC(cudaMemcpy(edge_buff, d_edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y, proc_z - 1),
                      id,
                      MPI_COMM_WORLD);
        }

        // Recieve buffers from neighbours

        if (proc_x < proc_size[x_dir] - 1) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x + 1, proc_y, proc_z), calc_rank(proc_x + 1, proc_y, proc_z), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(d_edge_buff, edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyHostToDevice));
            copy_yz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[x_dir], true); 
        }

        if (proc_x > 0) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x - 1, proc_y, proc_z), calc_rank(proc_x - 1, proc_y, proc_z), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(d_edge_buff, edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyHostToDevice));
            copy_yz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    -1, true); 
        }

        if (proc_y < proc_size[y_dir] - 1) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y + 1, proc_z), calc_rank(proc_x, proc_y + 1, proc_z), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(d_edge_buff, edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyHostToDevice));
            copy_xz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[y_dir], true);
        }

        if (proc_y > 0) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y - 1, proc_z), calc_rank(proc_x, proc_y - 1, proc_z), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(d_edge_buff, edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyHostToDevice));
            copy_xz_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    -1, true);
        }

        if (proc_z < proc_size[z_dir] - 1) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y, proc_z + 1), calc_rank(proc_x, proc_y, proc_z + 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(d_edge_buff, edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyHostToDevice));
            copy_xy_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    block_size[z_dir], true);
        }

        if (proc_z > 0) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y, proc_z - 1), calc_rank(proc_x, proc_y, proc_z - 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(d_edge_buff, edge_buff, sizeof(double) * edge_buff_size, cudaMemcpyHostToDevice));
            copy_xy_layer<<<BLOCKS, THREADS>>>(d_new_grid, d_edge_buff,
                    block_size[x_dir], block_size[y_dir], block_size[z_dir],
                    -1, true);
        }

        std::swap(d_old_grid, d_new_grid);
    }

    cudaMemcpy(new_grid, d_new_grid, sizeof(double) * elem_count, cudaMemcpyDeviceToHost);

    int n_size = 14;
    size_t buff_size = block_size[x_dir] * block_size[y_dir] * block_size[z_dir] * n_size;
    char *char_buff = new char[buff_size];
    memset(char_buff, ' ', buff_size);

    for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
        for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
            for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                size_t i = (((z-1)*block_size[y_dir]+(y-1))*block_size[x_dir] + (x-1));
                sprintf(char_buff + i * n_size, "%.6e", new_grid[calc_1d(x, y, z)]);
            }
        }
    }

    for (size_t i = 0; i < buff_size; ++i) {
        if (char_buff[i] == '\0') {
            char_buff[i] = ' ';
        }
    }

    MPI_Datatype cell;
    MPI_Type_contiguous(n_size, MPI_CHAR, &cell);
    MPI_Type_commit(&cell);

    MPI_Datatype something_complex_idk;

    size_t x_line_count = block_size[y_dir] * block_size[z_dir];
    int *lengths   = new int[x_line_count];
    int *disp = new int[x_line_count];
    for (size_t i = 0; i < x_line_count; ++i) {
        lengths[i] = block_size[x_dir];
    }

    const int line_size = block_size[x_dir] * proc_size[x_dir];
    const int face_size = line_size * block_size[y_dir] * proc_size[y_dir];

    const int x_first_i = block_size[x_dir] * proc_x;
    const int y_first_i = block_size[y_dir] * proc_y;
    const int z_first_i = block_size[z_dir] * proc_z;

    MPI_Aint global_offset = face_size * z_first_i + line_size * y_first_i + x_first_i;
    for (int z = 0; z < block_size[z_dir]; ++z) {
        for (int y = 0; y < block_size[y_dir]; ++y) {
            int i = z * block_size[y_dir] + y;
            disp[i] = y * line_size + z * face_size;
        }
    }

    MPI_Type_indexed(x_line_count, lengths, disp, cell, &something_complex_idk);
    MPI_Type_commit(&something_complex_idk);

    MPI_File fp;
    MPI_File_delete(out_file_name, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, out_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);

    MPI_File_set_view(fp, global_offset * n_size, cell, something_complex_idk, "native", MPI_INFO_NULL);
    MPI_File_write_all(fp, char_buff, buff_size, MPI_CHAR, &status);

    MPI_File_close(&fp);

    MPI_Buffer_detach(&send_buffer, &buffer_size);

    CSC(cudaFree(d_old_grid));
    CSC(cudaFree(d_new_grid));
    CSC(cudaFree(d_edge_buff));

    delete[] char_buff;
    delete[] block_maxes;
    delete[] send_buffer;
    delete[] old_grid;
    delete[] new_grid;
    delete[] edge_buff;

    MPI_Finalize();
}
