#include <iostream>
#include <algorithm>
#include <memory>
#include "mpi.h"

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
    std::string out_file_name;
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

    MPI_Bcast(proc_size,  3, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(block_size, 3, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&eps,       1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(l,          3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u,          6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    // Pad out elements for boundary values
    // Remember that it is padded out
    // REMEMBER THAT IT IS PADDED OUT
    
    block_size[x_dir] += 2;
    block_size[y_dir] += 2;
    block_size[z_dir] += 2;

    size_t elem_count = block_size[x_dir] * block_size[y_dir] * block_size[z_dir];

    int max_block_size = std::max({ block_size[x_dir], block_size[y_dir], block_size[z_dir] });

    double *prev_grid = new double[elem_count];
    double *new_grid  = new double[elem_count];
    double *edge_buff = new double[max_block_size];

    auto calc_1d = [block_size](int x, int y, int z) {
        return (z * block_size[y_dir] + y) * block_size[x_dir] + x;
    };

    // --------------- //
    // Initialize grid //
    // --------------- //
    
    for (size_t i = 0; i < elem_count; ++i) {
        prev_grid[i] = u0;
    }

    // Edge elements aren't used, 
    // do whatever with them

    if (proc_x == 0) {
        for (int j = 0; j < block_size[z_dir]; ++j) {
            for (int i = 0; i < block_size[y_dir]; ++i) {
                prev_grid[calc_1d(0, i, j)] = u[left_s];
            }
        }
    }

    if (proc_x == proc_size[x_dir] - 1) {
        for (int j = 0; j < block_size[z_dir]; ++j) {
            for (int i = 0; i < block_size[y_dir]; ++i) {
                prev_grid[calc_1d(block_size[x_dir] - 1, i, j)] = u[right_s];
            }
        }
    }

    if (proc_y == 0) {
        for (int j = 0; j < block_size[z_dir]; ++j) {
            for (int i = 0; i < block_size[x_dir]; ++i) {
                prev_grid[calc_1d(i, 0, j)] = u[up_s];
            }
        }
    }

    if (proc_y == proc_size[y_dir] - 1) {
        for (int j = 0; j < block_size[z_dir]; ++j) {
            for (int i = 0; i < block_size[x_dir]; ++i) {
                prev_grid[calc_1d(i, block_size[y_dir] - 1, j)] = u[down_s];
            }
        }
    }

    if (proc_z == 0) {
        // 
        for (int j = 0; j < block_size[y_dir]; ++j) {
            for (int i = 0; i < block_size[x_dir]; ++i) {
                prev_grid[calc_1d(i, j, 0)] = u[front_s];
            }
        }
    }

    if (proc_z == proc_size[z_dir] - 1) {
        // 
        for (int j = 0; j < block_size[y_dir]; ++j) {
            for (int i = 0; i < block_size[x_dir]; ++i) {
                prev_grid[calc_1d(i, j, block_size[z_dir] - 1)] = u[back_s];
            }
        }
    }

    delete[] prev_grid;
    delete[] new_grid;
    delete[] edge_buff;

    MPI_Finalize();
}
