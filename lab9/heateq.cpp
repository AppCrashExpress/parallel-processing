#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <omp.h>
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
    char out_file_name[1024];
    double eps;
    double l[3];
    double u[6];
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
    MPI_Bcast(out_file_name,  1024,  MPI_CHAR,    0,  MPI_COMM_WORLD);
    MPI_Bcast(&eps,           1,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(l,              3,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(u,              6,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
    MPI_Bcast(&u0,            1,     MPI_DOUBLE,  0,  MPI_COMM_WORLD);
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

    auto calc_1d = [pad_block_size](int x, int y, int z) {
        return (z * pad_block_size[y_dir] + y) * pad_block_size[x_dir] + x;
    };

    auto calc_1d_o = [pad_block_size](int x, int y, int z) {
        return ((z + 1) * pad_block_size[y_dir] + (y + 1)) * pad_block_size[x_dir] + (x + 1);
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

    // Fill border elements once for both arrays
    // Edge and corner elements aren't used, 
    // do whatever with them

    if (proc_x == 0) {
        for (int j = 0; j < pad_block_size[z_dir]; ++j) {
            for (int i = 0; i < pad_block_size[y_dir]; ++i) {
                old_grid[calc_1d(0, i, j)] = u[left_s];
                new_grid[calc_1d(0, i, j)] = u[left_s];
            }
        }
    }

    if (proc_x == proc_size[x_dir] - 1) {
        for (int j = 0; j < pad_block_size[z_dir]; ++j) {
            for (int i = 0; i < pad_block_size[y_dir]; ++i) {
                old_grid[calc_1d(pad_block_size[x_dir] - 1, i, j)] = u[right_s];
                new_grid[calc_1d(pad_block_size[x_dir] - 1, i, j)] = u[right_s];
            }
        }
    }

    if (proc_y == 0) {
        for (int j = 0; j < pad_block_size[z_dir]; ++j) {
            for (int i = 0; i < pad_block_size[x_dir]; ++i) {
                old_grid[calc_1d(i, 0, j)] = u[front_s];
                new_grid[calc_1d(i, 0, j)] = u[front_s];
            }
        }
    }

    if (proc_y == proc_size[y_dir] - 1) {
        for (int j = 0; j < pad_block_size[z_dir]; ++j) {
            for (int i = 0; i < pad_block_size[x_dir]; ++i) {
                old_grid[calc_1d(i, pad_block_size[y_dir] - 1, j)] = u[back_s];
                new_grid[calc_1d(i, pad_block_size[y_dir] - 1, j)] = u[back_s];
            }
        }
    }

    if (proc_z == 0) {
        for (int j = 0; j < pad_block_size[y_dir]; ++j) {
            for (int i = 0; i < pad_block_size[x_dir]; ++i) {
                old_grid[calc_1d(i, j, 0)] = u[down_s];
                new_grid[calc_1d(i, j, 0)] = u[down_s];
            }
        }
    }

    if (proc_z == proc_size[z_dir] - 1) {
        for (int j = 0; j < pad_block_size[y_dir]; ++j) {
            for (int i = 0; i < pad_block_size[x_dir]; ++i) {
                old_grid[calc_1d(i, j, pad_block_size[z_dir] - 1)] = u[up_s];
                new_grid[calc_1d(i, j, pad_block_size[z_dir] - 1)] = u[up_s];
            }
        }
    }

    // ---------------------- //
    // Create send/recv types //
    // ---------------------- //

    // It's yours :)

    int sizes[3] = {pad_block_size[x_dir], pad_block_size[y_dir], pad_block_size[z_dir]};
    int subsizes[3], coords[3];

    // X dir

    MPI_Datatype yz_send_left, yz_send_right;
    MPI_Datatype yz_recv_left, yz_recv_right;

    subsizes[0] = 1;
    subsizes[1] = block_size[y_dir];
    subsizes[2] = block_size[z_dir];

    coords[0] = 1;
    coords[1] = 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &yz_send_left);
    MPI_Type_commit(&yz_send_left);

    coords[0] = block_size[x_dir];
    coords[1] = 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &yz_send_right);
    MPI_Type_commit(&yz_send_right);

    coords[0] = 0;
    coords[1] = 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &yz_recv_left);
    MPI_Type_commit(&yz_recv_left);

    coords[0] = block_size[x_dir] + 1;
    coords[1] = 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &yz_recv_right);
    MPI_Type_commit(&yz_recv_right);

    // Y dir

    MPI_Datatype xz_send_left, xz_send_right;
    MPI_Datatype xz_recv_left, xz_recv_right;

    subsizes[0] = block_size[x_dir];
    subsizes[1] = 1;
    subsizes[2] = block_size[z_dir];

    coords[0] = 1;
    coords[1] = 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xz_send_left);
    MPI_Type_commit(&xz_send_left);

    coords[0] = 1;
    coords[1] = block_size[y_dir];
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xz_send_right);
    MPI_Type_commit(&xz_send_right);

    coords[0] = 1;
    coords[1] = 0;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xz_recv_left);
    MPI_Type_commit(&xz_recv_left);

    coords[0] = 1;
    coords[1] = block_size[y_dir] + 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xz_recv_right);
    MPI_Type_commit(&xz_recv_right);

    // Z dir

    MPI_Datatype xy_send_left, xy_send_right;
    MPI_Datatype xy_recv_left, xy_recv_right;

    subsizes[0] = block_size[x_dir];
    subsizes[1] = block_size[y_dir];
    subsizes[2] = 1;

    coords[0] = 1;
    coords[1] = 1;
    coords[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xy_send_left);
    MPI_Type_commit(&xy_send_left);

    coords[0] = 1;
    coords[1] = 1;
    coords[2] = block_size[z_dir];
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xy_send_right);
    MPI_Type_commit(&xy_send_right);

    coords[0] = 1;
    coords[1] = 1;
    coords[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xy_recv_left);
    MPI_Type_commit(&xy_recv_left);

    coords[0] = 1;
    coords[1] = 1;
    coords[2] = block_size[z_dir] + 1;
    MPI_Type_create_subarray(3, sizes, subsizes, coords, MPI_ORDER_FORTRAN, MPI_DOUBLE, &xy_recv_right);
    MPI_Type_commit(&xy_recv_right);

    // -------------- //
    // Calculate grid //
    // -------------- //

    int proc_count = proc_size[x_dir] * proc_size[y_dir] * proc_size[z_dir];
    double *block_maxes = new double[proc_count];

    int buffer_size = 2 * (block_size[x_dir] * block_size[z_dir] +
                           block_size[x_dir] * block_size[z_dir] +
                           block_size[x_dir] * block_size[y_dir]) * sizeof(double) + 6 * MPI_BSEND_OVERHEAD;
    double *send_buffer = new double[buffer_size / sizeof(double)];
    MPI_Buffer_attach(send_buffer, buffer_size);

    MPI_Status status;

    double hx = 1.0f * l[x_dir] / (proc_size[x_dir] * block_size[x_dir]);
    double hy = 1.0f * l[y_dir] / (proc_size[y_dir] * block_size[y_dir]);
    double hz = 1.0f * l[z_dir] / (proc_size[z_dir] * block_size[z_dir]);
    double inv_hx2 = 1.0f / (hx * hx);
    double inv_hy2 = 1.0f / (hy * hy);
    double inv_hz2 = 1.0f / (hz * hz);

    while (true) {
        double max_diff = 0;

        #pragma omp parallel reduction(max:max_diff)
        {
            int thread_i = omp_get_thread_num();
            int offset = omp_get_num_threads();

            int grid_size = block_size[x_dir] * block_size[y_dir] * block_size[z_dir];

            int x, y, z;

            for (size_t i = thread_i; i < grid_size; i += offset) {
                x =  i % block_size[x_dir];
                y = (i / block_size[x_dir]) % block_size[y_dir];
                z =  i / (block_size[x_dir] * block_size[y_dir]);

                double val = (
                        (old_grid[calc_1d_o(x+1, y,   z  )] + old_grid[calc_1d_o(x-1, y,   z  )]) * inv_hx2 +
                        (old_grid[calc_1d_o(x,   y+1, z  )] + old_grid[calc_1d_o(x,   y-1, z  )]) * inv_hy2 +
                        (old_grid[calc_1d_o(x,   y,   z+1)] + old_grid[calc_1d_o(x,   y,   z-1)]) * inv_hz2
                        ) / (2 * (inv_hx2 + inv_hy2 + inv_hz2));
                new_grid[calc_1d_o(x, y, z)] = val;
                max_diff = std::max( max_diff, std::abs(val - old_grid[calc_1d_o(x, y, z)]) );
            }
        }

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
        // n-2 n-1  |   0   1
        //  |   ↑       ↑   |
        //  |   |-------|---|
        //  |-----------|

        if (proc_x < proc_size[x_dir] - 1) {
            MPI_Bsend(new_grid, 1, yz_send_right, calc_rank(proc_x + 1, proc_y, proc_z), id, MPI_COMM_WORLD);
        }

        if (proc_x > 0) {
            MPI_Bsend(new_grid, 1, yz_send_left, calc_rank(proc_x - 1, proc_y, proc_z), id, MPI_COMM_WORLD);
        }

        if (proc_y < proc_size[y_dir] - 1) {
            MPI_Bsend(new_grid, 1, xz_send_right, calc_rank(proc_x, proc_y + 1, proc_z), id, MPI_COMM_WORLD);
        }

        if (proc_y > 0) {
            MPI_Bsend(new_grid, 1, xz_send_left, calc_rank(proc_x, proc_y - 1, proc_z), id, MPI_COMM_WORLD);
        }

        if (proc_z < proc_size[z_dir] - 1) {
            MPI_Bsend(new_grid, 1, xy_send_right, calc_rank(proc_x, proc_y, proc_z + 1), id, MPI_COMM_WORLD);
        }

        if (proc_z > 0) {
            MPI_Bsend(new_grid, 1, xy_send_left, calc_rank(proc_x, proc_y, proc_z - 1), id, MPI_COMM_WORLD);
        }

        // Recieve buffers from neighbours

        if (proc_x < proc_size[x_dir] - 1) {
            MPI_Recv(new_grid, 1, yz_recv_right, calc_rank(proc_x + 1, proc_y, proc_z), calc_rank(proc_x + 1, proc_y, proc_z), MPI_COMM_WORLD, &status);
        }

        if (proc_x > 0) {
            MPI_Recv(new_grid, 1, yz_recv_left, calc_rank(proc_x - 1, proc_y, proc_z), calc_rank(proc_x - 1, proc_y, proc_z), MPI_COMM_WORLD, &status);
        }

        if (proc_y < proc_size[y_dir] - 1) {
            MPI_Recv(new_grid, 1, xz_recv_right, calc_rank(proc_x, proc_y + 1, proc_z), calc_rank(proc_x, proc_y + 1, proc_z), MPI_COMM_WORLD, &status);
        }

        if (proc_y > 0) {
            MPI_Recv(new_grid, 1, xz_recv_left, calc_rank(proc_x, proc_y - 1, proc_z), calc_rank(proc_x, proc_y - 1, proc_z), MPI_COMM_WORLD, &status);
        }

        if (proc_z < proc_size[z_dir] - 1) {
            MPI_Recv(new_grid, 1, xy_recv_right, calc_rank(proc_x, proc_y, proc_z + 1), calc_rank(proc_x, proc_y, proc_z + 1), MPI_COMM_WORLD, &status);
        }

        if (proc_z > 0) {
            MPI_Recv(new_grid, 1, xy_recv_left, calc_rank(proc_x, proc_y, proc_z - 1), calc_rank(proc_x, proc_y, proc_z - 1), MPI_COMM_WORLD, &status);
        }

        std::swap(old_grid, new_grid);
    }


    int n_size = 17;
    size_t buff_size = block_size[x_dir] * block_size[y_dir] * block_size[z_dir] * n_size;
    char *char_buff = new char[buff_size];
    memset(char_buff, ' ', buff_size);

    for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
        for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
            for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                size_t i = (((z-1)*block_size[y_dir]+(y-1))*block_size[x_dir] + (x-1));
                sprintf(char_buff + i * n_size, "%.7e", new_grid[calc_1d(x, y, z)]);
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

    delete[] block_maxes;
    delete[] send_buffer;
    delete[] old_grid;
    delete[] new_grid;

    MPI_Finalize();
}
