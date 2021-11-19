#include <iostream>
#include <fstream>
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

    int pad_block_size[3];
    
    pad_block_size[x_dir] = block_size[x_dir] + 2;
    pad_block_size[y_dir] = block_size[y_dir] + 2;
    pad_block_size[z_dir] = block_size[z_dir] + 2;

    size_t elem_count = pad_block_size[x_dir] * pad_block_size[y_dir] * pad_block_size[z_dir];
    double *old_grid = new double[elem_count];
    double *new_grid = new double[elem_count];
    
    int edge_side_len = std::max({ block_size[x_dir], block_size[y_dir], block_size[z_dir] });
    int edge_buff_size = edge_side_len * edge_side_len;
    double *edge_buff = new double[edge_buff_size];

    auto calc_1d = [pad_block_size](int x, int y, int z) {
        return (z * pad_block_size[y_dir] + y) * pad_block_size[x_dir] + x;
    };

    auto calc_edge = [edge_side_len](int x, int y) {
        return y * edge_side_len + x;
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
        // 
        for (int j = 0; j < pad_block_size[y_dir]; ++j) {
            for (int i = 0; i < pad_block_size[x_dir]; ++i) {
                old_grid[calc_1d(i, j, 0)] = u[down_s];
                new_grid[calc_1d(i, j, 0)] = u[down_s];
            }
        }
    }

    if (proc_z == proc_size[z_dir] - 1) {
        // 
        for (int j = 0; j < pad_block_size[y_dir]; ++j) {
            for (int i = 0; i < pad_block_size[x_dir]; ++i) {
                old_grid[calc_1d(i, j, pad_block_size[z_dir] - 1)] = u[up_s];
                new_grid[calc_1d(i, j, pad_block_size[z_dir] - 1)] = u[up_s];
            }
        }
    }

    // -------------- //
    // Calculate grid //
    // -------------- //

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
    double inv_hx2 = 1.0f / (hx * hx);
    double inv_hy2 = 1.0f / (hy * hy);
    double inv_hz2 = 1.0f / (hz * hz);

    while (true) {
        double max_diff = 0;
        for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
            for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    double val = (
                        (old_grid[calc_1d(x+1, y,   z  )] + old_grid[calc_1d(x-1, y,   z  )]) * inv_hx2 +
                        (old_grid[calc_1d(x,   y+1, z  )] + old_grid[calc_1d(x,   y-1, z  )]) * inv_hy2 +
                        (old_grid[calc_1d(x,   y,   z+1)] + old_grid[calc_1d(x,   y,   z-1)]) * inv_hz2
                    ) / (2 * (inv_hx2 + inv_hy2 + inv_hz2));
                    new_grid[calc_1d(x, y, z)] = val;
                    max_diff = std::max( max_diff, std::abs(val - old_grid[calc_1d(x, y, z)]) );
                }
            }
        }

        MPI_Allgather(&max_diff, 1, MPI_DOUBLE, block_maxes, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < proc_count; ++i) {
            max_diff = std::max(max_diff, block_maxes[i]);
        }

        if (max_diff < eps) {
            std::swap(old_grid, new_grid);
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
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                    edge_buff[calc_edge(y-1, z-1)] = new_grid[calc_1d(pad_block_size[x_dir] - 2, y, z)];
                }
            }
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x + 1, proc_y, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_x > 0) {
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                    edge_buff[calc_edge(y-1, z-1)] = new_grid[calc_1d(1, y, z)];
                }
            }
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x - 1, proc_y, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_y < proc_size[y_dir] - 1) {
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    edge_buff[calc_edge(x-1, z-1)] = new_grid[calc_1d(x, pad_block_size[y_dir] - 2, z)];
                }
            }
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y + 1, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_y > 0) {
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    edge_buff[calc_edge(x-1, z-1)] = new_grid[calc_1d(x, 1, z)];
                }
            }
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y - 1, proc_z),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_z < proc_size[z_dir] - 1) {
            for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    edge_buff[calc_edge(x-1, y-1)] = new_grid[calc_1d(x, y, pad_block_size[z_dir] - 2)];
                }
            }
            MPI_Bsend(edge_buff,
                      edge_buff_size,
                      MPI_DOUBLE,
                      calc_rank(proc_x, proc_y, proc_z + 1),
                      id,
                      MPI_COMM_WORLD);
        }

        if (proc_z > 0) {
            for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    edge_buff[calc_edge(x-1, y-1)] = new_grid[calc_1d(x, y, 1)];
                }
            }
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
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                    new_grid[calc_1d(pad_block_size[x_dir] - 1, y, z)] = edge_buff[calc_edge(y-1, z-1)];
                }
            }
        }

        if (proc_x > 0) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x - 1, proc_y, proc_z), calc_rank(proc_x - 1, proc_y, proc_z), MPI_COMM_WORLD, &status);
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                    new_grid[calc_1d(0, y, z)] = edge_buff[calc_edge(y-1, z-1)];
                }
            }
        }

        if (proc_y < proc_size[y_dir] - 1) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y + 1, proc_z), calc_rank(proc_x, proc_y + 1, proc_z), MPI_COMM_WORLD, &status);
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    new_grid[calc_1d(x, pad_block_size[y_dir] - 1, z)] = edge_buff[calc_edge(x-1, z-1)];
                }
            }
        }

        if (proc_y > 0) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y - 1, proc_z), calc_rank(proc_x, proc_y - 1, proc_z), MPI_COMM_WORLD, &status);
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    new_grid[calc_1d(x, 0, z)] = edge_buff[calc_edge(x-1, z-1)];
                }
            }
        }

        if (proc_z < proc_size[z_dir] - 1) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y, proc_z + 1), calc_rank(proc_x, proc_y, proc_z + 1), MPI_COMM_WORLD, &status);
            for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    new_grid[calc_1d(x, y, pad_block_size[z_dir] - 1)] = edge_buff[calc_edge(x-1, y-1)];
                }
            }
        }

        if (proc_z > 0) {
            MPI_Recv(edge_buff, edge_buff_size, MPI_DOUBLE, calc_rank(proc_x, proc_y, proc_z - 1), calc_rank(proc_x, proc_y, proc_z - 1), MPI_COMM_WORLD, &status);
            for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    new_grid[calc_1d(x, y, 0)] = edge_buff[calc_edge(x-1, y-1)];
                }
            }
        }

        std::swap(old_grid, new_grid);
    }

    if (id != 0) {
        for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
            for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                    edge_buff[x - 1] = new_grid[calc_1d(x, y, z)];
                }
                MPI_Send(edge_buff, block_size[x_dir], MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    } else {
        std::ofstream out_file(out_file_name);
        out_file.setf(std::ios_base::scientific, std::ios_base::floatfield);

        // god damn it
        for (int zp = 0; zp < proc_size[z_dir]; ++zp) {
            for (int z = 1; z < pad_block_size[z_dir] - 1; ++z) {
                for (int yp = 0; yp < proc_size[y_dir]; ++yp) {
                    for (int y = 1; y < pad_block_size[y_dir] - 1; ++y) {
                        for (int xp = 0; xp < proc_size[x_dir]; ++xp) {
                            int curr_id = calc_rank(xp, yp, zp);

                            if (curr_id == 0) {
                                for (int x = 1; x < pad_block_size[x_dir] - 1; ++x) {
                                    edge_buff[x - 1] = new_grid[calc_1d(x, y, z)];
                                }
                            } else {
                                MPI_Recv(edge_buff, block_size[x_dir], MPI_DOUBLE, curr_id, curr_id, MPI_COMM_WORLD, &status);
                            }

                            for (int x = 0; x < block_size[x_dir]; ++x) {
                                out_file << edge_buff[x] << ' ';
                            }
                            out_file << '\n';
                        }
                    }
                }
            }
        }
    }

    std::cout << id << '\n';
    std::cout << proc_x << ' ' << proc_y << ' ' << proc_z << '\n';
    for (int z = 0; z < pad_block_size[z_dir]; ++z) {
        for (int y = 0; y < pad_block_size[y_dir]; ++y) {
            for (int x = 0; x < pad_block_size[x_dir]; ++x) {
                std::cout << old_grid[calc_1d(x, y, z)] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    std::cout << std::flush;

    MPI_Buffer_detach(&send_buffer, &buffer_size);

    delete[] block_maxes;
    delete[] send_buffer;
    delete[] old_grid;
    delete[] new_grid;
    delete[] edge_buff;

    MPI_Finalize();
}
