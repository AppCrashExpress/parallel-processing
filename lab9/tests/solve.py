import numpy as np

np.set_printoptions(formatter={'float': lambda x : np.format_float_scientific(x, precision=7)})

def solve_liebmann(nx, ny, nz, *, 
                   eps, 
                   lx, ly, lz,
                   left_bound, right_bound, front_bound, back_bound, bottom_bound, top_bound, init_val):
    def initialize_grid(nx, ny, nz):
        x_step = lx / (nx - 2)
        y_step = ly / (ny - 2)
        z_step = lz / (nz - 2)
        grid = np.zeros((nz, ny, nx))

        grid[ :,  :,  0] = left_bound
        grid[ :,  :, -1] = right_bound
        grid[ :,  0,  :] = front_bound
        grid[ :, -1,  :] = back_bound
        grid[ 0,  :,  :] = bottom_bound
        grid[-1,  :,  :] = top_bound

        grid[1:-1, 1:-1, 1:-1] = init_val
        
        return grid

    def calc_grid_val(u_up, u_left, u_down, u_right, u_front, u_back, h_x, h_y, h_z):
        inv_hx2 = 1 / (h_x * h_x)
        inv_hy2 = 1 / (h_y * h_y)
        inv_hz2 = 1 / (h_z * h_z)
    
        new_val = (
            (u_right + u_left) * inv_hx2 +
            (u_up    + u_down) * inv_hy2 +
            (u_front + u_back) * inv_hz2
        ) / ( 2 * (inv_hx2 + inv_hy2 + inv_hz2) )
        
        return new_val
    
    x_step = lx / (nx - 2)
    y_step = ly / (ny - 2)
    z_step = lz / (nz - 2)
    old_grid = initialize_grid(nx, ny, nz)
    new_grid = initialize_grid(nx, ny, nz)
    
    # Force at least one iteration
    u_max_prev = np.max(np.abs(old_grid))
    u_max = u_max_prev + eps * 2

    
    while abs(u_max - u_max_prev) >= eps:
        t = old_grid
        old_grid = new_grid
        new_grid = t
        u_max_prev = u_max
        
        for z in range(1, nz - 1):
            for y in range(1, ny - 1):
                for x in range(1, nx - 1):
                    u_up    = old_grid[    z, y - 1,     x]
                    u_down  = old_grid[    z, y + 1,     x]
                    u_right = old_grid[    z,     y, x + 1]
                    u_left  = old_grid[    z,     y, x - 1]
                    u_back  = old_grid[z + 1,     y,     x]
                    u_front = old_grid[z - 1,     y,     x]
                    
                    new_grid[z, y, x] = calc_grid_val(u_up, u_left, u_down, u_right, u_front, u_back, x_step, y_step, z_step)

        u_max = np.max( np.abs(new_grid[1:-1, 1:-1, 1:-1]) )

    return new_grid

def main():
    proc_size  = list( map(int, input().split()) )
    block_size = list( map(int, input().split()) )
    input() # file name
    eps = float(input())
    l = list( map(float, input().split()) )
    u_bottom, u_top, u_left, u_right, u_front, u_back = list( map(float, input().split()) )
    u0 = float(input())

    nx = proc_size[0] * block_size[0] + 2
    ny = proc_size[1] * block_size[1] + 2
    nz = proc_size[2] * block_size[2] + 2

    grid = solve_liebmann(nx, ny, nz, eps=eps, lx=l[0], ly=l[1], lz=l[2], 
                          bottom_bound=u_bottom,
                          top_bound=u_top,
                          left_bound=u_left,
                          right_bound=u_right,
                          front_bound=u_front,
                          back_bound=u_back,
                          init_val=u0)

    grid = grid[1:-1, 1:-1, 1:-1]
    for face in grid:
        for line in face:
            print(' '.join(map(lambda x : np.format_float_scientific(x, precision=6), line)))
        print('')

if __name__ == '__main__':
    main()
