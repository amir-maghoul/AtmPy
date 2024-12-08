import numpy as np
import time
from numba import njit, stencil


# NumPy Vectorized Version (Corrected)
def cell_to_node_average_2d_numpy(
    var_cells: np.ndarray, ngx: int, ngy: int
) -> np.ndarray:
    nx_total, ny_total = var_cells.shape
    var_nodes = np.zeros((nx_total + 1, ny_total + 1))

    # Corrected slicing
    var_nodes[ngx:-ngx, ngy:-ngy] = 0.25 * (
        var_cells[ngx - 1 : -ngx, ngy - 1 : -ngy]
        + var_cells[ngx - 1 : -ngx, ngy : -ngy + 1]
        + var_cells[ngx : -ngx + 1, ngy - 1 : -ngy]
        + var_cells[ngx : -ngx + 1, ngy : -ngy + 1]
    )
    return var_nodes


# Numba JIT-Compiled Version with For Loops (Unchanged)
@njit
def cell_to_node_average_2d_numba(
    var_cells: np.ndarray, ngx: int, ngy: int
) -> np.ndarray:
    nx_total, ny_total = var_cells.shape
    var_nodes = np.zeros((nx_total + 1, ny_total + 1))
    for i in range(ngx, nx_total - ngx):
        for j in range(ngy, ny_total - ngy):
            var_nodes[i, j] = 0.25 * (
                var_cells[i - 1, j - 1]
                + var_cells[i - 1, j]
                + var_cells[i, j - 1]
                + var_cells[i, j]
            )
    return var_nodes


# Corrected Numba Stencil Version
@stencil
def cell_to_node_average_2d_kernel(var_cells):
    return 0.25 * (
        var_cells[0, 0] + var_cells[1, 0] + var_cells[0, 1] + var_cells[1, 1]
    )


@njit
def cell_to_node_average_2d_stencil(var_cells: np.ndarray) -> np.ndarray:
    """
    Average 2D cell-centered variables to nodes using Numba stencil.

    Parameters:
        var_cells (np.ndarray): 2D array of cell-centered variable values.

    Returns:
        np.ndarray: 2D array of node-centered variable values.
    """
    # Apply the stencil to var_cells
    var_nodes = cell_to_node_average_2d_kernel(var_cells)

    return var_nodes


def benchmark_cell_to_node_average_2d():
    ngx, ngy = 2, 2  # Number of ghost cells
    nx, ny = 1000, 1000  # Grid size for benchmarking
    nx_total = nx + 2 * ngx
    ny_total = ny + 2 * ngy

    # Generate random data for var_cells including ghost cells
    var_cells = np.random.rand(nx_total, ny_total)

    # Warm-up runs
    cell_to_node_average_2d_numba(var_cells, ngx, ngy)
    cell_to_node_average_2d_stencil(var_cells)

    # Time the NumPy vectorized function
    start_time = time.time()
    var_nodes_numpy = cell_to_node_average_2d_numpy(var_cells, ngx, ngy)
    time_numpy = time.time() - start_time

    # Time the Numba JIT function
    start_time = time.time()
    var_nodes_numba = cell_to_node_average_2d_numba(var_cells, ngx, ngy)
    time_numba = time.time() - start_time

    # Time the Numba stencil function
    start_time = time.time()
    var_nodes_stencil_full = cell_to_node_average_2d_stencil(var_cells)
    # Extract the relevant portion corresponding to var_nodes_numpy and var_nodes_numba
    var_nodes_stencil = var_nodes_stencil_full[ngx : -ngx + 1, ngy : -ngy + 1]
    time_stencil = time.time() - start_time

    # Adjust var_nodes_numpy and var_nodes_numba to match the size
    var_nodes_numpy = var_nodes_numpy[ngx : -ngx + 1, ngy : -ngy + 1]
    var_nodes_numba = var_nodes_numba[ngx : -ngx + 1, ngy : -ngy + 1]

    # Print timing results
    print(f"NumPy vectorized function time: {time_numpy:.6f} seconds")
    print(f"Numba-optimized function time: {time_numba:.6f} seconds")
    print(f"Numba stencil function time: {time_stencil:.6f} seconds")
    print(f"Numba speedup over NumPy: {time_numpy / time_numba:.2f}")
    print(f"Stencil speedup over NumPy: {time_numpy / time_stencil:.2f}")
    print(f"Stencil speedup over Numba: {time_numba / time_stencil:.2f}\n")


# Run the benchmark
if __name__ == "__main__":
    print("Running 2D Benchmark...")
    benchmark_cell_to_node_average_2d()
