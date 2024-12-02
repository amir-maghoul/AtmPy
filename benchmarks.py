import numpy as np
import numba


# Continuous function


@numba.njit
def continuous_function(x, y):
    return np.sin(x) * np.cos(y)


# Numba-accelerated evaluation on 2D grid
@numba.njit
def evaluate_on_grid_2d(grid_x, grid_y):
    rows, cols = grid_x.shape
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i, j] = continuous_function(grid_x[i, j], grid_y[i, j])
    return result


# Numpy vectorized evaluation on 2D grid
def evaluate_on_grid_2d_numpy(grid_x, grid_y):
    return continuous_function(grid_x, grid_y)


# Create grid using numpy.meshgrid
x = np.linspace(0, 2 * np.pi, 5000)
y = np.linspace(0, 2 * np.pi, 5000)
grid_x, grid_y = np.meshgrid(x, y)

# Evaluate using Numba
result_numba = evaluate_on_grid_2d(grid_x, grid_y)

# Evaluate using Numpy vectorized operation
result_numpy = evaluate_on_grid_2d_numpy(grid_x, grid_y)
import timeit

# Timing numpy vectorized approach
numpy_time = timeit.timeit(lambda: evaluate_on_grid_2d_numpy(grid_x, grid_y), number=10)

# Timing numba approach
numba_time = timeit.timeit(lambda: evaluate_on_grid_2d(grid_x, grid_y), number=10)

print(f"Numpy vectorized time: {numpy_time:.4f} seconds")
print(f"Numba time: {numba_time:.4f} seconds")
