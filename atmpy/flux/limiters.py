""" This module contains slope limiter functions for the flux reconstruction."""

from numba import njit, prange
import numpy as np
import time


@njit
def minmod(a, b):
    """
    Classic minmod slope limiter for scalars.
    """
    if a * b <= 0.0:
        return 0.0
    else:
        return np.sign(a) * min(abs(a), abs(b))


def van_leer(a, b):
    pass


def mc_limiter(a, b):
    """monotonized central-difference limiter."""


def superbee(a, b):
    pass


@njit
def minmod2(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * min(abs(a), abs(b))


def main():
    N = 10**7
    a = np.random.randn(N).astype(np.float64)
    b = np.random.randn(N).astype(np.float64)
    result = np.empty_like(a)

    start_time = time.time()
    minmod2(a, b)
    numba2_time = time.time() - start_time
    print(
        f"Second Numba Time (first call, includes compilation): {numba2_time:.4f} seconds"
    )

    start_time = time.time()
    minmod2(a, b)
    numba2_time = time.time() - start_time
    print(
        f"Second Numba Time (second call, includes compilation): {numba2_time:.4f} seconds"
    )

    start_time = time.time()
    limited_slope_np = minmod2(a, b)
    numpy_time = time.time() - start_time
    print(f"NumPy Time: {numpy_time:.4f} seconds")

    # Time Numba implementation (includes compilation time)
    start_time = time.time()
    minmod(a, b, result)
    numba_time = time.time() - start_time
    print(f"Numba Time (first call, includes compilation): {numba_time:.4f} seconds")

    # Time Numba implementation again (no compilation)
    start_time = time.time()
    minmod(a, b, result)
    numba_time = time.time() - start_time
    print(f"Numba Time (second call): {numba_time:.4f} seconds")

    # Verify correctness
    limited_slope_nb = result
    assert np.allclose(limited_slope_np, limited_slope_nb)
    print("Minmod limiter implemented correctly.")
    pass


if __name__ == "__main__":
    main()
