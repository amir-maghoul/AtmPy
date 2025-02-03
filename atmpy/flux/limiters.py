""" This module contains slope limiter functions for the flux reconstruction."""

from numba import njit, prange
import numpy as np
import time


def minmod(a: np.ndarray, b: np.ndarray):
    """Minmod flux slope limiter using numpy vectorization"""
    result = np.zeros_like(a)
    positive_mask = (a * b) > 0
    result[positive_mask] = a[positive_mask]

    result[positive_mask] = np.sign(a[positive_mask]) * np.minimum(
        np.abs(a[positive_mask]), np.abs(b[positive_mask])
    )
    return result


def van_leer(a: np.ndarray, b: np.ndarray):
    """Van Leer flux limiter using numpy vectorization"""
    result = np.zeros_like(a)
    same_sign_mask = (a * b) > 0

    a_masked = a[same_sign_mask]
    b_masked = b[same_sign_mask]
    result[same_sign_mask] = (2 * a_masked * b_masked) / (a_masked + b_masked)

    return result


def mc_limiter(a, b):
    """monotonized central-difference limiter."""


def superbee(a, b):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
