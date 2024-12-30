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


def main():
    pass


if __name__ == "__main__":
    main()
