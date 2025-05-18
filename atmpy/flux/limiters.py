"""This module contains slope limiter functions for the flux reconstruction."""

import numpy as np
from numba import njit, prange


# def average(a: np.ndarray, b: np.ndarray):
#     """ Averaging as the limiter"""
#     return 0.5 * (a + b)
#
# def van_leer(a: np.ndarray, b: np.ndarray):
#     """Van Leer flux limiter using numpy vectorization"""
#     result = np.zeros_like(a)
#     same_sign_mask = (a * b) > 0
#
#     a_masked = a[same_sign_mask]
#     b_masked = b[same_sign_mask]
#     result[same_sign_mask] = (2 * a_masked * b_masked) / (a_masked + b_masked)
#
#     return result

def minmod(a: np.ndarray, b: np.ndarray):
    """Minmod flux slope limiter using numpy vectorization.
    This limiter is the most diffusive but ensures monotonicity.
    """
    result = np.zeros_like(a)
    positive_mask = (a * b) > 0

    # Apply minmod formula only where a and b have the same sign
    result[positive_mask] = np.sign(a[positive_mask]) * np.minimum(
        np.abs(a[positive_mask]), np.abs(b[positive_mask])
    )
    return result


def average(a: np.ndarray, b: np.ndarray):
    """Simple averaging as the limiter.
    Note: This is not TVD and should be used with caution.
    """
    return 0.5 * (a + b)


def van_leer(a: np.ndarray, b: np.ndarray):
    """Van Leer flux limiter using numpy vectorization.
    Less diffusive than minmod but still ensures monotonicity.
    """
    result = np.zeros_like(a)
    same_sign_mask = (a * b) > 0

    # Apply van Leer formula only where a and b have the same sign
    a_masked = a[same_sign_mask]
    b_masked = b[same_sign_mask]

    # Avoid division by zero
    denominator = a_masked + b_masked
    nonzero_mask = np.abs(denominator) > 1e-10

    # Apply formula only where denominator is not close to zero
    result_masked = np.zeros_like(a_masked)
    result_masked[nonzero_mask] = (2 * a_masked[nonzero_mask] * b_masked[nonzero_mask]) / denominator[nonzero_mask]

    # Assign to result where a and b have same sign
    result[same_sign_mask] = result_masked

    return result


def mc_limiter(a: np.ndarray, b: np.ndarray):
    """Monotonized central-difference limiter.
    A good balance between accuracy and monotonicity, often better for vortices.
    """
    result = np.zeros_like(a)
    same_sign_mask = (a * b) > 0

    # Apply MC limiter only where a and b have the same sign
    a_masked = a[same_sign_mask]
    b_masked = b[same_sign_mask]

    # MC limiter formula: minmod(2*min(|a|,|b|)*sign(a), 0.5*(a+b))
    avg = 0.5 * (a_masked + b_masked)
    twice_min = 2 * np.minimum(np.abs(a_masked), np.abs(b_masked)) * np.sign(a_masked)

    # Apply minmod to these two values
    result[same_sign_mask] = np.sign(avg) * np.minimum(np.abs(avg), np.abs(twice_min))

    return result


def superbee(a: np.ndarray, b: np.ndarray):
    """Superbee limiter by Roe.
    Less diffusive than most limiters, good for preserving contact discontinuities.
    """
    result = np.zeros_like(a)
    same_sign_mask = (a * b) > 0

    if not np.any(same_sign_mask):
        return result

    a_masked = a[same_sign_mask]
    b_masked = b[same_sign_mask]

    # First candidate: minmod(a, 2*b)
    first = np.sign(a_masked) * np.minimum(np.abs(a_masked), 2 * np.abs(b_masked))

    # Second candidate: minmod(2*a, b)
    second = np.sign(b_masked) * np.minimum(2 * np.abs(a_masked), np.abs(b_masked))

    # Superbee takes the maximum by magnitude
    result[same_sign_mask] = np.where(
        np.abs(first) > np.abs(second),
        first,
        second
    )

    return result


def ospre(a: np.ndarray, b: np.ndarray):
    """OSPRE limiter - a smooth limiter that works well for vortical flows.
    Combines properties of van Leer and harmonic limiters.
    """
    result = np.zeros_like(a)
    same_sign_mask = (a * b) > 0

    a_masked = a[same_sign_mask]
    b_masked = b[same_sign_mask]

    # Avoid division by zero
    r = np.ones_like(a_masked)
    nonzero_mask = np.abs(b_masked) > 1e-10
    r[nonzero_mask] = a_masked[nonzero_mask] / b_masked[nonzero_mask]

    # OSPRE formula: 1.5*(r^2 + r)/(r^2 + r + 1)
    r2 = r * r
    result[same_sign_mask] = 1.5 * (r2 + r) / (r2 + r + 1.0) * b_masked

    return result


def koren(a: np.ndarray, b: np.ndarray):
    """Koren's limiter - a third-order accurate limiter.
    Good for vortex preservation but might be slightly less robust.
    """
    result = np.zeros_like(a)
    same_sign_mask = (a * b) > 0

    a_masked = a[same_sign_mask]
    b_masked = b[same_sign_mask]

    # Avoid division by zero
    r = np.ones_like(a_masked)
    nonzero_mask = np.abs(b_masked) > 1e-10
    r[nonzero_mask] = a_masked[nonzero_mask] / b_masked[nonzero_mask]

    # Koren's formula: max(0, min(2*r, min((1+2*r)/3, 2)))
    term1 = 2 * r
    term2 = (1.0 + 2.0 * r) / 3.0
    term3 = np.full_like(r, 2.0)

    inner_min = np.minimum(term2, term3)
    outer_min = np.minimum(term1, inner_min)
    result[same_sign_mask] = np.maximum(0.0, outer_min) * b_masked

    return result


def mc_limiter(a, b):
    """monotonized central-difference limiter."""
    pass


def superbee(a, b):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
