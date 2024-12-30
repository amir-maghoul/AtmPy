""" This module contains function for different ways of flux reconstruction for different orders of accuracy."""

import numpy as np
from numba import njit
from atmpy.flux.limiters import *
from typing import Tuple


def piecewise_constant():
    """First order Gudonov scheme."""
    pass


def muscl():
    pass


@njit
def muscl_reconstruct(cell_values, dx, limiter=minmod):
    """
    A simple 1D MUSCL reconstruction (with minmod) example. This will be used for all dimensions since the project
    solver is based on dimensional splitting.

    Parameters
    ----------
    cell_values : np.ndarray
        Array of shape (nx,) or (nx, num_vars) holding cell-average values.
    dx : float
        Cell size in the x-direction.

    Returns
    -------
    left_states, right_states : np.ndarray, np.ndarray
        The reconstructed left and right states at each cell interface.
        Both arrays typically shape: (nx+1,) or (nx+1, num_vars).
    """
    n = cell_values.shape[0]
    left_states = np.zeros_like(cell_values)
    right_states = np.zeros_like(cell_values)

    for i in range(1, n - 1):
        # slopes
        slope_left = (cell_values[i] - cell_values[i - 1]) / dx
        slope_right = (cell_values[i + 1] - cell_values[i]) / dx

        # slope-limited
        slope = np.zeros_like(slope_left)
        for k in range(slope_left.shape[-1]) if slope_left.ndim > 0 else [0]:
            slope[k] = limiter(slope_left[k], slope_right[k])

        # Reconstructed states
        left_states[i + 1] = cell_values[i] + 0.5 * dx * slope
        right_states[i] = cell_values[i] - 0.5 * dx * slope

    return left_states, right_states


# flux/reconstruction.py

from numba import njit
import numpy as np
from .slope_limiters import minmod


@njit
def compute_slopes(
    qm: np.ndarray, qL: np.ndarray, qR: np.ndarray, limiter="minmod"
) -> np.ndarray:
    """
    Compute slopes using the specified limiter.

    Parameters
    ----------
    qm : np.ndarray
        Centered variable array.
    qL : np.ndarray
        Left variable array.
    qR : np.ndarray
        Right variable array.
    limiter : str, optional
        Type of slope limiter to use. Currently only 'minmod' is implemented.

    Returns
    -------
    np.ndarray
        Array of limited slopes.
    """
    slopes = np.zeros_like(qm)
    for i in range(qm.size):
        delta_left = qm[i] - qL[i]
        delta_right = qR[i] - qm[i]
        if limiter == "minmod":
            slopes[i] = minmod(delta_left, delta_right)
        else:
            slopes[i] = 0.0  # Default to zero if limiter not recognized
    return slopes


@njit
def MUSCL_reconstruction(
    qm: np.ndarray, qL: np.ndarray, qR: np.ndarray, dx: float, dt: float, Pu: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform MUSCL reconstruction with the additional C factor.

    Parameters
    ----------
    qm : np.ndarray
        Centered variable array.
    qL : np.ndarray
        Left variable array.
    qR : np.ndarray
        Right variable array.
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    Pu : np.ndarray
        Flux array at interfaces.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Reconstructed left and right states at each interface.
    """
    slopes = compute_slopes(qm, qL, qR, limiter="minmod")
    # Compute C factor
    C = (dt / dx) * Pu / ((qm + np.roll(qm, -1)) / 2.0)
    # Ensure C is bounded to prevent numerical issues
    C = np.minimum(np.maximum(C, 0.0), 1.0)
    # Reconstruct left and right states
    q_minus = qm + 0.5 * dx * (1.0 - C) * slopes
    q_plus = np.roll(qm, -1) + 0.5 * dx * (1.0 + C) * np.roll(slopes, -1)
    return q_minus, q_plus
