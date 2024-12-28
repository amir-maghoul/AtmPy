""" This module contains function for different ways of flux reconstruction for different orders of accuracy."""
import numpy as np
from numba import njit
from atmpy.flux.limiters import *

def piecewise_constant():
    """ First order Gudonov scheme."""

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

    for i in range(1, n-1):
        # slopes
        slope_left = (cell_values[i]   - cell_values[i-1]) / dx
        slope_right = (cell_values[i+1] - cell_values[i])  / dx

        # slope-limited
        slope = np.zeros_like(slope_left)
        for k in range(slope_left.shape[-1]) if slope_left.ndim > 0 else [0]:
            slope[k] = limiter(slope_left[k], slope_right[k])

        # Reconstructed states
        left_states[i+1]  = cell_values[i] + 0.5*dx*slope
        right_states[i]   = cell_values[i] - 0.5*dx*slope

    return left_states, right_states
