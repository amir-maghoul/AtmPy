""" This module is responsible for creating the left and right state to pass to the riemann solver."""

import numpy as np
from typing import Callable
from atmpy.flux.utility import directional_indices
from atmpy.physics.eos import EOS
from atmpy.variables.variables import Variables


def calculate_differential_variables(
    variables: Variables,
    eos: EOS,
    direction_str: str,
):
    """Calculate the difference of primitive variables in the given direction and store it in an array with the same
    shape as the variable array."""
    cell_vars = variables.cell_vars
    direction_map = {"x": 0, "y": 1, "z": 2}
    direction = direction_map[direction_str]

    primitives = variables.primitives(eos)

    diffs = np.zeros_like(cell_vars)
    diff_values = np.diff(primitives, axis=direction)

    # Create a slicer as a mask
    slicer = [slice(None)] * variables.ndim
    slicer[direction] = slice(0, -1)
    diffs[tuple(slicer)] = diff_values

    return diffs


def calculate_slopes(
    diffs: np.ndarray,
    direction_str: str,
    limiter: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ndim: int,
):
    """Calculate the slopes of the variables from their given difference array using the left and right values

    Parameters
    ----------
    diffs : np.ndarray of shape (nx, [ny], [nz], num_vars)
        Array of differences of the primitive variables. It should be calculated using the function
        :py:func:`atmpy.flux.reconstruction.calculate_differential_variables`
    direction_str : str
        Direction in which the slopes and the flow are calculated.
    limiter : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The flux slope limiter passed as a function.
    ndim : int
        The spatial dimension of the variables.

    Returns
    -------
    np.ndarray of shape (nx, [ny], [nz], num_vars)
        The slopes of the primitive variables at interfaces
    """
    left_idx, right_idx, _, inner_idx = directional_indices(ndim, direction_str)
    # Use twice indexing: once to eliminate the extra zero due to the size difference between vars and differences
    # (differences should have one less element) and once to obtain the left values
    left_variable_slopes = diffs[left_idx][left_idx]
    right_variable_slopes = diffs[right_idx][right_idx]

    pass
