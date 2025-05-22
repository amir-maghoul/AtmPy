"""This module is responsible for creating the left and right state to pass to the riemann solver."""

import numpy as np
from typing import Callable, List, TYPE_CHECKING
from atmpy.infrastructure.utility import directional_indices, one_element_inner_slice
from atmpy.physics.eos import EOS

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
from atmpy.flux.reconstruction_utility import (
    calculate_amplitudes,
    calculate_slopes,
    calculate_variable_differences,
)
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)


def modified_muscl(
    variables: "Variables",
    flux: dict[str, np.ndarray],
    eos: EOS,
    limiter: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lmbda: float,
    direction: str,
):
    """Compute the MUSCL scheme in the form specified in BK19 Paper

    Parameters
    ---------
    variables : Variables
        Object of the variables container.
    flux : dict[str, np.ndarray]
        The Pu = rhoYu as calculated in the flux container
    eos : EOS
        Object of the eos container.
    limiter : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The limiter function for slope limiting
    lmbda : float
        The ration of delta_t to delta_x (given as constant here)
    direction : str
        Direction of the flux calculation. Should be "x", "y" or "z".
    """
    cell_vars = variables.cell_vars
    ndim = variables.ndim

    # Compute the primitive variables
    variables.to_primitive(eos)
    primitives = variables.primitives

    # Left and right indices for single variables
    lefts_idx, rights_idx, directional_inner_idx = directional_indices(
        ndim, direction, full=False
    )
    inner_idx = one_element_inner_slice(ndim, full=False)

    # Unphysical Pressure
    Pu = flux[direction][inner_idx + (VI.RHOY,)]

    # Compute flow speed
    speed = np.zeros_like(cell_vars[..., VI.RHOY])
    speed[inner_idx] = (
        0.5 * (Pu[lefts_idx] + Pu[rights_idx]) / cell_vars[inner_idx + (VI.RHOY,)]
    )  # This is basically ((Pu)[i-1/2] + (Pu)[i+1/2])/(P[i]/2)

    # Compute variable differences (for slope) and slope at interfaces
    diffs = calculate_variable_differences(primitives, ndim, direction)
    slopes = calculate_slopes(diffs, direction, limiter, ndim)

    indices = [PVI.U, PVI.V, PVI.W, PVI.X]

    # left amplitudes and slope
    amplitudes = calculate_amplitudes(slopes, speed, lmbda, left=True)
    lefts = np.zeros_like(primitives)
    # for index in indices:
    #     lefts[..., index] = primitives[..., index] + amplitudes[..., index]
    #
    # lefts[..., PVI.Y] = 1.0 / ((1.0 / primitives[..., PVI.Y]) + amplitudes[..., PVI.Y])

    lefts[..., 1:] = primitives[..., 1:] + amplitudes

    # right amplitudes and slope
    amplitudes = calculate_amplitudes(slopes, speed, lmbda, left=False)
    rights = np.zeros_like(primitives)
    # for index in indices:
    #     rights[..., index] = primitives[..., index] + amplitudes[..., index]
    #
    # rights[..., PVI.Y] = 1.0 / ((1.0 / primitives[..., PVI.Y]) + amplitudes[..., PVI.Y])
    rights[..., 1:] = primitives[..., 1:] + amplitudes
    return lefts, rights


def piecewise_constant():
    pass


def muscl():
    pass
