""" This module is responsible for creating the left and right state to pass to the riemann solver."""

import numpy as np
from typing import Callable, List
from atmpy.flux.utility import directional_indices, direction_mapping
from atmpy.physics.eos import EOS
from atmpy.variables.variables import Variables
from atmpy.flux.reconstruction_utility import (
    calculate_amplitudes,
    calculate_slopes,
    calculate_variable_differences,
)
from atmpy.data.enums import PrimitiveVariableIndices as PVI, VariableIndices as VI


def modified_muscl(
    variables: Variables,
    iflux: np.ndarray,
    eos: EOS,
    limiter: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lmbda: float,
    direction: str,
):
    """Compute the MUSCL scheme in the form specified in BK19 Paper

    Parameters
    ---------
    variables : Variables
        Variables object
    iflux : np.ndarray of shape (nx, [ny], [nz])
        The list of unphysical flux values [Pu, [Pv], [Pw]]
    eos : EOS
        EOS object
    limiter : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The limiter function for slope limiting
    lmbda : float
        The ration of delta_t to delta_x (given as constant here)
    direction : str
        Direction of the flux calculation. Should be "x", "y" or "z".
    """
    cell_vars = variables.cell_vars
    variables.to_primitive(eos)
    primitives = variables.primitives
    ndim = variables.ndim
    lefts_idx, rights_idx, directional_inner_idx, inner_idx = directional_indices(
        2, direction
    )
    # Here we need the slices for only one variable not the whole variable attribute
    # Therefore we dont need the slices corresponding to the number of dimension (last entry of indices)
    inner_idx = inner_idx[:-1]
    lefts_idx = lefts_idx[:-1]
    rights_idx = rights_idx[:-1]

    Pu = iflux
    speed = np.zeros_like(cell_vars[..., VI.RHOY])
    speed[inner_idx] = (
        0.5 * (Pu[lefts_idx] + Pu[rights_idx]) / cell_vars[..., VI.RHOY][inner_idx]
    ) # This is basically ((Pu)[i-1/2] + (Pu)[i+1/2])/(P[i]/2)

    diffs = calculate_variable_differences(primitives, ndim, direction)
    slopes = calculate_slopes(diffs, direction, limiter, ndim)
    amplitudes = calculate_amplitudes(slopes, speed, lmbda, left=True)
    lefts = primitives + amplitudes
    amplitudes = calculate_amplitudes(slopes, speed, lmbda, left=False)
    rights = primitives + amplitudes

    # TODO: Write the "to_conservative" method in Variables class and use this here to calculate the conservative
    #       variables too.
    return lefts, rights


def piecewise_constant():
    pass


def muscl():
    pass
