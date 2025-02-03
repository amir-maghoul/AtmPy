""" This module is responsible for creating the left and right state to pass to the riemann solver."""

import numpy as np
from typing import Callable
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
    eos : EOS
        EOS object
    limiter : Callable[[np.ndarray, np.ndarray], np.ndarray]
        The limiter function for slope limiting
    direction : str
        Direction of the flux calculation. Should be "x", "y" or "z".
    """
    cell_vars = variables.cell_vars
    primitives = variables.to_primitive(eos)
    ndim = variables.ndim
    lefts_idx, rights_idx, directional_inner_idx, inner_idx = direction_mapping(
        direction
    )
    speed = np.zeros_like(cell_vars[..., VI.RHOY])
    speed[inner_idx] = (
        0.5
        * (
            cell_vars[..., VI.RHOY][inner_idx][lefts_idx]
            + cell_vars[..., VI.RHOY][inner_idx][rights_idx]
        )
        / cell_vars[..., VI.RHOY][inner_idx]
    )
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
