from typing import Tuple, List
import numpy as np
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)
from atmpy.infrastructure.utility import directional_indices


def modified_hll(
    left_state: np.ndarray,
    right_state: np.ndarray,
    *args,
    **kwargs,
):
    """Computes the flux using a modified HLL solver from the left and right states.

    Parameters
    ----------
    left_state: np.ndarray
        The array of primitive variables for the left state
    right_state: np.ndarray
        The array of primitive variables for the right state.

    Notes
    -----
    To have a uniform signature for all the riemann solvers, only the left and right states are passed as
    positional arguments. In modified HLL, the order of other arguments are as follows:

    flux = arg[0]
    direction = arg[1]
    ndim = arg[2]
    """

    flux, direction, ndim = args

    # Get left, right and directional inner slices for a single variable
    left_idx, right_idx, directional_inner_idx = directional_indices(
        ndim, direction, full=False
    )

    # Compute the Pu = rhoTheta*u. u example placeholder for all velocities
    Pu = flux[direction][..., VI.RHOY]

    # Compute the upwind factor based on the sign of Pu
    upwind = 0.5 * (1.0 + np.sign(Pu))
    upl = upwind[right_idx]
    upr = 1.0 - upwind[left_idx]

    # Compute U = sigma U- + (1-sigma) U+:
    # -------------------------------------
    # -------------------------------------

    # Compute the ADVECTING flux: Pu/Theta = rho*Theta*u/Theta = rho*u
    left_factor = (
        Pu[directional_inner_idx] * upl[left_idx] / left_state[left_idx + (PVI.Y,)]
    )
    right_factor = (
        Pu[directional_inner_idx] * upr[right_idx] / right_state[right_idx + (PVI.Y,)]
    )

    #  Compute the ADVECTED values
    # Exclude VI.RHO and VI.RHOY from other variables
    variable_indices = slice(2, None)

    # Compute the advected values. Rho separately
    flux[direction][directional_inner_idx + (VI.RHO,)] = left_factor + right_factor
    flux[direction][directional_inner_idx + (variable_indices,)] = (
        left_factor[..., np.newaxis] * left_state[left_idx + (variable_indices,)]
        + right_factor[..., np.newaxis] * right_state[right_idx + (variable_indices,)]
    )


def roe(left_state: np.ndarray, right_state: np.ndarray, *args, **kwargs):
    raise NotImplementedError(f"Roe solver for direction not implemented.")


def hll(left_state: np.ndarray, right_state: np.ndarray, *args, **kwargs):
    pass


def hllc(left_state: np.ndarray, right_state: np.ndarray, *args, **kwargs):
    raise NotImplementedError(f"HLLC solver for direction not implemented.")


def rusanov(left_state: np.ndarray, right_state: np.ndarray, *args, **kwargs):
    raise NotImplementedError(f"Rusanov solver for direction not implemented.")
